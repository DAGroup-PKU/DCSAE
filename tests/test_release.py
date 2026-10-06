"""Offline CPU checks. No pretrained downloads or private data required."""
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'dc-sae'), str(ROOT)]
import numpy as np
import torch
import yaml
from PIL import Image
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel
from config_utils import load_config
from train_sae import build_model
from checkpoint_utils import load_sae_checkpoint
from models.rae.utils.metrics_utils import calculate_batch_psnr
from models.rae.utils.vae_utils import LatentNormalizer


class ReleaseTests(unittest.TestCase):
    def test_entrypoint_imports(self):
        for name in ['train_sae', 'train_dit',
                     'compute_latent_stats', 'eval_sae', 'eval_gfid', 'data.imagenet_wds']:
            importlib.import_module(name)

    def test_psnr_averages_per_image(self):
        x = torch.zeros(2, 3, 4, 4)
        y = torch.stack([torch.full((3,4,4), .2), torch.full((3,4,4), .4)])
        total, n = calculate_batch_psnr(x, y)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(total/n, (20 + 10*math.log10(25))/2, places=4)

    def test_normalization_roundtrip(self):
        x = torch.randn(2, 3, 4, 4)
        stats = dict(mean=torch.tensor([1.,2.,3.]), std=torch.tensor([.5,1.,2.]), shift_factor=2., scale_factor=1.)
        for per_channel in [False, True]:
            norm = LatentNormalizer(stats, per_channel=per_channel)
            self.assertTrue(torch.allclose(norm.denormalize(norm.normalize(x)), x, atol=1e-5))

    def test_checkpoint_rejects_missing_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'model.pt'
            model = torch.nn.Linear(3, 2)
            torch.save({'model': {'bias': model.bias}}, p)
            with self.assertRaises(RuntimeError):
                load_sae_checkpoint(model, str(p))
            state = dict(model.state_dict());state['_lpips_loss_fn.unused'] = torch.zeros(1)
            torch.save({'model': state}, p)
            load_sae_checkpoint(model, str(p))

    def test_demerger_pos_embed_loads_strictly(self):
        from modules.demerger import SpatialDeMerger
        make = lambda: SpatialDeMerger(input_dim=16, num_output_tokens=16, num_transformer_layers=1,
                                       expand_size=2, nhead=4).eval()
        torch.manual_seed(0); trained = make()
        torch.manual_seed(1); fresh = make()
        self.assertIn('_pos_embed', dict(fresh.named_parameters()))
        fresh.load_state_dict(trained.state_dict(), strict=True)
        x = torch.randn(2, 4, 16)
        with torch.no_grad():
            self.assertTrue(torch.equal(fresh(x), trained(x)))
            with self.assertRaises(ValueError):
                fresh(torch.randn(2, 9, 16))

    def test_hf_encoder_patch_size_with_demerger(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            Dinov2WithRegistersModel(Dinov2WithRegistersConfig(
                hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                image_size=28, patch_size=14, num_register_tokens=2,
                intermediate_size=32)).save_pretrained(tmp/'encoder')
            x = torch.rand(1, 3, 32, 32)*2 - 1
            found = {}
            for hf_patch in (None, 16):
                config = {
                    'data': {'image_size':32},
                    'encoder': {'type':'dinov2', 'dinov2_model_name':str(tmp/'encoder')},
                    'model': {'decoder_type':'vit_decoder','hidden_size':16,'hidden_size_x':16,
                        'vit_decoder_hidden_size':16,'vit_decoder_num_layers':1,
                        'vit_decoder_num_heads':2,'vit_decoder_intermediate_size':32,
                        'hf_dim':32,'hf_encoder_patch_size':hf_patch,
                        'enable_de_merger':True,'de_merger_nhead':2},
                }
                config_path = tmp/f'sae_{hf_patch}.yaml';config_path.write_text(yaml.safe_dump(config))
                model = build_model(load_config(str(config_path)), torch.device('cpu')).eval()
                self.assertEqual(model.decode_patch_size, 8)
                with torch.no_grad():
                    found[hf_patch] = (model.hf_encoder.patch_size, tuple(model.hf_encoder(x).shape[-2:]))
                    self.assertEqual(tuple(model(x).sample.shape), (1,3,32,32))
            # Default follows the DeMerger's decode patch; an explicit size keeps the 2x2 latent grid.
            self.assertEqual(found[None], (8,(4,4)))
            self.assertEqual(found[16], (16,(2,2)))

    def test_sae_eval_cpu_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            encoder = tmp/'encoder'
            Dinov2WithRegistersModel(Dinov2WithRegistersConfig(
                hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                image_size=28, patch_size=14, num_register_tokens=2,
                intermediate_size=32)).save_pretrained(encoder)
            config = {
                'data': {'image_size':32},
                'encoder': {'type':'dinov2', 'dinov2_model_name':str(encoder)},
                'model': {'decoder_type':'vit_decoder','hidden_size':16,'hidden_size_x':16,
                    'vit_decoder_hidden_size':16,'vit_decoder_num_layers':1,
                    'vit_decoder_num_heads':2,'vit_decoder_intermediate_size':32,
                    'enable_hf_branch':False,'hf_dim':0},
            }
            config_path = tmp/'sae.yaml';config_path.write_text(yaml.safe_dump(config))
            model = build_model(load_config(str(config_path)), torch.device('cpu'))
            ckpt = tmp/'sae.pt';torch.save({'model':model.state_dict()}, ckpt)
            data = tmp/'images'/'class0';data.mkdir(parents=True)
            rng = np.random.default_rng(42)
            for i in range(3):
                Image.fromarray(rng.integers(0,256,(37,41,3),dtype=np.uint8)).save(data/f'{i}.png')
            output = tmp/'eval'
            cmd = [sys.executable, str(ROOT/'dc-sae/eval_sae.py'), '--config',str(config_path),
                   '--ckpt',str(ckpt),'--data-path',str(data.parent),'--output-dir',str(output),
                   '--device','cpu','--batch-size','2','--num-workers','0','--skip-fid']
            subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True, text=True)
            metrics = json.loads((output/'metrics.json').read_text())
            self.assertEqual(metrics['num_images'],3)
            self.assertTrue(math.isfinite(metrics['psnr_db']))
            retry = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode,0)
            self.assertIn('new/empty',retry.stderr)

    def test_generation_sampler_cpu(self):
        from models.rae.stage2.models.DDT import DiTwDDTHead
        from eval_gfid import sample_latent
        model = DiTwDDTHead(input_size=2, patch_size=1, in_channels=16,
                           hidden_size=[32,32], depth=[1,1], num_heads=[4,4],
                           num_classes=2, class_dropout_prob=.1)
        sample = sample_latent(model, batch_size=2, latent_shape=(16,2,2),
                               device=torch.device('cpu'), y=torch.tensor([0,1]),
                               time_shift=1., steps=2, use_cfg=True, null_class=2)
        self.assertEqual(tuple(sample.shape),(2,16,2,2))
        self.assertTrue(torch.isfinite(sample).all())

    def test_training_entrypoints(self):
        for script in ['train_sae.py', 'train_dit.py']:
            result = subprocess.run(
                [sys.executable, str(ROOT/'dc-sae'/script), '--help'],
                cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--config', result.stdout)
            if script == 'train_dit.py':
                self.assertIn('--latent-stats-path', result.stdout)
                self.assertIn('--fid-ref-path', result.stdout)


if __name__ == '__main__':
    unittest.main()
