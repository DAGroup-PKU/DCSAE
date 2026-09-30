#!/usr/bin/env python3
"""Evaluate the requested 512px and 256px SAE/DiT pairs. Default: preview only."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

# Run from the repository root. Replace every your_xxxx path before --check/--run.
REPO = Path.cwd()
PAIRS = {
    '512': dict(size=512, channels=1024, grid=16, demerger=False,
        sae_config=Path('your_configs/512/sae.yaml'),
        dit_config=Path('your_configs/512/dit.yaml'),
        sae_ckpt=Path('your_weights/512/sae.pt'),
        dit_ckpt=Path('your_weights/512/dit.pt'),
        latent_stats=Path('your_weights/512/latent_stats.pt'),
        fid_ref=Path('your_fid_refs/VIRTUAL_imagenet512.npz')),
    '256': dict(size=256, channels=832, grid=8, demerger=True,
        sae_config=Path('your_configs/256/sae.yaml'),
        dit_config=Path('your_configs/256/dit.yaml'),
        sae_ckpt=Path('your_weights/256/sae.pt'),
        dit_ckpt=Path('your_weights/256/dit.pt'),
        # Must be the statistics used to TRAIN DiT, even if the decoder changes.
        latent_stats=Path('your_weights/256/latent_stats.pt'),
        fid_ref=Path('your_fid_refs/VIRTUAL_imagenet256_labeled.npz')),
}


def commands(args, pair, output):
    configs = output/'configs'
    common = ['--sae-config', str(configs/'sae.yaml'), '--sae-ckpt', str(pair['sae_ckpt'])]
    rfid = [sys.executable, str(args.repo/'dc-sae/eval_sae.py'),
        '--config', str(configs/'sae.yaml'), '--ckpt', str(pair['sae_ckpt']),
        '--data-path', str(args.val_path), '--output-dir', str(output/'rfid'),
        '--batch-size', str(args.rfid_batch_size), '--num-workers', str(args.num_workers),
        '--precision', 'fp32', '--seed', str(args.seed)]
    gfid = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        f'--nproc_per_node={args.gpus}', str(args.repo/'dc-sae/eval_gfid.py'),
        '--config', str(configs/'dit.yaml'), *common,
        '--dit-ckpt', str(pair['dit_ckpt']),
        '--latent-stats-path', str(pair['latent_stats']), '--per-channel-norm',
        '--fid-ref-path', str(pair['fid_ref']),
        '--samples-per-class', '50', '--sample-steps', str(args.sample_steps),
        '--batch-size', str(args.gfid_batch_size), '--seed', str(args.seed),
        '--output-dir', str(output/'gfid')]
    if args.cfg_scale is not None:
        gfid += ['--use-cfg', '--cfg-scale', str(args.cfg_scale)]
    return [('rfid', rfid), ('gfid', gfid)]


def validate_and_prepare(args, key, pair, output):
    import torch
    import yaml
    import numpy as np
    if not args.val_path.is_dir():
        raise FileNotFoundError(f'Validation ImageFolder directory not found: {args.val_path}')
    for field in ['sae_config','dit_config','sae_ckpt','dit_ckpt','latent_stats','fid_ref']:
        if not pair[field].is_file():
            raise FileNotFoundError(f'{key}: missing {field}: {pair[field]}')
    with np.load(pair['fid_ref'], allow_pickle=False) as ref:
        if ref['mu'].shape != (2048,) or ref['sigma'].shape != (2048,2048):
            raise ValueError('Invalid FID reference statistics')
    for name in ['eval_sae.py','eval_gfid.py']:
        if not (args.repo/'dc-sae'/name).is_file():
            raise FileNotFoundError(args.repo/'dc-sae'/name)
    if args.dinov2_model.startswith('/') and not Path(args.dinov2_model).is_dir():
        raise FileNotFoundError(args.dinov2_model)
    sae = yaml.safe_load(pair['sae_config'].read_text())
    dit = yaml.safe_load(pair['dit_config'].read_text())
    assert sae['data']['image_size'] == pair['size'], 'Unexpected image resolution'
    assert sae['encoder']['type'] == 'dinov2_deep_compression'
    assert sae['model']['hf_dim'] + 768 == pair['channels']
    assert bool(sae['model'].get('enable_de_merger',False)) == pair['demerger']
    assert dit['misc']['latent_size'] == [pair['channels'],pair['grid'],pair['grid']]
    assert dit['stage_2']['params']['in_channels'] == pair['channels']
    assert dit['stage_2']['params']['input_size'] == pair['grid']
    assert dit['misc']['per_channel_norm'] is True
    assert dit['misc']['num_classes'] == 1000
    # Actually load the tensors now; merely checking that a filename exists is insufficient.
    stats = torch.load(pair['latent_stats'], map_location='cpu', weights_only=True)
    for name in ['mean','std']:
        tensor = torch.as_tensor(stats[name])
        if tensor.shape != (pair['channels'],) or not torch.isfinite(tensor).all():
            raise ValueError(f'{key}: invalid {name} in {pair["latent_stats"]}')
    if (torch.as_tensor(stats['std']) <= 0).any():
        raise ValueError(f'{key}: nonpositive latent standard deviation')
    digest = hashlib.sha256(pair['latent_stats'].read_bytes()).hexdigest()
    print(f'CHECKED {key}px: loaded {pair["channels"]}-channel mean/std from {pair["latent_stats"]}', flush=True)
    print(f'  latent_stats sha256={digest}; per_channel_norm=true', flush=True)
    # Make evaluation-only snapshots; never edit the experiment configs.
    sae['encoder']['dinov2_model_name'] = args.dinov2_model
    sae['data']['val_path'] = str(args.val_path)
    hf = Path(sae['model']['hf_encoder_config_path'])
    hf = hf if hf.is_absolute() else args.repo/hf
    if not hf.is_file():
        raise FileNotFoundError(hf)
    sae['model']['hf_encoder_config_path'] = str(hf)
    sae.setdefault('checkpoint',{})['sae_ckpt'] = str(pair['sae_ckpt'])
    dit['misc']['latent_stats_path'] = str(pair['latent_stats'])
    dit['misc']['per_channel_norm'] = True
    (output/'configs').mkdir(parents=True, exist_ok=False)
    (output/'configs/sae.yaml').write_text(yaml.safe_dump(sae,sort_keys=False))
    (output/'configs/dit.yaml').write_text(yaml.safe_dump(dit,sort_keys=False))
    manifest = dict(pair={k:str(v) if isinstance(v,Path) else v for k,v in pair.items()},
        latent_stats_sha256=digest, per_channel_norm=True, val_path=str(args.val_path),
        reference_protocol='pytorch-fid 2048, resolution-matched OpenAI reference statistics',
        cfg_scale=args.cfg_scale, sample_steps=args.sample_steps, seed=args.seed, gpus=args.gpus,
        gfid_num_images=50000, dinov2_model=args.dinov2_model)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=REPO)
    parser.add_argument('--val-path',type=Path,required=True,help='ImageFolder validation split')
    parser.add_argument('--dinov2-model',default='your_pretrained/dinov2-with-registers-base',help='Matching DINOv2 model directory or Hugging Face ID')
    parser.add_argument('--pair',choices=['both','512','256'],default='both')
    parser.add_argument('--output-dir',type=Path,default=REPO/'eval_outputs'/('two_pairs_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--gpus',type=int,default=1,help='gFID GPU count; rFID remains single GPU')
    parser.add_argument('--rfid-batch-size',type=int,default=8)
    parser.add_argument('--gfid-batch-size',type=int,default=8)
    parser.add_argument('--num-workers',type=int,default=4)
    parser.add_argument('--sample-steps',type=int,default=50)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--cfg-scale',type=float,default=None,help='Omit for no CFG; supply e.g. 1.5 for a separately reported guided run')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--check',action='store_true',help='Load/validate stats and save config snapshots, but do not evaluate')
    modes.add_argument('--run',action='store_true',help='Validate then run rFID and 50k-sample gFID')
    args=parser.parse_args()
    if min(args.gpus,args.rfid_batch_size,args.gfid_batch_size,args.sample_steps)<=0 or args.num_workers<0:
        parser.error('Invalid batch size, GPU count, steps or workers')
    args.repo=args.repo.absolute();args.val_path=args.val_path.absolute();args.output_dir=args.output_dir.absolute()
    for pair in PAIRS.values():
        for field, value in pair.items():
            if isinstance(value, Path):
                pair[field] = value if value.is_absolute() else args.repo/value
    if not args.dinov2_model.startswith('/') and (args.repo/args.dinov2_model).is_dir():
        args.dinov2_model = str(args.repo/args.dinov2_model)
    keys=list(PAIRS) if args.pair=='both' else [args.pair]
    # Validate ALL selected pairs before spending GPU time on either one.
    for key in keys:
        output=args.output_dir/key
        if args.check or args.run:
            if output.exists():
                parser.error(f'Use a fresh output directory: {output}')
            validate_and_prepare(args,key,PAIRS[key],output)
        print(f'\n{key}px | SAE={PAIRS[key]["sae_ckpt"]} | DiT={PAIRS[key]["dit_ckpt"]}',flush=True)
        for label,cmd in commands(args,PAIRS[key],output):
            command_text = ' '.join(shlex.quote(part) for part in cmd)
            print(f'[{label}] {command_text}',flush=True)
    if not args.run:
        print('\nNo evaluation started. Add --run to execute; --check validates inputs without GPU evaluation.')
        return
    for key in keys:
        output=args.output_dir/key
        for label,cmd in commands(args,PAIRS[key],output):
            with (output/f'{label}.log').open('w') as log:
                subprocess.run(cmd,cwd=args.repo,stdout=log,stderr=subprocess.STDOUT,check=True)
        gfid=json.loads((output/'gfid/metrics.json').read_text())
        if gfid.get('latent_stats_path') != str(PAIRS[key]['latent_stats']) or not gfid.get('per_channel_norm'):
            raise RuntimeError('gFID manifest does not confirm the requested latent normalization')
        result={'rfid':json.loads((output/'rfid/metrics.json').read_text()),'gfid':gfid}
        (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(f'Completed {key}px: {output}/summary.json',flush=True)


if __name__=='__main__':
    main()
