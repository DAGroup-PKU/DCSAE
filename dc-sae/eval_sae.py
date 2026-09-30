"""Evaluate paired reconstruction PSNR and reconstruction FID (rFID)."""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data-path', required=True, help='ImageFolder validation split')
    parser.add_argument('--output-dir', required=True, help='New/empty output directory')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-images', type=int, default=0, help='0 evaluates the full dataset')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-fid', action='store_true', help='Compute only PSNR; do not save images')
    parser.add_argument('--fid-batch-size', type=int, default=50)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.fid_batch_size <= 0 or args.max_images < 0 or args.num_workers < 0:
        parser.error('Batch sizes must be positive; max-images/num-workers must be nonnegative')
    if args.precision == 'bf16' and not args.device.startswith('cuda'):
        parser.error('bf16 evaluation requires a CUDA device')
    output = Path(args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('--output-dir must be new/empty to prevent stale samples affecting FID')
    return args


def main():
    args = parse_args()
    import json
    import random
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torchvision.utils import save_image
    from config_utils import load_config
    from checkpoint_utils import load_sae_checkpoint
    from train_sae import build_model
    from models.rae.utils.image_utils import center_crop_arr
    from models.rae.utils.metrics_utils import calculate_batch_psnr
    if not args.skip_fid:
        from pytorch_fid import fid_score

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(cfg, device)
    load_sae_checkpoint(model, args.ckpt)
    model.eval()
    transform = transforms.Compose([
        transforms.Lambda(lambda image: center_crop_arr(image, cfg.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    if args.max_images:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    if len(dataset) < 2 and not args.skip_fid:
        raise ValueError('FID requires at least two images')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not args.skip_fid:
        (output/'reference').mkdir()
        (output/'reconstruction').mkdir()
    psnr_sum, count = 0., 0
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.precision == 'bf16'):
                recon = model(images).sample.float().clamp(-1, 1)
            score, n = calculate_batch_psnr(recon, images.float())
            psnr_sum += score
            if not args.skip_fid:
                for i in range(n):
                    name = f'{count+i:08d}.png'
                    save_image((images[i].float()+1)/2, output/'reference'/name)
                    save_image((recon[i]+1)/2, output/'reconstruction'/name)
            count += n
    result = dict(num_images=count, psnr_db=psnr_sum/count, seed=args.seed,
                  image_size=cfg.data.image_size, precision=args.precision,
                  config=args.config, checkpoint=args.ckpt, data_path=args.data_path)
    if not args.skip_fid:
        result['rfid'] = float(fid_score.calculate_fid_given_paths(
            [str(output/'reference'), str(output/'reconstruction')],
            batch_size=args.fid_batch_size, device=device, dims=2048,
            num_workers=args.num_workers))
    (output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
