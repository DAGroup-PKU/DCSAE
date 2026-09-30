"""Create pytorch-fid reference statistics from an ImageFolder split."""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--output-dir', required=True, help='New/empty output directory')
    parser.add_argument('--max-images', type=int, default=0, help='0 uses all images')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.image_size <= 0 or args.batch_size <= 0 or args.max_images < 0 or args.num_workers < 0:
        parser.error('Invalid image size, batch size, image count or workers')
    output = Path(args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('Use a new/empty output directory')
    import torch
    from torchvision.datasets import ImageFolder
    from pytorch_fid.fid_score import save_fid_stats
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.rae.utils.image_utils import center_crop_arr
    dataset = ImageFolder(args.data_path)
    n = min(args.max_images, len(dataset)) if args.max_images else len(dataset)
    if n < 2:
        parser.error('FID requires at least two reference images')
    images = output/'images'
    images.mkdir(parents=True)
    for i in range(n):
        image, _ = dataset[i]
        center_crop_arr(image, args.image_size).save(images/f'{i:08d}.png')
    stats = output/'stats.npz'
    save_fid_stats([str(images), str(stats)], args.batch_size, torch.device(args.device),
                   2048, args.num_workers)
    (output/'reference.json').write_text(json.dumps(dict(
        data_path=args.data_path, num_images=n, image_size=args.image_size,
        preprocessing='RGB, ADM center crop, PNG', backend='pytorch-fid', dims=2048), indent=2)+'\n')
    print(stats)


if __name__ == '__main__':
    main()
