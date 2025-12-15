import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import argparse
import os
from safetensors.torch import save_file
from datetime import datetime
import sys
from omegaconf import OmegaConf

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'LightningDiT'))
from datasets.img_latent_dataset import ImgLatentDataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.vtp_tokenizer import VTP_Tokenizer


def main(args):
    assert torch.cuda.is_available(), "Requires at least one GPU"

    try:
        dist.init_process_group("nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = rank % torch.cuda.device_count()
        seed = args.seed + rank
        if rank == 0:
            print(f"rank={rank}, seed={seed}, world_size={world_size}")
    except:
        rank, device, world_size, seed = 0, 0, 1, args.seed

    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Determine output directory based on model name
    model_name = os.path.basename(args.hf_model_path.rstrip('/'))
    output_dir = os.path.join(args.output_path, 'latents', model_name, f'imgnet{args.image_size}_norm{args.normalize_type}')
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

    # Create tokenizer
    tokenizer = VTP_Tokenizer(
        hf_model_path=args.hf_model_path,
        img_size=args.image_size,
        horizon_flip=0.0,
        fp16=args.fp16,
        normalize_type=args.normalize_type
    )

    datasets = [
        ImageFolder(args.data_path, transform=tokenizer.img_transform(p_hflip=p))
        for p in [0.0, 1.0]
    ]
    samplers = [
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed)
        for ds in datasets
    ]
    loaders = [
        DataLoader(ds, batch_size=args.batch_size, shuffle=False, sampler=s,
                   num_workers=args.num_workers, pin_memory=True, drop_last=False)
        for ds, s in zip(datasets, samplers)
    ]

    if rank == 0:
        print(f"Total data: {len(loaders[0].dataset)}")

    run_images = saved_files = 0
    latents, latents_flip, labels = [], [], []

    for batch_idx, batch_data in enumerate(zip(*loaders)):
        run_images += batch_data[0][0].shape[0]
        if run_images % 100 == 0 and rank == 0:
            print(f'{datetime.now()} processing {run_images}/{len(loaders[0].dataset)}')

        for loader_idx, (x, y) in enumerate(batch_data):
            z = tokenizer.encode_images(x).detach().cpu()
            if batch_idx == 0 and rank == 0:
                print('latent shape', z.shape, 'dtype', z.dtype)
            (latents if loader_idx == 0 else latents_flip).append(z)
            if loader_idx == 0:
                labels.append(y)

        if len(latents) == 10000 // args.batch_size:
            save_dict = {
                'latents': torch.cat(latents, dim=0).contiguous(),
                'latents_flip': torch.cat(latents_flip, dim=0).contiguous(),
                'labels': torch.cat(labels, dim=0).contiguous()
            }
            if rank == 0:
                for k, v in save_dict.items():
                    print(k, v.shape)
            save_file(save_dict, os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors'),
                     metadata={'total_size': f'{save_dict["latents"].shape[0]}',
                              'dtype': f'{save_dict["latents"].dtype}',
                              'device': f'{save_dict["latents"].device}'})
            if rank == 0:
                print(f'Saved shard {saved_files}')
            latents, latents_flip, labels = [], [], []
            saved_files += 1

    if len(latents) > 0:
        save_dict = {
            'latents': torch.cat(latents, dim=0).contiguous(),
            'latents_flip': torch.cat(latents_flip, dim=0).contiguous(),
            'labels': torch.cat(labels, dim=0).contiguous()
        }
        if rank == 0:
            for k, v in save_dict.items():
                print(k, v.shape)
        save_file(save_dict, os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors'),
                 metadata={'total_size': f'{save_dict["latents"].shape[0]}',
                          'dtype': f'{save_dict["latents"].dtype}',
                          'device': f'{save_dict["latents"].device}'})
        if rank == 0:
            print(f'Saved shard {saved_files}')

    dist.barrier()
    if rank == 0:
        ImgLatentDataset(output_dir, latent_norm=True)
        print(f"Latent stats saved to {output_dir}/latents_stats.pt")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    args.data_path = config.data.raw_image_path
    args.output_path = config.train.output_dir
    args.image_size = config.data.image_size
    args.batch_size = config.vae.per_proc_batch_size
    args.seed = 42
    args.num_workers = 8

    args.hf_model_path = config.vae.hf_model_path
    args.normalize_type = config.vae.get('normalize_type', 'imagenet')

    main(args)
