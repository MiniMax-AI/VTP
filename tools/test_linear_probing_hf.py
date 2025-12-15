#!/usr/bin/env python3
"""
Linear probing evaluation script for VTP HuggingFace model.

Usage:
    # Single GPU
    python tools/test_linear_probing_hf.py \
        --model_path pretrained/vtp-l-hf \
        --imagenet_root /path/to/imagenet

    # Multi-GPU with DDP
    torchrun --nproc_per_node=8 tools/test_linear_probing_hf.py \
        --model_path pretrained/vtp-l-hf \
        --imagenet_root /path/to/imagenet \
        --use_ddp
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vtp.models.vtp_hf import VTPConfig, VTPModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("linear_probing_hf")


# ============================================================================
# Constants
# ============================================================================

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
CROP_SIZE = 224
RESIZE_SIZE = 256

DEFAULT_LEARNING_RATES = (
    1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4,
    1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1
)


# ============================================================================
# Distributed utilities
# ============================================================================

def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


# ============================================================================
# Transforms
# ============================================================================

def make_train_transform(crop_size: int = CROP_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def make_eval_transform(resize_size: int = RESIZE_SIZE, crop_size: int = CROP_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


# ============================================================================
# Feature Extraction Model
# ============================================================================

class FeatureExtractor(nn.Module):
    """Wrapper that extracts intermediate layer features from VTPModel."""

    def __init__(self, model: VTPModel, n_last_blocks: int, autocast_dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_dtype = autocast_dtype

    def forward(self, images: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Extract intermediate layer features.

        Returns:
            List of (patch_tokens, cls_token) tuples for each requested layer
        """
        with torch.inference_mode():
            with torch.amp.autocast(device_type='cuda', dtype=self.autocast_dtype):
                features = self.model.get_intermediate_layers_feature(
                    images, n=self.n_last_blocks, return_class_token=True
                )
        return features


# ============================================================================
# Linear Classifier
# ============================================================================

def create_linear_input(x_tokens_list, use_n_blocks: int, use_avgpool: bool) -> torch.Tensor:
    """Create input for linear classifier from intermediate features."""
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)

    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)

    return output.float()


class LinearClassifier(nn.Module):
    """Linear classifier on top of frozen features."""

    def __init__(self, out_dim: int, use_n_blocks: int, use_avgpool: bool, num_classes: int = 1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)


class AllClassifiers(nn.Module):
    """Container for multiple linear classifiers."""

    def __init__(self, classifiers_dict: Dict[str, nn.Module]):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs) -> Dict[str, torch.Tensor]:
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


# ============================================================================
# Infinite Sampler
# ============================================================================

class InfiniteSampler(Sampler):
    """Wraps another sampler to yield an infinite stream of indices."""

    def __init__(self, sampler: Sampler, shuffle: bool = True, seed: int = 0):
        self.sampler = sampler
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        while True:
            if hasattr(self.sampler, 'set_epoch'):
                self.sampler.set_epoch(self.epoch)
            yield from iter(self.sampler)
            self.epoch += 1

    def __len__(self):
        return int(1e18)  # Effectively infinite


# ============================================================================
# Training
# ============================================================================

def scale_lr(learning_rate: float, batch_size: int) -> float:
    """Scale learning rate based on batch size."""
    return learning_rate * (batch_size * get_world_size()) / 256.0


def setup_linear_classifiers(
    sample_output,
    n_last_blocks_list: Tuple[int, ...],
    learning_rates: Tuple[float, ...],
    batch_size: int,
    num_classes: int = 1000,
    device: torch.device = None,
) -> Tuple[AllClassifiers, List[Dict]]:
    """Setup linear classifiers with different configurations."""
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []

    for n in n_last_blocks_list:
        for avgpool in [True]:
            for _lr in learning_rates:
                lr = scale_lr(_lr, batch_size)
                out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                linear_classifier = LinearClassifier(
                    out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
                )
                linear_classifier = linear_classifier.to(device)
                classifier_key = f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")
                if is_main_process():
                    logger.info(f"Create linear classifier {classifier_key} with input_dim={out_dim}")
                linear_classifiers_dict[classifier_key] = linear_classifier
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)
    if dist.is_initialized():
        linear_classifiers = nn.parallel.DistributedDataParallel(
            linear_classifiers, device_ids=[get_rank() % torch.cuda.device_count()]
        )

    return linear_classifiers, optim_param_groups


def train_one_epoch(
    feature_model: nn.Module,
    linear_classifiers: AllClassifiers,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    train_loader,
    epoch: int,
    epoch_length: int,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    linear_classifiers.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, total=epoch_length, desc=f"Epoch {epoch}") if is_main_process() else train_loader

    for batch_idx, (images, labels) in enumerate(progress_bar):
        if batch_idx >= epoch_length:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        features = feature_model(images)
        outputs = linear_classifiers(features)

        losses = {f"loss_{k}": criterion(v, labels) for k, v in outputs.items()}
        loss = sum(losses.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if is_main_process() and batch_idx % 50 == 0:
            progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    feature_model: nn.Module,
    linear_classifiers: AllClassifiers,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate all classifiers and return accuracies."""
    linear_classifiers.eval()

    classifiers_dict = linear_classifiers.module.classifiers_dict if hasattr(linear_classifiers, 'module') else linear_classifiers.classifiers_dict

    correct = {k: 0 for k in classifiers_dict.keys()}
    total = 0

    progress_bar = tqdm(val_loader, desc="Evaluating") if is_main_process() else val_loader

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        features = feature_model(images)
        outputs = linear_classifiers(features)

        for k, logits in outputs.items():
            preds = logits.argmax(dim=1)
            correct[k] += (preds == labels).sum().item()

        total += labels.size(0)

    # Aggregate across processes
    if dist.is_initialized():
        # Gather correct counts and total from all processes
        total_tensor = torch.tensor([total], device=device, dtype=torch.long)
        dist.all_reduce(total_tensor)
        total = total_tensor.item()

        for k in correct:
            correct_tensor = torch.tensor([correct[k]], device=device, dtype=torch.long)
            dist.all_reduce(correct_tensor)
            correct[k] = correct_tensor.item()

    accuracies = {k: 100.0 * v / total for k, v in correct.items()}
    return accuracies


# ============================================================================
# Main
# ============================================================================

def test_linear_probing(
    model_path: str,
    imagenet_root: str,
    output_dir: str = "./linear_probing_results",
    batch_size: int = 128,
    epochs: int = 10,
    epoch_length: int = 1250,
    n_last_blocks_list: Tuple[int, ...] = (1, 4),
    learning_rates: Tuple[float, ...] = DEFAULT_LEARNING_RATES,
    precision: str = "bf16",
    use_ddp: bool = False,
    device: str = "cuda:0",
    num_workers: int = 8,
):
    """Run linear probing evaluation.

    Args:
        model_path: Path to VTP HuggingFace model directory
        imagenet_root: Path to ImageNet dataset root (with train/ and val/ subdirs)
        output_dir: Output directory for results
        batch_size: Batch size per GPU
        epochs: Number of training epochs
        epoch_length: Number of iterations per epoch
        n_last_blocks_list: Number of last blocks to use for features
        learning_rates: Learning rates to sweep
        precision: Precision for inference (fp32, fp16, bf16)
        use_ddp: Whether to use Distributed Data Parallel
        device: Device to use (ignored if use_ddp)
        num_workers: Number of dataloader workers
    """
    # Initialize DDP if needed
    if use_ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        local_rank = get_rank()
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        device = torch.device(f'cuda:{local_rank % torch.cuda.device_count()}')
    else:
        device = torch.device(device)

    if is_main_process():
        print("=" * 60)
        print("Linear Probing Evaluation (VTP HuggingFace)")
        print("=" * 60)
        print(f"Model path: {model_path}")
        print(f"ImageNet root: {imagenet_root}")
        print(f"Output dir: {output_dir}")
        print(f"Device: {device}" + (f", DDP: {get_world_size()} GPUs" if use_ddp else ""))
        print(f"Precision: {precision}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}, Epoch length: {epoch_length}")
        print()

    os.makedirs(output_dir, exist_ok=True)
    cudnn.benchmark = True

    # Load model
    if is_main_process():
        print("Loading model...")
    model = VTPModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    # Get precision dtype
    if precision in ('bf16', 'bfloat16'):
        autocast_dtype = torch.bfloat16
    elif precision in ('fp16', 'float16'):
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32

    # Create feature extractor
    n_last_blocks = max(n_last_blocks_list)
    feature_model = FeatureExtractor(model, n_last_blocks, autocast_dtype)

    # Setup datasets
    imagenet_train_path = os.path.join(imagenet_root, "train")
    imagenet_val_path = os.path.join(imagenet_root, "val")

    train_transform = make_train_transform()
    eval_transform = make_eval_transform()

    if is_main_process():
        print(f"Loading train dataset: {imagenet_train_path}")
    train_dataset = ImageFolder(root=imagenet_train_path, transform=train_transform)
    num_classes = len(train_dataset.classes)

    if is_main_process():
        print(f"Loading val dataset: {imagenet_val_path}")
        print(f"Number of classes: {num_classes}")
    val_dataset = ImageFolder(root=imagenet_val_path, transform=eval_transform)

    # Create dataloaders
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = None

    infinite_sampler = InfiniteSampler(train_sampler)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=infinite_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        sampler=val_sampler,
        shuffle=False if val_sampler else False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Get sample output for dimension calculation
    if is_main_process():
        print("\nSetting up classifiers...")
    sample_image = train_dataset[0][0].unsqueeze(0).to(device)
    sample_output = feature_model(sample_image)

    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output,
        n_last_blocks_list,
        learning_rates,
        batch_size,
        num_classes,
        device,
    )

    # Setup optimizer and scheduler
    max_iter = epochs * epoch_length
    optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    if is_main_process():
        print(f"\nStarting training for {epochs} epochs...")

    best_accuracy = 0.0
    best_classifier = ""

    for epoch in range(epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            feature_model,
            linear_classifiers,
            optimizer,
            scheduler,
            criterion,
            train_loader,
            epoch,
            epoch_length,
            device,
        )

        if is_main_process():
            logger.info(f"Epoch {epoch} - Train loss: {train_loss:.4f}")

        # Evaluate
        accuracies = evaluate(feature_model, linear_classifiers, val_loader, device)

        # Find best classifier
        current_best_acc = max(accuracies.values())
        current_best_key = max(accuracies, key=accuracies.get)

        if current_best_acc > best_accuracy:
            best_accuracy = current_best_acc
            best_classifier = current_best_key

        if is_main_process():
            logger.info(f"Epoch {epoch} - Best accuracy: {current_best_acc:.2f}% ({current_best_key})")

    # Final results
    if is_main_process():
        print("\n" + "=" * 60)
        print("Final Results:")
        print("=" * 60)
        print(f"  Best Accuracy: {best_accuracy:.2f}%")
        print(f"  Best Classifier: {best_classifier}")
        print("=" * 60)

        # Save results
        results = {
            "best_accuracy": best_accuracy,
            "best_classifier": best_classifier,
            "all_accuracies": accuracies,
        }
        results_path = os.path.join(output_dir, "linear_probing_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    # Cleanup
    if use_ddp:
        dist.destroy_process_group()

    return {"accuracy": best_accuracy, "classifier": best_classifier}


def main():
    parser = argparse.ArgumentParser(description="Linear probing evaluation for VTP HuggingFace model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to VTP HuggingFace model directory")
    parser.add_argument("--imagenet_root", type=str, required=True,
                        help="Path to ImageNet dataset root (with train/ and val/ subdirs)")
    parser.add_argument("--output_dir", type=str, default="./linear_probing_results",
                        help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--epoch_length", type=int, default=1250,
                        help="Number of iterations per epoch")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (e.g., cuda:0)")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Precision for inference")
    parser.add_argument("--use_ddp", action="store_true",
                        help="Use Distributed Data Parallel for multi-GPU")
    parser.add_argument("--local_rank", type=int, default=None,
                        help="Local rank for DDP (usually set automatically)")
    args = parser.parse_args()

    if args.use_ddp:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    test_linear_probing(
        model_path=args.model_path,
        imagenet_root=args.imagenet_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        epoch_length=args.epoch_length,
        precision=args.precision,
        use_ddp=args.use_ddp,
        device=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
