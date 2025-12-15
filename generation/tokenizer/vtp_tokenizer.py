import torch
from torchvision import transforms
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from vtp.utils.image_utils import center_crop_arr


# Normalization constants
NORMALIZE_HALF = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
NORMALIZE_IMAGENET = {"mean": list(IMAGENET_DEFAULT_MEAN), "std": list(IMAGENET_DEFAULT_STD)}


class VTP_Tokenizer:
    def __init__(
        self,
        hf_model_path,
        img_size=256,
        horizon_flip=0.5,
        fp16=True,
        normalize_type="imagenet"
    ):
        """Initialize VTP Tokenizer.

        Args:
            hf_model_path: Path to HuggingFace VTPModel directory
            img_size: Input image size
            horizon_flip: Horizontal flip probability for data augmentation
            fp16: Whether to use FP16 precision
            normalize_type: Normalization type, one of "half" (0.5 mean/std) or "imagenet"
        """
        self.img_size = img_size
        self.horizon_flip = horizon_flip
        self.fp16 = fp16
        self.normalize_type = normalize_type

        # Setup normalization transforms
        self._setup_normalization(normalize_type)

        # Load HuggingFace model
        from vtp.models.vtp_hf import VTPModel
        self.model = VTPModel.from_pretrained(hf_model_path)
        self.model = self.model.cuda().eval()

        config = self.model.config
        self.patch_size = config.vision_patch_size
        self.embed_dim = config.vision_feature_bottleneck

        self.downsample_ratio = self.patch_size
        self.latent_size = img_size // self.downsample_ratio

        print(f"VTP Tokenizer: patch_size={self.patch_size}, embed_dim={self.embed_dim}, "
              f"downsample_ratio={self.downsample_ratio}, latent_size={self.latent_size}, "
              f"normalize={self.normalize_type}")

    def _setup_normalization(self, normalize_type):
        """Setup normalization and inverse normalization transforms."""
        if normalize_type == "half":
            norm_cfg = NORMALIZE_HALF
        elif normalize_type == "imagenet":
            norm_cfg = NORMALIZE_IMAGENET
        else:
            raise ValueError(f"Unknown normalize_type: {normalize_type}. Use 'half' or 'imagenet'.")

        self.norm_mean = norm_cfg["mean"]
        self.norm_std = norm_cfg["std"]

        # Inverse normalization: x_orig = x_norm * std + mean
        # Which is: Normalize with mean=-mean/std, std=1/std
        inv_mean = [-m / s for m, s in zip(self.norm_mean, self.norm_std)]
        inv_std = [1.0 / s for s in self.norm_std]
        self.transform_inv = Normalize(inv_mean, inv_std)

    def img_transform(self, p_hflip=0, img_size=None):
        img_size = img_size if img_size is not None else self.img_size
        return transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, img_size)),
            transforms.RandomHorizontalFlip(p=p_hflip),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std, inplace=True)
        ])

    def encode_images(self, images):
        with torch.no_grad():
            if not images.is_cuda:
                images = images.cuda()

            B, C, H, W = images.shape
            self._current_img_h = H
            self._current_img_w = W

            latents = self.model.get_reconstruction_latents(images)
            return latents.detach().cpu()

    def decode_to_images(self, z):
        with torch.no_grad():
            if not z.is_cuda:
                z = z.cuda()

            B, C, H_latent, W_latent = z.shape
            self._current_img_h = H_latent * self.patch_size
            self._current_img_w = W_latent * self.patch_size

            decoded = self.model.get_latents_decoded_images(z)

            # Apply inverse normalization to get [0, 1] range
            decoded = self.transform_inv(decoded)
            # Convert to [0, 255] uint8
            images = torch.clamp(decoded * 255, 0, 255)
            images = images.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            return images
