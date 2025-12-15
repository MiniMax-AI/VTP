import sys
import os
import types

os.environ['XFORMERS_DISABLED'] = '1'
os.environ['DISABLE_XFORMERS'] = '1'

mock_diffusers = types.ModuleType('diffusers')
mock_models = types.ModuleType('diffusers.models')
mock_models.AutoencoderKL = type('MockAutoencoderKL', (), {})
mock_diffusers.models = mock_models
sys.modules['diffusers'] = mock_diffusers
sys.modules['diffusers.models'] = mock_models

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'LightningDiT'))

import argparse
from train import do_train, load_config
from accelerate import Accelerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)

    train_config['vae']['model_name'] = 'vtp'

    # Load HuggingFace VTP model config
    hf_model_path = train_config['vae'].get('hf_model_path', None)
    if hf_model_path is None:
        raise ValueError("vae.hf_model_path must be specified")

    from vtp.models.vtp_hf import VTPConfig
    hf_config = VTPConfig.from_pretrained(hf_model_path)
    train_config['vae']['downsample_ratio'] = hf_config.vision_patch_size

    if accelerator.process_index == 0:
        print(f"Using VTP model: {hf_model_path}")

    do_train(train_config, accelerator)


if __name__ == "__main__":
    main()
