import sys
import os
import importlib.util
import argparse
import logging
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
generation_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lightningdit_path = os.path.join(project_root, 'LightningDiT')

if lightningdit_path in sys.path:
    sys.path.remove(lightningdit_path)
sys.path.insert(0, lightningdit_path)

vtp_tokenizer_path = os.path.join(generation_dir, 'tokenizer', 'vtp_tokenizer.py')
project_root_in_path = project_root in sys.path
if not project_root_in_path:
    sys.path.insert(1, project_root)

spec = importlib.util.spec_from_file_location("vtp_tokenizer_module", vtp_tokenizer_path)
vtp_tokenizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vtp_tokenizer_module)
VTP_Tokenizer = vtp_tokenizer_module.VTP_Tokenizer

if not project_root_in_path and project_root in sys.path:
    sys.path.remove(project_root)

if 'models' in sys.modules:
    del sys.modules['models']
for k in [k for k in list(sys.modules.keys()) if k.startswith('models.')]:
    del sys.modules[k]

if sys.path[0] != lightningdit_path:
    if lightningdit_path in sys.path:
        sys.path.remove(lightningdit_path)
    sys.path.insert(0, lightningdit_path)

from inference import do_sample, load_config
from accelerate import Accelerator
from models.lightningdit import LightningDiT_models


def setup_logger(log_dir, rank):
    """Setup logger that outputs to both console and file (rank 0 only)."""
    logger = logging.getLogger('inference')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler only for rank 0
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f'Log file: {log_file}')

    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--demo', action='store_true', default=False)
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)

    # Setup logger - log to sample output directory
    exp_name = train_config['train']['exp_name']
    output_dir = train_config['train']['output_dir']
    log_dir = os.path.join(output_dir, exp_name, 'logs')
    logger = setup_logger(log_dir, accelerator.process_index)

    # Log sampling config
    if accelerator.process_index == 0:
        sample_cfg = train_config.get('sample', {})
        logger.info(f"Sampling config: "
            f"method={sample_cfg.get('sampling_method')}, "
            f"steps={sample_cfg.get('num_sampling_steps')}, "
            f"shift={sample_cfg.get('timestep_shift')}, "
            f"cfg={sample_cfg.get('cfg_scale')}, "
            f"fid_num={sample_cfg.get('fid_num')}, "
            f"seed={train_config.get('train', {}).get('global_seed')}"
        )

    train_config['vae']['model_name'] = 'vtp'

    # Load HuggingFace VTP model config
    hf_model_path = train_config['vae'].get('hf_model_path', None)
    if hf_model_path is None:
        raise ValueError("vae.hf_model_path must be specified")

    from vtp.models.vtp_hf import VTPConfig
    hf_config = VTPConfig.from_pretrained(hf_model_path)
    patch_size = hf_config.vision_patch_size
    in_chans = hf_config.vision_feature_bottleneck
    train_config['vae']['downsample_ratio'] = patch_size
    if accelerator.process_index == 0:
        logger.info(f'Using VTP model: {hf_model_path}')

    # Get checkpoint path from config
    ckpt_path = train_config.get('ckpt_path')
    if ckpt_path is None:
        raise ValueError("ckpt_path must be specified in config")

    if accelerator.process_index == 0:
        logger.info(f'Using ckpt: {ckpt_path}')
    
    latent_size = train_config['data']['image_size'] // train_config['vae']['downsample_ratio']
    
    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model'].get('use_swiglu', False),
        use_rope=train_config['model'].get('use_rope', False),
        use_rmsnorm=train_config['model'].get('use_rmsnorm', False),
        wo_shift=train_config['model'].get('wo_shift', False),
        in_channels=train_config['model'].get('in_chans', in_chans),
        learn_sigma=train_config['model'].get('learn_sigma', False),
    )

    # Create VTP Tokenizer
    normalize_type = train_config['vae'].get('normalize_type', 'half')
    vae = VTP_Tokenizer(
        hf_model_path=hf_model_path,
        img_size=train_config['data']['image_size'],
        normalize_type=normalize_type
    )
    
    sample_folder_dir = do_sample(train_config, accelerator, ckpt_path=ckpt_path, model=model, vae=vae, demo_sample_mode=args.demo)
    
    if not args.demo and accelerator.process_index == 0:
        # Import from LightningDiT/tools/calculate_fid.py
        fid_module_path = os.path.join(lightningdit_path, 'tools', 'calculate_fid.py')
        spec = importlib.util.spec_from_file_location("calculate_fid", fid_module_path)
        fid_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fid_module)
        calculate_fid_given_paths = fid_module.calculate_fid_given_paths
        logger.info(f'Calculating FID with {train_config["sample"]["fid_num"]} samples')
        assert 'fid_reference_file' in train_config['data'], "fid_reference_file must be specified"
        fid = calculate_fid_given_paths(
            [train_config['data']['fid_reference_file'], sample_folder_dir],
            batch_size=50,
            dims=2048,
            device='cuda',
            num_workers=8,
            sp_len=train_config['sample']['fid_num']
        )
        logger.info(f'FID: {fid}')


if __name__ == "__main__":
    main()

