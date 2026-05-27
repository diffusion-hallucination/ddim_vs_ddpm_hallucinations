import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from common.diffusion import Diffusion
from common.unet import UNet
from common.unet_new import UNet_medium
from common.train import train
from common.gaussian_data import load_gaussian_data
from common.image_data import ImageDataset
from common.utils import build_model_from_args, seed_all, setup_model_channels


def resolve_gaussian_epoch_resampling(dataset_cfg) -> tuple[bool, str]:
    # resolve whether a gaussian dataset should be resampled between epochs.
    # this supports the main training entrypoint for the paper's learned-score models.
    explicit = dataset_cfg.get("resample_each_epoch", None)
    dataset_name = str(dataset_cfg.get("name", "")).lower()
    num_dims = int(dataset_cfg.get("num_dims", 2) or 2)

    auto_enabled = dataset_name == "gaussian5d" and num_dims >= 4

    if explicit is None:
        return auto_enabled, "auto"

    if isinstance(explicit, bool):
        return explicit, "explicit"

    text = str(explicit).strip().lower()
    if text in {"", "auto", "none", "null"}:
        return auto_enabled, "auto"
    if text in {"1", "true", "yes", "y", "on"}:
        return True, "explicit"
    if text in {"0", "false", "no", "n", "off"}:
        return False, "explicit"

    raise ValueError(
        "dataset.resample_each_epoch must be one of "
        "{true, false, auto, null}"
    )


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="base_config",
)
def main(cfg: DictConfig):
    # train the learned score model and save checkpoints plus gaussian metadata for later experiments.
    """
    Hydra-based entry point for Gaussian Diffusion experiments.
    """

    print("\n=== Configuration ===")
    print(OmegaConf.to_yaml(cfg))

    if not hasattr(cfg, "dataset"):
        raise ValueError("'dataset' must be specified in config")

    # ----------------------------
    # Seeding & folders
    # ----------------------------
    seed_all(cfg.seed)
    if cfg.log.run_name != "": 
        full_save_folder = os.path.join(cfg.dataset.save_folder, cfg.log.run_name)
    else:
        full_save_folder = cfg.dataset.save_folder

    existing_final_ckpt = next((p for p in [
        os.path.join(full_save_folder, "checkpoints", "final_model.pth"),
        os.path.join(full_save_folder, "final_model.pth"),
    ] if os.path.exists(p)), None)
    if existing_final_ckpt is not None and not bool(cfg.get("force_retrain", False)):
        print(f"Found existing final checkpoint at {existing_final_ckpt}.")
        print("Skipping training. Pass force_retrain=true to retrain explicitly.")
        return

    os.makedirs(full_save_folder, exist_ok=True)
    device_str = cfg.device if hasattr(cfg, "device") else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using Device", device_str)
    device = torch.device(device_str)
    print(f"Running on device: {device}")

    # ----------------------------
    # Model channels
    # ----------------------------
    model_channels = setup_model_channels(cfg)
    print(f"Using {model_channels} channel(s) for model input/output")

    # ----------------------------
    # Diffusion
    # ----------------------------
    diffusion = Diffusion(args=cfg, device=device)

    # ----------------------------
    # Model
    # ----------------------------
    model_type = cfg.get("model_type", "mlp").lower()

    if model_type == "mlp":
        model = build_model_from_args(
            cfg,
            device=device,
            schedule_metadata=diffusion.export_schedule_metadata(),
        )
   
    elif model_type == "unet": 

        model = UNet(
            in_channels=model_channels,
            hid_channels=cfg["hid_channels"],
            out_channels=model_channels,
            ch_multipliers=cfg["ch_multipliers"],
            num_res_blocks=cfg["num_res_blocks"],
            apply_attn=cfg["apply_attn"],
            drop_rate=0.1
        ).to(device)
    
    elif model_type == "unet_med": 
        model = UNet_medium(
            image_size=cfg.dataset.img_size,
            in_channels=model_channels,
            out_channels=model_channels,
        ).to(device)
        
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # if not cfg.get("train_model_load_path"):
    print("\n=== TRAINING ===")

    if cfg.train_dataset_type=="gaussian":
        gaussian_datasets = [
            "gaussian25",
            "gaussian5d",
            "gaussian_mixture_2d",
        ]
        is_gaussian_mixture_2d = str(cfg.dataset.get("kind", "")).lower() == "gaussian_mixture_2d"

        if cfg.dataset.name.lower() not in gaussian_datasets and not is_gaussian_mixture_2d:
            raise ValueError(f"Dataset {cfg.dataset} not supported")

        mixture_spec_path = (
            os.path.join(full_save_folder, "mixture_spec.json")
            if is_gaussian_mixture_2d or str(cfg.dataset.name).lower() == "gaussian5d"
            else None
        )
        resample_each_epoch, resample_source = resolve_gaussian_epoch_resampling(cfg.dataset)
        print(
            "Gaussian epoch-wise resampling:",
            resample_each_epoch,
            f"({resample_source}; dataset={cfg.dataset.name}, num_dims={cfg.dataset.get('num_dims', None)})",
        )
        dataloader = load_gaussian_data(
            dataset_name=cfg.dataset.name,
            batch_size=cfg.batch_size,
            num_samples=cfg.dataset.get("num_samples", 100_000),
            num_dims=cfg.dataset.get("num_dims", None), 
            modes=cfg.get("modes", None),
            dataset_cfg=cfg.dataset,
            mixture_spec_path=mixture_spec_path,
            resample=resample_each_epoch,
            sample_seed=cfg.seed,
        )

        sample_batch = next(iter(dataloader))
        print(f"Sample batch shape: {sample_batch.shape}")
        print(f"Sample batch dtype: {sample_batch.dtype}")

        train(model, diffusion, dataloader, cfg, device=device)
        print("Training completed!")
    
    elif cfg.train_dataset_type=="image_dataset": 

        print(f"Loading custom dataset from {cfg.dataset.data_dir}...")

        if not hasattr(cfg.dataset, 'data_dir'):
            raise ValueError("'data_dir' must be specified in config for custom datasets")

        # if not hasattr(cfg.dataset, 'image_size'):
            # args.image_size = cfg.dataset.img_size

        # if not hasattr(cfg.dataset, 'channels'):
            # args.channels = model_channels
        
        dataset = ImageDataset(
            root_dir=cfg.dataset.data_dir,
            image_size=cfg.dataset.img_size,
            channels=cfg.dataset.channels
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=getattr(cfg, 'num_workers', 0),
            pin_memory=True if device.type == 'cuda' else False
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of batches per epoch: {len(dataloader)}")
    
         
        # Train model
        train(model, diffusion, dataloader, cfg, device=device)
        
        print("Training completed!")
        


if __name__ == "__main__":
    main()
