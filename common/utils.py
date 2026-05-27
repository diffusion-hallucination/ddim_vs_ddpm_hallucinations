import os
import random
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from omegaconf import OmegaConf

from common.config_utils import cfg_get
from common.diffusion import Diffusion


def generate_image_organizer(images, save_folder, filename, title="Generated Images"):

    # generate image organizer.

    # this supports the shared sampling and visualization utilities used across the repo.
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{filename}.png")
    
    images_norm = denormalize(images)
    num_images = images_norm.shape[0]
    num_channels = images_norm.shape[1]
    
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes = axes.flatten() if len(axes.shape) > 1 else axes
    
    for i in range(num_images):
        img = prepare_image_for_display(images_norm[i])
        if num_channels == 1:
            axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        else:
            axes[i].imshow(img)
        
        axes[i].axis('off')
 
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def prepare_image_for_display(img_tensor):

    # prepare image for display for the shared sampling and visualization utilities used across the repo.
    if img_tensor.shape[0] == 1:
        # Grayscale: (1, H, W) -> (H, W)
        return img_tensor.squeeze(0).numpy()
    elif img_tensor.shape[0] == 3:
        # RGB: (3, H, W) -> (H, W, 3)
        return img_tensor.permute(1, 2, 0).numpy()
    else:
        raise ValueError(f"Unexpected number of channels: {img_tensor.shape[0]}")


def denormalize(imgs):
    # denormalize.
    # this supports the shared sampling and visualization utilities used across the repo.
    return torch.clamp((imgs + 1.0) * 0.5, 0, 1).cpu()


def sample(model, diffusion, diffusion_args, input_type="random_noise", num_samples=8, 
                        img_size=32, noise_level=1, target_classes=None, data_root='./data', 
                        save_folder="./results", filename=None, device='cuda', save_intermediate=False, step_interval=1, callback=None, config = None):
                            # sample.
                            # this supports the shared sampling and visualization utilities used across the repo.
    model.eval()
    # Prepare visualization data based on input type
    visualization_data = {}
    
    with torch.no_grad():
        if input_type == "random_noise":
            print("device in sampling: ", device)
            # Scenario 1: Random noise -> Denoised image (2 rows)
            shape = (num_samples, 1 , img_size, img_size)
            print("shape:", shape)
            input_noise = torch.randn(shape, device=device)
            
            # #ddim
            if config is not None:
                if config.get('use_ddim', False):
                    samples = diffusion.ddim_sampling(
                        model=model,
                        shape=shape,
                        device=device,
                        ddim_steps=config.get('ddim_steps', 50),
                        ddim_eta=config.get('ddim_eta', 0.0),
                        skip_type=config.get('ddim_skip_type', 'uniform'),
                        input_noise=input_noise,
                        save_intermediate=save_intermediate,
                        step_interval=step_interval,
                        callback=callback
                    )
            # else:
            # print("here")
            samples = diffusion.construct_image(
                model=model, 
                shape=shape, 
                device=device,
                input=input_noise,
                callback=callback,
                save_intermediate=save_intermediate,  
                step_interval=step_interval      
            )
            
            visualization_data = {
                'type': 'two_row',
                'input': input_noise,
                'output': samples,
                'title': f"Random Noise → Denoised ({num_samples} samples)",
                'row_labels': ['Input (Random Noise)', 'Generated Images']
            }
            
        elif input_type == "black":
            # Scenario 2: Black -> Denoised image (2 rows)
            shape = (num_samples, 1, img_size, img_size)
            black_images = torch.full(shape, -1.0, device=device)
            
            samples = diffusion.construct_image(
                model=model,
                shape=shape,
                device=device,
                input=black_images
            )
            
            visualization_data = {
                'type': 'two_row',
                'input': black_images,
                'output': samples,
                'title': f"Black Images → Denoised ({num_samples} samples)",
                'row_labels': ['Input (Black)', 'Generated Images']
            }
            
        elif input_type == "white":
            # Scenario 3: White -> Denoised image (2 rows)
            shape = (num_samples, 1, img_size, img_size)
            white_images = torch.full(shape, 1.0, device=device)
            
            samples = diffusion.construct_image(
                model=model,
                shape=shape,
                device=device,
                input=white_images
            )
            
            visualization_data = {
                'type': 'two_row',
                'input': white_images,
                'output': samples,
                'title': f"White Images → Denoised ({num_samples} samples)",
                'row_labels': ['Input (White)', 'Generated Images']
            }
            
        elif input_type == "noisy_mnist":
            raise ValueError("noisy_mnist sampling was removed from the camera-ready Gaussian code path.")
            
        else:
            raise ValueError(f"Unknown input_type: {input_type}. "
                           f"Choose from: 'random_noise', 'black', 'white', 'noisy_mnist'")
    
    model.train()
    
    # Generate visualization
    if filename is None:
        filename = f"samples_{input_type}"
    
    visualize_samples(visualization_data, save_folder, filename)
    
    return samples

def visualize_samples(viz_data: Dict, save_folder: str, filename: str):
    # visualize samples for the shared sampling and visualization utilities used across the repo.
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{filename}.png")
    
    if viz_data['type'] == 'two_row':
        save_two_row_grid(
            viz_data['input'], 
            viz_data['output'], 
            save_path, 
            viz_data['title'],
            viz_data['row_labels']
        )
    elif viz_data['type'] == 'three_row':
        save_three_row_grid(
            viz_data['clean'],
            viz_data['noisy'], 
            viz_data['output'], 
            save_path, 
            viz_data['title'],
            viz_data['row_labels']
        )
    else:
        raise ValueError(f"Unknown visualization type: {viz_data['type']}")
    
    print(f"Saved: {save_path}")




def save_two_row_grid(input_images, output_images, save_path, title, row_labels):
    # save two row grid for the shared sampling and visualization utilities used across the repo.
    """
    Save 2-row grid: Input images (top) vs Generated images (bottom)
    Supports both grayscale and RGB images
    """
    input_norm = denormalize(input_images)
    output_norm = denormalize(output_images)
    
    num_images = min(input_norm.shape[0], output_norm.shape[0])
    num_channels = input_norm.shape[1]  # Get number of channels
    
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    if num_images == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_images):
        # First row - input
        img1 = prepare_image_for_display(input_norm[i])
        if num_channels == 1:
            axes[0, i].imshow(img1, cmap='gray', vmin=0, vmax=1)
        else:
            axes[0, i].imshow(img1)  # No cmap for RGB
        
        if i == 0:
            axes[0, i].set_ylabel(row_labels[0], fontsize=12, rotation=0, ha='right', va='center')
        axes[0, i].axis('off')
        
        # Second row - output
        img2 = prepare_image_for_display(output_norm[i])
        if num_channels == 1:
            axes[1, i].imshow(img2, cmap='gray', vmin=0, vmax=1)
        else:
            axes[1, i].imshow(img2)  # No cmap for RGB
        
        if i == 0:
            axes[1, i].set_ylabel(row_labels[1], fontsize=12, rotation=0, ha='right', va='center')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_three_row_grid(clean_images, noisy_images, output_images, save_path, title, row_labels):
    # save three row grid for the shared sampling and visualization utilities used across the repo.
    """
    Save 3-row grid: Clean (top) vs Noisy (middle) vs Denoised (bottom)
    Supports both grayscale and RGB images
    """
    clean_norm = denormalize(clean_images)
    noisy_norm = denormalize(noisy_images)
    output_norm = denormalize(output_images)
    
    num_images = min(clean_norm.shape[0], noisy_norm.shape[0], output_norm.shape[0])
    num_channels = clean_norm.shape[1]
    
    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 2, 6))
    if num_images == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(num_images):
        # First row - clean
        img1 = prepare_image_for_display(clean_norm[i])
        if num_channels == 1:
            axes[0, i].imshow(img1, cmap='gray', vmin=0, vmax=1)
        else:
            axes[0, i].imshow(img1)
        
        if i == 0:
            axes[0, i].set_ylabel(row_labels[0], fontsize=12, rotation=0, ha='right', va='center')
        axes[0, i].axis('off')
        
        # Second row - noisy
        img2 = prepare_image_for_display(noisy_norm[i])
        if num_channels == 1:
            axes[1, i].imshow(img2, cmap='gray', vmin=0, vmax=1)
        else:
            axes[1, i].imshow(img2)
        
        if i == 0:
            axes[1, i].set_ylabel(row_labels[1], fontsize=12, rotation=0, ha='right', va='center')
        axes[1, i].axis('off')
        
        # Third row - output
        img3 = prepare_image_for_display(output_norm[i])
        if num_channels == 1:
            axes[2, i].imshow(img3, cmap='gray', vmin=0, vmax=1)
        else:
            axes[2, i].imshow(img3)
        
        if i == 0:
            axes[2, i].set_ylabel(row_labels[2], fontsize=12, rotation=0, ha='right', va='center')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()



def build_model_from_args(args, *, device="cpu", schedule_metadata=None):
    # build the score model from the saved run configuration.
    args = OmegaConf.to_container(args, resolve=True) if OmegaConf.is_config(args) else args
    model_type = str(args["model_type"]).lower()
    model_channels = setup_model_channels(args)

    if model_type == "mlp":
        from common.mlp import MLP

        lambda_vals = None
        if schedule_metadata is not None:
            lambda_vals = schedule_metadata.get("lambda_vals", None)

        model = MLP(
            in_dim=model_channels,
            hidden_dim=int(args["hidden_dim"]),
            time_dim=int(args.get("time_dim", args["hidden_dim"])),
            out_dim=model_channels,
            num_blocks=int(args["num_blocks"]),
            conditioning_type=str(cfg_get(args, "mlp.conditioning_type", "timestep")),
            conditioning_strategy=str(cfg_get(args, "mlp.conditioning_strategy", "input_add")),
            input_encoding=str(cfg_get(args, "mlp.input_encoding", "raw")),
            fourier_features=int(cfg_get(args, "mlp.fourier_features", 256)),
            fourier_scale=float(cfg_get(args, "mlp.fourier_scale", 0.5)),
            fourier_seed=cfg_get(args, "mlp.fourier_seed", cfg_get(args, "seed", None)),
            include_input_identity=bool(cfg_get(args, "mlp.include_input_identity", True)),
            lambda_vals=lambda_vals,
            num_diffusion_steps=int(cfg_get(args, "timesteps", 1000)),
        ).to(device)
        return model

    raise ValueError(f"Unknown model_type: {model_type}")


def load_checkpoint(
    checkpoint_path,
    device="cpu",
    optimizer=None,
    prefer_ema=True,
):
    # Deserialize on CPU first so schedule tensors can be reconstructed
    # without mixing saved CUDA tensors with freshly created CPU tensors
    # inside Diffusion.from_checkpoint_payload(...). The rebuilt diffusion
    # object and model are moved to the device below.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    assert isinstance(args, dict), "Expected args to be a plain dict"

    diffusion = Diffusion.from_checkpoint_payload(checkpoint, args=args, device=device)
    schedule_metadata = diffusion.export_schedule_metadata()
    model = build_model_from_args(args, device=device, schedule_metadata=schedule_metadata)

    state_key = "ema_state_dict" if prefer_ema and "ema_state_dict" in checkpoint else "model_state_dict"
    state_dict = checkpoint[state_key]
    model.load_state_dict(state_dict)
    model.to(device)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "model": model,
        "optimizer": optimizer,
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step", checkpoint.get("epoch")),
        "losses": checkpoint.get("losses"),
        "diffusion": diffusion,
        "args": args,
        "checkpoint": checkpoint,
    }


# Seed Everything that's required. # 
def seed_all(seed):
    # seed all randomness used by the shared sampling and visualization utilities used across the repo.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Setup Model Channels for the specific dataset # 
def setup_model_channels(args):
    # setup model channels.
    # this supports the shared sampling and visualization utilities used across the repo.
    dataset_name = str(cfg_get(args, "dataset.name", "")).lower()
    dataset_kind = str(cfg_get(args, "dataset.kind", "")).lower()
    if 'gaussian5d' in dataset_name:
        return int(cfg_get(args, "dataset.num_dims", 2))
    elif dataset_kind == 'gaussian_mixture_2d' or 'gaussian25' in dataset_name:
        return 2
    else:
        return int(cfg_get(args, "dataset.channels", 1))
    



def sample_gaussian(model, diffusion, args, num_samples=1000, device='cuda', 
                   vis_trajectory=False, save_folder='./results', true_data=None):
                       # draw Gaussian reverse samples with the selected learned score model and sampler.
    model.eval()
    if 'gaussian5d' in args.dataset.name.lower():
        dim = args.dataset.num_dims
    elif str(args.dataset.get('kind', '')).lower() == 'gaussian_mixture_2d':
        dim = 2
    else: 
        dim = 2

        
    print(f"Dataset: {args.dataset}")
    print(f"Detected dimension: {dim}")
    
    trajectories = [] if vis_trajectory else None
    
    with torch.no_grad():
        x = torch.randn(num_samples, dim, device=device)
        print(f"Initial noise shape: {x.shape}")
        
        if vis_trajectory:
            trajectories.append(x.cpu().clone())
        
        for t_idx in reversed(range(args.timesteps)):
            t = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
            x = diffusion.denoising_step(model, x, t)
            
            if vis_trajectory:
                trajectories.append(x.cpu().clone())
                
        print(f"Final samples shape: {x.shape}")
    
    if vis_trajectory and dim == 2 and trajectories:
        visualize_trajectories(trajectories, args.timesteps, save_folder, true_data)
    
    model.train()
    return x.cpu()


def visualize_gaussian_samples(
    samples,
    save_folder,
    filename,
    title="Generated Samples",
    true_data=None,
    xlim=None,
    ylim=None,
):
    # plot Gaussian samples together with the underlying mode geometry.
    """
    Visualize 1D (histogram) or 2D (scatter) Gaussian samples.
    """
    os.makedirs(save_folder, exist_ok=True)

    if torch.is_tensor(samples):
        samples = samples.cpu().numpy()

    plt.figure(figsize=(6, 6))

    if samples.shape[1] == 1:
        # 1D histogram
        if true_data is not None:
            true_data = true_data.cpu().numpy() if torch.is_tensor(true_data) else true_data
            plt.hist(true_data.flatten(), bins=100, alpha=0.5, label="True", color="red")

        plt.hist(samples.flatten(), bins=100, alpha=0.7, label="Generated", color="blue")
        plt.yscale("log")
        plt.xlabel("x")
        plt.ylabel("Frequency")

    else:
        # 2D scatter
        if true_data is not None:
            true_data = true_data.cpu().numpy() if torch.is_tensor(true_data) else true_data
            plt.scatter(true_data[:, 0], true_data[:, 1], alpha=1.0, label="True")

        plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.7, label="Generated")
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_folder, f"{filename}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved samples to: {save_path}")
    return save_path


# Visualize Trajectories by taking
def visualize_trajectories(
    trajectories,
    timesteps,
    save_folder,
    true_data=None,
    num_display=10,
    save_name=None, 
    tau_kappa_dict = None, 
):
    # plot reverse trajectories in the normalized Gaussian geometry used in the paper.
    """
    Visualize denoising trajectories in 2D.
    """
    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    os.makedirs(save_folder, exist_ok=True)

    # Convert list of tensors to numpy array: [T, N, 2]
    trajectories = [t.cpu().numpy() if torch.is_tensor(t) else t for t in trajectories]
    traj = np.stack(trajectories)  # (T, N, 2)

    T, N, _ = traj.shape
    num_display = min(num_display, N)
    indices = np.random.choice(N, num_display, replace=False)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot true modes if provided
    if true_data is not None:
        true_data = true_data.cpu().numpy() if torch.is_tensor(true_data) else true_data
        if true_data.ndim == 3:
            true_data = true_data.squeeze(0)

        ax.scatter(
            true_data[:, 0],
            true_data[:, 1],
            s=150,
            c="red",
            marker="*",
            alpha=1.0,
            label="True Modes",
        )

    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=T - 1)
    colors = cmap(np.linspace(0, 1, T))

    for idx in indices:
        path = traj[:, idx]  # [T, 2]
        for t in range(T - 1):
            ax.plot(
                path[t:t+2, 0],
                path[t:t+2, 1],
                color=colors[t],
                alpha=0.15,   # ↓ more transparent
                linewidth=1, # optional: thinner lines help clarity
            )

        ax.scatter(
            path[0, 0],
            path[0, 1],
            c="green",
            s=60,
            marker="x",
            label="Start" if idx == indices[0] else "",
            zorder=5,
        )
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            c="blue",
            s=60,
            label="End" if idx == indices[0] else "",
            zorder=5,
        )


    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # ---- Colorbar on the side ----
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for older matplotlib
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)

    if save_name is None: 
        save_name = "denoising_trajectories.png"
    save_path = os.path.join(save_folder, save_name) 
    plt.tight_layout() 
    plt.savefig(save_path, dpi=150) 
    plt.close() 
    print(f"Saved trajectory plot to: {save_path}") 
    return save_path

    
