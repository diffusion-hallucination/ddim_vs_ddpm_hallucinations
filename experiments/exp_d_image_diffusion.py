import json
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from common.diffusion import Diffusion
from common.unet import UNet
from common.mlp import MLP
from common.utils import setup_model_channels
from skimage.measure import label, regionprops



def load_image_checkpoint(checkpoint_path, device="cpu"):
    # Load checkpoint saved by common.train.save_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    assert isinstance(args, dict), "Expected args to be a plain dict"

    # Build the model from saved args so we can load image checkpoints
    model_type = args.get("model_type", "unet").lower()
    model_channels = setup_model_channels(args)

    if model_type == "unet":
        model = UNet(
            in_channels=model_channels,
            hid_channels=args.get("hid_channels", 64),
            out_channels=model_channels,
            ch_multipliers=args.get("ch_multipliers", [1, 2, 2]),
            num_res_blocks=args.get("num_res_blocks", 2),
            apply_attn=args.get("apply_attn", [False, False, True]),
            drop_rate=args.get("drop_rate", 0.1),
        ).to(device)

    elif model_type == "mlp":
        model = MLP(
            in_dim=model_channels,
            hidden_dim=int(args.get("hidden_dim", 128)),
            time_dim=int(args.get("time_dim", 128)),
            out_dim=model_channels,
            num_blocks=int(args.get("num_blocks", 3)),
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return {
        "model": model,
        "args": args,
        "checkpoint": checkpoint,
    }


def set_seed(seed):
    # set seed for the image-domain hallucination experiments.
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sample_image_diffusion(
    model,
    diffusion: Diffusion,
    num_samples,
    img_size,
    channels,
    device="cuda",
    sampling_mode="ddpm",
    skip_type="quad",
    ddim_steps=50,
    vis_trajectory=False,
    chunk_size = 100,
    seed=0, 
):
    # sample image diffusion for the image-domain hallucination experiments.
    if chunk_size is None or chunk_size <= 0:
        chunk_size = num_samples

    final_samples = []
    trajectories = [] if vis_trajectory else None
    seq = None

    model.eval()
    for start_idx in range(0, num_samples, chunk_size):
        current_size = min(chunk_size, num_samples - start_idx)
        shape = (current_size, channels, img_size, img_size)
        x = torch.randn(shape, device=device)

        if sampling_mode == "ddpm":
            set_seed(seed)

            with torch.no_grad():
                if vis_trajectory:
                    chunk_trajectories = [x.cpu().clone()]

                for t_idx in reversed(range(diffusion.steps)):
                    t = torch.full((current_size,), t_idx, device=device, dtype=torch.long)
                    x = diffusion.denoising_step(model, x, t)

                    if vis_trajectory:
                        chunk_trajectories.append(x.cpu().clone())

            if seq is None:
                seq = list(range(diffusion.steps))

        elif sampling_mode == "ddim":
            if vis_trajectory:
                chunk_trajectories, seq = diffusion.ddim_sampling(
                    model,
                    x.shape,
                    ddim_steps=ddim_steps,
                    retrieve_all_samples=True,
                    device=device,
                    skip_type=skip_type,
                    input_noise=x,
                )
                x = chunk_trajectories[-1]
            else:
                x, seq = diffusion.ddim_sampling(
                    model,
                    x.shape,
                    ddim_steps=ddim_steps,
                    retrieve_all_samples=False,
                    device=device,
                    skip_type=skip_type,
                    input_noise=x,
                    ddim_eta=0.0,
                )
                chunk_trajectories = None
        else:
            raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

        final_samples.append(x.detach().cpu())

        if vis_trajectory:
            if trajectories is None or len(trajectories) == 0:
                trajectories = chunk_trajectories
            else:
                for idx in range(len(trajectories)):
                    trajectories[idx] = torch.cat(
                        [trajectories[idx], chunk_trajectories[idx]], dim=0
                    )

    final_sample = torch.cat(final_samples, dim=0) if final_samples else torch.empty(0)
    if len(final_samples) > 0:
        print(torch.min(final_sample), torch.max(final_sample), torch.std(final_sample))
    model.train()
    return {
        "final_sample": final_sample,
        "trajectories": trajectories,
        "seq": seq,
    }

# 
def count_triangles(binary_img, labeled_img):
    # count triangles for the image-domain hallucination experiments.
    props = regionprops(labeled_img)
    triangle_count = 0
    
    for prop in props:
        if prop.area < 3:  
            continue
        
        minr, minc, maxr, maxc = prop.bbox
        shape_mask = (labeled_img[minr:maxr, minc:maxc] == prop.label).astype(int)
        
        height, width = shape_mask.shape
        if height < 3 or width < 3:  
            continue
            
        row_widths = []
        for row in range(height):
            row_pixels = np.sum(shape_mask[row, :])
            if row_pixels > 0:
                row_widths.append(row_pixels)
        
        if len(row_widths) < 3:
            continue
            
        max_width = max(row_widths)
        min_width = min(row_widths)
        width_variation = max_width / min_width if min_width > 0 else 1
        
        area = prop.area
        perimeter = prop.perimeter
        
        #print(f"area: {area}, width_variation : {width_variation}, prop.extent:{prop.extent}, prop.solidity: { prop.solidity}")
        
        is_triangle = (
            # area < 25 and
            width_variation >= 3.0 and
            prop.extent <= 0.7 and
            prop.solidity > 0.5 and
            area >= 5
        )
        
        if is_triangle:
            triangle_count += 1
    
    return triangle_count


def process_tensor(tensor):
    # process tensor.
    # this supports the image-domain hallucination experiments.
    if hasattr(tensor, 'cpu'):
        tensor = tensor.cpu().numpy()
    elif hasattr(tensor, 'numpy'):
        tensor = tensor.numpy()

    tensor = np.array(tensor)
    if tensor.ndim > 2:
        tensor = tensor.squeeze()
    tensor = (tensor > 0.5).astype(np.uint8)

    return tensor

def count_shapes(tensor, dataset_type):
    # count shapes for the image-domain hallucination experiments.
    tensor = process_tensor(tensor)
    binary_image = (tensor == 1).astype(np.uint8)
    
    if dataset_type == "triangle_only":
        labeled_img = label(binary_image)
        triangle_count = count_triangles(binary_image, labeled_img)
        return {'triangles': triangle_count}
    
    else:
        height, width = binary_image.shape
        mid_point = width // 2
        
        left_half = binary_image[:, :mid_point]
        right_half = binary_image[:, mid_point:]
        
        left_labeled = label(left_half)
        right_labeled = label(right_half)
        
        triangles_left = count_triangles(left_half, left_labeled)
        circles_left = count_circles(left_half, left_labeled)
        circles_right = count_circles(right_half, right_labeled)
        triangle_right = count_triangles(right_half, right_labeled)
        
        left_ans = -1 if circles_left else triangles_left
        right_ans = -1 if triangle_right else circles_right
        

def compute_hallucination_rate(final_samples, run_name, dataset_name, dataset_path): 
    # compute hallucination rate for the image-domain hallucination experiments.
    full_path = os.path.join(dataset_path, dataset_name)
    image_metadata_path = os.path.join(full_path, "image_metadata.json")
    with open(image_metadata_path, "r") as f:
        image_metadata = json.load(f)
    triangle_counts = np.array([entry["triangles"] for entry in image_metadata])
    unique_triangles = np.unique(triangle_counts)

    if isinstance(final_samples, dict) and "final_sample" in final_samples:
        final_samples = final_samples["final_sample"]
    if torch.is_tensor(final_samples):
        final_samples = final_samples.detach().cpu()

    hallucinated_indices = []
    num_samples = len(final_samples)
    for idx, sample in enumerate(final_samples):
        shape_counts = count_shapes(sample, "triangle_only")
        triangle_count = shape_counts.get("triangles", 0)
        if triangle_count not in unique_triangles:
            hallucinated_indices.append(idx)
 
    hallucination_rate = (
        len(hallucinated_indices) / num_samples if num_samples > 0 else 0.0
    )
    print(
        f"Hallucination rate: {hallucination_rate:.4f} "
        f"({len(hallucinated_indices)}/{num_samples})"
    )
    return {
        "hallucination_rate": hallucination_rate,
        "hallucinated_indices": hallucinated_indices,
    }



def save_image_grid(samples, save_path, ncols=8, padding=2):
    # Convert from [-1, 1] to [0, 1] for visualization
    if torch.is_tensor(samples):
        samples = samples.detach().cpu()
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0).numpy()

    n, c, h, w = samples.shape
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    grid_h = nrows * h + padding * (nrows - 1)
    grid_w = ncols * w + padding * (ncols - 1)
    grid = np.ones((grid_h, grid_w, c), dtype=np.float32)

    for idx in range(n):
        row = idx // ncols
        col = idx % ncols
        y0 = row * (h + padding)
        x0 = col * (w + padding)
        img = samples[idx].transpose(1, 2, 0)
        grid[y0:y0 + h, x0:x0 + w] = img

    if c == 1:
        plt.imsave(save_path, grid.squeeze(-1), cmap="gray")
    else:
        plt.imsave(save_path, grid)

    return save_path
