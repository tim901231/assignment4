import argparse
import os
import os.path as osp
import random
import time

import imageio
import numpy as np
import torch
from nerf.config_parser import add_config_arguments
from nerf.network_grid import NeRFNetwork
from nerf.provider import NeRFDataset
from optimizer import Adan
from PIL import Image
from SDS import SDS
from utils import prepare_embeddings, seed_everything

import torch.nn.functional as F
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Q1")))
from model import Scene, Gaussians

def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]

def tv_loss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=100000):
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    return helper

def make_trainable(gaussians):

    ### YOUR CODE HERE ###
    # HINT: You can access and modify parameters from gaussians
    gaussians.means.requires_grad_(True)
    gaussians.pre_act_scales.requires_grad_(True)
    gaussians.colours.requires_grad_(True)
    gaussians.pre_act_opacities.requires_grad_(True)
    gaussians.pre_act_quats.requires_grad_(True)

def setup_optimizer(gaussians):

    gaussians.check_if_trainable()

    ### YOUR CODE HERE ###
    # HINT: Modify the learning rates to reasonable values. We have intentionally
    # set very high learning rates for all parameters.
    # HINT: Consider reducing the learning rates for parameters that seem to vary too
    # fast with the default settings.
    # HINT: Consider setting different learning rates for different sets of parameters.
    parameters = [
        {'params': [gaussians.pre_act_opacities], 'lr': 0.025, "name": "opacities"},
        {'params': [gaussians.pre_act_scales], 'lr': 0.005, "name": "scales"},
        {'params': [gaussians.colours], 'lr': 0.0025, "name": "colours"},
        {'params': [gaussians.means], 'lr': 1.6e-4, "name": "means"},
        {'params': [gaussians.pre_act_quats], 'lr': 1e-3, "name": "quats"},
    ]
    optimizer = torch.optim.Adam(parameters, lr=0.0, eps=1e-15)
    # optimizer = None

    return optimizer

def optimize_nerf(
    sds,
    prompt,
    neg_prompt="",
    device="cpu",
    log_interval=20,
    args=None,
):
    """
    Optimize the view for a NeRF model to match the prompt.
    """

    # Step 1. Create text embeddings from prompt
    embeddings = prepare_embeddings(sds, prompt, neg_prompt, view_dependent=True)

    # Step 2. Set up NeRF model
    # model = NeRFNetwork(args).to(device)

    gaussians = Gaussians(
        num_points=2000, init_type="random",
        device="cuda", isotropic=False
    )

    parameters = [
        {'params': [gaussians.pre_act_opacities], 'lr': 0.05, "name": "opacities"},
        {'params': [gaussians.pre_act_scales], 'lr': 0.005, "name": "scales"},
        {'params': [gaussians.colours], 'lr': 0.01, "name": "colours"},
        {'params': [gaussians.means], 'lr': 1.6e-4, "name": "means"},
        {'params': [gaussians.pre_act_quats], 'lr': 5e-3, "name": "quats"},
    ]
    param_list = [p for group in parameters for p in group['params']]

    scene = Scene(gaussians)
    means_lr_schedule = get_expon_lr_func(
        lr_init=1e-3,     # base learning rate
        lr_final=2e-5,    # target at the end of training
        lr_delay_mult=0.01,   # start at 1% of lr_init
        max_steps=10000
    )

    # Step 3. Create optimizer and training parameters
    # Making gaussians trainable and setting up optimizer
    make_trainable(gaussians)
    optimizer = setup_optimizer(gaussians)

    # Step 4. Load the dataset
    train_loader = NeRFDataset(
        args,
        device=device,
        type="train",
        H=args.h,
        W=args.w,
        size=args.dataset_size_train * args.batch_size,
    ).dataloader()
    test_loader = NeRFDataset(
        args,
        device=device,
        type="test",
        H=args.h,
        W=args.w,
        size=args.dataset_size_test,
    ).dataloader(batch_size=1)

    # Step 5. Training loop
    loss_dict = {}
    global_step = 0
    # create logging and saving directories
    checkpoint_path = osp.join(sds.output_dir, f"nerf_checkpoint.pth")
    os.makedirs(f"{sds.output_dir}/images", exist_ok=True)
    os.makedirs(f"{sds.output_dir}/videos", exist_ok=True)

    # tv_loss = torchmetrics.TotalVariation().to(device="cuda")

    max_epoch = np.ceil(args.iters / len(train_loader)).astype(np.int32)
    for epoch in range(max_epoch):
        for data in train_loader:
            global_step += 1
            # print(data.keys())
            # print(data['H'])

            # Initialize optimizer
            optimizer.zero_grad()
            # experiment iterations ratio
            # i.e. what proportion of this experiment have we completed (in terms of iterations) so far?

            pred_rgb = scene.render(data['cameras'], per_splat=args.gaussians_per_splat, \
                img_size=(128, 128), bg_colour=(0.0, 0.0, 0.0))[0].permute(2, 0, 1).unsqueeze(0)

            # Compuate the loss
            # interpolate text_z
            azimuth = data["azimuth"]  # [-180, 180]
            assert azimuth.shape[0] == 1, "Batch size should be 1"
            text_uncond = embeddings["uncond"]

            if not args.view_dep_text:
                text_cond = embeddings["default"]
            else:
                ### YOUR CODE HERE ###
                # print("azimuth", azimuth)
                if abs(azimuth) < 45:
                    text_cond = embeddings["front"]
                elif abs(azimuth) > 135:
                    text_cond = embeddings["back"]
                else:
                    text_cond = embeddings["side"]

  
            ### YOUR CODE HERE ###
            # print(pred_rgb.shape)
            # print(text_cond.shape)
            # print(pred_rgb.shape)
            if not args.pixel_space_sds:
                latents = sds.encode_imgs(F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False))
                loss = 0.1 * sds.sds_loss(latents, text_cond, text_embeddings_uncond=text_uncond)
            else:
                loss = 0.1 * sds.pixel_sds_loss(F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False), text_cond, text_embeddings_uncond=text_uncond)

            # print(pred_rgb.shape)
            loss += tv_loss(pred_rgb)
            # loss += 0.1 * (gaussians.colours - 0.5).pow(2).mean()

            loss.backward()
            lr_means = means_lr_schedule(global_step)
            for group in optimizer.param_groups:
                if group["name"] == "means":
                    group["lr"] = lr_means

            # max_grad_norm = max(p.grad.norm() for p in param_list if p.grad is not None)
            # print("Max grad norm:", max_grad_norm)
            # print("opacity", torch.max(gaussians.pre_act_opacities))
            # print("color", torch.max(gaussians.colours))

            torch.nn.utils.clip_grad_norm_(param_list, 2.0)
            
            optimizer.step()

            # Log
            print(f"Epoch {epoch}, global_step {global_step}, loss {loss.item()}")
            if global_step % 100 == 0:
                loss_dict[global_step] = loss.item()
                # save the nerf rendering as the logging output, instead of the decoded latent
                imgs = (
                    pred_rgb.detach().cpu().permute(0, 2, 3, 1).numpy()
                )  # torch to numpy, shape [1, 512, 512, 3]
                imgs = (imgs * 255).round()  # [0, 1] => [0, 255]
                rgb = Image.fromarray(imgs[0].astype("uint8"))
                output_path = (
                    f"{sds.output_dir}/images/rgb_epoch_{epoch}_iter_{global_step}.png"
                )
                rgb.save(output_path)


        if epoch % log_interval == 0 or epoch == max_epoch - 1:
            all_preds = []
            all_preds_depth = []

            print(f"Epoch {epoch}, testing and save rgb and depth to video...")

            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    

                    preds, preds_depth, _ = scene.render(data['cameras'], per_splat=args.gaussians_per_splat, \
                        img_size=(128, 128), bg_colour=(0.0, 0.0, 0.0))

                    pred = preds.detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth.detach().cpu().numpy()
                    pred_depth = (pred_depth - pred_depth.min()) / (
                        pred_depth.max() - pred_depth.min() + 1e-6
                    )
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            # save the video
            imageio.mimwrite(
                os.path.join(sds.output_dir, "videos", f"rgb_ep_{epoch}.mp4"),
                all_preds,
                fps=25,
                quality=8,
                macro_block_size=1,
            )
            imageio.mimwrite(
                os.path.join(sds.output_dir, "videos", f"depth_ep_{epoch}.mp4"),
                all_preds_depth,
                fps=25,
                quality=8,
                macro_block_size=1,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a hamburger")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--loss_scaling", type=int, default=1)

    ### YOUR CODE HERE ###
    # You wil need to tune the following parameters to obtain good NeRF results
    ### regularizations
    parser.add_argument('--lambda_entropy', type=float, default=0.01, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_orient', type=float, default=0.001, help="loss scale for orientation")
    ### shading options
    parser.add_argument('--latent_iter_ratio', type=float, default=0.1, help="training iters that only use albedo shading")


    parser.add_argument(
        "--postfix", type=str, default="", help="Postfix for the output directory"
    )
    parser.add_argument(
        "--view_dep_text",
        type=int,
        default=0,
        help="option to use view dependent text embeddings for nerf optimization",
    )
    parser.add_argument(
        "--pixel_space_sds",
        type=int,
        default=0,
        help="option to pixel space sds for nerf optimization",
    )
    parser.add_argument(
        "--gaussians_per_splat", default=-1, type=int,
        help=(
            "Number of gaussians to splat in one function call. If set to -1, "
            "then all gaussians in the scene are splat in a single function call. "
            "If set to any other positive interger, then it determines the number of "
            "gaussians to splat per function call (the last function call might splat "
            "lesser number of gaussians). In general, the algorithm can run faster "
            "if more gaussians are splat per function call, but at the cost of higher GPU "
            "memory consumption."
        )
    )

    parser = add_config_arguments(
        parser
    )  # add additional arguments for nerf optimization, You don't need to change the setting here by default

    args = parser.parse_args()

    seed_everything(args.seed)

    # create output directory
    args.output_dir = osp.join(args.output_dir, "nerf")
    output_dir = os.path.join(
        args.output_dir, args.prompt.replace(" ", "_") + args.postfix
    )
    os.makedirs(output_dir, exist_ok=True)

    # initialize SDS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sds = SDS(sd_version="2.1", device=device, output_dir=output_dir)

    # optimize a NeRF model
    start_time = time.time()
    optimize_nerf(sds, prompt=args.prompt, device=device, args=args)
    print(f"Optimization took {time.time() - start_time:.2f} seconds")
