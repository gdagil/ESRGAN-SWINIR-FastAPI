import os
from pydantic import BaseModel
from typing import Literal

import numpy as np
import torch, cv2

from utils.SWINIR.models.network_swinir import SwinIR
from utils.SWINIR.utils import util_calculate_psnr_ssim as util


class SWINIR(BaseModel):
    task:Literal["color_dn", "classical_sr", "lightweight_sr", "real_sr", "gray_dn", "jpeg_car"] = os.getenv("SWINIR_TASK")
    scale:int = int(os.getenv("SWINIR_SCALE"))
    noise:int = int(os.getenv("SWINIR_NOISE"))
    jpeg:int = int(os.getenv("SWINIR_JPEG"))
    training_patch_size:int = int(os.getenv("SWINIR_TRAIN_PATCH_SIZE"))
    large_model:bool = True if os.getenv("SWINIR_LARGE_MODEL") == "True" else False
    model_path:str = f"utils/SWINIR/model_zoo/{os.getenv('SWINIR_MODEL')}.pth"
    tile:int = int(os.getenv("SWINIR_TILE"))
    tile_overlap:int = int(os.getenv("SWINIR_TILE_OVERLAP"))

    @staticmethod
    def test(img_lq, model, args, window_size):
        if args.tile == 0:
            # test the image as a whole
            output = model(img_lq)
        else:
            # test the image tile by tile
            b, c, h, w = img_lq.size()
            tile = min(args.tile, h, w)
            assert tile % window_size == 0, "tile size should be a multiple of window_size"
            tile_overlap = args.tile_overlap
            sf = args.scale

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
            output = E.div_(W)

        return output
    
    @staticmethod
    def define_model(args):
    # 001 classical image sr
        if args.task == 'classical_sr':
            model = SwinIR(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
            param_key_g = 'params'

        # 002 lightweight image sr
        # use 'pixelshuffledirect' to save parameters
        elif args.task == 'lightweight_sr':
            model = SwinIR(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
            param_key_g = 'params'

        # 003 real-world image sr
        elif args.task == 'real_sr':
            if not args.large_model:
                # use 'nearest+conv' to avoid block artifacts
                model = SwinIR(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                            mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
            else:
                # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
                model = SwinIR(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                            img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                            num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                            mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
            param_key_g = 'params_ema'

        # 004 grayscale image denoising
        elif args.task == 'gray_dn':
            model = SwinIR(upscale=1, in_chans=1, img_size=128, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='', resi_connection='1conv')
            param_key_g = 'params'

        # 005 color image denoising
        elif args.task == 'color_dn':
            model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='', resi_connection='1conv')
            param_key_g = 'params'

        # 006 JPEG compression artifact reduction
        # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
        elif args.task == 'jpeg_car':
            model = SwinIR(upscale=1, in_chans=1, img_size=126, window_size=7,
                        img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='', resi_connection='1conv')
            param_key_g = 'params'

        pretrained_model = torch.load(args.model_path)
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

        return model

    @staticmethod
    def get_image_pair(args, path):
        (imgname, imgext) = os.path.splitext(os.path.basename(path))

        # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
        if args.task in ['classical_sr', 'lightweight_sr']:
            img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img_lq = cv2.imread(f'{args.folder_lq}/{imgname}x{args.scale}{imgext}', cv2.IMREAD_COLOR).astype(
                np.float32) / 255.

        # 003 real-world image sr (load lq image only)
        elif args.task in ['real_sr']:
            img_gt = None
            img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

        # 004 grayscale image denoising (load gt image and generate lq image on-the-fly)
        elif args.task in ['gray_dn']:
            img_gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)
            img_gt = np.expand_dims(img_gt, axis=2)
            img_lq = np.expand_dims(img_lq, axis=2)

        # 005 color image denoising (load gt image and generate lq image on-the-fly)
        elif args.task in ['color_dn']:
            img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)

        # 006 JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
        elif args.task in ['jpeg_car']:
            img_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img_gt.ndim != 2:
                img_gt = util.bgr2ycbcr(img_gt, y_only=True)
            result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
            img_lq = cv2.imdecode(encimg, 0)
            img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

        return imgname, img_lq, img_gt

    @staticmethod
    def define_device() -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

