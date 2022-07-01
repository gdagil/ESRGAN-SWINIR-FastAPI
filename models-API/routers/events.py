import os, pprint

from fastapi import FastAPI

from basicsr.archs.rrdbnet_arch import RRDBNet

from models.esrgan import ESRGAN
from utils.ESRGAN.realesrgan import RealESRGANer


def startup(app:FastAPI) -> None:
    app.state.ESRGAN_config = ESRGAN()
    pprint.pprint(f"ESRGAN:\n{app.state.ESRGAN_config.dict()}")

    model = None
    netscale = None
    if app.state.ESRGAN_config.model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif app.state.ESRGAN_config.model_name in ['RealESRGAN_x4plus_anime_6B']:  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif app.state.ESRGAN_config.model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2

    model_path = os.path.join('utils/ESRGAN/experiments/pretrained_models', app.state.ESRGAN_config.model_name + '.pth')

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=app.state.ESRGAN_config.tile,
        tile_pad=app.state.ESRGAN_config.tile_pad,
        pre_pad=app.state.ESRGAN_config.pre_pad,
        half=not app.state.ESRGAN_config.fp32,
        gpu_id=app.state.ESRGAN_config.gpu_id)

    if app.state.ESRGAN_config.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        app.state.ESRGAN = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=app.state.ESRGAN_config.out_scale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    else:
        app.state.ESRGAN = upsampler

def shutdown(app:FastAPI) -> None:
    del app.state.ESRGAN


