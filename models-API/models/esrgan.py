import os
from pydantic import BaseModel
from typing import Literal


class ESRGAN(BaseModel):
    model_name: Literal["RealESRGAN_x4plus", "RealESRNet_x4plus", "RealESRGAN_x4plus_anime_6B", \
        "RealESRGAN_x2plus", "realesr-animevideov3"] = os.getenv("ESRGAN_MODEL_NAME")
    tile:int = int(os.getenv("ESRGAN_TILE"))
    tile_pad:int = int(os.getenv("ESRGAN_TILE_PAD"))
    pre_pad:int = int(os.getenv("ESRGAN_PRE_PAD"))
    out_scale: int = int(os.getenv("ESRGAN_OUTPUT_SCALE"))
    fp32:bool = True if os.getenv("ESRGAN_FP32") == "True" else False
    face_enhance:bool = True if os.getenv("ESRGAN_FACE_ENCHANCE") == "True" else False
    alpha_upsampler: Literal["realesrgan", "bicubic"] = os.getenv("ESRGAN_ALPHA_UPSAMPLER")
    gpu_id:int = int(os.getenv("ESRGAN_GPU_ID"))


