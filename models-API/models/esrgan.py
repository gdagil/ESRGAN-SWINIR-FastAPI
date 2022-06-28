from email.policy import default
from pydantic import BaseModel
from typing import Literal
from fastapi import Form


class ESRGAN_request(BaseModel):
    model_name: Literal["RealESRGAN_x4plus", "RealESRNet_x4plus", "RealESRGAN_x4plus_anime_6B", \
        "RealESRGAN_x2plus", "realesr-animevideov3"] = "RealESRGAN_x4plus"
    tile:int = 0
    tile_pad:int = 10
    pre_pad:int = 0
    out_scale: int = 2
    fp32:bool = False
    alpha_upsampler: Literal["realesrgan", "bicubic"] = "realesrgan"



    @classmethod
    def as_form(cls, 
                model_name: str = Form(default="RealESRGAN_x4plus", 
                description="RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | realesr-animevideov3"),  
                tile: int = Form(default=0), 
                tile_pad:int = Form(default=10), 
                pre_pad: int = Form(default=0),
                out_scale:int = Form(default=2), 
                fp32:bool = Form(default=False),
                alpha_upsampler: str = Form(default="realesrgan",
                description="realesrgan | bicubic")):
        return cls(
            model_name=model_name, 
            tile=tile, 
            tile_pad=tile_pad, 
            pre_pad=pre_pad, 
            out_scale=out_scale,
            fp32=fp32,
            alpha_upsampler=alpha_upsampler)


