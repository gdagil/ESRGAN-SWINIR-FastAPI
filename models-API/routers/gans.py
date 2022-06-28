import os, shutil, cv2, json, uuid
from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import FileResponse
from basicsr.archs.rrdbnet_arch import RRDBNet

from models.esrgan import ESRGAN_request
from utils.ESRGAN.realesrgan import RealESRGANer


models = APIRouter()

@models.post("/esrgan", response_class=FileResponse)
async def create_upload_files(params:ESRGAN_request = Depends(ESRGAN_request.as_form), image:UploadFile = File(...)):
    
    filename, ext = image.filename.split(".")
    encode_name = uuid.uuid5(uuid.NAMESPACE_DNS, (str(json.dumps(params.dict())) + filename))
    save_path = f"data/upped/{encode_name}.{ext}"

    if not os.path.exists(save_path):
        model = None
        netscale = None
        if params.model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif params.model_name in ['RealESRGAN_x4plus_anime_6B']:  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
        elif params.model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2

        model_path = os.path.join('utils/ESRGAN/experiments/pretrained_models', params.model_name + '.pth')

        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=params.tile,
            tile_pad=params.tile_pad,
            pre_pad=params.pre_pad,
            half=not params.fp32,
            gpu_id=0)

        with open(f"data/raw/{image.filename}", "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        img = cv2.imread(f"data/raw/{image.filename}", cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            output, _ = upsampler.enhance(img, outscale=params.out_scale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')


        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            ext = 'png'

        cv2.imwrite(save_path, output)


    return save_path


