from fastapi import APIRouter, UploadFile, File
import os, shutil, cv2, json, uuid

models = APIRouter()


@models.post("/esrgan")
async def ESRGAN(image:UploadFile=File(...)):
    filename, ext = image.filename.split(".")
    encode_name = uuid.uuid5(uuid.NAMESPACE_DNS, (str(json.dumps(models.state.ESRGAN_config.dict())) + filename))
    save_path = f"data/upped/{encode_name}.{ext}"

    if not os.path.exists(save_path):
        with open(f"data/raw/{image.filename}", "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        img = cv2.imread(f"data/raw/{image.filename}", cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        output = None
        try:
            if models.state.ESRGAN_config.face_enhance:
                _, _, output = models.state.ESRGAN.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = models.state.ESRGAN.enhance(img, outscale=models.state.ESRGAN_config.out_scale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')

        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            ext = 'png'

        cv2.imwrite(save_path, output)

    return save_path


