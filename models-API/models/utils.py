import os, shutil, cv2, json, uuid
import torch, numpy as np

from fastapi import FastAPI, UploadFile

from models.swinir import SWINIR


def file_meta(image:UploadFile, configs):
    filename, ext = image.filename.split(".")
    encode_name = uuid.uuid5(uuid.NAMESPACE_DNS, (str(json.dumps(configs.dict())) + filename))
    save_path = f"data/upped/{encode_name}"
    return save_path, ext

    
    
def ESRGAN_upscale(image:UploadFile, app:FastAPI) -> str:    
    save_path, ext = file_meta(image, app.state.ESRGAN_config)

    if not os.path.exists(f"{save_path}.{ext}"):
        with open(f"data/raw/{image.filename}", "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)  
        img = cv2.imread(f"data/raw/{image.filename}", cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None 
        output = None
        try:
            if app.state.ESRGAN_config.face_enhance:
                _, _, output = app.state.ESRGAN.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = app.state.ESRGAN.enhance(img, outscale=app.state.ESRGAN_config.out_scale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')  
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            ext = 'png' 
        print(f"{save_path}.{ext}")
        cv2.imwrite(f"{save_path}.{ext}", output)
    
    return f"{save_path}.{ext}"


def SWINIR_upscale(image:UploadFile, app:FastAPI) -> str:  
    save_path, ext = file_meta(image, app.state.SWINIR_config)

    if not os.path.exists(f"{save_path}.{ext}"):
        with open(f"data/raw/{image.filename}", "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        imgname, img_lq, img_gt = SWINIR.get_image_pair(app.state.SWINIR_config, f"data/raw/{image.filename}")  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(SWINIR.define_device())  # CHW-RGB to NCHW-RGB

        window_size:int = 8
        if app.state.SWINIR_config.task == "jpeg_car":
            window_size = 7

        output = None
        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = SWINIR.test(img_lq, app.state.SWINIR, app.state.SWINIR_config, window_size)
            scale = app.state.SWINIR_config.scale
            output = output[..., :h_old * scale , :w_old * scale ]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f"{save_path}.{ext}", output)

    return f"{save_path}.{ext}"