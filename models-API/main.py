import uvicorn

from functools import partial

import os, shutil, cv2, json, uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from routers.events import startup, shutdown


app = FastAPI(title="Mai-models-API", default_response_class=FileResponse)
app.add_event_handler("startup", func=partial(startup, app=app))
app.add_event_handler("shutdown", func=partial(shutdown, app=app))

# app.add_api_route("/esrgan", ESRGAN_request, tags=["api_models"])


@app.post("/esrgan")
async def ESRGAN_request(image:UploadFile = File(...)):
    
    filename, ext = image.filename.split(".")
    encode_name = uuid.uuid5(uuid.NAMESPACE_DNS, (str(json.dumps(app.state.ESRGAN_config.dict())) + filename))
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
            if app.state.ESRGAN_config.face_enhance:
                _, _, output = app.state.ESRGAN.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = app.state.ESRGAN.enhance(img, outscale=app.state.ESRGAN_config.out_scale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')

        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            ext = 'png'

        cv2.imwrite(save_path, output)


    return save_path


if __name__ == "__main__":
    uvicorn.run("main:app", port=80, host="0.0.0.0", reload=False, debug=True)