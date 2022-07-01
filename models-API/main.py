import uvicorn

from functools import partial

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from routers.events import startup, shutdown
from models import ESRGAN_upscale, SWINIR_upscale


app = FastAPI(title="Mai-models-API", default_response_class=FileResponse)
app.add_event_handler("startup", func=partial(startup, app=app))
app.add_event_handler("shutdown", func=partial(shutdown, app=app))


@app.post("/esrgan", tags=["api_models"])
def ESRGAN_request(image:UploadFile = File(...)):
    return ESRGAN_upscale(image, app)


@app.post("/swinir", tags=["api_models"])
def SWINIR_request(image:UploadFile = File(...)):
    return SWINIR_upscale(image, app)


if __name__ == "__main__":
    uvicorn.run("main:app", port=80, host="0.0.0.0", reload=False, debug=True)