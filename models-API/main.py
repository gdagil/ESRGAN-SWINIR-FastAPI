import uvicorn

from fastapi import FastAPI

from routers.gans import models


app = FastAPI(title="Mai-models-API")
app.include_router(models, prefix="/models", tags=["models-api"])


if __name__ == "__main__":
    uvicorn.run("main:app", port=80, host="0.0.0.0", reload=True)