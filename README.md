![Build](https://github.com/gdagil/ESRGAN-SWINIR-FastAPI/actions/workflows/docker-publish.yaml/badge.svg)

# ESRGAN-SWINIR-FastAPI
## Fast way to use it with default settings
```bash
git clone https://github.com/gdagil/ESRGAN-SWINIR-FastAPI.git
cd ESRGAN-SWINIR-FastAPI
sudo docker compose up
```
### Usung the image (environment variables)
* ESRGAN - all settings you can find in original [repo](https://github.com/xinntao/Real-ESRGAN)
	* ESRGAN_MODEL_NAME
	* ESRGAN_TILE
	* ESRGAN_TILE_PAD
	* ESRGAN_PRE_PAD
	* ESRGAN_OUTPUT_SCALE
	* ESRGAN_FP32
	* ESRGAN_FACE_ENCHANCE
	* ESRGAN_ALPHA_UPSAMPLER
	* ESRGAN_GPU_ID
* SWINIR - [repo](https://github.com/JingyunLiang/SwinIR)
	* SWINIR_TASK
	* SWINIR_SCALE
	* SWINIR_NOISE
	* SWINIR_JPEG
	* SWINIR_TRAIN_PATCH_SIZE
	* SWINIR_LARGE_MODEL
	* SWINIR_MODEL
	* SWINIR_TILE
	* SWINIR_TILE_OVERLAP
