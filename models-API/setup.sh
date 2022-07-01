pip install -r requirements.txt


git clone https://github.com/xinntao/Real-ESRGAN ./utils/ESRGAN
git clone https://github.com/JingyunLiang/SwinIR.git ./utils/SWINIR


utils/SWINIR/model_zoo/
# Download weights for ESRGAN
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P utils/ESRGAN/experiments/pretrained_models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B_netD.pth -P utils/ESRGAN/experiments/pretrained_models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth -P utils/ESRGAN/experiments/pretrained_models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -P utils/ESRGAN/experiments/pretrained_models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth -P utils/ESRGAN/experiments/pretrained_models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P utils/ESRGAN/experiments/pretrained_models

# Download weights for SWINIR
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth -P utils/SWINIR/model_zoo/
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth -P utils/SWINIR/model_zoo/
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth -P utils/SWINIR/model_zoo/
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth -P utils/SWINIR/model_zoo/
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth -P utils/SWINIR/model_zoo/
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth -P utils/SWINIR/model_zoo/
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth -P utils/SWINIR/model_zoo/
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth -P utils/SWINIR/model_zoo/
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth -P utils/SWINIR/model_zoo/
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth -P utils/SWINIR/model_zoo/
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth -P utils/SWINIR/model_zoo/



cd utils/ESRGAN
python setup.py develop
cd utils/SWINIR
python setup.py develop
cd /