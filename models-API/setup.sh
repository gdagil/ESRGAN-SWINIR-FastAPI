pip install -r requirements.txt


git clone https://github.com/xinntao/Real-ESRGAN.git ./utils/ESRGAN
git clone https://github.com/JingyunLiang/SwinIR.git ./utils/SWINIR


# Download weights for ESRGAN
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P utils/ESRGAN/experiments/pretrained_models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B_netD.pth -P utils/ESRGAN/experiments/pretrained_models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth -P utils/ESRGAN/experiments/pretrained_models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -P utils/ESRGAN/experiments/pretrained_models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth -P utils/ESRGAN/experiments/pretrained_models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P utils/ESRGAN/experiments/pretrained_models


cd utils/ESRGAN
python setup.py develop
cd utils/SWINIR
python setup.py develop
cd /
