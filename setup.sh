git clone https://github.com/xinntao/Real-ESRGAN ./utils/ESRGAN
pip install basicsr facexlib gfpgan
pip install -r ./utils/ESRGAN/requirements.txt
cd utils/ESRGAN
python setup.py develop

# cd models-api/utils/Real-ESRGAN 