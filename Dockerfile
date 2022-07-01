FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV LANG=C.UTF-8

RUN apt update
RUN apt-get install git wget ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip

WORKDIR /api
COPY models-API/setup.sh models-API/requirements.txt ./

RUN chmod +x setup.sh
RUN ./setup.sh

COPY models-API .

EXPOSE 80

ENTRYPOINT python main.py