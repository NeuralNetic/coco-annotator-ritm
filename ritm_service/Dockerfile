FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

EXPOSE 9019

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update && \
    apt-get install --no-install-recommends -y curl screen python3 build-essential python3-pip
RUN apt-get install ffmpeg libsm6 libxext6  -y && apt clean

COPY ./ritm_service /app
# COPY ./models/coco_lvis_h18_itermask.pth /app/coco_lvis_h18_itermask.pth
# COPY ./models/ritm.onnx /app/ritm.onnx
WORKDIR /app/
ENV PYTHONPATH=/app

RUN pip3 install -r requirements.txt

CMD python3 flask_ritm_service.py --model /models/ritm.onnx --ip "0.0.0.0" --port 9019