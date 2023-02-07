FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

EXPOSE 9019

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update && \
    apt-get install --no-install-recommends -y cmake curl build-essential screen python3-dev python3-pip
RUN apt-get install ffmpeg libsm6 libxext6  -y && apt clean

COPY ./ritm_service /app
WORKDIR /app/
ENV PYTHONPATH=/app

RUN pip3 install -r requirements.txt

CMD python3 -m ritmapp.app