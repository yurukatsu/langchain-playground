ARG IMAGE=python:3.11-bullseye
FROM ${IMAGE}

WORKDIR /workspace
COPY ./requirements.txt /workspace/

RUN pip install --upgrade pip &&\
    pip install --no-cache-dir -r requirements.txt
