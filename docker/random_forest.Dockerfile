FROM python:3.7

ENV SHELL /bin/bash

WORKDIR /code

COPY requirements.txt /code
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir polyaxon==1.1.9

WORKDIR /plx-context/artifacts
