#Download base image ubuntu 16.04
FROM ubuntu:16.04

COPY . /cluhtm
WORKDIR /cluhtm

# Update ubuntu and install python3.6
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.6 python3.6-dev python3-pip

RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3 && ln -sfn /usr/bin/python3 /usr/bin/python && ln -sfn /usr/bin/pip3 /usr/bin/pip

# To beautifully print utf-8 characters
ENV PYTHONIOENCODING utf-8

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt