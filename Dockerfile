From nvidia/cuda:11.0.3-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update --fix-missing && \
    apt upgrade -y --fix-missing && \
    apt install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libxrender1 \
    libxrender-dev \
    wget \
    curl \
    git \
    zip \
    nkf \
    gcc \
    make \
    cmake \
    sudo \
    openssl \
    libssl-dev \
    libgl1-mesa-glx \
    vim \
    silversearcher-ag \
    jq \
    tree \
    python3-dev python3-numpy python3-pip python3-setuptools \
        && \
    apt autoremove && apt clean

RUN pip3 install --upgrade pip && \
    pip3 install -U \
    opencv-python \
    Cython \
    pillow \
    opencv-contrib-python \
    setuptools \
    matplotlib \
    IPython \
    jupyter \
    tqdm \
    albumentations \
    scikit-learn \
    more-itertools \
    jupyterlab \
    japanize-matplotlib \
    face-alignment dlib tensorflow

# JupyterNotebookのパスワード
RUN mkdir /root/.jupyter
COPY jupyter_lab_config.py /root/.jupyter/
ENV PASSWORD password

WORKDIR /root
