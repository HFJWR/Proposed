FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Proxy Setting
# ENV https_proxy "http://proxy.itc.kansai-u.ac.jp:8080"
# ENV http_proxy "http://proxy.itc.kansai-u.ac.jp:8080"

RUN apt-get update && \
    apt-get install -y sudo \
                       python-is-python3 \
                       python3-pip



RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install torch-geometric
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install torch-geometric

# Install other python libraries
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
RUN pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html

##
# Set User
##
ENV USER student
ENV UID 1002
ENV HOME /home/${USER}
ENV SHELL /bin/bash

RUN groupadd -g ${UID} ${USER}
RUN useradd -g ${UID} -u ${UID} -m -s /bin/bash ${USER}
RUN gpasswd -a ${USER} sudo
RUN echo "${USER}:student" | chpasswd

USER ${USER}

WORKDIR /home/${USER}/

RUN echo "alias python=python3" >> ~/.bashrc
RUN alias python=python3

