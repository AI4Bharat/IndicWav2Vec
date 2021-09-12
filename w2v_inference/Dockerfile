FROM nvcr.io/nvidia/pytorch:21.06-py3

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    liblzma-dev libbz2-dev libzstd-dev \
    libsndfile1-dev libopenblas-dev libfftw3-dev \
    libgflags-dev libgoogle-glog-dev \
    build-essential cmake libboost-system-dev \
    libboost-thread-dev libboost-program-options-dev \
    libboost-test-dev libeigen3-dev zlib1g-dev \
    libbz2-dev liblzma-dev

RUN pip install packaging soundfile swifter joblib==1.0.0 indic-nlp-library\
    tqdm==4.56.0 numpy==1.20.0 pandas==1.2.2 progressbar2==3.53.1 \
    python_Levenshtein==0.12.2 editdistance==0.3.1 omegaconf==2.0.6 \
    tensorboard==2.4.1 tensorboardX==2.1 wandb jiwer jupyterlab

WORKDIR /home
RUN git clone https://github.com/pytorch/fairseq.git && \
    cd fairseq && pip install --editable ./

# WORKDIR /tmp/apex_build
# RUN git clone https://github.com/NVIDIA/apex && cd apex && \
#     pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
#     --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
#     --global-option="--fast_multihead_attn" ./

RUN cd /tmp && git clone https://github.com/kpu/kenlm.git && \
    cd kenlm && git checkout 0c4dd4e && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_INSTALL_PREFIX=/opt/kenlm \
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make install -j$(nproc)
ENV KENLM_ROOT=/opt/kenlm

WORKDIR /tmp/flashlight_build
RUN git clone https://github.com/flashlight/flashlight.git && \
    cd flashlight/bindings/python && \
    export USE_MKL=0 && python setup.py install

WORKDIR /workspace
COPY infer /workspace/infer
COPY scripts /workspace/scripts