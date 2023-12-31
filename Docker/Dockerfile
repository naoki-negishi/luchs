FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
RUN echo "start building docker container!"

USER root
ENV TZ=Asia/Tokyo
ENV LANG="en_US.UTF-8"
# ENV RUST_HOME /usr/local/lib/rust
# ENV RUSTUP_HOME ${RUST_HOME}/rustup
# ENV CARGO_HOME ${RUST_HOME}/cargo

# Pre-configure apt installation to not prompt for timezone selection
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# install command tools, python, rust, cuda
RUN apt update \
    && apt install -y \
       curl \
       tar \
       bzip2 \
       vim \
       make \
       gcc \
       g++
       # zsh \
       # git \
       # wget \
       # tmux \
       # rsync \
       # tree \
       # lsb-release

RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y \
       python3.9 \
       python3-pip \
       python3-distro-info \
       python3.9-distutils \
    && apt clean

RUN rm -rf /var/lib/apt/lists/* /usr/bin/python3 /usr/bin/python3.8 \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
    && ln -s /usr/bin/python3.9 /usr/bin/python3
    # && mkdir /usr/local/lib/rust \
    # && chmod 0755 $RUST_HOME \
    # && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > ${RUST_HOME}/rustup.sh \
    # && chmod +x ${RUST_HOME}/rustup.sh \
    # && ${RUST_HOME}/rustup.sh -y --default-toolchain nightly --no-modify-path \
    # && ldconfig /usr/local/cuda-11.3/lib64

# PATH
# ENV PATH /usr/local/cuda-12.2/bin:$CARGO_HOME/bin:$PATH
# ENV LD_LIBRARY_PATH /usr/local/cuda-11.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# copy
WORKDIR /code
COPY pyproject.toml .

# install python packages
# memo: poetry 1.2.X 以降は PEP-440 に従っていないのでエラー出る. 1.1.15 なら OK
# memo: pytorch だけ cuda に依存するので個別に install する
RUN pip install --upgrade pip \
    && pip install poetry==1.1.15 \
    && poetry config virtualenvs.create false \
    && poetry install --no-root \
    && poetry cache clear pypi --all -n \
    && pip3 install torch torchvision torchaudio \
    # && pip3 install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 \
    && rm -rf /root/.cache/pip
