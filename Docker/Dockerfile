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
    && echo $TZ > /etc/timezone \

# install command tools, python, rust, cuda
RUN apt update \
    # && apt-get install build-essential libbz2-dev libdb-dev \
    #     libreadline-dev libffi-dev libgdbm-dev liblzma-dev \
    #     libncursesw5-dev libsqlite3-dev libssl-dev \
    #     zlib1g-dev uuid-dev tk-dev locales-all -y\
    && apt install -y \
       zsh \
       curl \
       vim \
       git \
       # tmux \
       wget \
       # rsync \
       # tree \
       # lsb-release \
       make \
       gcc \
       g++ \
       software-properties-common \
       python3.9 \
       python3-pip \
       python3-distro-info \
    && apt clean \
    # && rm -rf /var/lib/apt/lists/* /usr/bin/python3 /usr/bin/python3.8 \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
    && ln -s /usr/bin/python3.9 /usr/bin/python3 \
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
    && poetry install --no-root \
    && poetry cache clear pypi --all -n \
    && pip3 install torch torchvision torchaudio \
    # && pip3 install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 \
    && rm -rf /root/.cache/pip
