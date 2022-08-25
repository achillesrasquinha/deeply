FROM l.gcr.io/google/bazel:latest

ENV TENSORFLOW_VERSION="2.3.0"

RUN apt-get update && \
        apt-get -y install --no-install-recommends \
            wget \
            git \
            python3-dev \
            python3-pip \
        && git clone https://github.com/tensorflow/tensorflow.git ./tensorflow \
        && cd ./tensorflow \
        && git checkout v${TENSORFLOW_VERSION} \
        && ./configure