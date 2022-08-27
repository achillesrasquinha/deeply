FROM python:3.8.11

ARG DEVELOPMENT=false

LABEL maintainer=achillesrasquinha@gmail.com

ENV DEEPLY_PATH=/deeply

RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            ffmpeg \
            libsm6 \
            libxext6 \
            bash \
            git \
            graphviz \
    && mkdir -p $DEEPLY_PATH

COPY . $DEEPLY_PATH
COPY ./docker/entrypoint.sh /entrypoint.sh

WORKDIR $DEEPLY_PATH

SHELL ["/bin/bash", "-c"]

RUN pip install tensorflow==2.6.0 -f https://tf.kmtea.eu/whl/stable.html

RUN if [[ "${DEVELOPMENT}" ]]; then \
        pip install -r ./requirements-dev.txt --use-deprecated=legacy-resolver; \
        python setup.py develop; \
    else \
        pip install -r ./requirements.txt --use-deprecated=legacy-resolver; \
        python setup.py install; \
    fi

ENTRYPOINT ["/entrypoint.sh"]

CMD ["deeply"]