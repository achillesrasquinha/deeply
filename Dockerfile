

<<<<<<< HEAD
FROM python:3.9
=======
FROM  python:3.7-alpine
>>>>>>> template/master

ARG DEVELOPMENT=false

LABEL maintainer=achillesrasquinha@gmail.com

ENV DEEPLY_PATH=/deeply

<<<<<<< HEAD
# COPY --from 

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

RUN if [[ "${DEVELOPMENT}" ]]; then \
        pip install -r ./requirements-dev.txt --use-deprecated=legacy-resolver; \
        python setup.py develop; \
    else \
        pip install -r ./requirements.txt --use-deprecated=legacy-resolver; \
        python setup.py install; \
    fi

ENTRYPOINT ["/entrypoint.sh"]
=======
RUN apk add --no-cache \
        bash \
        git \
    && mkdir -p $DEEPLY_PATH && \
    pip install --upgrade pip

COPY . $DEEPLY_PATH
COPY ./docker/entrypoint.sh /entrypoint
RUN sed -i 's/\r//' /entrypoint \
	&& chmod +x /entrypoint

WORKDIR $DEEPLY_PATH

RUN if [[ "${DEVELOPMENT}" ]]; then \
        pip install -r ./requirements-dev.txt; \
        python setup.py develop; \
    else \
        pip install -r ./requirements.txt; \
        python setup.py install; \
    fi

ENTRYPOINT ["/entrypoint"]
>>>>>>> template/master

CMD ["deeply"]