FROM python:3.9

LABEL maintainer=achillesrasquinha@gmail.com

ENV DEEPLY_PATH=/deeply

RUN apt-get update && \
        apt-get install -y --no-install-recommends \
                ffmpeg \
                libsm6 \
                libxext6 \
                bash \
                git \
    && mkdir -p $DEEPLY_PATH

COPY . $DEEPLY_PATH
COPY ./docker/entrypoint.sh /entrypoint.sh

WORKDIR $DEEPLY_PATH

RUN pip install -r ./requirements.txt && \
    python setup.py install

ENTRYPOINT ["/entrypoint.sh"]

CMD ["deeply"]