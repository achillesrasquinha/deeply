

FROM  python:3.7-alpine

LABEL maintainer=achillesrasquinha@gmail.com

ENV DEEPLY_PATH=/usr/local/src/deeply

RUN apk add --no-cache \
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