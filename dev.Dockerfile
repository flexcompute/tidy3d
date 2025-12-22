FROM ghcr.io/astral-sh/uv:0.8-debian AS tidy3d-python-client-dev

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    pandoc \
    xsel \
    groff \
    mandoc \
    xclip

RUN apt-get update && apt-get install -y zip unzip curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${POETRY_HOME}/bin:${PATH}"
RUN poetry self add poetry-codeartifact-login

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install -y curl \
    && curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz \
    && tar -C /opt -xzf nvim-linux-x86_64.tar.gz \
    && rm nvim-linux-x86_64.tar.gz
ENV PATH="/opt/nvim-linux-x86_64/bin:$PATH"

RUN addgroup --gid 1000 flexdaemon && \
    adduser --uid 1000 --gid 1000 \
        --home /home/flexdaemon \
        --shell /bin/bash \
        --disabled-password \
        flexdaemon \
    && if getent group video >/dev/null 2>&1; then usermod -aG video flexdaemon; fi \
    && if getent group render >/dev/null 2>&1; then usermod -aG render flexdaemon; fi \
    && mkdir -p /home/flexdaemon \
    && chown -R flexdaemon:flexdaemon /home/flexdaemon \
    && chmod a+rX /home \
    && chmod a+rwX /home/flexdaemon

RUN apt-get update && apt-get install -y git-lfs && git lfs install 

USER flexdaemon
WORKDIR /home/flexdaemon
CMD ["sleep", "infinity"]
