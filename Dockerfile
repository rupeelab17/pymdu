# syntax=docker/dockerfile:1.2

FROM mambaorg/micromamba AS runtime_micromamba
USER root
# install apt dependencies
RUN --mount=type=cache,target=/var/cache/apt \
   export DEBIAN_FRONTEND=noninteractive && \
   apt-get update && \
   apt-get upgrade -y && \
   apt-get install -y --no-install-recommends --no-install-suggests build-essential \
    vim zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev \
    libsqlite3-dev wget libbz2-dev libpq-dev libssl-dev libffi-dev libxml2-dev \
    libxslt1-dev zlib1g-dev apt-transport-https dirmngr gnupg ca-certificates unzip wkhtmltopdf

RUN --mount=type=cache,target=/var/cache/apt \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends git wget gcc g++ make pkg-config apt-utils procps \
    && apt-get install -y  software-properties-common \
    && apt-get install -y  --no-install-recommends openjdk-17-jre-headless \
    && apt-get install -y --no-install-recommends libxmlsec1-dev \
    && apt-get install -y --no-install-recommends libgl1-mesa-dri gosu iputils-ping xvfb

# INSTALLATION QGIS EN DUR
#RUN --mount=type=cache,target=/var/cache/apt \
#    export DEBIAN_FRONTEND=noninteractive \
#    && wget -qO - https://qgis.org/downloads/qgis-2022.gpg.key | gpg --no-default-keyring --keyring gnupg-ring:/etc/apt/trusted.gpg.d/qgis-archive.gpg --import \
#    && chmod a+r /etc/apt/trusted.gpg.d/qgis-archive.gpg \
#    && add-apt-repository "deb https://qgis.org/${repo} ${debian_dist} main" \
#    && apt-get install -y --no-install-recommends \
#      python3-pip qgis python3-qgis python3-qgis-common \
#      python3-venv \
#      python3-psutil \
#      python3-qgis \
#      qgis-providers \
#      qgis-server \

ARG MAMBA_DOCKERFILE_ACTIVATE=1
# Create the environment:
COPY ./environment.yml .
RUN micromamba env create -f environment.yml -p /opt/conda/envs/umep_pymdu

# Définir l'environnement par défaut
ENV PATH=/opt/conda/envs/umep_pymdu/bin:$PATH
ENV CONDA_DEFAULT_ENV=/opt/conda/envs/umep_pymdu

SHELL ["micromamba", "run", "-n", "umep_pymdu", "/bin/bash", "-c"]
# RUN micromamba install qgis -c conda-forge=
# RUN micromamba install ocl-icd-system pyopencl -c conda-forge
RUN micromamba clean --all --yes

# Set the working directory in the container
WORKDIR /app

# install dependencies
COPY ./pyproject.toml .
COPY ./README.md .

COPY ./pymdu ./pymdu
VOLUME ./pymdu ./pymdu
COPY ./scripts/build_rasterize_lidar.py ./scripts/build_rasterize_lidar.py


# install poetry
ARG PYPI_MIRROR
RUN if [ -n "$PYPI_MIRROR" ]; then pip config set global.index-url $PYPI_MIRROR; fi
RUN --mount=type=cache,target=/app/.cache python -m pip install --upgrade pip
RUN --mount=type=cache,target=/app/.cache python -m pip install uv
RUN --mount=type=cache,target=/app/.cache python -m pip install --upgrade uv
RUN --mount=type=cache,target=/app/.cache python -m uv pip install matplotlib scipy pandas numpy
RUN --mount=type=cache,target=/app/.cache python -m uv pip install poetry
RUN --mount=type=cache,target=/app/.cache python -m uv pip install jaydebeapi netCDF4 numba xarray
RUN --mount=type=cache,target=/app/.cache python -m uv pip install notebook jupyter_server jupyterlab-link-share jupyterlab
RUN --mount=type=cache,target=/app/.cache python -m uv pip install jupyter --upgrade

RUN --mount=type=cache,target=/app/.cache poetry install --no-interaction --no-ansi -vvv
# RUN --mount=type=cache,target=/app/.cache poetry export -f requirements.txt --output requirements.txt --without-hashes
#RUN --mount=type=cache,target=/app/.cache python setup.py install

# QGIS UMEP
#ENV QGIS_DISABLE_MESSAGE_HOOKS=1
# Set environment variables for QGIS
#ENV QGIS_PREFIX_PATH=/usr
ENV QT_QPA_PLATFORM=offscreen
ENV QGIS_NO_OVERRIDE_IMPORT=1
ENV XDG_RUNTIME_DIR=/tmp/runtime-root

# Copy the current directory contents into the container at /app
COPY docker/processing_umep.zip /app

# RUN #echo "[PythonPlugins]\nprocessing=true\nprocessing_umep=true" > ~/.local/share/QGIS/QGIS3/profiles/default/QGIS/QGIS3.ini
#RUN curl https://plugins.qgis.org/plugins/processing_umep/version/2.0.10/download/ --output umep-processing.zip
RUN unzip processing_umep.zip -d /opt/conda/envs/umep_pymdu/share/qgis/python/plugins
RUN rm processing_umep.zip

EXPOSE 8898

# Install dependencies
COPY docker/docker-entrypoint.sh /docker-entrypoint.sh
COPY docker/jupyter_server_config.py /app/jupyter_server_config.py
RUN chmod +x /docker-entrypoint.sh

WORKDIR /app
ENTRYPOINT ["micromamba", "run", "-n", "umep_pymdu", "/bin/bash", "-c", "jupyter notebook --port=8898 --allow-root --notebook-dir='/app' --ServerApp.password='' --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.allow_root=True --NotebookApp.open_browser=False"]
