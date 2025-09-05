FROM ghcr.io/cpslab-asu/multicosim/gazebo:harmonic

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y git && \
    rm -rf /var/apt/lists/*

WORKDIR /opt/gcs

# Add GCS program metadata to image
COPY ./pyproject.toml ./uv.lock ./mavsdk.patch ./

# Create virtual environment
RUN --mount=from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
    uv venv --python python3.10 && \
    uv sync --frozen --no-dev --group gcs

RUN patch .venv/lib/python3.10/site-packages/mavsdk/system.py mavsdk.patch

WORKDIR /opt/imu

# Add IMU program metadata to image
COPY ./pyproject.toml ./uv.lock ./

# Create virtual environment, giving access to system site packages for gazebo transport access
RUN --mount=from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
    --mount=from=greensight,source=./pyproject.toml,target=/opt/greensight/pyproject.toml \
    --mount=from=greensight,source=./src/,target=/opt/greensight/src/ \
    uv venv --system-site-packages --python python3.10 && \
    uv sync --frozen --no-dev --group imu && \
    uv pip install /opt/greensight

# Add GCS program source files to image
COPY ./src/gcs.py /opt/gcs/gcs.py

# Add IMU program source files to image
COPY ./src/imu.py /opt/imu/imu.py

# Add binary scripts to image and make them executable
COPY --chmod=0755 ./bin/ /usr/local/bin/
