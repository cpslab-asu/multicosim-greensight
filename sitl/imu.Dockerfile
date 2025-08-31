FROM ghcr.io/cpslab-asu/multicosim/gazebo:harmonic

WORKDIR /app

COPY ./pyproject.toml ./
RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    uv venv --system-site-packages --python python3.10 && \
    uv sync --group imu

COPY ./src/imu.py ./imu.py
