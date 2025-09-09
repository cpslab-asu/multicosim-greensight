# Greensight simulation components for [MultiCoSim][multicosim]

This repository contains the various components for multi-fidelity simulations
of the Greensight drone.

## Organization

Each component is stored in it's own directory, which is eventually combined
using the magic of python packaging tools into a single installable wheel. Some
components are implemented using [Docker][docker] because they require
additional resources or specialized environments.

## Setup

This project depends on both the [`uv`][uv] tool, and a docker installation with a recent version of
the [buildx plugin][buildx]. You can find the instructions for installing `uv` [here][install-uv]
and the buildx plugin [here][install-buildx].

The Python dependencies for this project are stored in the `pyproject.toml` file and managed using
`uv`. In order to create a virtual environment with all of the dependencies installed, you can run
the following command: 

```console
$ uv sync
```

This will create a folder called `.venv` in the current directory with all of the dependencies as
well as symbolic links to a valid python interpreter. Thsi virtual environment will be re-used by
`uv` to execute commands that are prefixed by `uv run`. For example, the following command will
start a python interpreter with access to the virtual environment:

```console
$ uv run python3
```

The docker images for this project are defined in the `docker-bake.hcl` file, which is a declarative
build file format specific for container images. Building the images is accomplished by running the
command:

```console
$ docker bake
```

This will build all the images sequentially and store them in the local container registry on your
machine.

## Tests

### IMU Attack

The IMU attack is defined in the `tests/imu_attack.py` script, which contains both the low-fidelity
and high-fidelity system models. This script accepts 3 sub-commands. A single simulation of the
low-fidelity model can be executed using the `lofi` as shown:

```console
$ uv run tests/imu_attack.py lofi [--magnitude <FLOAT>]
```

The `--magnitude` argument represents the power of the IMU attack, and is 0 by default for an
un-attacked simulation. The high-fidelity simulation can be executed similarly, replacing the `lofi`
sub-command with `hifi`:

```console
$ uv run tests/imu_attack.py hifi [--magnitude <FLOAT>]
```

Finally, an attack search pipeline can be run using the sub-command `search`. This will use the
low-fidelity model to quickly identify IMU disturbance values that cause the drone to crash,
which are then simulated again at high-fidelity to verify that a crash occurs. The pipeline can be
run using the command:

```console
$ uv run tests/imu_attack.py search [-b|--budget <INT>]
```

The `--budget` argument represents the number of low-fidelity simulations to run. The results of the
search are output as a set of web-based [Plotly][plotly] charts.

[multicosim]: https://github.com/cpslab-asu/multicosim
[docker]: https://docker.com
[uv]: https://docs.astral.sh/uv
[buildx]: https://github.com/docker/buildx
[install-uv]: https://docs.astral.sh/uv/getting-started/installation/
[install-buildx]: https://github.com/docker/buildx?tab=readme-ov-file#installing
[plotly]: https://plotly.com
