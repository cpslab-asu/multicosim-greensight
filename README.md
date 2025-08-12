# Greensight simulation components for [MultiCoSim][multicosim]

This repository contains the various components for multi-fidelity simulations
of the Greensight drone.

## Organization

Each component is stored in it's own directory, which is eventually combined
using the magic of python packaging tools into a single installable wheel. Some
components are implemented using [Docker][docker] because they require
additional resources or specialized environments.

## Installation

This repo can be installed directly via pip using the command:

```console
$ pip install https://github.com/cpslab-asu/multicosim-greensight
```

## Usage

**TODO**

[multicosim]: https://github.com/cpslab-asu/multicosim
[docker]: https://docker.com
