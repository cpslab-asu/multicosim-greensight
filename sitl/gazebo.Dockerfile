FROM ghcr.io/cpslab-asu/multicosim/ardupilot/gazebo:harmonic

COPY ./models/ ${GZ_ROOT}/models
COPY ./worlds/ ${GZ_ROOT}/worlds
