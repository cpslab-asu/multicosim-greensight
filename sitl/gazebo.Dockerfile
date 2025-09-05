FROM ghcr.io/cpslab-asu/multicosim/ardupilot/gazebo:harmonic

COPY ./models/ ${GZ_ROOT}/resources/models/
COPY ./worlds/ ${GZ_ROOT}/resources/worlds/
