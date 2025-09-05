from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Callable

import click
import greensight.sitl as _sitl
import gz.transport13 as _ts
import gz.msgs10.imu_pb2 as _imu
import gz.msgs10.pose_v_pb2 as _pose
import multicosim.docker.firmware as fw
import rich.logging

IMU_TOPIC: typing.Final[str] = "/world/generated/model/iris_with_battery/model/iris_with_ardupilot/model/iris_with_standoffs/link/imu_link/sensor/imu_sensor/imu"
POSE_TOPIC: typing.Final[str] = "/world/generated/pose/info"

logger = logging.getLogger("sitl.imu")


def _publish_modified(dst: _ts.Publisher, magnitude: float, enabled: threading.Event) -> Callable[[_imu.IMU], None]:
    """Create a subscriber callback to modify IMU messages and re-publish.

    Args:
        dst: destination topic for the messages as a gz-tranport `Publisher`
        magnitude: the magnitude of the linear acceleration perturbation

    Returns:
        A callback function that accepts an IMU message and modifies it before publishing
    """

    def _publish(msg: _imu.IMU):
        # Apply disturbance only if the attack is enabled
        if enabled.is_set():
            msg.linear_acceleration.x += magnitude
            msg.linear_acceleration.y += magnitude
            msg.linear_acceleration.z += magnitude

        dst.publish(msg)

    return _publish


class PositionHandler:
    def __init__(self, model_name: str):
        self.positions: list[dict[str, float]] = []
        self.model_name = model_name
        self.stop = threading.Event()

    def __call__(self, msg: _pose.Pose_V):
        if not self.stop.is_set():
            for pose in msg.pose:
                if pose.name == self.model_name:
                    position = {
                        "t": msg.header.stamp.sec + msg.header.stamp.nsec / 1e9,
                        "x": pose.position.x,
                        "y": pose.position.y,
                        "z": pose.position.z,
                    }

                    self.positions.append(position)

    def finalize(self) -> list[dict[str, float]]:
        self.stop.set()
        return self.positions


@fw.firmware(msgtype=_sitl.Start)
def run(msg: _sitl.Start) -> _sitl.Result:
    # Create gz-transport node
    node = _ts.Node()

    # Create publisher to publish modified IMU messages to original topic
    pub = node.advertise(IMU_TOPIC, _imu.IMU)
    logger.info("Created IMU publisher on topic %s", IMU_TOPIC)

    imu_attack_enabled = threading.Event()

    # Subscribe to input topic and call attack function for each message
    if not node.subscribe(_imu.IMU, msg.topic_name,  _publish_modified(pub, msg.imu.magnitude, imu_attack_enabled)):
        raise RuntimeError(f"Could not create subscriber for topic: {msg.topic_name}")

    logger.info("Created IMU subscriber on topic %s", msg.topic_name)

    # Handler records position data broadcast using gazebo transport
    pos_handler = PositionHandler(msg.model_name)

    if not node.subscribe(_pose.Pose_V, POSE_TOPIC, pos_handler):
        raise RuntimeError(f"Could not create subscriber for topic: {POSE_TOPIC}")

    logger.info("Created Pose subscriber on topic %s", POSE_TOPIC)

    # Run GCS program in separate process and wait for completion
    proc = subprocess.Popen(
        args="/usr/local/bin/gcs --takeoff-alt 15.0",
        shell=True,
        encoding="utf-8",
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    exit_code = proc.wait()

    if exit_code != 0:
        logger.error("GCS program terminated with non-zero exit code %d", exit_code)
    else:
        logger.info("GCS program terminated. Mission complete.")

    # Enable IMU attack after drone has achieved takeoff altitidue
    imu_attack_enabled.set()

    # Wait for attack effect
    time.sleep(10.0)

    # Turn off the handler to prevent additional messages, and return the set of position
    positions = pos_handler.finalize()

    logger.info("Positions finalized, transmitting results.")

    # Send drone positions back to MultiCoSim
    return _sitl.Result(positions)


@click.command()
@click.option("-p", "--port", type=int, default=5005)
def imu(port: int):
    logging.basicConfig(
        level=logging.INFO,
        handlers=[rich.logging.RichHandler()],
    )

    logger.info("Listening for start message...")
    run.listen(port)


if __name__ == "__main__":
    imu()
