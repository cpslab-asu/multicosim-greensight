import typing

if typing.TYPE_CHECKING:
    from collections.abc import Callable

import click
import gz.transport13 as _ts
import gz.msgs10.imu as _imu

IMU_TOPIC: typing.Final[str] = "/world/greensight_runway/model/greensight_with_ardupilot/model/greensight_with_standoffs/link/imu_link/sensor/imu_sensor/imu"


def _publish_modified(dst: _ts.Publisher, magnitude: float) -> Callable[[_imu.IMU], None]:
    """Create a subscriber callback to modify IMU messages and re-publish.

    Args:
        dst: destination topic for the messages as a gz-tranport `Publisher`
        magnitude: the magnitude of the linear acceleration perturbation

    Returns:
        A callback function that accepts an IMU message and modifies it before publishing
    """

    def _publish(msg: _imu.IMU):
        msg.linear_acceleration[0] += magnitude
        msg.linear_acceleration[1] += magnitude
        msg.linear_acceleration[2] += magnitude
        dst.publish(msg)

    return _publish


@click.command()
@click.option("-m", "--magnitude", type=float, default=0.0)
@click.option("-i", "--input-topic", default="/imu_original")
def imu(magnitude: float, input_topic: str):
    # Create gz-transport node
    node = _ts.Node()

    # Create publisher to publish modified IMU messages to original topic
    pub = node.advertise(IMU_TOPIC, _imu.IMU)

    # Subscribe to input topic and call attack function for each message
    if not node.subscribe(input_topic, _imu.IMU, _publish_modified(pub, magnitude)):
        raise RuntimeError(f"Could not create subscriber for topic: {input_topic}")

    # Busy loop to wait for shutdown
    while True:
        pass


if __name__ == "__main__":
    imu()
