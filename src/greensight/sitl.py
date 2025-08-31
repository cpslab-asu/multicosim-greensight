import attrs
import multicosim as mcs
import multicosim.ardupilot as _ap
import multicosim.docker.component as _cp
import multicosim.docker.firmware as _fw
import multicosim.docker.gazebo as _gz

from .__about__ import __version__


@attrs.define()
class IMUAttack:
    magnitude: float


class Simulator(_fw.GazeboFirmwareSimulator):
    def __init__(self, *, imu_attack: IMUAttack | None = None):
        gazebo = _gz.GazeboConfig(
            image=f"ghcr.io/cpslab-asu/multicosim-greensight/ardupilot/gazebo:{__version__}",
            template="/app/resources/worlds/iris-battery.sdf",
        )

        firmware = _fw.FirmwareConfig(
            image=f"ghcr.io/cpslab-asu/multicosim/ardupilot/firmware:{mcs.__version__}",
            command="",
            port=_ap.PORT,
            message_type=_ap.Start,
            response_type=_ap.Result,
        )

        imu_magnitude = 0.0 if imu_attack is None else imu_attack.magnitude
        imu = _cp.ContainerComponent(
            image=f"ghcr.io/cpslab-asu/multicosim-greensight/sitl/imu:{__version__}",
            command=f"/usr/local/bin/imu --magnitude {imu_magnitude}",
        )

        super().__init__(_fw.JointGazeboFirmwareComponent(gazebo, firmware))
        self.add(imu)
