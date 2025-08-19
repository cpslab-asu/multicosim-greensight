import multicosim as mcs
import multicosim.ardupilot as _ap
import multicosim.docker.firmware as _fw
import multicosim.docker.gazebo as _gz

from .__about__ import __version__


class Simulator(_fw.GazeboFirmwareSimulator):
    def __init__(self):
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

        super().__init__(_fw.JointGazeboFirmwareComponent(gazebo, firmware))
