import typing

import attrs
import multicosim.ardupilot as _ap
import multicosim.simulations as _sims
import multicosim.docker.component as _cp
import multicosim.docker.firmware as _fw
import multicosim.docker .simulation as _sim

from .__about__ import __version__

PORT: typing.Final[int] = 5005


@attrs.define()
class IMUAttack:
    magnitude: float


@attrs.define()
class Start:
    firmware_host: str
    model_name: str
    topic_name: str
    imu: IMUAttack


@attrs.define()
class Result:
    positions: list[dict[str, float]]


class IMUNode(_sims.CommunicationNode[IMUAttack | None, Result]):
    def __init__(
        self,
        imu_node: _fw.FirmwareContainerNode[Start, Result],
        fw_node: _cp.ReporterNode,
    ):
        self.imu_node = imu_node
        self.fw_node = fw_node

    def send(self, msg: IMUAttack | None) -> Result:
        msg_ = Start(
            firmware_host=self.fw_node.name(),
            model_name="iris_with_battery",
            topic_name="/imu_original",
            imu=IMUAttack(0.0) if msg is None else msg,
        )

        return self.imu_node.send(msg_)

    def stop(self):
        self.imu_node.stop()


class Simulation(_ap.Simulation):
    def __init__(
        self,
        simulation: _sim.ContainerSimulation,
        node_id: _sims.NodeId[_ap.ArduPilotGazeboNode],
        imu_id: _sims.NodeId[_fw.FirmwareContainerNode[Start, Result]],
    ):
        super().__init__(simulation, node_id)
        self.imu_node = self.inner.get(imu_id)

    @property
    def imu(self) -> IMUNode:
        return IMUNode(self.imu_node, self.node.firmware.node)


class Simulator(_ap.Simulator):
    def __init__(self):
        gz = _ap.GazeboOptions(
            image="ghcr.io/cpslab-asu/multicosim-greensight/sitl/gazebo:harmonic",
            world="/app/resources/worlds/iris_battery.sdf",
        )

        firmware = _ap.FirmwareOptions(
            image="ghcr.io/cpslab-asu/multicosim-greensight/sitl/firmware:0.1.0",
            param_files=[
                "/app/ardupilot-default-config.param"
            ],
        )

        imu = _fw.FirmwareContainerComponent(
            image="ghcr.io/cpslab-asu/multicosim-greensight/sitl/imu:0.1.0",
            command=f"/usr/local/bin/imu --port {PORT}",
            port=PORT,
            message_type=Start,
            response_type=Result,
        )

        super().__init__(gz, firmware)
        self.imu_id = self.add(imu)

    def start(self) -> Simulation:
        return Simulation(self.simulator.start(), self.node_id, self.imu_id)
