import time
import typing

import attrs
import multicosim.ardupilot as _ap
import multicosim.simulations as _sims
import multicosim.docker.firmware as _fw
import multicosim.docker.gazebo as _gz
import multicosim.docker.simulation as _sim

PORT: typing.Final[int] = 5005


@attrs.define()
class IMUAttack:
    magnitude: float


@attrs.define()
class Start:
    model_name: str
    topic_name: str
    imu: IMUAttack


@attrs.define()
class Result:
    positions: list[dict[str, float]]


@attrs.define()
class CombinedNode(_ap.ArduPilotGazeboNode):
    imu: _fw.FirmwareContainerNode

    def stop(self):
        self.imu.stop()
        super().stop()


@attrs.define()
class CombinedComponent(_sims.Component[_sim.Environment, CombinedNode]):
    gazebo: _gz.GazeboContainerComponent
    firmware: _ap.ArduPilotComponent
    imu: _fw.FirmwareContainerComponent[Start, Result]

    def start(self, environment: _sim.Environment) -> CombinedNode:
        gz = self.gazebo.start(environment)
        time.sleep(2.0)

        imu = self.imu.start(environment)
        time.sleep(2.0)

        env_ext = _ap.Environment(
            environment.client, environment.network_name, gz.node.name(), imu.node.name()
        )
        ap = self.firmware.start(env_ext)

        return CombinedNode(gz, ap, imu)


class IMUNode(_sims.CommunicationNode[IMUAttack | None, Result]):
    def __init__(self, node: _fw.FirmwareContainerNode[Start, Result]):
        self.node = node

    def send(self, msg: IMUAttack | None) -> Result:
        msg_ = Start(
            model_name="iris_with_battery",
            topic_name="/imu_original",
            imu=IMUAttack(0.0) if msg is None else msg,
        )

        return self.node.send(msg_)

    def stop(self):
        self.node.stop()


class Simulation(_sims.Simulation):
    def __init__(self, simulation: _sim.ContainerSimulation, node_id: _sims.NodeId[CombinedNode]):
        self.inner = simulation
        self.node = self.inner.get(node_id)

    @property
    def gazebo(self) -> _gz.GazeboContainerNode:
        return self.node.gazebo

    @property
    def firmware(self) -> _fw.FirmwareContainerNode[_ap.Start, _ap.Result]:
        return self.node.firmware

    @property
    def imu(self) -> IMUNode:
        return IMUNode(self.node.imu)

    def stop(self):
        self.inner.stop()


class Simulator(_sims.Simulator):
    def __init__(self, *, remove: bool = False):
        gazebo = _ap.GazeboOptions(
            image="ghcr.io/cpslab-asu/multicosim-greensight/sitl/gazebo:harmonic",
            world="/app/resources/worlds/iris_battery.sdf",
        )

        firmware = _ap.FirmwareOptions(
            image="ghcr.io/cpslab-asu/multicosim-greensight/sitl/firmware:0.1.0",
            # param_files=[
            #     "/app/ardupilot-default-config.param"
            # ],
        )

        gz = _gz.GazeboContainerComponent(
            image=gazebo.image,
            template=gazebo.world,
            remove=remove,
        )

        fw = _ap.ArduPilotComponent(
            image=firmware.image,
            vehicle=firmware.vehicle,
            frame=firmware.frame,
            param_files=firmware.param_files,
            remove=remove,
        )

        imu = _fw.FirmwareContainerComponent(
            image="ghcr.io/cpslab-asu/multicosim-greensight/sitl/imu:0.1.0",
            command=f"/usr/local/bin/imu --port {PORT}",
            port=PORT,
            message_type=Start,
            response_type=Result,
            remove=remove,
        )

        comp = CombinedComponent(gz, fw, imu)

        self.simulator = _sim.ContainerSimulator()
        self.node_id = self.simulator.add(comp)

    def start(self) -> Simulation:
        return Simulation(self.simulator.start(), self.node_id)
