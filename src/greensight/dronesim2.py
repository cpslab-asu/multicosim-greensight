from __future__ import annotations

import logging
import time
import typing

import attrs
import numpy as np
import matplotlib.pyplot as plt
import multicosim.simulations as _sim

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

from ._dronesim2.Simulation.ctrl import Control
from ._dronesim2.Simulation.trajectory import Trajectory
from ._dronesim2.Simulation.quadFiles.quad import Quadcopter
from ._dronesim2.Simulation.utils.animation import sameAxisAnimation
from ._dronesim2.Simulation.utils.display import makeFigures
from ._dronesim2.Simulation.utils.windModel import Wind


def _step(t: float, step_size: float, quad: Quadcopter, ctrl: Control, wind: Wind, traj: Trajectory) -> float:
    """Step the simulation at a given time.

    This function updates the models provided as arguments, advancing them to the new time step.

    Args:
        t: The current time
        step_size: The amount of time to step for (in seconds)
        quad: The quadrotor physical model
        ctrl: The system controller model
        wind: The physics model of the wind
        traj: The desired trajectory

    Returns:
        The next time step
    """

    # Dynamics (using last timestep's commands)
    # ---------------------------
    quad.update(t, step_size, ctrl.w_cmd, wind)
    t += step_size

    # Trajectory for Desired States 
    # ---------------------------
    sDes = traj.desiredState(t, step_size, quad)        

    # Generate Commands (for next iteration)
    # ---------------------------
    ctrl.controller(traj, quad, sDes, step_size)

    return t


@attrs.frozen()
class Simulation(_sim.Simulation):
    times: NDArray[np.float64]
    states: NDArray[np.float64]
    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    quaternions: NDArray[np.float64]
    omegas: NDArray[np.float64]
    eulers: NDArray[np.float64]
    desired_trajectories: NDArray[np.float64]
    desired_calcuations: NDArray[np.float64]
    commands: NDArray[np.float64]
    motor_values: NDArray[np.float64]
    throttle_values: NDArray[np.float64]
    torque_values: NDArray[np.float64]
    _quad: Quadcopter
    _traj: Trajectory
    _step_size: float

    def stop(self):
        pass

    def animate(self, *, save: bool = False):
        """Create an animation of the states produced by the simulator.

        This function will show the animation as a Matplotlib graph, and can optionally save the
        animation as a GIF.

        Args:
            save: Flag to indicate if the animation should be saved to a file
        """

        makeFigures(
            self._quad.params,
            self.times,
            self.positions,
            self.velocities,
            self.quaternions,
            self.omegas,
            self.eulers,
            self.commands,
            self.motor_values,
            self.throttle_values,
            self.torque_values,
            self.desired_trajectories,
            self.desired_calcuations,
        )

        sameAxisAnimation(
            self.times,
            self._traj.wps,
            self.positions,
            self.quaternions,
            self.desired_trajectories,
            self._step_size,
            self._quad.params,
            self._traj.xyzType,
            self._traj.yawType,
            save,
        )

        plt.show()


def _simulate(t_initial: float, t_final: float, step_size: float) -> Simulation:
    """Simulate a quadrotor from a given time.

    Args:
        t_initial: The start time for the simulation
        t_final: The final time for the simulation
        step_size: The size of the time step for each update

    Returns:
        The set of system states over the duration of the simulation
    """

    logger = logging.getLogger("greensight.dronesim2")
    logger.addHandler(logging.NullHandler())

    start_time = time.time()

    # Choose trajectory settings
    # --------------------------- 
    ctrlOptions = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]
    trajSelect = np.zeros(3)

    # Select Control Type             (0: xyz_pos,                  1: xy_vel_z_pos,            2: xyz_vel)
    ctrlType = ctrlOptions[0]

    # Select Position Trajectory Type (0: hover,                    1: pos_waypoint_timed,      2: pos_waypoint_interp,    
    #                                  3: minimum velocity          4: minimum accel,           5: minimum jerk,           6: minimum snap
    #                                  7: minimum accel_stop        8: minimum jerk_stop        9: minimum snap_stop
    #                                 10: minimum jerk_full_stop   11: minimum snap_full_stop
    #                                 12: pos_waypoint_arrived     13: pos_waypoint_arrived_wait
    trajSelect = [
        5,  # Position trajectory type (See above ^)
        3,  # Yaw Trajectory Type - (0: none, 1: yaw_waypoint_timed, 2: yaw_waypoint_interp, 3: follow, 4: zero)
        1,  # if waypoint time is used, or if average speed is used to calculate waypoint time - (0: waypoint time, 1: average speed) 
    ]

    logger.debug(f"Control type: {ctrlType}")

    # Initialize Quadcopter, Controller, Wind, Result Matrixes
    # ---------------------------
    quad = Quadcopter(t_initial)
    traj = Trajectory(quad, ctrlType, trajSelect)
    ctrl = Control(quad, traj.yawType)
    wind = Wind('None', 2.0, 90, -15)

    # Trajectory for First Desired States
    # ---------------------------
    sDes = traj.desiredState(0, step_size, quad)        

    # Generate First Commands
    # ---------------------------
    ctrl.controller(traj, quad, sDes, step_size)
    
    # Initialize Result Matrixes
    # ---------------------------
    numTimeStep = int(t_final / step_size + 1)

    t_all          = np.zeros(numTimeStep)
    s_all          = np.zeros([numTimeStep, len(quad.state)])
    pos_all        = np.zeros([numTimeStep, len(quad.pos)])
    vel_all        = np.zeros([numTimeStep, len(quad.vel)])
    quat_all       = np.zeros([numTimeStep, len(quad.quat)])
    omega_all      = np.zeros([numTimeStep, len(quad.omega)])
    euler_all      = np.zeros([numTimeStep, len(quad.euler)])
    sDes_traj_all  = np.zeros([numTimeStep, len(traj.sDes)])
    sDes_calc_all  = np.zeros([numTimeStep, len(ctrl.sDesCalc)])
    w_cmd_all      = np.zeros([numTimeStep, len(ctrl.w_cmd)])
    wMotor_all     = np.zeros([numTimeStep, len(quad.wMotor)])
    thr_all        = np.zeros([numTimeStep, len(quad.thr)])
    tor_all        = np.zeros([numTimeStep, len(quad.tor)])

    t_all[0]            = t_initial
    s_all[0,:]          = quad.state
    pos_all[0,:]        = quad.pos
    vel_all[0,:]        = quad.vel
    quat_all[0,:]       = quad.quat
    omega_all[0,:]      = quad.omega
    euler_all[0,:]      = quad.euler
    sDes_traj_all[0,:]  = traj.sDes
    sDes_calc_all[0,:]  = ctrl.sDesCalc
    w_cmd_all[0,:]      = ctrl.w_cmd
    wMotor_all[0,:]     = quad.wMotor
    thr_all[0,:]        = quad.thr
    tor_all[0,:]        = quad.tor

    # Run Simulation
    # ---------------------------
    t = t_initial
    i = 1

    while round(t, 3) < t_final :
        t = _step(t, step_size, quad, ctrl, wind, traj)
        t_all[i]             = t
        s_all[i,:]           = quad.state
        pos_all[i,:]         = quad.pos
        vel_all[i,:]         = quad.vel
        quat_all[i,:]        = quad.quat
        omega_all[i,:]       = quad.omega
        euler_all[i,:]       = quad.euler
        sDes_traj_all[i,:]   = traj.sDes
        sDes_calc_all[i,:]   = ctrl.sDesCalc
        w_cmd_all[i,:]       = ctrl.w_cmd
        wMotor_all[i,:]      = quad.wMotor
        thr_all[i,:]         = quad.thr
        tor_all[i,:]         = quad.tor
        
        i += 1
        if quad.pos[2] < -0.1 :
            break
    
    duration = time.time() - start_time
    logger.debug(f"Simulated {t:.2f}s in {duration:.6f}s.")

    return Simulation(
        t_all,
        s_all,
        pos_all,
        vel_all,
        quat_all,
        omega_all,
        euler_all,
        sDes_traj_all,
        sDes_calc_all,
        w_cmd_all,
        wMotor_all,
        thr_all,
        tor_all,
        quad,
        traj,
        step_size,
    )


class Simulator(_sim.Simulator[None, Simulation]):
    """Simulator implemenation for the DroneSim2 quadrotor simulator.

    The `Simulation` instance returned by the `start` method will contain all of the simulation
    data produced by a run of the simulator.

    Args:
        t_initial: The start time for the simulation
        t_final: The final time for the simulation
        step_size: The size of the integration step for the simulator
    """

    def __init__(self, t_initial: float = 0.0, t_final: float = 91.4, step_size: float = 0.005):
        self.t_initial = t_initial
        self.t_final = t_final
        self.step_size = step_size

    def start(self) -> Simulation:
        return _simulate(self.t_initial, self.t_final, self.step_size)
