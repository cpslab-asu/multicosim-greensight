from pymavlink import mavutil

import asyncio
from asyncio import AbstractEventLoop
from threading import Thread, Lock
from dataclasses import dataclass, field
from logging import Logger, StreamHandler, getLogger, ERROR
from enum import Enum
import copy
import signal
import time
import sys

class MAVLinkCommands(Enum):
    ARM = 0
    ARM_FORCE = 1
    DISARM = 2
    HOLD = 3
    KILL = 4
    LAND = 5
    RETURN = 6
    SHUTDOWN = 7
    TAKEOFF = 8
    TERMINATE = 9
    SET_PARAM = 10
    SET_MODE = 11
    REBOOT = 12
    GO_TO = 13
    THROTTLE = 14

class APFlightModes(Enum):
    STABILIZE = 0
    ALTHOLD = 2
    AUTO = 3
    GUIDED = 4
    LOITER = 5
    RTL = 6
    LAND = 9
    FLIP = 14

def _logger() -> Logger:
    logger = getLogger("mavlink")
    handler = StreamHandler(sys.stdout)
    handler.setLevel(ERROR)
    logger.addHandler(handler)
    return logger

def _run_event_loop(loop: AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

@dataclass
class MAVLinkHandler:
    _address: str = field(default='udp:localhost:14551')
    _time_out: float = field(default=60.0)
    _logger: Logger = field(default=_logger(), init=False)
    _connection: mavutil.mavfile|None = field(default=None, init=False)
    _loop: AbstractEventLoop|None = field(default=None, init=False)
    _thread: Thread|None = field(default=None, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)
    _sys_id: int|None = field(default=None, init=False)
    _sys_state: dict[int:mavutil.mavfile_state]|None = field(default=None, init=False)
    _vehicle_status: int = field(default=0, init=False)
    _last_heartbeat_time: float = field(default=0.0, init=False)

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        if self._loop and self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=1)
            self._loop.close()

    def _signal_handler(self, sig, frame):
        print(f"Caught signal {sig} in MAVLinkHandler. Exiting thread gracefully...")
        self.shutdown()

    def connect(self):
        try:
            try:
                self._connection = mavutil.mavlink_connection(self._address)
            except Exception as error:
                self._logger.error(f"MavLink failed to connect! {error}")
                raise error
            
            msg = self._connection.wait_heartbeat(timeout=self._time_out)
            if msg == None:
                raise TimeoutError(f"Heartbeat not recieved after {self._time_out} seconds.")
            
            self._sys_id = msg.get_srcSystem()
            self._sys_state = {}
            self._sys_state[self._sys_id] = copy.deepcopy(self._connection.sysid_state[self._sys_id])
            self._vehicle_status = self._sys_state[self._sys_id].messages['HEARTBEAT'].system_status

        except TimeoutError as error:
            self._logger.error(f"MavLink timed out waiting to connect: {error}")
            raise error
        else:
            self._logger.info(f"MavLink connected successfully to {self._address}")
            self._loop = asyncio.new_event_loop()
            self._thread = Thread(target=_run_event_loop, args=[self._loop])
            self._thread.start()
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            time.sleep(0.5)
            
            try:
                asyncio.run_coroutine_threadsafe(self._get_heartbeat(),self._loop)
            except Exception as e:
                self.shutdown()
                raise e

    def get_logger(self):
        return self._logger

    def is_ready(self):
        with self._lock:
            return self._vehicle_status == mavutil.mavlink.MAV_STATE_STANDBY
        
    def is_armed(self):
        with self._lock:
            return self._sys_state[self._sys_id].armed

    def vehicle_status(self):
        with self._lock:
            return self._vehicle_status

    def time_since_heartbeat(self):
        with self._lock:
            return time.time() - self._last_heartbeat_time
        
    def flight_mode(self):
        with self._lock:
            return self._sys_state[self._sys_id].flightmode
        
    def message_data(self, message_type: str, attr: str):
        with self._lock:
            if message_type in self._sys_state[self._sys_id].messages:
                msg = self._sys_state[self._sys_id].messages[message_type]
                if(hasattr(msg,attr)):
                    return getattr(msg,attr)
                else:
                    return None
            else:
                return None

    async def _get_heartbeat(self):
        while True:
            await asyncio.sleep(1.0)
            msg = self._connection.wait_heartbeat(blocking=False)
            if msg != None:
                src_system = msg.get_srcSystem()
                with self._lock:
                    self._last_heartbeat_time = time.time()
                    self._sys_state[src_system] = copy.deepcopy(self._connection.sysid_state[src_system])
                    if self._sys_id == src_system:
                        self._vehicle_status = self._sys_state[self._sys_id].messages['HEARTBEAT'].system_status

    def send_command(self, command: MAVLinkCommands, args: list[str|float|int] = []):
        asyncio.run_coroutine_threadsafe(self._send_command(command,args),self._loop)

    async def _send_command(self, command: MAVLinkCommands, args: list[str|float|int]):
        match command:
            case MAVLinkCommands.ARM:
                try:
                    await self._send_command_helper(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                                    (0,1,0,0,0,0,0,0))
                except Exception as error:
                    self._logger.error(f"Failed to arm: {error}")
            case MAVLinkCommands.ARM_FORCE:
                try:
                    await self._send_command_helper(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                                    (0,1,21196,0,0,0,0,0))
                except Exception as error:
                    self._logger.error(f"Failed to force arm: {error}")
            case MAVLinkCommands.DISARM:
                try:
                    await self._send_command_helper(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                                    (0,0,0,0,0,0,0,0))
                except Exception as error:
                    self._logger.error(f"Failed to disarm: {error}")
            case MAVLinkCommands.KILL:
                try:
                    await self._send_command_helper(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                                    (0,0,21196,0,0,0,0,0))
                except Exception as error:
                    self._logger.error(f"Failed to kill: {error}")
            case MAVLinkCommands.SET_MODE:
                if len(args) != 1:
                    self._logger.error(f"MAVLink set mode command requires 1 argument! {len(args)} provided.")
                
                if not isinstance(args[0],APFlightModes):
                    self._logger.error(f"MAVLink set mode command argument should be of type APFlightModes! {type(args[0])}")
                
                try:
                    await self._send_command_helper(mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                                                    (0,1,args[0].value,0,0,0,0,0))
                except Exception as error:
                    self._logger.error(f"Failed to set mode to : {error}")
            case MAVLinkCommands.TAKEOFF:
                if len(args) != 1:
                    self._logger.error(f"MAVLink takeoff command requires 1 argument! {len(args)} provided.")
                    return
                
                if not isinstance(args[0],float):
                    self._logger.error(f"MAVLink takeoff command argument should be of type float! {type(args[0])}")
                    return
                
                try:
                    await self._send_command_helper(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                                                    (0,0,0,0,0,0,0,args[0]))
                except Exception as error:
                    self._logger.error(f"Failed to takeoff: {error}")
            case MAVLinkCommands.SET_PARAM:
                if len(args) != 3:
                    self._logger.error(f"MAVLink set param command requires 3 arguments! {len(args)} provided.")
                    return
                
                try:
                    await self._send_set_param_helper((self._connection.target_system, 
                                                       self._connection.target_component,
                                                       args[0],
                                                       args[1],
                                                       args[2]))
                except Exception as error:
                    self._logger.error(f"Failed to set param: {error}")
            case MAVLinkCommands.LAND:
                try:
                    await self._send_command_helper(mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                                                    (0,1,APFlightModes.LAND.value,0,0,0,0,0))
                except Exception as error:
                    self._logger.error(f"Failed to land : {error}")
            case MAVLinkCommands.RETURN:
                try:
                    await self._send_command_helper(mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                                                    (0,1,int(APFlightModes.RTL),0,0,0,0,0))
                except Exception as error:
                    self._logger.error(f"Failed to return to launch : {error}")
            case MAVLinkCommands.TERMINATE:
                try:
                    await self._send_command_helper(mavutil.mavlink.MAV_CMD_DO_FLIGHTTERMINATION,
                                                    (0,1,0,0,0,0,0,0))
                except Exception as error:
                    self._logger.error(f"Failed to terminate: {error}")
            case MAVLinkCommands.REBOOT:
                try:
                    await self._send_command_helper(mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
                                                    (0,1,1,1,1,0,0,0))
                except Exception as error:
                    self._logger.error(f"Failed to reboot: {error}")
            case MAVLinkCommands.SHUTDOWN:
                try:
                    await self._send_command_helper(mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
                                                    (0,2,2,2,2,0,0,0))
                except Exception as error:
                    self._logger.error(f"Failed to shutdown: {error}")
            case MAVLinkCommands.GO_TO:
                if len(args) != 4:
                    self._logger.error(f"MAVLink go to command requires 4 arguments! {len(args)} provided.")
                
                if not isinstance(args[0],float):
                    self._logger.error(f"First MAVLink go to command argument should be of type float! {type(args[0])}")
                
                if not isinstance(args[1],float):
                    self._logger.error(f"Second MAVLink go to command argument should be of type float! {type(args[1])}")
                
                if not isinstance(args[2],float):
                    self._logger.error(f"Third MAVLink go to command argument should be of type float! {type(args[2])}")
                
                if not isinstance(args[3],float):
                    self._logger.error(f"Fourth MAVLink go to command argument should be of type float! {type(args[3])}")
                
                try:
                    await self._send_command_helper(mavutil.mavlink.MAV_CMD_DO_REPOSITION,
                                                    (0,0,mavutil.mavlink.MAV_DO_REPOSITION_FLAGS_CHANGE_MODE,0,args[0],args[1],args[2],args[3]))
                except Exception as error:
                    self._logger.error(f"Failed to reboot: {error}")
            case MAVLinkCommands.THROTTLE:
                if len(args) != 1:
                    self._logger.error(f"MAVLink set throttle command requires 1 argument! {len(args)} provided.")
                
                if not isinstance(args[0],int):
                    self._logger.error(f"MAVLink set throttle command argument should be of type int! {type(args[0])}")
                
                try:
                    await self._send_rc_command_helper((0,0,args[0],0,0,0,0,0))
                except Exception as error:
                    self._logger.error(f"Failed to set throttle: {error}")

    async def _send_command_helper(self, command: int, params: tuple[int|float]):
        if len(params) != 8:
            self._logger.error(f"Command calls require 8 parameters! command: {command}, params: {params}")
        
        self._connection.mav.command_long_send(self._connection.target_system,
                                               self._connection.target_component,
                                               command,*params)

    async def _send_set_param_helper(self, params: tuple[int|float]):
        self._connection.mav.param_set_send(self._connection.target_system,
                                            self._connection.target_component,
                                            *params[0])

    async def _send_rc_command_helper(self, params: tuple[int|float]):
        self._connection.mav.rc_channels_override_send(self._connection.target_system,
                                                       self._connection.target_component,
                                                       *params)
