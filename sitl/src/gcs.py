from __future__ import annotations

import asyncio
import logging

import click
import mavsdk
import mavsdk.action
import rich.logging

logger = logging.getLogger("sitl.gcs")


async def mission(takeoff_alt: float):
    drone = mavsdk.System()
    address = "udpin://0.0.0.0:14551"

    await drone.connect(address)
    logger.info("Connected to system at %s", address)

    async for health in drone.telemetry.health():
        healthy = all([
            health.is_local_position_ok,
            health.is_global_position_ok,
            health.is_home_position_ok,
            health.is_armable,
        ])

        if healthy:
            break

    logger.info("Drone is ready to arm.")

    await drone.action.set_takeoff_altitude(takeoff_alt)
    logger.info("Set takeoff altitude to %f", takeoff_alt)

    takeoff_attempts = 1

    while True:
        await drone.action.arm()
        logger.info("Drone armed.")

        try:
            await drone.action.takeoff()
            logger.info("Drone launched.")
        except mavsdk.action.ActionError as e:
            logger.warning(f"Failed takeoff action with error: {e}")
            await drone.action.disarm()
            await asyncio.sleep(2.0 * takeoff_attempts)
            takeoff_attempts += 1
        else:
            break

    async for position in drone.telemetry.position():
        logger.info("Drone relative altitude: %.4f", position.relative_altitude_m)

        if position.relative_altitude_m >= takeoff_alt:
            break

    logger.info("Takeoff altitude achieved, shutting down.")


@click.command("gcs")
@click.option("--takeoff-alt", type=float, default=15.0)
def gcs(takeoff_alt: float):
    logging.basicConfig(
        level=logging.INFO,
        handlers=[rich.logging.RichHandler()],
    )
    
    logger.info("Starting mission.")
    asyncio.run(mission(takeoff_alt))


if __name__ == "__main__":
    gcs()
