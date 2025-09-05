from __future__ import annotations

import asyncio
import logging

import click
import mavsdk
import rich.logging

logger = logging.getLogger("sitl.gcs")


async def mission(takeoff_alt: float):
    drone = mavsdk.System()
    address = "udpin://0.0.0.0:14551"

    await drone.connect(address)
    logger.info("Connected to system at %s", address)

    await drone.action.set_takeoff_altitude(takeoff_alt)
    logger.info("Set takeoff altitude to %f", takeoff_alt)

    await drone.action.arm()
    logger.info("Drone armed.")

    await drone.action.takeoff()
    logger.info("Drone launched.")

    async for altitude in drone.telemetry.altitude():
        if altitude.altitude_relative_m >= takeoff_alt:
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
