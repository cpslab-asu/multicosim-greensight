import logging
import pprint

from click import group
from staliro import Sample, TestOptions, Trace, models, optimizers, staliro
from staliro.specifications import rtamt

from greensight import dronesim2, sitl


@models.model()
def model_hifi(sample: Sample) -> Trace[dict[str, float]]:
    attack = sitl.IMUAttack(magnitude=sample.static["magnitude"])
    sim = sitl.Simulator()
    sys = sim.start()
    res = sys.imu.send(attack)
    sys.stop()

    return Trace({
        pos["t"]: {"alt": pos["z"]} for pos in res.positions
    })


@models.model()
def model_lofi(inputs: Sample) -> Trace[dict[str, float]]:
    sim = dronesim2.Simulator(signal_noise_ratio=inputs.static["magnitude"])
    res = sim.start()

    return Trace({
        time: {"alt": state[2]} for time, state in zip(res.times, res.states)
    })


def run(model: models.Model):
    req = "always (alt > 0)"
    spec = rtamt.parse_dense(req)
    opt = optimizers.UniformRandom()
    opts = TestOptions(
        runs=1,
        iterations=10,
        static_inputs={
            "magnitude": (0.0, 100.0),
        },
    )
    res = staliro(model, spec, opt, opts)

    pprint.pprint(res)


@group()
def imu_attack():
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[],
    )


@imu_attack.command()
def lofi():
    run(model_lofi)


@imu_attack.command()
def hifi():
    run(model_hifi)


if __name__ == "__main__":
    imu_attack()
