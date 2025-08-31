from click import group
from staliro import Sample, TestOptions, Trace, models, optimizers, staliro
from staliro.specifications import rtamt

from greensight import dronesim2, sitl


@models.model()
def model_hifi(sample: Sample) -> Trace[dict[str, float]]:
    attack = sitl.IMUAttack(magnitude=sample.static["magnitude"])
    sim = sitl.Simulator(imu_attack=attack)
    sys = sim.start()
    res = sys.firmware.send()
    sys.stop()

    raise NotImplementedError()


@models.model()
def model_lofi(inputs: Sample) -> Trace[dict[str, float]]:
    sim = dronesim2.Simulator()
    res = sim.start()
    traj = {
        time: {"alt": state[2]}
        for time, state in zip(res.times, res.states)
    }

    return Trace(traj)


def run(model: models.Model):
    req = "always (alt > 0)"
    spec = rtamt.parse_dense(req)
    opt = optimizers.UniformRandom()
    opts = TestOptions()
    res = staliro(model, spec, opt, opts)


@group()
def imu_attack():
    pass


@imu_attack.command()
def lofi():
    run(model_lofi)


@imu_attack.command()
def hifi():
    run(model_hifi)


if __name__ == "__main__":
    imu_attack()
