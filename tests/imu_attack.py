import logging
import pprint

from click import group, option
from staliro import Sample, TestOptions, Trace, models, optimizers, staliro
from staliro.specifications import rtamt

from greensight import dronesim2, sitl


def run_hifi(magnitude: float) -> dict[float, dict[str, float]]:
    attack = sitl.IMUAttack(magnitude)
    sim = sitl.Simulator()
    sys = sim.start()
    res = sys.imu.send(attack)
    sys.stop()

    return {
        pos["t"]: {"alt": pos["z"]} for pos in res.positions
    }


@models.model()
def model_hifi(inputs: Sample) -> Trace[dict[str, float]]:
    return Trace(run_hifi(inputs.static["magnitude"]))


def run_lofi(magnitude: float) -> dict[float, dict[str, float]]:
    sim = dronesim2.Simulator(signal_noise_ratio=magnitude)
    res = sim.start()

    return {
        time: {"alt": state[2]} for time, state in zip(res.times, res.states)
    }


@models.model()
def model_lofi(inputs: Sample) -> Trace[dict[str, float]]:
    return Trace(run_lofi(inputs.static["magnitude"]))


@group("imu_attack")
def imu_attack():
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[],
    )


@imu_attack.command("search")
@option("-i", "--iterations", type=int, default=10)
def search(iterations: int):
    req = "always (alt > 0)"
    spec = rtamt.parse_dense(req)
    opt = optimizers.UniformRandom()
    opts = TestOptions(
        runs=1,
        iterations=iterations,
        static_inputs={
            "magnitude": (0.0, 100.0),
        },
    )
    runs_lofi = staliro(model_lofi, spec, opt, opts)
    evals_lofi = (eval for run in runs_lofi for eval in run.evaluations)

    for eval in evals_lofi:
        if eval.cost <= 0:
            trace = model_hifi.simulate(eval.sample).value
            cost = spec.evaluate(trace).value
            magnitude = eval.sample.static['magnitude']

            print(f"IMU attack magnitude: {magnitude}\tLow-Fidelity cost: {eval.cost}\tHigh-Fidelity cost: {cost}")


@imu_attack.command("lofi")
@option("--magnitude", type=float, default=0.0)
def lofi(magnitude: float):
    pprint.pprint(run_lofi(magnitude))


@imu_attack.command("hifi")
@option("--magnitude", type=float, default=0.0)
def hifi(magnitude: float):
    pprint.pprint(run_hifi(magnitude))


if __name__ == "__main__":
    imu_attack()
