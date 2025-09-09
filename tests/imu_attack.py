import logging
import pprint

from click import group, option
from plotly import graph_objects as go
from plotly import subplots
from rich.logging import RichHandler
from staliro import Sample, TestOptions, Trace, models, optimizers, staliro
from staliro.specifications import rtamt

from greensight import dronesim2, sitl

logger = logging.getLogger("greensight.imu_attack")


def run_hifi(magnitude: float) -> dict[float, dict[str, float]]:
    logger.info("Evaluating high-fidelity model using IMU attack magnitude: %.4f", magnitude)

    attack = sitl.IMUAttack(magnitude)
    sim = sitl.Simulator(remove=True)
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
    logger.info("Evaluating low-fidelity model using IMU attack magnitude: %.4f", magnitude)

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
        level=logging.INFO,
        handlers=[RichHandler()]
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
    evals_lofi = [eval for run in runs_lofi for eval in run.evaluations]
    magnitude_lofi = [eval.sample.static["magnitude"] for eval in evals_lofi]
    robustness_lofi = [eval.cost for eval in evals_lofi]

    candidates = [eval.sample for eval in evals_lofi if eval.cost <= 0]
    logger.info("Found %d candidate solutions for High-Fidelity evaluation", len(candidates))

    magnitude_hifi = [s.static["magnitude"] for s in candidates]
    robustness_hifi = [spec.evaluate(model_hifi.simulate(s).value).value for s in candidates]

    fig = subplots.make_subplots(rows=1, cols=2, subplot_titles=["Low-Fidelity", "High-Fidelity"])
    fig.add_trace(go.Scatter(x=magnitude_lofi, y=robustness_lofi, mode="markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=magnitude_hifi, y=robustness_hifi, mode="markers"), row=1, col=2)
    fig.show()


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
