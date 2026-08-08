#!/usr/bin/env python3









from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Sequence

import numpy as np
import onnxruntime as ort
import bindings

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "quadrotor"
LINK_NAME = "quadrotor/base_link"
PHYSICS_STEPS_PER_CONTROL = 10
EXPECTED_PHYSICS_DT = 1.0e-3
CONTROL_RATE_HZ = 100.0
GRAVITY_VECTOR = np.array([0.0, 0.0, -9.81], dtype=np.float64)
DEFAULT_MASS = 1.5


def _prepend_environment_path(name: str, path: Path) -> None:
    value = str(path.resolve())
    existing = [entry for entry in os.environ.get(name, "").split(":") if entry]
    if value not in existing:
        os.environ[name] = ":".join([value, *existing])



_prepend_environment_path(
    "GZ_SIM_RESOURCE_PATH", REPOSITORY_ROOT / "resources" / "sdf"
)
_prepend_environment_path("GZ_SIM_SYSTEM_PLUGIN_PATH", REPOSITORY_ROOT / "_lib")
sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an ONNX trajectory-tracking policy through a standalone "
            "processor and a lightweight RK4 quadrotor simulator."
        )
    )
    parser.add_argument(
        "-onnxfile",
        "--onnx-file",
        required=True,
        type=Path,
        help="Exported ONNX policy",
    )
    parser.add_argument(
        "-processor_module",
        "--processor-module",
        required=True,
        help=(
            "Importable module name or path to the generated processor .so/.py"
        ),
    )
    parser.add_argument(
        "--processor-class",
        default="TrajectoryTrackingProcessor",
        help="Processor class exposed by the generated module",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help=(
            "ONNX Runtime provider in preference order; may be repeated. "
            "Defaults to CUDA when available, followed by CPU."
        ),
    )


    parser.add_argument("--sdf-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--partition", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--link-name", default=LINK_NAME)
    parser.add_argument("--trajectory-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument(
        "--spawn-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Offset from the trajectory reference at t=0 in metres",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=DEFAULT_MASS,
        help="Mass used for the environment-equivalent gravity compensation",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pace the synchronous simulation to wall-clock 100 Hz "
            "(default: enabled; use --no-realtime for faster verification)"
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display tracking plots after evaluation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional .npz output containing states, references, and actions",
    )
    args = parser.parse_args(argv)

    args.onnx_file = args.onnx_file.expanduser().resolve()
    if not args.onnx_file.is_file():
        parser.error(f"ONNX model does not exist: {args.onnx_file}")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not math.isfinite(args.mass) or args.mass <= 0.0:
        parser.error("--mass must be finite and positive")
    if args.trajectory_seed < 0 or args.trajectory_seed > np.iinfo(np.uint32).max:
        parser.error("--trajectory-seed must fit in an unsigned 32-bit integer")
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
        if args.output.suffix != ".npz":
            args.output = args.output.with_suffix(".npz")
    return args


def load_processor_module(reference: str) -> ModuleType:

    candidate = Path(reference).expanduser()
    module_name = os.path.splitext(os.path.splitext(os.path.split(candidate)[1])[0])[0]
    if sys.modules.get(module_name, None) is not None:
        return sys.modules.get(module_name)
    if candidate.exists():
        module_path = candidate.resolve()
        if not module_path.is_file():
            raise ValueError(
                f"Processor module path is not a file: {module_path}"
            )



        module_name = module_path.name.split(".", maxsplit=1)[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load processor module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    if os.sep in reference or (os.altsep is not None and os.altsep in reference):
        raise FileNotFoundError(f"Processor module does not exist: {candidate}")
    return importlib.import_module(reference)


def make_processor(processor_class: type, state_key: str, physics_dt: float, seed: int):

    return processor_class(
        state_key=state_key,
        use_rotation_matrix=True,
        state_history_len=20,
        future_waypoint_num=20,
        action_scaling=[9.0, 9.0, 15.0],
        action_bias=[0.0, 0.0, 0.0],
        physics_steps_per_control=PHYSICS_STEPS_PER_CONTROL,
        physics_dt=physics_dt,
        horizontal_amplitude_min=1.35,
        horizontal_amplitude_max=1.75,
        vertical_amplitude_min=0.3,
        vertical_amplitude_max=0.4,
        height_offset_min=0.7,
        height_offset_max=1.0,
        trajectory_speed_min=1.0,
        trajectory_speed_max=1.5,
        maximum_normal_acceleration=4.0,
        vertical_position_weight=2.0,
        vertical_velocity_weight=0.04,
        fixed_zero_yaw_probability=0.5,
        trajectory_sampling_attempts=64,
        trajectory_seed=seed,
    )


def make_controller() -> bindings.SlidingModeController:

    array = np.array
    return bindings.SlidingModeController(array([2.5, 2.5, 3. ]), array([4., 4., 5.]), array([8., 8., 5.]), array([0.8, 0.8, 0.4]), array([0.5, 0.5, 0.5]), array([0.2, 0.2, 0.2]), array([10., 10., 10.]), array([ 0.  ,  0.  , -9.81]), 1.53, array([[0.0147209, 0.       , 0.       ],
       [0.       , 0.0169101, 0.       ],
       [0.       , 0.       , 0.029448 ]]))


def select_providers(requested: Sequence[str] | None) -> list[str]:
    available = ort.get_available_providers()
    if requested:
        unavailable = [provider for provider in requested if provider not in available]
        if unavailable:
            raise ValueError(
                f"Unavailable ONNX Runtime provider(s): {unavailable}. "
                f"Available providers: {available}"
            )
        return list(requested)

    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        raise RuntimeError(f"No supported ONNX Runtime provider in {available}")
    return providers


def quaternion_wxyz_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm <= 1.0e-9:
        raise ValueError("The UAV quaternion is invalid.")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_wxyz_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    rotation = quaternion_wxyz_to_rotation(quaternion)
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
    return np.array([roll, pitch, yaw], dtype=np.float64)



def quaternion_rpy_to_wxyz(rpy: np.ndarray) -> np.ndarray:

    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64).reshape(3)
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    quaternion = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    return quaternion / np.linalg.norm(quaternion)


def quaternion_product_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:

    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q2 = np.asarray(q2, dtype=np.float64).reshape(4)
    w1, v1 = q1[0], q1[1:]
    w2, v2 = q2[0], q2[1:]
    return np.concatenate(
        ([w1 * w2 - np.dot(v1, v2)], w1 * v2 + w2 * v1 + np.cross(v1, v2))
    )


def _inertia_matrix(value) -> np.ndarray:

    inertia = np.asarray(value, dtype=np.float64)
    if inertia.size == 3:
        inertia = np.diag(inertia.reshape(3))
    elif inertia.size == 9:
        inertia = inertia.reshape(3, 3)
    else:
        raise ValueError(f"Expected 3 or 9 inertia values, received shape {inertia.shape}.")
    if not np.isfinite(inertia).all():
        raise ValueError("Inertia contains non-finite values.")
    if np.min(np.linalg.eigvalsh(0.5 * (inertia + inertia.T))) <= 0.0:
        raise ValueError("Inertia matrix must be positive definite.")
    return inertia


class AdHocQuadrotorSimulator:













    def __init__(
        self,
        model_name: str,
        link_name: str,
        mass: float,
        gravity: np.ndarray,
        inertia: np.ndarray,
        dt: float = EXPECTED_PHYSICS_DT,
    ) -> None:
        self.model_name = model_name
        self.link_name = link_name
        self.mass = float(mass)
        self.gravity = np.asarray(gravity, dtype=np.float64).reshape(3).copy()
        self.inertia = _inertia_matrix(inertia)
        self.dt = float(dt)
        if not math.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("Simulator mass must be finite and positive.")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("Simulator timestep must be finite and positive.")

        self.state = np.zeros(13, dtype=np.float64)
        self.state[6] = 1.0
        self.control = np.zeros(4, dtype=np.float64)
        self.time = 0.0
        self.control_states: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        self.update_control_states()

    def step_size(self) -> float:
        return self.dt

    def respawn_model(
        self,
        model_name: str,
        position: np.ndarray,
        orientation_rpy: np.ndarray,
    ) -> None:
        self._validate_names(model_name, self.link_name)
        self.state.fill(0.0)
        self.state[:3] = np.asarray(position, dtype=np.float64).reshape(3)
        self.state[6:10] = quaternion_rpy_to_wxyz(orientation_rpy)
        self.control.fill(0.0)
        self.time = 0.0
        self.update_control_states()

    def set_ctbt(self, thrust: float, body_moments: np.ndarray) -> None:

        body_moments = np.asarray(body_moments, dtype=np.float64).reshape(3)
        control = np.concatenate(([float(thrust)], body_moments))
        if not np.isfinite(control).all():
            raise ValueError("CTBT command contains non-finite values.")
        self.control[:] = control

    def set_wrench(
        self,
        model_name: str,
        link_name: str,
        force_world: np.ndarray,
        moments_world: np.ndarray,
    ) -> None:





        self._validate_names(model_name, link_name)
        rotation = quaternion_wxyz_to_rotation(self.state[6:10])
        force_body = rotation.T @ np.asarray(force_world, dtype=np.float64).reshape(3)
        moments_body = rotation.T @ np.asarray(moments_world, dtype=np.float64).reshape(3)
        self.set_ctbt(max(0.0, float(force_body[2])), moments_body)

    def run_once(self) -> None:
        self.state = self._rk4_step(self.state, self.control, self.dt)
        quaternion_norm = np.linalg.norm(self.state[6:10])
        if not np.isfinite(quaternion_norm) or quaternion_norm <= 1.0e-12:
            raise FloatingPointError("Quaternion became invalid during RK4 integration.")
        self.state[6:10] /= quaternion_norm
        if not np.isfinite(self.state).all():
            raise FloatingPointError("Simulator state became non-finite.")
        self.time += self.dt
        self.update_control_states()

    def run_N(self, count: int) -> None:
        if count < 0:
            raise ValueError("run_N count must be non-negative.")
        for _ in range(int(count)):
            self.run_once()

    def update_control_states(self) -> None:
        state_dot = self._dynamics_ctbt(self.state, self.control)
        self.control_states = {
            self.model_name: {
                self.link_name: (self.state.copy(), state_dot.copy())
            }
        }

    def _validate_names(self, model_name: str, link_name: str) -> None:
        if model_name != self.model_name or link_name != self.link_name:
            raise KeyError(
                f"Unknown model/link pair {model_name!r}/{link_name!r}; expected "
                f"{self.model_name!r}/{self.link_name!r}."
            )

    def _dynamics_ctbt(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:

        x = np.asarray(x, dtype=np.float64).reshape(13)
        u = np.asarray(u, dtype=np.float64).reshape(4)

        velocity = x[3:6]
        quaternion = x[6:10]
        q_norm = np.linalg.norm(quaternion)
        if q_norm <= 1.0e-12 or not np.isfinite(q_norm):
            raise FloatingPointError("Invalid quaternion encountered in dynamics.")
        quaternion_normalized = quaternion / q_norm
        omega = x[10:13]
        thrust = u[0]
        moments = u[1:4]

        rotation = quaternion_wxyz_to_rotation(quaternion_normalized)
        acceleration = (
            rotation @ np.array([0.0, 0.0, thrust], dtype=np.float64) / self.mass
            + self.gravity
        )

        inertia_omega = self.inertia @ omega
        omega_dot = np.linalg.solve(
            self.inertia,
            moments - np.cross(omega, inertia_omega),
        )

        quaternion_dot = 0.5 * quaternion_product_wxyz(
            quaternion,
            np.concatenate(([0.0], omega)),
        )

        return np.concatenate((velocity, acceleration, quaternion_dot, omega_dot))

    def _rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._dynamics_ctbt(x, u)
        k2 = self._dynamics_ctbt(x + 0.5 * dt * k1, u)
        k3 = self._dynamics_ctbt(x + 0.5 * dt * k2, u)
        k4 = self._dynamics_ctbt(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def make_simulator(args: argparse.Namespace) -> AdHocQuadrotorSimulator:

    array = np.array
    return AdHocQuadrotorSimulator(
        model_name=args.model_name,
        link_name=args.link_name,
        mass=1.53,
        gravity=GRAVITY_VECTOR,
        inertia=array([[0.0147209, 0.       , 0.       ],
       [0.       , 0.0169101, 0.       ],
       [0.       , 0.       , 0.029448 ]]),
        dt=EXPECTED_PHYSICS_DT,
    )


def read_control_state(
    simulator: AdHocQuadrotorSimulator, model_name: str, link_name: str
) -> tuple[np.ndarray, np.ndarray]:
    simulator.update_control_states()
    state, state_dot = simulator.control_states[model_name][link_name]
    state = np.asarray(state, dtype=np.float64).reshape(-1).copy()
    state_dot = np.asarray(state_dot, dtype=np.float64).reshape(-1).copy()
    if state.shape != (13,) or state_dot.shape != (13,):
        raise RuntimeError(
            "Simulator control state must contain two 13-element vectors; "
            f"received {state.shape} and {state_dot.shape}."
        )
    if not np.isfinite(state).all() or not np.isfinite(state_dot).all():
        raise RuntimeError("Simulator returned a non-finite control state.")
    return state, state_dot


def is_failure_state(state: np.ndarray) -> bool:
    position = state[:3]
    if (
        abs(position[0]) > 9.0
        or abs(position[1]) > 9.0
        or position[2] < 0.0
        or position[2] > 10.0
    ):
        return True
    rotation = quaternion_wxyz_to_rotation(state[6:10])
    tilt = math.acos(float(np.clip(rotation[:, 2] @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0)))
    return tilt >= 1.75 * math.pi / 2.0 or np.linalg.norm(state[10:13]) > 10.0


def validate_onnx_interface(
    session: ort.InferenceSession, observation_dimension: int
) -> tuple[str, str]:
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("The verifier requires one ONNX input and one output.")
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    input_dimension = input_meta.shape[-1]
    output_dimension = output_meta.shape[-1]
    if isinstance(input_dimension, int) and input_dimension != observation_dimension:
        raise ValueError(
            f"ONNX input has {input_dimension} features, but the processor "
            f"produces {observation_dimension}."
        )
    if isinstance(output_dimension, int) and output_dimension != 3:
        raise ValueError(
            f"The high-level policy must output 3 actions, got {output_dimension}."
        )
    return input_meta.name, output_meta.name


def run_evaluation(
    args: argparse.Namespace,
    processor,
    simulator: AdHocQuadrotorSimulator,
    controller:None,
    session: ort.InferenceSession,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_key = args.model_name + args.link_name
    input_name, output_name = validate_onnx_interface(
        session, processor.observation_dimension
    )

    processor.update_trajectory()
    initial_reference = np.asarray(
        processor.reference_state_at(0.0), dtype=np.float64
    )
    spawn_position = initial_reference[:3] + np.asarray(
        args.spawn_offset, dtype=np.float64
    )
    spawn_orientation = quaternion_wxyz_to_rpy(initial_reference[6:10])
    simulator.respawn_model(
        args.model_name, spawn_position, spawn_orientation
    )

    simulator.run_N(10)

    current_state, current_state_dot = read_control_state(
        simulator, args.model_name, args.link_name
    )
    previous_state = current_state.copy()
    previous_state_dot = current_state_dot.copy()
    processor.reset(current_state.astype(np.float32))

    observation_array = np.empty(
        processor.observation_dimension, dtype=np.float32
    )
    observation = {"obs": observation_array}
    current_state_map = {state_key: current_state.astype(np.float32)}
    current_state_dot_map = {state_key: current_state_dot.astype(np.float32)}
    processor.process_observation(
        current_state_map,
        current_state_dot_map,
        current_state_map,
        current_state_dot_map,
        observation,
    )

    processed_action_array = np.empty(3, dtype=np.float32)
    processed_action = {state_key: processed_action_array}
    desired_state = np.zeros(13, dtype=np.float64)
    states = [current_state.copy()]
    references = [
        np.asarray(processor.reference_state(), dtype=np.float64).copy()
    ]
    actions: list[np.ndarray] = []
    start_time = time.perf_counter()
    next_deadline = start_time

    for step in range(args.max_steps):
        policy_input = np.ascontiguousarray(
            observation_array.reshape(1, -1), dtype=np.float32
        )
        policy_action = np.asarray(
            session.run([output_name], {input_name: policy_input})[0],
            dtype=np.float32,
        ).reshape(-1)
        if policy_action.shape != (3,) or not np.isfinite(policy_action).all():
            raise RuntimeError(
                "The ONNX policy returned an invalid action with shape "
                f"{policy_action.shape}."
            )
        if np.any(np.abs(policy_action) > 1.0 + 1.0e-5):
            raise RuntimeError(
                "The ONNX policy returned an action outside [-1, 1]. The "
                "export must include policy action bounding, while processor "
                "force scaling must remain outside the ONNX model."
            )

        processor.process_action(
            {"action": policy_action}, processed_action
        )
        residual_force = processed_action_array.astype(np.float64, copy=True)
        desired_force = residual_force - args.mass * GRAVITY_VECTOR
        reference = np.asarray(
            processor.reference_state(), dtype=np.float64
        )
        desired_state.fill(0.0)
        desired_state[6:10] = reference[6:10]

        body_moments = np.asarray(
            controller.calculate_moments(
                current_state, current_state_dot, desired_state, desired_force
            ),
            dtype=np.float64,
        )
        rotation = quaternion_wxyz_to_rotation(current_state[6:10])
        body_force = rotation.T @ desired_force
        collective_thrust = max(0.0, float(body_force[2]))


        simulator.set_ctbt(collective_thrust, body_moments)
        simulator.run_N(PHYSICS_STEPS_PER_CONTROL)

        previous_state = current_state
        previous_state_dot = current_state_dot
        current_state, current_state_dot = read_control_state(
            simulator, args.model_name, args.link_name
        )
        processor.process_observation(
            {state_key: current_state.astype(np.float32)},
            {state_key: current_state_dot.astype(np.float32)},
            {state_key: previous_state.astype(np.float32)},
            {state_key: previous_state_dot.astype(np.float32)},
            observation,
        )

        states.append(current_state.copy())
        references.append(reference.copy())
        actions.append(policy_action.copy())
        if is_failure_state(current_state):
            print(f"Stopped at control step {step + 1}: failure state detected.")
            break

        if args.realtime:
            next_deadline += 1.0 / CONTROL_RATE_HZ
            remaining = next_deadline - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)

    elapsed = time.perf_counter() - start_time
    completed_steps = len(actions)
    print(f"Completed {completed_steps} control steps in {elapsed:.3f} s.")
    if elapsed > 0.0:
        print(f"Wall-clock control rate: {completed_steps / elapsed:.1f} Hz")
    return np.asarray(states), np.asarray(references), np.asarray(actions)


def print_metrics(states: np.ndarray, references: np.ndarray) -> None:
    count = min(len(states), len(references))
    position_error = references[:count, :3] - states[:count, :3]
    velocity_error = references[:count, 3:6] - states[:count, 3:6]
    position_rmse = np.sqrt(np.mean(np.square(position_error), axis=0))
    velocity_rmse = np.sqrt(np.mean(np.square(velocity_error), axis=0))
    print("Position RMSE [x, y, z] (m):", position_rmse)
    print("Velocity RMSE [x, y, z] (m/s):", velocity_rmse)


def plot(states: np.ndarray, references: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation

    time_axis = np.arange(len(states), dtype=np.float64) / CONTROL_RATE_HZ
    figure, axes = plt.subplots(3, 1, sharex=True)
    attitude_figure, attitude_axes = plt.subplots(3, 1, sharex=True)
    tracked_rpy = Rotation.from_quat(
        states[:, 6:10], scalar_first=True
    ).as_euler("xyz")
    desired_rpy = Rotation.from_quat(
        references[:, 6:10], scalar_first=True
    ).as_euler("xyz")

    labels = ("x", "y", "z")
    for axis, label in enumerate(labels):
        axes[axis].plot(time_axis, references[:, axis], "--", label="Reference")
        axes[axis].plot(time_axis, states[:, axis], label="State")
        axes[axis].set_ylabel(f"{label} (m)")
        axes[axis].grid(True, alpha=0.3)
        attitude_axes[axis].plot(
            time_axis, desired_rpy[:, axis], "--", label="Reference"
        )
        attitude_axes[axis].plot(
            time_axis, tracked_rpy[:, axis], label="State"
        )
        attitude_axes[axis].set_ylabel(f"{label} (rad)")
        attitude_axes[axis].grid(True, alpha=0.3)
    axes[0].legend()
    attitude_axes[0].legend()
    axes[-1].set_xlabel("Time (s)")
    attitude_axes[-1].set_xlabel("Time (s)")
    figure.tight_layout()
    attitude_figure.tight_layout()
    figure.savefig("position_figure.png")
    attitude_figure.savefig("attitude_figure.png")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    processor_module = load_processor_module(args.processor_module)
    try:
        processor_class = getattr(processor_module, args.processor_class)
    except AttributeError as error:
        raise AttributeError(
            f"Processor module {processor_module.__name__!r} does not expose "
            f"{args.processor_class!r}."
        ) from error

    providers = select_providers(args.provider)
    print("ONNX Runtime providers:", providers)
    session = ort.InferenceSession(str(args.onnx_file), providers=providers)
    simulator = make_simulator(args)
    physics_dt = float(simulator.step_size())
    if not math.isclose(
        physics_dt, EXPECTED_PHYSICS_DT, rel_tol=0.0, abs_tol=1.0e-7
    ):
        raise RuntimeError(
            f"Expected a {EXPECTED_PHYSICS_DT:g} s RK4 timestep, got "
            f"{physics_dt:g} s."
        )
    control_dt = physics_dt * PHYSICS_STEPS_PER_CONTROL
    if not math.isclose(
        control_dt, 1.0 / CONTROL_RATE_HZ, rel_tol=0.0, abs_tol=1.0e-7
    ):
        raise RuntimeError(
            f"Physics configuration produces {1.0 / control_dt:g} Hz rather "
            f"than {CONTROL_RATE_HZ:g} Hz."
        )

    state_key = args.model_name + args.link_name
    processor = make_processor(
        processor_class, state_key, physics_dt, args.trajectory_seed
    )
    controller = make_controller()
    print(
        f"RK4 simulator: {PHYSICS_STEPS_PER_CONTROL} x {physics_dt:g} s "
        f"integration steps per action ({CONTROL_RATE_HZ:g} Hz simulated control)."
    )
    states, references, actions = run_evaluation(
        args, processor, simulator, controller, session
    )
    print_metrics(states, references)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            states=states,
            references=references,
            actions=actions,
            control_rate_hz=np.array(CONTROL_RATE_HZ),
        )
        print(f"Saved evaluation data to {args.output}")
    if not args.no_plot:
        plot(states, references)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
