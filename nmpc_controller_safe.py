"""Safety-oriented CTBR NMPC for the Quanser QDrone2.

State convention
----------------
Physical state (10):
    [p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z]

Augmented OCP state (14):
    [physical_state, previous_collective_thrust,
     previous_body_rate_x, previous_body_rate_y, previous_body_rate_z]

OCP control (4):
    the next zero-order-held CTBR command

The controller returns CTBR commands:
    [collective_thrust_N, body_rate_x_rad_s, body_rate_y_rad_s,
     body_rate_z_rad_s]

Important integration assumptions
---------------------------------
* World z is up and gravity is typically [0, 0, -9.81].
* Quaternions are scalar-first and represent body-to-world orientation.
* Positive collective thrust acts along body +z.
* The downstream Quanser controller is a fast body-rate controller.
* A Simulink-side watchdog must switch to the original Quanser controller if
  packets are stale or this controller reports an invalid solution.
"""

from collections import namedtuple
from collections.abc import Mapping
from pathlib import Path
import math
import shutil
import time

import acados_template as at
import casadi as ca
import numpy as np


# Replace this stub with the project base class when integrating into DATT.
class AuxiliaryController:
    pass


PHYSICAL_STATE_DIM = 10
CTBR_DIM = 4
AUGMENTED_STATE_DIM = PHYSICAL_STATE_DIM + CTBR_DIM
CONTROL_DIM = 4
PARAMETER_DIM = 8  # mass(1), gravity(3), reference quaternion(4)


class NMPCConfig:
    """Configuration and conservative commissioning limits.

    These defaults are starting points, not certified QDrone2 limits. Replace
    the thrust/rate/slew limits using measurements from the actual vehicle.
    """

    def __init__(
        self,
        control_dt=0.02,
        horizon_steps=10,
        thrust_min=0.0,
        thrust_max=20.0,
        body_rate_max=None,
        thrust_slew_max=30.0,
        body_rate_slew_max=None,
        velocity_max=None,
        max_tilt_rad=math.radians(25.0),
        position_weight=None,
        velocity_weight=None,
        attitude_weight=None,
        ctbr_weight=None,
        slew_weight=None,
        terminal_weight_scale=2.0,
        qp_tolerance=1e-6,
        qp_max_iterations=50,
        max_solver_wall_time_s=0.018,
        max_state_age_s=0.060,
        quaternion_stabilization_gain=1.0,
        code_export_root=".",
    ):
        self.control_dt = control_dt
        self.horizon_steps = horizon_steps
        self.thrust_min = thrust_min
        self.thrust_max = thrust_max
        self.body_rate_max = (
            np.array([1.5, 1.5, 1.0], dtype=float)
            if body_rate_max is None
            else body_rate_max
        )
        self.thrust_slew_max = thrust_slew_max
        self.body_rate_slew_max = (
            np.array([4.0, 4.0, 3.0], dtype=float)
            if body_rate_slew_max is None
            else body_rate_slew_max
        )
        self.velocity_max = (
            np.array([3.0, 3.0, 2.0], dtype=float)
            if velocity_max is None
            else velocity_max
        )
        self.max_tilt_rad = max_tilt_rad
        self.position_weight = (
            np.array([25.0, 25.0, 35.0], dtype=float)
            if position_weight is None
            else position_weight
        )
        self.velocity_weight = (
            np.array([10.0, 10.0, 15.0], dtype=float)
            if velocity_weight is None
            else velocity_weight
        )
        self.attitude_weight = (
            np.array([12.0, 12.0, 6.0], dtype=float)
            if attitude_weight is None
            else attitude_weight
        )
        self.ctbr_weight = (
            np.array([0.20, 0.80, 0.80, 0.50], dtype=float)
            if ctbr_weight is None
            else ctbr_weight
        )
        self.slew_weight = (
            np.array([0.02, 0.08, 0.08, 0.05], dtype=float)
            if slew_weight is None
            else slew_weight
        )
        self.terminal_weight_scale = terminal_weight_scale
        self.qp_tolerance = qp_tolerance
        self.qp_max_iterations = qp_max_iterations
        self.max_solver_wall_time_s = max_solver_wall_time_s
        self.max_state_age_s = max_state_age_s
        self.quaternion_stabilization_gain = quaternion_stabilization_gain
        self.code_export_root = code_export_root
        self.__post_init__()

    def __post_init__(self):
        self.body_rate_max = _vector(self.body_rate_max, 3, "body_rate_max")
        self.body_rate_slew_max = _vector(
            self.body_rate_slew_max, 3, "body_rate_slew_max"
        )
        self.velocity_max = _vector(self.velocity_max, 3, "velocity_max")
        self.position_weight = _vector(self.position_weight, 3, "position_weight")
        self.velocity_weight = _vector(self.velocity_weight, 3, "velocity_weight")
        self.attitude_weight = _vector(self.attitude_weight, 3, "attitude_weight")
        self.ctbr_weight = _vector(self.ctbr_weight, 4, "ctbr_weight")
        self.slew_weight = _vector(self.slew_weight, 4, "slew_weight")

        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if self.horizon_steps < 2:
            raise ValueError("horizon_steps must be at least 2")
        if not 0.0 <= self.thrust_min < self.thrust_max:
            raise ValueError("Require 0 <= thrust_min < thrust_max")
        if np.any(self.body_rate_max <= 0.0):
            raise ValueError("body_rate_max entries must be positive")
        if self.thrust_slew_max <= 0.0 or np.any(self.body_rate_slew_max <= 0.0):
            raise ValueError("CTBR slew limits must be positive")
        if np.any(self.velocity_max <= 0.0):
            raise ValueError("velocity_max entries must be positive")
        if not 0.0 < self.max_tilt_rad < math.pi / 2.0:
            raise ValueError("max_tilt_rad must be between 0 and pi/2")
        if self.max_solver_wall_time_s <= 0.0:
            raise ValueError("max_solver_wall_time_s must be positive")
        if self.max_state_age_s < 0.0:
            raise ValueError("max_state_age_s cannot be negative")


ControlInfo = namedtuple(
    "ControlInfo",
    (
        "valid",
        "solver_status",
        "solver_time_s",
        "wall_time_s",
        "state_age_s",
        "used_fallback",
        "reason",
    ),
)
ControlInfo.__doc__ = "Diagnostic information for the most recent control call."


ReferenceTrajectory = namedtuple("ReferenceTrajectory", ("states", "ctbr"))
ReferenceTrajectory.__new__.__defaults__ = (None,)
ReferenceTrajectory.__doc__ = """Horizon references accepted by calculate_control.

states has shape (N+1, >=10) or can be a single state with shape (>=10,).
ctbr is optional and has shape (N+1, 4), (N, 4), or (4,).
"""


def _vector(value, size, name):
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} must contain {size} values, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array.copy()


def _normalize_quaternion_np(quaternion):
    quaternion = _vector(quaternion, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("Quaternion norm is too small")
    return quaternion / norm


def _align_quaternion_np(quaternion, anchor):
    quaternion = _normalize_quaternion_np(quaternion)
    anchor = _normalize_quaternion_np(anchor)
    return -quaternion if float(np.dot(quaternion, anchor)) < 0.0 else quaternion


def _quat_product_ca(q1, q2):
    scalar = q1[0] * q2[0] - ca.dot(q1[1:4], q2[1:4])
    vector = q1[0] * q2[1:4] + q2[0] * q1[1:4] + ca.cross(q1[1:4], q2[1:4])
    return ca.vertcat(scalar, vector)


def _normalize_quaternion_ca(quaternion):
    return quaternion / ca.sqrt(ca.sumsqr(quaternion) + 1e-12)


def _rotation_matrix_ca(quaternion):
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    return ca.vertcat(
        ca.horzcat(
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qw * qz),
            2 * (qx * qz + qw * qy),
        ),
        ca.horzcat(
            2 * (qx * qy + qw * qz),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qw * qx),
        ),
        ca.horzcat(
            2 * (qx * qz - qw * qy),
            2 * (qy * qz + qw * qx),
            1 - 2 * (qx * qx + qy * qy),
        ),
    )


def _attitude_error_ca(quaternion, reference_quaternion):
    """Return a local 3D quaternion error.

    Runtime code aligns reference quaternion signs with the measured quaternion,
    so q and -q do not create a discontinuous cost around the operating point.
    """

    q = _normalize_quaternion_ca(quaternion)
    q_ref = _normalize_quaternion_ca(reference_quaternion)
    q_ref_conjugate = ca.vertcat(q_ref[0], -q_ref[1:4])
    q_error = _quat_product_ca(q_ref_conjugate, q)
    return 2.0 * q_error[1:4]


def _build_model(config, model_name):
    """Build a discrete-time CTBR model with exact zero-order-hold semantics.

    The augmented state stores the previously issued CTBR command. The current
    OCP input is the next CTBR command, so ``u - x[10:14]`` is the true command
    increment and can be penalized and constrained at every shooting stage.
    """

    model = at.AcadosModel()
    model.name = model_name

    x = ca.SX.sym("x", AUGMENTED_STATE_DIM)
    u = ca.SX.sym("u", CONTROL_DIM)
    p = ca.SX.sym("p", PARAMETER_DIM)

    physical_state = x[0:10]
    position = physical_state[0:3]
    velocity = physical_state[3:6]
    quaternion = physical_state[6:10]
    previous_ctbr = x[10:14]

    mass = p[0]
    gravity = p[1:4]
    reference_quaternion = p[4:8]

    def physical_dynamics(state_, ctbr_):
        velocity_ = state_[3:6]
        quaternion_ = state_[6:10]
        quaternion_normalized_ = _normalize_quaternion_ca(quaternion_)
        rotation_ = _rotation_matrix_ca(quaternion_normalized_)
        thrust_ = ctbr_[0]
        body_rates_ = ctbr_[1:4]

        acceleration_ = rotation_ @ ca.vertcat(0.0, 0.0, thrust_) / mass + gravity
        quaternion_derivative_ = 0.5 * _quat_product_ca(
            quaternion_normalized_, ca.vertcat(0.0, body_rates_)
        )
        quaternion_derivative_ += (
            config.quaternion_stabilization_gain
            * (1.0 - ca.dot(quaternion_, quaternion_))
            * quaternion_
        )
        return ca.vertcat(velocity_, acceleration_, quaternion_derivative_)

    # Discrete RK4 propagation using the CTBR command as a zero-order-held input.
    dt = config.control_dt
    k1 = physical_dynamics(physical_state, u)
    k2 = physical_dynamics(physical_state + 0.5 * dt * k1, u)
    k3 = physical_dynamics(physical_state + 0.5 * dt * k2, u)
    k4 = physical_dynamics(physical_state + dt * k3, u)
    physical_next = physical_state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    physical_next = ca.vertcat(
        physical_next[0:6],
        _normalize_quaternion_ca(physical_next[6:10]),
    )

    attitude_error = _attitude_error_ca(quaternion, reference_quaternion)
    command_increment = u - previous_ctbr
    rotation = _rotation_matrix_ca(_normalize_quaternion_ca(quaternion))
    cos_tilt = rotation[2, 2]

    model.x = x
    model.u = u
    model.p = p
    model.disc_dyn_expr = ca.vertcat(physical_next, u)

    model.cost_y_expr = ca.vertcat(
        position,
        velocity,
        attitude_error,
        u,
        command_increment,
    )
    model.cost_y_expr_e = ca.vertcat(
        position,
        velocity,
        attitude_error,
        previous_ctbr,
    )

    # Path constraints: tilt and true per-update CTBR increments.
    model.con_h_expr = ca.vertcat(cos_tilt, command_increment)
    model.con_h_expr_0 = ca.vertcat(cos_tilt, command_increment)
    model.con_h_expr_e = ca.vertcat(cos_tilt)
    return model


def build_acados_ocp_solver(
    mass,
    gravity,
    config,
    model_name,
    output_dir,
):
    """Create and compile the safety-oriented acados OCP solver."""

    if mass <= 0.0 or not np.isfinite(mass):
        raise ValueError("mass must be finite and positive")
    gravity = _vector(gravity, 3, "gravity")

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    json_file = output_dir.parent / f"{model_name}.json"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    if json_file.exists():
        json_file.unlink()

    ocp = at.AcadosOcp()
    ocp.model = _build_model(config, model_name)

    path_weights = np.concatenate(
        [
            config.position_weight,
            config.velocity_weight,
            config.attitude_weight,
            config.ctbr_weight,
            config.slew_weight,
        ]
    )
    terminal_weights = (
        np.concatenate(
            [
                config.position_weight,
                config.velocity_weight,
                config.attitude_weight,
                config.ctbr_weight,
            ]
        )
        * config.terminal_weight_scale
    )

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag(path_weights)
    ocp.cost.W_e = np.diag(terminal_weights)
    ocp.cost.yref = np.zeros(path_weights.size)
    ocp.cost.yref_e = np.zeros(terminal_weights.size)

    hover_thrust = mass * float(np.linalg.norm(gravity))
    x0 = np.zeros(AUGMENTED_STATE_DIM)
    x0[6] = 1.0
    x0[10] = np.clip(hover_thrust, config.thrust_min, config.thrust_max)
    ocp.constraints.x0 = x0

    # OCP inputs are physical CTBR commands.
    ocp.constraints.idxbu = np.arange(CONTROL_DIM, dtype=int)
    ocp.constraints.lbu = np.concatenate([[config.thrust_min], -config.body_rate_max])
    ocp.constraints.ubu = np.concatenate([[config.thrust_max], config.body_rate_max])

    # Bound physical velocity and the CTBR command states at shooting nodes.
    state_bound_indices = np.array([3, 4, 5, 10, 11, 12, 13], dtype=int)
    state_lower = np.concatenate(
        [
            -config.velocity_max,
            [config.thrust_min],
            -config.body_rate_max,
        ]
    )
    state_upper = np.concatenate(
        [
            config.velocity_max,
            [config.thrust_max],
            config.body_rate_max,
        ]
    )
    ocp.constraints.idxbx = state_bound_indices
    ocp.constraints.lbx = state_lower
    ocp.constraints.ubx = state_upper
    ocp.constraints.idxbx_e = state_bound_indices
    ocp.constraints.lbx_e = state_lower
    ocp.constraints.ubx_e = state_upper

    min_cos_tilt = math.cos(config.max_tilt_rad)
    maximum_delta = config.control_dt * np.concatenate(
        [[config.thrust_slew_max], config.body_rate_slew_max]
    )
    path_lower = np.concatenate([[min_cos_tilt], -maximum_delta])
    path_upper = np.concatenate([[1.0], maximum_delta])
    ocp.constraints.lh = path_lower
    ocp.constraints.uh = path_upper
    ocp.constraints.lh_0 = path_lower
    ocp.constraints.uh_0 = path_upper
    ocp.constraints.lh_e = np.array([min_cos_tilt])
    ocp.constraints.uh_e = np.array([1.0])

    default_parameter = np.concatenate([[mass], gravity, [1.0, 0.0, 0.0, 0.0]])
    ocp.parameter_values = default_parameter

    ocp.solver_options.N_horizon = config.horizon_steps
    # Keep this for compatibility with older acados_template releases.
    if hasattr(ocp, "dims") and hasattr(ocp.dims, "N"):
        ocp.dims.N = config.horizon_steps
    ocp.solver_options.tf = config.horizon_steps * config.control_dt
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver_iter_max = config.qp_max_iterations
    ocp.solver_options.qp_tol = config.qp_tolerance
    ocp.solver_options.print_level = 0

    ocp.code_export_directory = str(output_dir)
    return at.AcadosOcpSolver(ocp, json_file=str(json_file))


class NMPCController(AuxiliaryController):
    """CTBR NMPC with trajectory tracking, delay handling and fail-safe output.

    The legacy call signature remains valid::

        command = controller.calculate_control(state, state_dot, desired_state)

    For streamed hardware operation, also provide ``state_age_s`` and inspect
    ``controller.last_info``. Simulink should use ``last_info.valid`` as one of
    the conditions for accepting the packet.
    """

    _next_id = 1

    def __init__(
        self,
        mass,
        gravity,
        config=None,
    ):
        super().__init__()
        self.mass = float(mass)
        self.gravity = _vector(gravity, 3, "gravity")
        self.config = config if config is not None else NMPCConfig()

        self.instance_id = NMPCController._next_id
        NMPCController._next_id += 1
        model_name = f"qdrone2_ctbr_nmpc_{self.instance_id}"
        output_dir = (
            Path(self.config.code_export_root) / f"c_generated_code_{model_name}"
        )

        self.ocp_solver = build_acados_ocp_solver(
            self.mass,
            self.gravity,
            self.config,
            model_name,
            output_dir,
        )
        self.N_horizon = self.config.horizon_steps

        hover = self.mass * float(np.linalg.norm(self.gravity))
        self.hover_command = self._clip_command(
            np.array([hover, 0.0, 0.0, 0.0], dtype=float)
        )
        self.last_command = self.hover_command.copy()
        self.last_valid_command = self.hover_command.copy()
        self.last_state_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.previous_x_solution = None
        self.previous_u_solution = None
        self.last_info = ControlInfo(
            valid=False,
            solver_status=5,
            solver_time_s=0.0,
            wall_time_s=0.0,
            state_age_s=0.0,
            used_fallback=True,
            reason="controller initialized; no solve performed",
        )

    def calculate_control(
        self,
        state,
        state_dot,
        desired_state,
        *,
        desired_ctbr=None,
        state_age_s=0.0,
        fallback_control=None,
        return_info=False,
    ):
        """Calculate a bounded CTBR command.

        Parameters
        ----------
        state:
            Current state. The first 10 values must be
            [position(3), velocity(3), quaternion_wxyz(4)].
        state_dot:
            Optional derivative of the first 10 state entries. It is used only
            for timestamp/delay compensation. Pass ``None`` if unavailable.
        desired_state:
            A single desired state, an (N+1)-state trajectory,
            :class:`ReferenceTrajectory`, or a mapping containing ``states``
            (or ``x``) and optional ``ctbr``.
        desired_ctbr:
            Optional feed-forward CTBR trajectory. Explicitly supplied data
            takes precedence over a mapping's ``ctbr`` value.
        state_age_s:
            Age of the received state when the solve begins. The state is
            propagated forward by this amount. Calls exceeding the configured
            maximum age are rejected.
        fallback_control:
            Optional valid CTBR command from the original Quanser controller.
            It is used when NMPC cannot produce an acceptable solution.
        return_info:
            If true, return ``(command, ControlInfo)``.
        """

        wall_start = time.perf_counter()
        try:
            age = float(state_age_s)
            if not np.isfinite(age) or age < 0.0:
                raise ValueError("state_age_s must be finite and nonnegative")
            if age > self.config.max_state_age_s:
                return self._fallback_result(
                    fallback_control,
                    status=7,
                    solver_time_s=0.0,
                    wall_start=wall_start,
                    state_age_s=age,
                    reason=(
                        f"state is stale ({age:.4f} s > "
                        f"{self.config.max_state_age_s:.4f} s)"
                    ),
                    return_info=return_info,
                )

            measured_state = self._validate_physical_state(state)
            measured_state[6:10] = _align_quaternion_np(
                measured_state[6:10], self.last_state_quaternion
            )
            self.last_state_quaternion = measured_state[6:10].copy()

            compensated_state = self._compensate_state_delay(
                measured_state, state_dot, age
            )
            state_reference, ctbr_reference = self._prepare_references(
                desired_state,
                desired_ctbr,
                compensated_state[6:10],
            )

            self._set_problem_data(
                compensated_state,
                state_reference,
                ctbr_reference,
            )
            self._warm_start(compensated_state)

            x0_augmented = np.concatenate([compensated_state, self.last_command])
            self._constraints_set(0, "lbx", x0_augmented)
            self._constraints_set(0, "ubx", x0_augmented)

            status = int(self.ocp_solver.solve())
            wall_time = time.perf_counter() - wall_start
            solver_time = self._get_solver_time(default=wall_time)

            if status != 0:
                return self._fallback_result(
                    fallback_control,
                    status=status,
                    solver_time_s=solver_time,
                    wall_start=wall_start,
                    state_age_s=age,
                    reason=f"acados returned status {status}",
                    return_info=return_info,
                )
            if wall_time > self.config.max_solver_wall_time_s:
                return self._fallback_result(
                    fallback_control,
                    status=7,
                    solver_time_s=solver_time,
                    wall_start=wall_start,
                    state_age_s=age,
                    reason=(
                        f"solver missed deadline ({wall_time:.4f} s > "
                        f"{self.config.max_solver_wall_time_s:.4f} s)"
                    ),
                    return_info=return_info,
                )

            candidate = np.asarray(self.ocp_solver.get(0, "u"), dtype=float).reshape(-1)
            if candidate.size != CTBR_DIM:
                raise RuntimeError(
                    "acados returned an unexpected CTBR command dimension"
                )
            if not np.all(np.isfinite(candidate)):
                raise FloatingPointError("NMPC produced NaN or infinity")

            command = self._apply_output_safety(candidate)
            self.last_command = command.copy()
            self.last_valid_command = command.copy()
            self._store_solution()

            self.last_info = ControlInfo(
                valid=True,
                solver_status=status,
                solver_time_s=solver_time,
                wall_time_s=time.perf_counter() - wall_start,
                state_age_s=age,
                used_fallback=False,
                reason="success",
            )
            return (command.copy(), self.last_info) if return_info else command.copy()

        except Exception as exc:
            return self._fallback_result(
                fallback_control,
                status=-1,
                solver_time_s=0.0,
                wall_start=wall_start,
                state_age_s=self._safe_nonnegative_float(state_age_s),
                reason=f"controller exception: {exc}",
                return_info=return_info,
            )

    def reset(self, initial_command=None):
        """Reset warm-start memory and the held CTBR command."""

        command = self.hover_command if initial_command is None else initial_command
        command = self._clip_command(command)
        self.last_command = command.copy()
        self.last_valid_command = command.copy()
        self.last_state_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.previous_x_solution = None
        self.previous_u_solution = None
        if hasattr(self.ocp_solver, "reset"):
            try:
                self.ocp_solver.reset()
            except Exception:
                pass
        self.last_info = ControlInfo(
            valid=False,
            solver_status=5,
            solver_time_s=0.0,
            wall_time_s=0.0,
            state_age_s=0.0,
            used_fallback=True,
            reason="controller reset",
        )

    def _validate_physical_state(self, state):
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.size < PHYSICAL_STATE_DIM:
            raise ValueError(f"state must have at least {PHYSICAL_STATE_DIM} values")
        state = state[:PHYSICAL_STATE_DIM].copy()
        if not np.all(np.isfinite(state)):
            raise ValueError("state contains NaN or infinity")
        state[6:10] = _normalize_quaternion_np(state[6:10])
        return state

    def _prepare_references(
        self,
        desired_state,
        desired_ctbr,
        quaternion_anchor,
    ):
        mapping_ctbr = None
        if isinstance(desired_state, ReferenceTrajectory):
            state_data = desired_state.states
            mapping_ctbr = desired_state.ctbr
        elif isinstance(desired_state, Mapping):
            if "states" in desired_state:
                state_data = desired_state["states"]
            elif "x" in desired_state:
                state_data = desired_state["x"]
            else:
                raise ValueError("reference mapping must contain 'states' or 'x'")
            mapping_ctbr = desired_state.get("ctbr")
        else:
            state_data = desired_state

        reference = np.asarray(state_data, dtype=float)
        if reference.ndim == 1:
            if reference.size < PHYSICAL_STATE_DIM:
                raise ValueError("desired_state must have at least 10 values")
            reference = np.repeat(
                reference[None, :PHYSICAL_STATE_DIM],
                self.N_horizon + 1,
                axis=0,
            )
        elif reference.ndim == 2:
            if reference.shape[1] < PHYSICAL_STATE_DIM:
                raise ValueError("desired trajectory must have at least 10 columns")
            reference = reference[:, :PHYSICAL_STATE_DIM]
            reference = self._pad_or_truncate(reference, self.N_horizon + 1)
        else:
            raise ValueError("desired_state must be one- or two-dimensional")

        if not np.all(np.isfinite(reference)):
            raise ValueError("desired trajectory contains NaN or infinity")

        previous_quaternion = _normalize_quaternion_np(quaternion_anchor)
        for index in range(reference.shape[0]):
            reference[index, 6:10] = _align_quaternion_np(
                reference[index, 6:10], previous_quaternion
            )
            previous_quaternion = reference[index, 6:10]

        chosen_ctbr = desired_ctbr if desired_ctbr is not None else mapping_ctbr
        if chosen_ctbr is None:
            ctbr_reference = np.repeat(
                self.hover_command[None, :], self.N_horizon + 1, axis=0
            )
        else:
            ctbr_reference = np.asarray(chosen_ctbr, dtype=float)
            if ctbr_reference.ndim == 1:
                ctbr_reference = np.repeat(
                    ctbr_reference[None, :], self.N_horizon + 1, axis=0
                )
            elif ctbr_reference.ndim == 2:
                ctbr_reference = self._pad_or_truncate(
                    ctbr_reference, self.N_horizon + 1
                )
            else:
                raise ValueError("desired_ctbr must be one- or two-dimensional")
            if ctbr_reference.shape[1] != CTBR_DIM:
                raise ValueError("desired_ctbr must have four columns")
            if not np.all(np.isfinite(ctbr_reference)):
                raise ValueError("desired_ctbr contains NaN or infinity")
            ctbr_reference = np.vstack(
                [self._clip_command(row) for row in ctbr_reference]
            )

        return reference, ctbr_reference

    @staticmethod
    def _pad_or_truncate(array, rows):
        if array.shape[0] == 0:
            raise ValueError("reference trajectory cannot be empty")
        if array.shape[0] >= rows:
            return array[:rows].copy()
        padding = np.repeat(array[-1:, :], rows - array.shape[0], axis=0)
        return np.concatenate([array, padding], axis=0)

    def _set_problem_data(
        self,
        state,
        state_reference,
        ctbr_reference,
    ):
        zero_attitude_error = np.zeros(3)
        zero_slew = np.zeros(4)

        for stage in range(self.N_horizon):
            parameter = np.concatenate(
                [[self.mass], self.gravity, state_reference[stage, 6:10]]
            )
            self.ocp_solver.set(stage, "p", parameter)
            y_ref = np.concatenate(
                [
                    state_reference[stage, 0:3],
                    state_reference[stage, 3:6],
                    zero_attitude_error,
                    ctbr_reference[stage],
                    zero_slew,
                ]
            )
            self._cost_set(stage, "yref", y_ref)

        terminal_parameter = np.concatenate(
            [[self.mass], self.gravity, state_reference[-1, 6:10]]
        )
        self.ocp_solver.set(self.N_horizon, "p", terminal_parameter)
        terminal_y_ref = np.concatenate(
            [
                state_reference[-1, 0:3],
                state_reference[-1, 3:6],
                zero_attitude_error,
                ctbr_reference[-1],
            ]
        )
        self._cost_set(self.N_horizon, "yref", terminal_y_ref)

    def _warm_start(self, state):
        if self.previous_x_solution is None or self.previous_u_solution is None:
            x_guess = np.concatenate([state, self.last_command])
            for stage in range(self.N_horizon + 1):
                self.ocp_solver.set(stage, "x", x_guess)
            for stage in range(self.N_horizon):
                self.ocp_solver.set(stage, "u", self.last_command)
            return

        for stage in range(self.N_horizon):
            source_x = min(stage + 1, self.N_horizon)
            source_u = min(stage + 1, self.N_horizon - 1)
            self.ocp_solver.set(stage, "x", self.previous_x_solution[source_x])
            self.ocp_solver.set(stage, "u", self.previous_u_solution[source_u])
        self.ocp_solver.set(self.N_horizon, "x", self.previous_x_solution[-1])

    def _store_solution(self):
        x_solution = np.zeros((self.N_horizon + 1, AUGMENTED_STATE_DIM), dtype=float)
        u_solution = np.zeros((self.N_horizon, CONTROL_DIM), dtype=float)
        for stage in range(self.N_horizon):
            x_solution[stage] = np.asarray(
                self.ocp_solver.get(stage, "x"), dtype=float
            ).reshape(-1)
            u_solution[stage] = np.asarray(
                self.ocp_solver.get(stage, "u"), dtype=float
            ).reshape(-1)
        x_solution[-1] = np.asarray(
            self.ocp_solver.get(self.N_horizon, "x"), dtype=float
        ).reshape(-1)
        self.previous_x_solution = x_solution
        self.previous_u_solution = u_solution

    def _compensate_state_delay(
        self,
        state,
        state_dot,
        age,
    ):
        if age <= 0.0:
            return state.copy()

        if state_dot is not None:
            derivative = np.asarray(state_dot, dtype=float).reshape(-1)
            if derivative.size >= PHYSICAL_STATE_DIM:
                derivative = derivative[:PHYSICAL_STATE_DIM]
                if np.all(np.isfinite(derivative)):
                    predicted = state + age * derivative
                    predicted[6:10] = _align_quaternion_np(predicted[6:10], state[6:10])
                    return predicted

        # Fall back to RK4 propagation with the last held CTBR command.
        steps = max(1, int(math.ceil(age / self.config.control_dt)))
        dt = age / steps
        predicted = state.copy()
        for _ in range(steps):
            k1 = self._physical_dynamics_np(predicted, self.last_command)
            k2 = self._physical_dynamics_np(
                predicted + 0.5 * dt * k1, self.last_command
            )
            k3 = self._physical_dynamics_np(
                predicted + 0.5 * dt * k2, self.last_command
            )
            k4 = self._physical_dynamics_np(predicted + dt * k3, self.last_command)
            predicted += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            predicted[6:10] = _align_quaternion_np(predicted[6:10], state[6:10])
        return predicted

    def _physical_dynamics_np(self, state, ctbr):
        quaternion = _normalize_quaternion_np(state[6:10])
        qw, qx, qy, qz = quaternion
        rotation = np.array(
            [
                [
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qw * qz),
                    2 * (qx * qz + qw * qy),
                ],
                [
                    2 * (qx * qy + qw * qz),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qw * qx),
                ],
                [
                    2 * (qx * qz - qw * qy),
                    2 * (qy * qz + qw * qx),
                    1 - 2 * (qx * qx + qy * qy),
                ],
            ],
            dtype=float,
        )
        thrust = float(ctbr[0])
        body_rates = np.asarray(ctbr[1:4], dtype=float)
        acceleration = rotation @ np.array([0.0, 0.0, thrust]) / self.mass
        acceleration += self.gravity

        omega_quaternion = np.concatenate([[0.0], body_rates])
        quaternion_derivative = 0.5 * self._quat_product_np(
            quaternion, omega_quaternion
        )
        return np.concatenate([state[3:6], acceleration, quaternion_derivative])

    @staticmethod
    def _quat_product_np(q1, q2):
        return np.concatenate(
            [
                [q1[0] * q2[0] - np.dot(q1[1:4], q2[1:4])],
                q1[0] * q2[1:4] + q2[0] * q1[1:4] + np.cross(q1[1:4], q2[1:4]),
            ]
        )

    def _clip_command(self, command):
        command = _vector(command, CTBR_DIM, "CTBR command")
        command[0] = np.clip(command[0], self.config.thrust_min, self.config.thrust_max)
        command[1:4] = np.clip(
            command[1:4], -self.config.body_rate_max, self.config.body_rate_max
        )
        return command

    def _apply_output_safety(self, candidate):
        candidate = self._clip_command(candidate)
        maximum_delta = self.config.control_dt * np.concatenate(
            [[self.config.thrust_slew_max], self.config.body_rate_slew_max]
        )
        candidate = np.clip(
            candidate,
            self.last_command - maximum_delta,
            self.last_command + maximum_delta,
        )
        return self._clip_command(candidate)

    def _fallback_result(
        self,
        fallback_control,
        *,
        status,
        solver_time_s,
        wall_start,
        state_age_s,
        reason,
        return_info,
    ):
        try:
            target = (
                self.last_valid_command
                if fallback_control is None
                else self._clip_command(fallback_control)
            )
        except Exception:
            target = self.last_valid_command
            reason += "; supplied fallback_control was invalid"

        command = self._apply_output_safety(target)
        self.last_command = command.copy()
        self.last_info = ControlInfo(
            valid=False,
            solver_status=int(status),
            solver_time_s=float(solver_time_s),
            wall_time_s=time.perf_counter() - wall_start,
            state_age_s=float(state_age_s),
            used_fallback=True,
            reason=reason,
        )
        return (command.copy(), self.last_info) if return_info else command.copy()

    @staticmethod
    def _safe_nonnegative_float(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(number, 0.0) if np.isfinite(number) else 0.0

    def _get_solver_time(self, default):
        try:
            value = float(self.ocp_solver.get_stats("time_tot"))
            return value if np.isfinite(value) and value >= 0.0 else default
        except Exception:
            return default

    def _constraints_set(self, stage, field, value):
        if hasattr(self.ocp_solver, "constraints_set"):
            self.ocp_solver.constraints_set(stage, field, value)
        else:
            self.ocp_solver.set(stage, field, value)

    def _cost_set(self, stage, field, value):
        if hasattr(self.ocp_solver, "cost_set"):
            self.ocp_solver.cost_set(stage, field, value)
        else:
            self.ocp_solver.set(stage, field, value)


def make_controller():
    """Factory retaining the original project entry point."""

    return NMPCController(1.54, np.array([0.0, 0.0, -9.81]))


if __name__ == "__main__":
    # Creating the controller generates and compiles the acados solver.
    controller = make_controller()
    print(
        "Generated QDrone2 CTBR NMPC. Hover command:",
        controller.hover_command,
    )
