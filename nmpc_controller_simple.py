import os
import shutil

import acados_template as at
import casadi as ca
import numpy as np


class AuxiliaryController: ...


NX = 10
NU = 4
N = 10
DT = 0.02


def quadrotor_dynamics_ctbr():
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    mass = ca.SX.sym("mass")
    gravity = ca.SX.sym("gravity", 3)

    velocity = x[3:6]
    q = x[6:10]
    q = q / ca.sqrt(ca.sumsqr(q) + 1e-12)
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    R = ca.vertcat(
        ca.horzcat(1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)),
        ca.horzcat(2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)),
        ca.horzcat(2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)),
    )

    acceleration = R @ ca.vertcat(0.0, 0.0, u[0]) / mass + gravity
    omega = u[1:4]
    q_dot = 0.5 * ca.vertcat(
        -ca.dot(q[1:4], omega),
        q[0] * omega + ca.cross(q[1:4], omega),
    )

    return ca.Function(
        "quadrotor_dynamics_ctbr",
        [x, u, mass, gravity],
        [ca.vertcat(velocity, acceleration, q_dot)],
    )


def build_solver(mass, gravity, output_dir, model_name):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    ocp = at.AcadosOcp()
    model = at.AcadosModel()
    model.name = model_name
    model.x = ca.SX.sym("x", NX)
    model.u = ca.SX.sym("u", NU)

    p_mass = ca.SX.sym("mass")
    p_gravity = ca.SX.sym("gravity", 3)
    model.p = ca.vertcat(p_mass, p_gravity)
    model.f_expl_expr = quadrotor_dynamics_ctbr()(
        model.x, model.u, p_mass, p_gravity
    )
    ocp.model = model
    ocp.parameter_values = np.r_[mass, gravity]

    # y = [state, control]. This is linear, so do not use NONLINEAR_LS.
    ny = NX + NU
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.Vx = np.zeros((ny, NX))
    ocp.cost.Vx[:NX] = np.eye(NX)
    ocp.cost.Vu = np.zeros((ny, NU))
    ocp.cost.Vu[NX:] = np.eye(NU)
    ocp.cost.Vx_e = np.eye(NX)

    ocp.cost.W = np.diag([
        30, 30, 30,
        15, 15, 15,
        0.5, 0.5, 0.5, 0.5,
        0.00625, 0.1, 0.1, 0.1,
    ])
    ocp.cost.W_e = np.diag([
        30, 30, 30,
        15, 15, 15,
        0.5, 0.5, 0.5, 0.5,
    ])
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(NX)

    ocp.constraints.idxbu = np.arange(NU)
    ocp.constraints.lbu = np.array([0.0, -1.0, -1.3, -0.25])
    ocp.constraints.ubu = np.array([20.0, 1.0, 1.3, 0.25])
    ocp.constraints.x0 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=float)

    ocp.solver_options.N_horizon = N
    ocp.dims.N = N
    ocp.solver_options.tf = N * DT
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.qp_tol = 1e-4
    ocp.solver_options.print_level = 0
    ocp.code_export_directory = output_dir

    solver = at.AcadosOcpSolver(ocp, json_file=f"{model_name}.json")
    return solver


class NMPCController(AuxiliaryController):
    _next_id = 1

    def __init__(self, mass, gravity):
        super().__init__()
        self.mass = float(mass)
        self.gravity = np.asarray(gravity, dtype=float)
        self.u_hover = np.array([
            self.mass * np.linalg.norm(self.gravity), 0.0, 0.0, 0.0
        ])
        self.last_u = self.u_hover.copy()
        self.u_min = np.array([0.0, -1.0, -1.3, -0.25])
        self.u_max = np.array([20.0, 1.0, 1.3, 0.25])

        i = NMPCController._next_id
        NMPCController._next_id += 1
        self.ocp_solver = build_solver(
            self.mass,
            self.gravity,
            f"c_generated_code_{i}",
            f"quadrotor_ctbr_{i}",
        )
        self.yref = np.r_[np.zeros(NX), self.u_hover]

    @staticmethod
    def _state(value):
        x = np.asarray(value, dtype=float).reshape(-1)[:NX].copy()
        if x.size != NX or not np.all(np.isfinite(x)):
            raise ValueError
        norm = np.linalg.norm(x[6:10])
        if norm < 1e-8:
            raise ValueError
        x[6:10] /= norm
        return x

    def calculate_control(self, state, state_dot, desired_state):
        """Return [thrust, body_rate_x, body_rate_y, body_rate_z]."""
        try:
            x = self._state(state)
            x_ref = self._state(desired_state)

            # q and -q are the same attitude. Pick the closest representation.
            if np.dot(x[6:10], x_ref[6:10]) < 0.0:
                x_ref[6:10] *= -1.0

            self.yref[:NX] = x_ref
            for stage in range(N):
                self.ocp_solver.set(stage, "yref", self.yref)
            self.ocp_solver.set(N, "yref", x_ref)

            u = np.asarray(
                self.ocp_solver.solve_for_x0(x0_bar=x), dtype=float
            ).reshape(-1)
            if u.size != NU or not np.all(np.isfinite(u)):
                return self.last_u.copy()

            self.last_u = np.clip(u, self.u_min, self.u_max)
            return self.last_u.copy()

        except Exception:
            return self.last_u.copy()

    def reset(self):
        self.last_u = self.u_hover.copy()


def make_controller():
    return NMPCController(1.54, np.array([0.0, 0.0, -9.81]))
