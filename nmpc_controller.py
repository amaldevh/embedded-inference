import acados_template as at
import numpy as np
import casadi as ca
from scipy.spatial.transform import Rotation
import scipy
import os
import shutil
import sys
import time


# from DATT.learning.auxiliary_controller import AuxiliaryController
class AuxiliaryController: ...


def quadrotor_dynamics_ctbr():
    """Defines the quadrotor dynamics using CasADi for Collective thrust and body rates."""
    # Define the state and control variables
    x = ca.SX.sym("x", 10)  # State: [position(3), velocity(3), quaternion(4)]
    u = ca.SX.sym("u", 4)  # Control: [thrust, bodyrate_x, bodyrate_y, bodyrate_z]

    # Parameters
    mass = ca.SX.sym("mass")
    gravity = ca.SX.sym("gravity", 3)

    # Create the dynamics equations
    pos = x[0:3]
    vel = x[3:6]
    quat = x[6:10]
    quat_n = quat / ca.norm_2(quat)  # Normalize quaternion
    thrust = u[0]
    bodyrates = u[1:4]
    R = ca.SX(3, 3)

    def quat_prod(q1, q2):
        qw1 = q1[0]
        qw2 = q2[0]
        qv1 = q1[1:]
        qv2 = q2[1:]
        qw3 = qw1 * qw2 - ca.dot(qv1, qv2)
        qv3 = qw1 * qv2 + qw2 * qv1 + ca.cross(qv1, qv2)
        return ca.vertcat(qw3, qv3)

    # Rotation matrix from quaternion
    R[0, 0] = 1 - 2 * (quat_n[2] ** 2 + quat_n[3] ** 2)
    R[0, 1] = 2 * (quat_n[1] * quat_n[2] - quat_n[0] * quat_n[3])
    R[0, 2] = 2 * (quat_n[1] * quat_n[3] + quat_n[0] * quat_n[2])
    R[1, 0] = 2 * (quat_n[1] * quat_n[2] + quat_n[0] * quat_n[3])
    R[1, 1] = 1 - 2 * (quat_n[1] ** 2 + quat_n[3] ** 2)
    R[1, 2] = 2 * (quat_n[2] * quat_n[3] - quat_n[0] * quat_n[1])
    R[2, 0] = 2 * (quat_n[1] * quat_n[3] - quat_n[0] * quat_n[2])
    R[2, 1] = 2 * (quat_n[2] * quat_n[3] + quat_n[0] * quat_n[1])
    R[2, 2] = 1 - 2 * (quat_n[1] ** 2 + quat_n[2] ** 2)
    acc = (1 / mass) * (R @ ca.vertcat(0, 0, thrust)) + gravity
    quaternion_dot = 0.5 * quat_prod(quat, ca.vertcat(0, bodyrates))

    x_dot = ca.vertcat(vel, acc, quaternion_dot)
    return ca.Function("quadrotor_dynamics_ctbr", [x, u, mass, gravity], [x_dot])


def acados_ocp_solver(mass, gravity, output_dir="./c_generated_code_ocp"):
    """Creates an acados OCP Solver for a quadrotor dynamics.
    Args:
        mass (float): Mass of the quadrotor.
        gravity (np.ndarray): Gravity vector.
        output_dir (str): Directory to export the generated C code.
    """
    if os.path.exists("acados_ocp.json"):
        os.remove("acados_ocp.json")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    ocp = at.AcadosOcp()
    model = ocp.model
    model.name = "quadrotor"
    # Set states and control syms
    model.x = ca.SX.sym("x", 10)
    model.u = ca.SX.sym("u", 4)

    # Get dynamics expression
    p_mass = ca.SX.sym("mass")
    p_gravity = ca.SX.sym("gravity", 3)
    model.p = ca.vertcat(p_mass, p_gravity)
    dm = quadrotor_dynamics_ctbr()
    model.f_expl_expr = dm(model.x, model.u, p_mass, p_gravity)
    ocp.parameter_values = np.concatenate([[mass], gravity])

    # Set cost parameters
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    # set cost
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    # Tune this weight matrix to get desired performance
    # For State tune Q, for control tune R
    # Intermediate weights: Increased Q from stable set, kept R high for stability
    Q_mat = np.diag(
        [
            30.0,
            30.0,
            30.0,
            15.0,
            15.0,
            15.0,
            0.5 / 16 * 16,
            0.5 / 16 * 16,
            0.5 / 16 * 16,
            0.5 / 16 * 16,
        ]
    )
    R_mat = np.diag([0.05 / 8, 0.1 / 2 * 2, 0.1 / 2 * 2, 0.1 / 2 * 2])
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat

    # Cost expressions (using Q and R)
    # For terminal cost, we don't have control input
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # set constraints
    ocp.constraints.lbu = np.array([0.0, -3.0, -3.0, -3.0])
    ocp.constraints.ubu = np.array([+20.0, 3.0, 3.0, 3.0])
    # If using solve_for_x0, need to set initial condition constraints
    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    # Indices where control bounds are applied (here all controls)
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # set prediction horizon
    ocp.solver_options.N_horizon = 10
    ocp.dims.N = 10
    ocp.solver_options.tf = 1000e-3 / 5  # 0.2 seconds horizon
    ocp.solver_options.qp_solver_iter_max = 50
    # set ocp options
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_tol = 1e-8
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"

    ocp.code_export_directory = output_dir
    ocp_solver = at.AcadosOcpSolver(ocp)

    print(f" Acados C code generated successfully in: {output_dir}")
    print(f"  - Model name: {model.name}")
    print(f"  - State dimension: {nx}")
    print(f"  - Control dimension: {nu}")
    print(f"  - Prediction horizon: {ocp.solver_options.N_horizon}")

    return ocp_solver, ocp.solver_options.N_horizon


class NMPCController(AuxiliaryController):
    """NMPC Controller using acados for quadrotor trajectory tracking."""

    idx = 1

    def __init__(self, mass, gravity):
        """NMPC Controller using acados for quadrotor trajectory tracking.
        Args:
            mass (float): Mass of the quadrotor.
            gravity (np.ndarray): Gravity vector.
        """
        super().__init__()
        self.idx = NMPCController.idx
        NMPCController.idx += 1
        self.mass = mass
        self.gravity = gravity
        self.u_ref = np.array([mass * np.linalg.norm(gravity), 0.0, 0.0, 0.0])
        self.u_opt = self.u_ref.copy()
        ocp_solver, N_horizon = acados_ocp_solver(
            self.mass, self.gravity, f"c_generated_code_{self.idx}"
        )
        self.ocp_solver = ocp_solver
        self.N_horizon = N_horizon

    def calculate_control(self, state, state_dot, desired_state):
        """Calculate thrust and moments using NMPC.
        Args:
            state (np.ndarray): Current state of the UAV.
            state_dot (np.ndarray): Current state derivative of the UAV.
            desired_state (np.ndarray): Desired state of the UAV.
        Returns:
            np.ndarray: Calculated thrust and moments.
        """
        # Set initial state constraint
        # self.ocp_solver.set(0, "x", state)
        # # Set reference trajectory
        des_state_ = desired_state[:10]
        state_ = state[:10]
        for i in range(self.N_horizon):
            self.ocp_solver.set(i, "yref", np.concatenate((des_state_, self.u_ref)))
        self.ocp_solver.set(self.N_horizon, "yref", des_state_)
        # status = self.ocp_solver.solve()
        self.u_opt = self.ocp_solver.solve_for_x0(
            x0_bar=state[:10]
        )  # , fail_on_nonzero_status = True)
        return self.u_opt

    def reset(self): ...


def make_controller():
    return NMPCController(1.54, np.array((0, 0, -9.81)))


if __name__ == "__main__":
    controller = NMPCController(1, np.array((0, 0, -9.81)))
    controller2 = NMPCController(1, np.array((0, 0, -9.81)))
