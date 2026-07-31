from nmpc_controller import NMPCController
import gzdrl as grl
import numpy as np
import matplotlib.pyplot as plt 
import time
def make_controller():
    return NMPCController(1.504, np.array([0,0,-9.81]))
    
from scipy.spatial.transform import Rotation

def test_trajectory_tracking(server: grl.DRLServer, 
                             controller: grl.UAVController,
                             hz: float,
                             alpha: float,
                             uav_model: str = "quadrotor",
                             uav_canonical_link: str = "quadrotor/base_link",
                             trajectory_function: callable = None) -> tuple[np.ndarray, np.ndarray]:
    """ Test the trajectory tracking performance of the given controller on the server.
     Args:
        server (grl.DRLServer): The DRL server instance.
        controller (grl.UAVController): The UAV controller to be tested.
        hz (float): Control frequency in Hz
        alpha (float): Control delay smoother- alpha*last + (1-alpha)*curr
        uav_model (str): The name of the UAV model in the simulation.
        uav_canonical_link (str): The canonical link of the UAV model to which MultiRotorPlugin is attached.
        trajectory_function (callable): A function that defines the desired trajectory. It should take time t as input and return desired position and velocity.
     Returns:
        np.ndarray: Recorded states during the test.
        np.ndarray: Desired states during the test.
    """
    des_state = np.zeros((13))
    des_state_ = np.zeros((controller.N_horizon, 13))
    des_state_[:, 6] =1.0
    des_state[6] = 1.0  # set desired quaternion w component to
    if (trajectory_function is None):
        x = lambda t:  np.cos(2*np.pi/4.0 * t) * 2.5
        y = lambda t: np.sin(3*np.pi/4.0 * t) * 2.5
        z = lambda t: 2.0 + 0.5 * np.sin (1*np.pi/4.0 * t)
        pos = lambda t: np.array([x(t), y(t), z(t)])
        vx = lambda t: -np.sin(2*np.pi/4.0 * t) * (2*np.pi/4.0) * 2.5
        vy = lambda t:  np.cos(3*np.pi/4.0 * t) * (3*np.pi/4.0) * 2.5
        vz = lambda t: 0.5 * np.cos(1*np.pi/4.0 * t) * (1*np.pi/4.0)
        vel = lambda t: np.array([vx(t), vy(t), vz(t)])
        trajectory_function = lambda t: np.concatenate((pos(t), vel(t)))
    server.reset_pos(uav_model, [0, 0, 0.15], [0, 0, 0])
    server.run_N(10)
    states = []
    desired_states = []
    controls = []
    print("Starting test...")
    last_u = np.zeros((4))
    ts = time.perf_counter()
    steps_per_control = int(1000/hz)
    for i in range(1000):
        if i%100 == 0:
            print("Iteration:", i)
        t = i * 0.01
        des_state[:6] = trajectory_function(t)
        for j in range(controller.N_horizon):
            des_state_[j,:6] =trajectory_function(t + j*0.01)
        for _ in range(steps_per_control):
            server.update_control_states()
            state, state_dot = server.control_states[uav_model][uav_canonical_link]
            states.append(state.copy())
            desired_states.append(des_state.copy())
            curr_u = controller.calculate_control(state, state_dot, des_state_)
            u = last_u *alpha + (1-alpha)*curr_u
            last_u = u
            controls.append(u)
            server.set_ctbr_cmd(uav_model, uav_canonical_link, u)
            server.run_N(1)
    te = time.perf_counter()
    print(f"Test duration: {te - ts:.2f}s")
    print("Averaged FPS:", 10000 / (te - ts))
    # plt.show()
    return np.array(states), np.array(desired_states), np.array(controls)


def plot_performace(states, desired_state, controls):
    time = np.arange(states.shape[0]) * 0.001
    fig, axs = plt.subplots(3, 2, figsize=(8, 6))
    fig2, ax2 = plt.subplots(3, 1, figsize=(8, 3))
    fig3, ax3 = plt.subplots(3, 1, figsize=(8, 3))
    fig4, ax4 = plt.subplots(4,1, figsize=(10,3))
    labels = ['x', 'y', 'z']
    rpy_labels = ['Roll', 'Pitch', 'Yaw']
    rpys = Rotation.from_quat(states[:, 6:10], scalar_first=True).as_euler('xyz')
    for i in range(3):
        axs[i][0].plot(time, states[:, i], label='Actual ' + labels[i])
        axs[i][0].plot(time, desired_state[:, i], label='Desired ' + labels[i], linestyle='--')
        axs[i][0].set_xlabel('Time (s)')
        axs[i][0].set_ylabel(labels[i] + ' Position (m)')
        axs[i][0].legend()
        axs[i][0].grid()

        axs[i][1].plot(states[:, i], states[:, i+3], label='Phase ' + labels[i])
        axs[i][1].set_xlabel(labels[i] + ' Position (m)')
        axs[i][1].set_ylabel(labels[i] + ' Velocity (m/s)')
        axs[i][1].legend()
        axs[i][1].grid()

        ax2[i].plot(time, rpys[:, i], label=rpy_labels[i])

        ax3[i].plot(time, states[:, i+3], label=labels[i] + ' Velocity')
        ax3[i].plot(time, desired_state[:, i+3], label='Desired ' + labels[i] + ' Velocity', linestyle='--')
        ax3[i].set_xlabel('Time (s)')
        ax3[i].set_ylabel(labels[i] + ' Velocity (m/s)')
        ax3[i].legend()
        ax3[i].grid()
    control_labels = ['T','WX','WY','WZ']
    for i in range(4):
        ax4[i].plot(time, controls[:,i], label =control_labels[i])
        ax4[i].legend()
        ax4[i].grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    controller = make_controller()
    server = grl.DRLServer("0", "world_simple.sdf", ["quadrotor"], False)
    A = 1.0
    B = 1.0
    
    a = 2*np.pi/15.0
    b = a*2
    trajectory_function = lambda t: np.array([ A *np.cos(a*t),
                                                  B *np.sin(b*t),
                                                  1.0,
                                                  -A*a*np.sin(a*t),
                                                   B*b*np.cos(b*t),
                                                 0.0])

    states, desired_states, controls = test_trajectory_tracking(server, controller, 50, 0.8,
                                                      uav_model="quadrotor",
                                                     uav_canonical_link="quadrotor/base_link" ,
                                                     trajectory_function=trajectory_function)
    plot_performace(states, desired_states, controls)

    
