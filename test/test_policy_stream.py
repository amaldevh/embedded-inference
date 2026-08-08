import os
import sys
import time

import bindings
import numpy as np

from quarc_stream import Streamer
from jetson_inference.runtime import TensorRTRunner


SAMPLE_TIME = 0.01
PHYSICS_STEPS_PER_CONTROL = 10

CUDA_ENGINE = "/path/to/model.engine"

GRAVITY_VECTOR = np.array(
    [0.0, 0.0, -9.81],
    dtype=np.float64,
)

OUTPUT_DIMS = 3


def make_processor(
    processor_class: type,
    state_key: str,
    physics_dt: float,
    seed: int,
):
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
        fixed_zero_yaw_probability=1.0,
        trajectory_sampling_attempts=64,
        trajectory_seed=seed,
    )


processor = make_processor(
    bindings.TrajectoryTrackingProcessor,
    "quadrotor",
    1e-3,
    42,
)

processor.update_trajectory()

INPUT_DIMS = processor.observation_dimension

runner = TensorRTRunner(CUDA_ENGINE)


trt_input = {
    "quadrotor": np.zeros(
        (1, INPUT_DIMS),
        dtype=np.float32,
    )
}

for _ in range(100):
    runner.infer(
        {"observation": trt_input["quadrotor"]}
    )


def control_tick(observation_):
    outputs = runner.infer(
        {
            "observation":
                observation_["quadrotor"]
        },
        return_cpu=True,
        synchronize=True,
    )

    action = outputs["action"][0]
    action[np.isnan(action)] = 0.0

    return action


drone_states = {
    "quadrotor":
        np.zeros(13, dtype=np.float32)
}

drone_state_dots = {
    "quadrotor":
        np.zeros(13, dtype=np.float32)
}

previous_drone_states = {
    "quadrotor":
        np.zeros(13, dtype=np.float32)
}

previous_drone_state_dots = {
    "quadrotor":
        np.zeros(13, dtype=np.float32)
}


observation_array = np.empty(
    INPUT_DIMS,
    dtype=np.float32,
)

observation = {
    "obs": observation_array
}

policy_action = {
    "action":
        np.zeros(3, dtype=np.float32)
}

processed_action = {
    "quadrotor":
        np.zeros(3, dtype=np.float32)
}


def update_states(new_state):
    np.copyto(
        drone_states["quadrotor"],
        np.asarray(
            new_state,
            dtype=np.float32,
        ).reshape(13),
    )


if __name__ == "__main__":

    with Streamer(
        "tcpip://localhost:18002",
        OUTPUT_DIMS,
        13,
        Streamer.CLIENT,
    ) as client:

        # Get the first valid physical state.
        update_states(client.receive())

        # Initialize processor history exactly once.
        processor.reset(
            drone_states["quadrotor"]
        )

        np.copyto(
            previous_drone_states["quadrotor"],
            drone_states["quadrotor"],
        )

        np.copyto(
            previous_drone_state_dots["quadrotor"],
            drone_state_dots["quadrotor"],
        )

        start_time = time.perf_counter()

        completed_steps = 0

        for i in range(100000):

            update_states(client.receive())

            processor.process_observation(
                drone_states,
                drone_state_dots,
                previous_drone_states,
                previous_drone_state_dots,
                observation,
            )

            np.copyto(
                trt_input["quadrotor"][0],
                observation_array,
            )

            action = control_tick(
                trt_input
            )

            np.copyto(
                policy_action["action"],
                action,
            )

            processor.process_action(
                policy_action,
                processed_action,
            )

            client.send(
                processed_action["quadrotor"]
            )

            np.copyto(
                previous_drone_states["quadrotor"],
                drone_states["quadrotor"],
            )

            np.copyto(
                previous_drone_state_dots["quadrotor"],
                drone_state_dots["quadrotor"],
            )

            completed_steps += 1

            # Do NOT print every control cycle onboard.
            if i % 100 == 0:
                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    "U:",
                    processed_action["quadrotor"],
                )

                print(
                    "Avg freq:",
                    completed_steps / elapsed,
                )

        end_time = time.perf_counter()


    elapsed = end_time - start_time

    print("Total time:", elapsed)

    print(
        "Avg. time per step:",
        elapsed / completed_steps,
    )

    print(
        "Avg. frequency:",
        completed_steps / elapsed,
    )