import numpy as np
from jetson_inference.runtime import TensorRTRunner
import time

runner = TensorRTRunner("artifacts/policy_fp16.engine")

# Construct, allocate, and warm before arming.
state_host = np.empty((1, 13), dtype=np.float32)
state_host.fill(0)
for _ in range(100):
    runner.infer({"state": state_host})

# Optional. Keep disabled until the normal path is verified on the target model.
#runner.capture_cuda_graph()

def control_tick(latest_state):
    # Prefer writing preprocessing results directly into a reusable pinned tensor
    # for truly asynchronous H2D transfer. This NumPy assignment is illustrative.
    state_host[0, :] = latest_state
    outputs = runner.infer(
        {"state": state_host},
        return_cpu=True,
        synchronize=True,
#        use_cuda_graph=True,
    )
    action = outputs["output_0"][0]  # persistent view, overwritten next call
    if not np.isfinite(action).all():
        raise RuntimeError("non-finite policy output")
    return action

if __name__ == "__main__":
    ts = time.time()
    for i in range(1000):
        control_tick(np.zeros((13))+i)
    tf = time.time()
    print(tf-ts)
    print(1000/(tf-ts))
