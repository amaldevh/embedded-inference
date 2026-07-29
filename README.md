# Low-latency PyTorch inference on Jetson Xavier NX

This repository provides a Python deployment path for an arbitrary
`torch.nn.Module`:

```text
PyTorch module + checkpoint
          |
          | fixed-shape ONNX export
          v
       model.onnx
          |
          | TensorRT build on the target Xavier NX
          v
       model.engine
          |
          | persistent CUDA buffers, async enqueue, optional CUDA Graph
          v
  state/sensor input -> inference -> control application
```

The recommended starting point is **TensorRT on the Xavier NX GPU, fixed batch
size 1, fixed input shapes, and FP16**. It gives a much lower-overhead deployment
than calling the eager PyTorch module, while retaining a Python application around
the optimized engine. INT8 may be faster, but it should be adopted only after
representative calibration or quantization-aware training and task-level accuracy
validation.

## Why this design

As of July 2026, Xavier NX is supported by the JetPack 5 sustaining line, not
JetPack 6 or 7. The current production release is
[JetPack 5.1.7](https://developer.nvidia.com/embedded/jetpack-sdk-517), which
ships Ubuntu 20.04, CUDA 11.4.19, cuDNN 8.6, and TensorRT 8.5.2. JetPack 5.1.7
uses the same compute stack as 5.1.6 and adds BSP/security fixes. This code targets
the TensorRT 8.5 API but also handles the current TensorRT tensor API.

TensorRT is preferable here because:

- TensorRT fuses layers and chooses kernels specifically for the Xavier NX.
- FP16 and INT8 use Xavier's Tensor Cores where supported.
- A fixed batch-1 engine avoids dynamic-shape profile and shape-update overhead.
- The runtime allocates inputs, outputs, and pinned host output buffers only once.
- `execute_async_v2` is enqueued on a non-default CUDA stream.
- Optional CUDA Graph capture replaces many small kernel launches with one replay.
  NVIDIA documents kernel-launch overhead of roughly 5–15 microseconds per kernel
  and recommends CUDA Graphs for enqueue-bound networks:
  [TensorRT performance guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html#inference-with-cuda-graphs).

For a single latency-sensitive controller, Triton/DeepStream adds machinery that
is not needed. DLA can free the GPU for other work, but its operator set is smaller,
fallbacks can be costly, and it is not automatically the lowest-latency choice.
Benchmark the GPU path first.

Python itself is not expected to make GPU execution slower: NVIDIA notes that
TensorRT inference through Python should be nearly identical to the C++ API when
buffers and synchronization are managed correctly. Python can still add jitter
around inference, so measure the complete control-loop path and move only the
deadline-critical shell to C++ if its tail latency is unacceptable.

## Repository contents

```text
jetson_inference/
  export_onnx.py       Load a Module/checkpoint and export fixed-shape ONNX
  build_engine.py      Build an FP32, FP16, or explicit-Q/DQ INT8 engine
  runtime.py           Reusable TensorRTRunner with persistent CUDA buffers
  infer.py             Command-line inference smoke test
  model_loader.py      Generic module/factory/state_dict loading
benchmarks/
  benchmark_pytorch.py Baseline original-module compute latency
  benchmark_tensorrt.py TensorRT GPU-only and host-to-host latency
examples/
  example_model.py     Small policy network used in this walkthrough
```

## Deployment plan

1. Freeze the inference contract. Define input order, names, shapes, dtypes,
   normalization, coordinate frames, output semantics, and valid ranges. Use batch
   1 and fixed shapes. For a quadrotor policy, inputs are often a single state/history
   tensor; a vision policy may have an image and a state tensor.
2. Establish a PyTorch golden set. Save representative, edge-case, and out-of-range
   inputs plus the FP32 outputs. Include flight-log samples.
3. Export ONNX on a development machine or on the Jetson. Validate the ONNX graph.
4. Transfer the ONNX file, source tree, and golden inputs to the Xavier NX.
5. Build the TensorRT engine **on the deployment Xavier NX**. Serialized TensorRT
   engines are not generally portable across TensorRT versions, CUDA stacks, GPU
   architectures, or platforms.
6. Compare TensorRT outputs with the PyTorch golden set. Validate closed-loop behavior,
   not only elementwise error.
7. Benchmark FP32 and FP16. Record GPU-only latency, host-to-host latency, p99/max
   latency, temperatures, clocks, and power mode. Try CUDA Graph replay if enqueue
   time is important.
8. Integrate one long-lived `TensorRTRunner` into the flight process. Construct it and
   warm it before arming; never load/build an engine in the control loop.
9. Run hardware-in-the-loop, tethered, and bounded-flight tests. Add deadline,
   finite-value, input-range, and output-range checks plus a tested fallback
   controller.
10. Only then evaluate explicit-Q/DQ INT8 if FP16 leaves insufficient margin.

At 100 Hz, the complete sensing-to-actuation deadline is 10 ms. The inference
benchmark should not consume that whole budget: acquisition, preprocessing, state
estimation, scheduling jitter, postprocessing, the control law, and actuator I/O
also need bounded time. This framework does not optimize to a 10 ms target; it uses
the lowest-overhead practical path and reports the achieved distribution.

## 1. Prepare the Xavier NX

### Install JetPack

Use JetPack 5.1.7 / Jetson Linux 35.6.5 when possible. NVIDIA's
[Xavier NX getting-started guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit)
describes the SD-card flow. A developer kit that has never run JetPack 5 may need a
QSPI boot-firmware update before it can boot a JetPack 5 image; follow the notice on
the [JetPack 5.1.7 page](https://developer.nvidia.com/embedded/jetpack-sdk-517).
For a production module, flash the module/storage using NVIDIA SDK Manager or the
Jetson Linux flashing tools appropriate to the carrier board.

After boot:

```bash
cat /etc/nv_tegra_release
sudo apt update
sudo apt install nvidia-jetpack python3-pip python3-venv libopenblas-dev
```

Verify the core stack:

```bash
/usr/local/cuda/bin/nvcc --version
dpkg-query -W nvinfer-bin python3-libnvinfer
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

Do not install a random desktop `tensorrt` wheel over the JetPack packages. The
JetPack-provided Python bindings must match the installed TensorRT libraries.

### Install a Jetson-compatible PyTorch

This runtime uses PyTorch CUDA tensors as TensorRT buffers, which avoids a second
CUDA-context manager and lets an upstream PyTorch/CUDA preprocessing stage pass
device data without a host round trip. Install an NVIDIA aarch64 wheel that matches
JetPack 5.1.x and Python 3.8. NVIDIA's
[PyTorch for Jetson installation guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
and
[compatibility table](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)
are the source of truth. PyTorch 2.1.0a/23.06 is listed for JetPack 5.1.x.

Create a virtual environment that can see the apt-installed TensorRT binding:

```bash
cd /path/to/Embedded-Inference
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-export.txt
```

Download the exact aarch64 wheel selected from NVIDIA's compatibility table, then:

```bash
python3 -m pip install /path/to/torch-*-cp38-cp38-linux_aarch64.whl
```

Check that PyTorch and TensorRT see the same working CUDA environment:

```bash
python3 - <<'PY'
import tensorrt as trt
import torch
print("TensorRT:", trt.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY
```

### Configure performance mode for benchmarking

Provide adequate power and active cooling. Query the modes available on the exact
module and carrier board instead of assuming that a mode number is universal:

```bash
sudo nvpmodel -q --verbose
```

Select the highest-performance supported mode shown by that command, then lock
clocks for reproducible peak-performance tests:

```bash
sudo nvpmodel -m <MAX_PERFORMANCE_MODE_ID>
sudo /usr/bin/jetson_clocks --fan
sudo /usr/bin/jetson_clocks --show
```

`jetson_clocks` sets static maximum CPU, GPU, and memory clocks, as described in
NVIDIA's
[Xavier power and performance guide](https://docs.nvidia.com/jetson/archives/r35.3.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonXavierNxSeriesAndJetsonAgxXavierSeries.html).
This raises power draw and heat. A quadrotor's final power mode must be chosen from
flight energy and thermal tests, not desktop benchmark results. Monitor a long run:

```bash
tegrastats --interval 1000
```

## 2. Adapt your model

Expose an importable Module instance, Module class, or factory. For example:

```python
# my_project/policy.py
import torch

class Policy(torch.nn.Module):
    def __init__(self, state_dim=13, action_dim=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim),
        )

    def forward(self, state):
        return self.net(state)

def create_policy(state_dim=13, action_dim=4):
    return Policy(state_dim, action_dim)
```

The loader accepts:

- a raw `state_dict`;
- a dictionary containing `state_dict`, `model_state_dict`, or `model`;
- a nested weights dictionary selected with `--checkpoint-key`;
- a serialized `nn.Module` when `--model` is omitted.

PyTorch checkpoints use pickle internally. Load only checkpoints from a trusted
source.

Keep the forward path tensor-only. Python control flow dependent on tensor values,
custom operators, data-dependent output shapes, or unsupported ONNX operators may
require rewriting the model or a TensorRT plugin. The input spec syntax is
`NAME:D0,D1,...[:DTYPE]`; repeat `--input` in forward-argument order.

## 3. Export fixed-shape ONNX

Run from the repository root so your model package is importable:

```bash
python3 -m jetson_inference.export_onnx \
  --model my_project.policy:create_policy \
  --factory-kwargs '{"state_dim": 13, "action_dim": 4}' \
  --checkpoint checkpoints/policy.pt \
  --input state:1,13:float32 \
  --output artifacts/policy.onnx
```

The default opset is 17, appropriate for TensorRT 8.5. The default is the legacy
TorchScript-based ONNX exporter because the PyTorch releases available for JetPack 5
predate today's recommended `torch.export` exporter. On a newer development machine,
`--dynamo` opts into the current exporter. Always parse and build with the target
TensorRT before treating an export as deployable.

Multiple inputs:

```bash
python3 -m jetson_inference.export_onnx \
  --model my_project.vision_policy:create_model \
  --checkpoint checkpoints/vision_policy.pt \
  --input image:1,3,224,224:float32 \
  --input state:1,13:float32 \
  --output artifacts/vision_policy.onnx
```

Smoke-test the included example without a checkpoint:

```bash
python3 -m jetson_inference.export_onnx \
  --model examples.example_model:create_model \
  --input state:1,13:float32 \
  --output artifacts/example.onnx
```

## 4. Build the TensorRT engine on the Xavier NX

```bash
python3 -m jetson_inference.build_engine \
  --onnx artifacts/policy.onnx \
  --output artifacts/policy_fp16.engine \
  --precision fp16 \
  --workspace-mib 2048 \
  --timing-cache artifacts/xavier_nx.timing
```

The workspace is a build-time upper bound used while TensorRT selects tactics; lower
it if the board cannot provide that much memory. Optimization level 5 is requested
where the installed API supports it. Building may take minutes. The `.timing` cache
speeds repeated builds on the same target stack, and the generated `.engine.json`
records bindings.

Build FP32 as an accuracy reference:

```bash
python3 -m jetson_inference.build_engine \
  --onnx artifacts/policy.onnx \
  --output artifacts/policy_fp32.engine \
  --precision fp32
```

INT8 mode assumes the ONNX graph already contains explicit QuantizeLinear /
DequantizeLinear (Q/DQ) nodes:

```bash
python3 -m jetson_inference.build_engine \
  --onnx artifacts/policy_qdq.onnx \
  --output artifacts/policy_int8.engine \
  --precision int8
```

The generic builder intentionally does not invent calibration data. A useful INT8
model requires representative flight data and task-level accuracy validation.

## 5. Validate inference

Random input:

```bash
python3 -m jetson_inference.infer \
  --engine artifacts/policy_fp16.engine
```

Real inputs can be stored in an NPZ whose keys match engine input names:

```python
import numpy as np
np.savez("golden_input.npz", state=state.astype(np.float32)[None, :])
```

```bash
python3 -m jetson_inference.infer \
  --engine artifacts/policy_fp16.engine \
  --input-npz golden_input.npz \
  --cuda-graph
```

Compare FP32 PyTorch, FP32 TensorRT, and FP16 TensorRT over the entire golden data
set. Use absolute/relative error tolerances appropriate to the model, check for
NaN/Inf, and validate the resulting control action and closed-loop trajectory.

## 6. Integrate the runtime

```python
import numpy as np
from jetson_inference.runtime import TensorRTRunner

runner = TensorRTRunner("artifacts/policy_fp16.engine")

# Construct, allocate, and warm before arming.
state_host = np.empty((1, 13), dtype=np.float32)
state_host.fill(0)
for _ in range(100):
    runner.infer({"state": state_host})

# Optional. Keep disabled until the normal path is verified on the target model.
runner.capture_cuda_graph()

def control_tick(latest_state):
    # Prefer writing preprocessing results directly into a reusable pinned tensor
    # for truly asynchronous H2D transfer. This NumPy assignment is illustrative.
    state_host[0, :] = latest_state
    outputs = runner.infer(
        {"state": state_host},
        return_cpu=True,
        synchronize=True,
        use_cuda_graph=True,
    )
    action = outputs["output_0"][0]  # persistent view, overwritten next call
    if not np.isfinite(action).all():
        raise RuntimeError("non-finite policy output")
    return action
```

Important runtime properties:

- `runner.inputs` and `runner.outputs` are persistent CUDA tensors.
- A CUDA preprocessing stage can copy or write into `runner.inputs[name]` and then
  call `runner.enqueue()` without a host round trip.
- `infer(..., return_cpu=True)` uses reusable page-locked host output tensors.
- Returned tensors/arrays are overwritten by the next inference. Copy only when a
  consumer needs to retain them.
- CUDA Graph capture fixes tensor addresses and shapes. Do not reallocate buffers or
  change shapes after capture.
- A synchronization is required before the CPU consumes an output. If the next stage
  remains on the GPU, use `return_cpu=False, synchronize=False` and coordinate with
  CUDA streams/events instead.

For a camera pipeline, avoid CPU decode/resize followed by an upload. Use Jetson
Argus/GStreamer/NVMM or a CUDA-aware camera path and GPU preprocessing, then place
the final tensor into the persistent input. Copy elimination can matter more than a
small improvement in network compute time.

## 7. Benchmark

### Original PyTorch module

```bash
python3 benchmarks/benchmark_pytorch.py \
  --model my_project.policy:create_policy \
  --factory-kwargs '{"state_dim": 13, "action_dim": 4}' \
  --checkpoint checkpoints/policy.pt \
  --input state:1,13:float32 \
  --device cuda \
  --precision fp16 \
  --warmup 200 \
  --iterations 2000 \
  --json results/pytorch_fp16.json
```

This reports device forward latency with preallocated device inputs.

### TensorRT application benchmark

```bash
python3 benchmarks/benchmark_tensorrt.py \
  --engine artifacts/policy_fp16.engine \
  --warmup 200 \
  --iterations 2000 \
  --json results/tensorrt_fp16.json
```

It reports two different quantities:

- `engine_gpu`: CUDA-event time around TensorRT only, with data already on GPU;
- `host_to_host`: pinned host input copy + enqueue + host output copy +
  synchronization + Python overhead.

Test CUDA Graph replay separately:

```bash
python3 benchmarks/benchmark_tensorrt.py \
  --engine artifacts/policy_fp16.engine \
  --cuda-graph \
  --warmup 200 \
  --iterations 2000 \
  --json results/tensorrt_fp16_graph.json
```

For an independent TensorRT measurement, use the JetPack `trtexec` binary. NVIDIA
recommends `trtexec` as the first comparison point in its
[TensorRT 8.6 performance guidance](https://archive.docs.nvidia.com/tensorrt/tensorrt-861/developer-guide/index.html#reporting-a-performance-issue):

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=artifacts/policy_fp16.engine \
  --useCudaGraph \
  --noDataTransfers \
  --useSpinWait \
  --warmUp=1000 \
  --duration=30
```

`--noDataTransfers` isolates compute. Remove it to include synthetic transfer time.
`--useSpinWait` can reduce timing variation at the cost of CPU utilization and power.

### Benchmark discipline

- Benchmark batch 1 and the exact production shapes.
- Warm until clocks, caches, and TensorRT lazy initialization are stable.
- Report p50, p95, p99, and maximum, not only mean throughput.
- Run for several minutes in the enclosure with production cooling and power supply.
- Record `tegrastats`, power mode, clocks, JetPack/TensorRT versions, precision, and
  engine hash alongside results.
- Benchmark with the camera, estimator, logging, and control process active. An idle
  microbenchmark does not expose contention or scheduling jitter.
- Test both peak-clock and intended flight-power configurations.
- Do not use multi-stream throughput tricks for a serial 100 Hz controller; they can
  improve throughput while worsening single-sample latency and predictability.

## Latency optimization order

1. Fixed shapes and batch 1.
2. TensorRT FP16, engine built on the target.
3. Persistent device and pinned-host buffers; no allocation in the loop.
4. Keep preprocessing/postprocessing on GPU when inputs are images.
5. Remove avoidable synchronization; retain the one needed before CPU actuation.
6. CUDA Graph replay if `trtexec` shows enqueue time near GPU compute time.
7. Profile with Nsight Systems or `trtexec --dumpProfile`.
8. Simplify the network or use operator-friendly replacements.
9. Explicit-Q/DQ INT8 with representative calibration/QAT and validation.
10. If Python tail jitter is still material, retain the same engine and move the thin
    runtime/control-loop shell to C++.

## Flight integration and failure handling

This framework is not a safety-certified flight-control component. Treat learned
inference as a fallible subsystem:

- Use a monotonic deadline timer around the complete tick.
- Reject stale sensor timestamps, invalid shapes/ranges, and NaN/Inf.
- Clamp or validate actions against physical limits.
- Keep a conventional, tested fallback controller and define its transition logic.
- Add an inference heartbeat/watchdog outside the inference process.
- Pre-fault memory and avoid logging, model reload, engine rebuild, garbage-producing
  allocations, and filesystem/network operations in the real-time loop.
- Pinning CPU affinity and using real-time scheduling can reduce jitter, but configure
  priorities only after analyzing interactions with sensor, actuator, and kernel
  threads. A misconfigured real-time process can starve critical I/O.
- Version the checkpoint, ONNX graph, engine, preprocessing, TensorRT/JetPack stack,
  golden data, and acceptance thresholds as one deployment artifact.
- Rebuild and requalify after any JetPack, TensorRT, CUDA, model, shape, carrier,
  power-mode, or thermal-design change.

## Troubleshooting

### `No module named tensorrt` inside the virtual environment

Recreate the environment with `--system-site-packages`, or use the same `/usr/bin/python3`
for which `python3-libnvinfer` was installed. Do not solve this by installing an
unmatched desktop wheel.

### ONNX parser failure

Read every parser error printed by `build_engine.py`. Typical causes are unsupported
operators, too-new an opset, dynamic/data-dependent shapes, or Python behavior that
was not representable in ONNX. Try opset 17, simplify the operation, and verify the
graph with ONNX tooling. NVIDIA recommends ONNX plus its TensorRT parser as the
primary framework import route:
[TensorRT architecture guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html#onnx).

### Engine deserialization failure

Build the engine again on the target with its installed TensorRT. Do not copy a
desktop-built engine or assume an engine survives a JetPack upgrade.

### CUDA Graph capture failure

The model may contain loops, conditionals, data-dependent shapes, or another
capture-incompatible operation. Use the normal async enqueue path. CUDA Graphs are an
optional optimization, not a correctness requirement.

### FP16 output is inaccurate

Compare layer/model outputs against FP32, inspect for overflow/underflow and NaN/Inf,
and keep numerically sensitive operations in FP32 by modifying the model or TensorRT
precision constraints. Do not deploy a precision solely because it benchmarks faster.

### Benchmark is fast but the flight loop misses deadlines

Measure timestamped stages separately: acquisition, conversion, preprocessing, H2D,
enqueue/GPU, D2H, postprocessing, controller, and actuator output. Check CPU/GPU/EMC
contention and thermal throttling with `tegrastats`. Tail latency usually identifies a
different problem than average engine compute time.
