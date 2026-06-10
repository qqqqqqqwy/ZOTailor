# TinyLlama on Android Snapdragon

This guide shows how to:

- prepare `TinyLlama/TinyLlama-1.1B-Chat-v1.0` from Hugging Face
- convert it to `F16 GGUF`
- quantize it to `Q4_0` and optionally `Q8_0`
- build two Android Snapdragon packages from the latest `llama.cpp`
- run a CPU-only prefill benchmark
- run a Hexagon NPU prefill benchmark
- validate the Hexagon F16 load-time HMX weight pre-tiling path
- compare prompt processing speed under identical settings

The model preparation flow in this guide follows the current upstream-compatible route:

`HF -> F16 GGUF -> optional Q4_0/Q8_0 -> Hexagon load-time repack`

Weight repacking for Hexagon is performed at load time by the Hexagon backend. This guide does not use a private offline HTP layout or a custom GGUF format.

Current implementation status:

- `F16` LLaMA trunk matmul weights are pre-tiled once during Hexagon buffer upload, then the HTP F16 HMX matmul path consumes the pre-tiled layout directly.
- `Q4_0` and `Q8_0` still use the current `x4x2` runtime repack path. The older `w4d16a32/w8d16a32`-style quantized legacy layout is not yet fully migrated.
- The layout can be controlled with `GGML_HEXAGON_WEIGHT_LAYOUT=auto|legacy|x4x2`. The default is `auto`.

## Host Preparation

Clone the model from Hugging Face with Git LFS:

```bash
mkdir -p ~/models
cd ~/models
git lfs install
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Create a Python environment for conversion:

```bash
cd ~/src/llama.cpp
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install ninja
```

If you are using the Snapdragon toolchain container, also make sure the Android, OpenCL, and Hexagon SDK environment variables are set before configuring CMake presets.

## Model Conversion and Quantization

Convert the Hugging Face model to `F16 GGUF`:

```bash
cd ~/src/llama.cpp
. .venv/bin/activate

python convert_hf_to_gguf.py \
    ~/models/TinyLlama-1.1B-Chat-v1.0 \
    --outtype f16 \
    --outfile ~/models/TinyLlama-1.1B-Chat-v1.0-F16.gguf
```

Build the quantizer:

```bash
cmake -S . -B build-host
cmake --build build-host --target llama-quantize -j8
```

Quantize the model to `Q4_0`:

```bash
./build-host/bin/llama-quantize \
    ~/models/TinyLlama-1.1B-Chat-v1.0-F16.gguf \
    ~/models/TinyLlama-1.1B-Chat-v1.0-Q4_0.gguf \
    Q4_0
```

Optionally create a `Q8_0` reference model:

```bash
./build-host/bin/llama-quantize \
    ~/models/TinyLlama-1.1B-Chat-v1.0-F16.gguf \
    ~/models/TinyLlama-1.1B-Chat-v1.0-Q8_0.gguf \
    Q8_0
```

Recommended first-pass benchmark model:

- `TinyLlama-1.1B-Chat-v1.0-F16.gguf`

Recommended quantized comparison model:

- `TinyLlama-1.1B-Chat-v1.0-Q4_0.gguf`

Optional reference model:

- `TinyLlama-1.1B-Chat-v1.0-Q8_0.gguf`

## Build Guide 1: CPU-only Package

Use the Snapdragon Android preset, but explicitly disable Hexagon and OpenCL so the package is CPU-only.

```bash
cd ~/src/llama.cpp
cp docs/backend/snapdragon/CMakeUserPresets.json .

cmake --preset arm64-android-snapdragon-release \
    -B build-android-cpu \
    -DGGML_HEXAGON=OFF \
    -DGGML_OPENCL=OFF \
    -DCMAKE_TOOLCHAIN_FILE=/home/fuyongjian/qwy/android-ndk-r25c/build/cmake/android.toolchain.cmake

cmake --build build-android-cpu -j8
cmake --install build-android-cpu --prefix pkg-android-cpu/llama.cpp
```

Expected result:

- `pkg-android-cpu/llama.cpp/bin/llama-batched-bench`
- `pkg-android-cpu/llama.cpp/bin/llama-bench`
- no `libggml-hexagon.so`

Push the CPU-only package and model to the device:

```bash
adb shell rm -rf /data/local/tmp/llama.cpp-cpu
adb shell mkdir -p /data/local/tmp/llama.cpp-cpu
adb push pkg-android-cpu/llama.cpp/* /data/local/tmp/llama.cpp-cpu

adb shell mkdir -p /data/local/tmp/gguf
adb push ~/models/TinyLlama-1.1B-Chat-v1.0-F16.gguf /data/local/tmp/gguf/
adb push ~/models/TinyLlama-1.1B-Chat-v1.0-Q4_0.gguf /data/local/tmp/gguf/
```

Basic CPU-only runtime check:

```bash
adb shell "
  cd /data/local/tmp/llama.cpp-cpu &&
  chmod +x ./bin/llama-batched-bench &&
  LD_LIBRARY_PATH=./lib ./bin/llama-batched-bench --help >/dev/null
"
```

## Build Guide 2: Hexagon NPU Package

Build the Android Snapdragon package with Hexagon enabled. Download Hexagon_SDK first:

```bash
cd /home/fuyongjian/qwy/Hexagon_SDK/6.5.0.0
source setup_sdk_env.source
cd ~/src/llama.cpp
cp docs/backend/snapdragon/CMakeUserPresets.json .

cmake --preset arm64-android-snapdragon-release \
    -B build-android-htp \
    -DGGML_HEXAGON=ON \
    -DGGML_OPENCL=OFF \
    -DCMAKE_TOOLCHAIN_FILE=/home/fuyongjian/qwy/android-ndk-r25c/build/cmake/android.toolchain.cmake

cmake --build build-android-htp -j8
cmake --install build-android-htp --prefix pkg-android-htp/llama.cpp
```

Expected result:

- `pkg-android-htp/llama.cpp/bin/llama-batched-bench`
- `pkg-android-htp/llama.cpp/bin/llama-bench`
- `pkg-android-htp/llama.cpp/lib/libggml-hexagon.so`
- `pkg-android-htp/llama.cpp/lib/libggml-htp-v73.so`
- `pkg-android-htp/llama.cpp/lib/libggml-htp-v75.so`
- `pkg-android-htp/llama.cpp/lib/libggml-htp-v79.so`
- `pkg-android-htp/llama.cpp/lib/libggml-htp-v81.so`

Push the Hexagon package and model to the device:

```bash
adb shell rm -rf /data/local/tmp/llama.cpp
adb push pkg-android-htp/llama.cpp /data/local/tmp/

adb shell mkdir -p /data/local/tmp/gguf
adb push ~/models/TinyLlama-1.1B-Chat-v1.0-F16.gguf /data/local/tmp/gguf/
adb push ~/models/TinyLlama-1.1B-Chat-v1.0-Q4_0.gguf /data/local/tmp/gguf/
adb shell chmod +x /data/local/tmp/llama.cpp/bin/llama-batched-bench
```

Basic Hexagon runtime check:

```bash
./scripts/snapdragon/adb/run-tool.sh llama-batched-bench --list-devices
```

You should see a Hexagon device such as `HTP0`.

## Run Guide 1: CPU-only Prefill Benchmark

Run `llama-batched-bench` from the CPU-only package using the same prompt-heavy settings that will be used for the NPU run.

```bash
adb shell "
  cd /data/local/tmp/llama.cpp-cpu &&
  chmod +x ./bin/llama-batched-bench
  LD_LIBRARY_PATH=./lib ./bin/llama-batched-bench \
    -m /data/local/tmp/gguf/TinyLlama-1.1B-Chat-v1.0-F16.gguf \
    --device none \
    -c 4096 \
    -b 2048 \
    -ub 512 \
    -npp 128,256,512 \
    -ntg 0 \
    -npl 1,2,4,8 \
    --output-format jsonl
"
```

What to record from the output:

- `speed_pp`
- `t_pp`
- backend should remain CPU-only
- no Hexagon initialization

## Run Guide 2: Hexagon NPU Prefill Benchmark

Run the same workload on the Hexagon package. Start with the `F16` model if you want to specifically validate the migrated HMX-friendly pre-tiled weight path.

```bash
cd ~/src/llama.cpp
M="TinyLlama-1.1B-Chat-v1.0-F16.gguf"
D="HTP0"
GGML_HEXAGON_WEIGHT_LAYOUT=auto \
./scripts/snapdragon/adb/run-tool.sh llama-batched-bench \
  -m /data/local/tmp/gguf/$M \
  --device HTP0 \
  --repack \
  -ngl 99 \
  -c 4096 \
  -b 2048 \
  -ub 512 \
  -npp 128,256,512 \
  -ntg 0 \
  -npl 1,2,4,8 \
  --output-format jsonl
```

What to record from the output and logs:

- `speed_pp`
- `t_pp`
- Hexagon backend initialization
- `HTP0-REPACK` model buffer allocation
- for `F16`, logs containing `legacy-pre-tiled-f16` on upload and `pretiled=1` in the HTP F16 HMX matmul path
- for `Q4_0` or `Q8_0`, whether quantized matmul is claimed by Hexagon and uses the existing `x4x2` repack path

Useful debug variants:

```bash
M=TinyLlama-1.1B-Chat-v1.0-F16.gguf \
D=HTP0 \
V=1 \
GGML_HEXAGON_WEIGHT_LAYOUT=auto \
./scripts/snapdragon/adb/run-tool.sh llama-batched-bench \
  -m /data/local/tmp/gguf/$M \
  --device HTP0 \
  --repack \
  -ngl 99 \
  -c 4096 \
  -b 2048 \
  -ub 512 \
  -npp 128 \
  -ntg 0 \
  -npl 1
```

Force the previous non-pre-tiled F16 behavior for A/B comparison:

```bash
M=TinyLlama-1.1B-Chat-v1.0-F16.gguf \
D=HTP0 \
V=1 \
GGML_HEXAGON_WEIGHT_LAYOUT=x4x2 \
./scripts/snapdragon/adb/run-tool.sh llama-batched-bench \
  -m /data/local/tmp/gguf/$M \
  --device HTP0 \
  --repack \
  -ngl 99 \
  -c 4096 \
  -b 2048 \
  -ub 512 \
  -npp 128 \
  -ntg 0 \
  -npl 1
```

Run the quantized reference path:

```bash
M=TinyLlama-1.1B-Chat-v1.0-Q4_0.gguf \
D=HTP0 \
./scripts/snapdragon/adb/run-tool.sh llama-batched-bench \
  -m /data/local/tmp/gguf/$M \
  --device HTP0 \
  --repack \
  -ngl 99 \
  -c 4096 \
  -b 2048 \
  -ub 512 \
  -npp 128,256,512 \
  -ntg 0 \
  -npl 1,2,4,8 \
  --output-format jsonl
```

```bash
M=TinyLlama-1.1B-Chat-v1.0-Q4_0.gguf \
D=HTP0 \
./scripts/snapdragon/adb/run-tool.sh llama-batched-bench \
  -m /data/local/tmp/gguf/$M \
  --device HTP0 \
  --no-repack \
  -ngl 99 \
  -c 4096 \
  -b 2048 \
  -ub 512 \
  -npp 128 \
  -ntg 0 \
  -npl 1
```

The `--no-repack` run is a diagnostic check. The main benchmark should keep `--repack` enabled.

## Comparing CPU vs NPU

Keep all of the following fixed across both runs:

- device under test
- model file
- context length
- batch size
- micro-batch size
- `npp`
- `npl`
- `ntg`

Recommended first comparison:

- model: `TinyLlama-1.1B-Chat-v1.0-F16.gguf`
- command family: `llama-batched-bench`
- metric: `speed_pp`
- purpose: validate CPU vs HTP prefill and the migrated F16 pre-tiled path

Optional follow-up:

- repeat the same process with `TinyLlama-1.1B-Chat-v1.0-Q4_0.gguf`
- repeat the same process with `TinyLlama-1.1B-Chat-v1.0-Q8_0.gguf`

Success criteria for this phase:

- CPU-only package runs the benchmark successfully
- Hexagon package runs the benchmark successfully
- Hexagon run shows `HTP*-REPACK`
- `F16` Hexagon verbose logs show `legacy-pre-tiled-f16` and `pretiled=1`
- Hexagon prefill throughput is clearly higher than CPU-only under the same settings

## Notes on Runtime Repack

Current upstream-compatible Snapdragon flow uses load-time/runtime repack, not a private offline HTP layout.

That means:

- the GGUF file remains a normal upstream-compatible `F16`, `Q4_0`, or `Q8_0` file
- the Hexagon backend repacks eligible weights when model tensors are uploaded/offloaded to the Hexagon repack buffer
- `F16` eligible LLaMA trunk matmul weights are converted to an HMX tile-major layout once, then reused by subsequent prefill runs in the same process
- `Q4_0` and `Q8_0` are still converted to the current `q4x4x2/q8x4x2` layout
- you can explicitly compare behavior with `--repack` and `--no-repack`
- you can force the old F16 path with `GGML_HEXAGON_WEIGHT_LAYOUT=x4x2`

This is the correct first-stage route if you want to stay close to the latest upstream `llama.cpp`.

## Future Work

If you later want true offline Hexagon-friendly layouts before quantization, that becomes a separate private extension. It would require:

- a derived conversion script based on `convert_hf_to_gguf.py`
- explicit offline block-layout transforms for matmul weights
- GGUF metadata to mark the custom layout
- matching quantization and loader support

That is intentionally out of scope for this first-stage guide.
