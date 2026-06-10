# TinyLlama LoRA-FA Zeroth-Order Finetuning

This directory contains a self-contained CPU ExecuTorch flow for running five
zeroth-order LoRA-FA finetuning steps on Android arm64.

## 1. Prepare Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\install_executorch.bat
pip install -e . --no-build-isolation
pip install transformers safetensors huggingface_hub sentencepiece accelerate
```

On this machine, the last observed environment was still missing `torch`,
`executorch`, and `transformers`, and `ANDROID_HOME`/`ANDROID_NDK` were unset.
`adb` was available from `D:\Android\SDK\platform-tools\adb.exe`.

## 2. Export TinyLlama

```powershell
python mytest\export_lora_fa_tinyllama.py --output_dir mytest\out
```

Expected outputs:

```text
mytest/out/tinyllama_lora_fa.pte
mytest/out/tinyllama_lora_fa_weights.ptd
mytest/out/tinyllama_lora_fa_layout.json
```

Use `--hf_model <local-or-hf-path>` to reuse a local Hugging Face checkpoint.
Use `--run_executorch_smoke` after pybindings are available to check one
ExecuTorch forward on the host.

## 3. Build Android Runtime

Install Android NDK r28c, then set:

```powershell
$env:ANDROID_HOME="D:\Android\SDK"
$env:ANDROID_NDK="$env:ANDROID_HOME\ndk\28.2.13676358"
```

Build and install ExecuTorch for Android arm64:

```powershell
cmake . --preset android-arm64-v8a `
  -DCMAKE_TOOLCHAIN_FILE="$env:ANDROID_NDK\build\cmake\android.toolchain.cmake" `
  -DANDROID_PLATFORM=android-26 `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_INSTALL_PREFIX=cmake-out-android-arm64-v8a `
  -B cmake-out-android-arm64-v8a

cmake --build cmake-out-android-arm64-v8a --target install --config Release -j
```

Build the runner:

```powershell
cmake -S mytest -B cmake-out-android-arm64-v8a\mytest `
  -DCMAKE_TOOLCHAIN_FILE="$env:ANDROID_NDK\build\cmake\android.toolchain.cmake" `
  -DANDROID_ABI=arm64-v8a `
  -DANDROID_PLATFORM=android-26 `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_FIND_ROOT_PATH="$PWD\cmake-out-android-arm64-v8a"

cmake --build cmake-out-android-arm64-v8a\mytest --target lora_fa_runner --config Release -j
```

## 4. Run on Device

```powershell
adb push mytest\out\tinyllama_lora_fa.pte /data/local/tmp/
adb push mytest\out\tinyllama_lora_fa_weights.ptd /data/local/tmp/
adb push cmake-out-android-arm64-v8a\mytest\lora_fa_runner /data/local/tmp/
adb shell chmod +x /data/local/tmp/lora_fa_runner
adb shell "cd /data/local/tmp && ./lora_fa_runner --model_path=tinyllama_lora_fa.pte --data_path=tinyllama_lora_fa_weights.ptd"
```

The runner prints per-step `loss_plus`, `loss_minus`, `g`, `speed_pp`,
`avg_step_time_ms`, and `peak_rss_mb`.
