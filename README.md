# flakepart-fixups: python package fixes for use with uvpart

Most entries in `fixup-overlay.nix` are applied unconditionally by `flake-module.nix`
as `uvpart.pythonOverlays`. `flash-attn` is guarded with `lib.optionalAttrs (prev ? flash-attn)` so projects whose lock does not contain it are unaffected. The CUDA
toolkit is unfree, so consuming projects must allow unfree packages.

Requiring packages exist in the *lock* is the general shape here: `prev.<name>` is
only defined for packages the consuming project resolved.

## gptqmodel kernels (opt-in)

gptqmodel JIT-compiles its torch.ops kernels on first use, which needs nvcc and a
writable package directory; a Nix store offers neither. `gptqmodel-ops.nix` compiles
the extensions you name at package build time, one derivation per extension, and the
patched loader then loads them from `gptqmodel_ext/prebuilt`. Nothing is needed at
run time.

```nix
pythonOverlays = [
  # ...the fixup overlay itself...
  (inputs.uvpart-fixups.gptqmodel-ops {
    inherit pkgs;
    extensions = [ "marlin_fp16" "marlin_bf16" ];
  })
];
```

The selection is required: an empty or missing list fails evaluation with
instructions, and `extensions = [ "none" ]` is the explicit opt-out. Most AWQ and
GPTQ checkpoints resolve to the Marlin kernels. `machete`/`swordfish` are only
reachable on sm_90+ hardware and fetch CUTLASS from the network, so they are not
supported here.

## flash-attn (automatic, one project-side prerequisite)

`flash-attn` ships a legacy `setup.py` with no `pyproject.toml`, so nothing declares
what its build needs. Without this stanza in the *project's* `pyproject.toml`, uv2nix
builds it in an empty environment and `setup.py` fails on `import setuptools`:

```toml
[tool.uv.extra-build-dependencies]
"flash-attn" = ["torch", "setuptools", "wheel", "packaging", "psutil", "ninja"]
```

The overlay then supplies the toolkit, forces a source build (its prebuilt-wheel
download cannot work in a sandbox), and pins `-std=c++20`, which torch 2.14's headers
require. Arch selection is a one-line change in `fixup-overlay.nix`
(`FLASH_ATTN_CUDA_ARCHS` / `TORCH_CUDA_ARCH_LIST`), currently `80` — Ampere cubins
run on Ada via CUDA's minor-version binary compatibility.

## llama-cpp-python (automatic, one project-side prerequisite)

`llama-cpp-python` compiles the llama.cpp it vendors, through `scikit-build-core`.
That sdist *does* declare its backend, but a lock which never resolved it contains no
such package, and uv2nix then builds the package in an environment without one — so
the project declares it, exactly as for flash-attn:

```toml
[tool.uv.extra-build-dependencies]
"llama-cpp-python" = ["scikit-build-core"]
```

The overlay adds `cmake`, `ninja` and the CUDA toolkit, and pins the ggml build:
`GGML_CUDA=on` (ggml defaults to off), `CMAKE_CUDA_ARCHITECTURES=89` (Ada; there is no
GPU in the sandbox for CMake to detect an architecture from), and `GGML_NATIVE=OFF` so
`-march=native` does not leak into the store. Change the arch line for other GPUs.

ggml links `libcuda` only for its VMM API, and autoPatchelf resolves that from the
toolkit's `lib/stubs` — so the *stub* is what loads at run time, ggml's init fails with
`CUDA driver is a stub library`, and inference quietly falls back to the CPU. The wheel
still advertises a CUDA build, so nothing looks wrong until you notice the speed. The
overlay therefore builds with `GGML_CUDA_NO_VMM=ON` and attaches `cuda-loader-helper`
to the extension's libraries, which preopens the real driver. Both halves are
load-bearing.

`llama_supports_gpu_offload()` comes from the wheel itself, so it distinguishes a
CUDA build from a CPU one:

```console
$ python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
True
```
