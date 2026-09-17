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

## vllm (automatic, one project-side prerequisite)

`vllm` drags in a large CUDA dependency tree, and most of the work is telling
autoPatchelf where the sibling packages keep their libraries: torch's (under
`site-packages/torch/lib`), the CUDA runtime libraries inside the nvidia-* wheels,
and the cutlass DSL and TVM FFI runtimes. One `withCudaLibs` helper in
`fixup-overlay.nix` adds those search paths and leaves the `libcuda.so.1` reference
unresolved, so the driver is preloaded at run time as everywhere else here. The
wheels that need more than that get their own entry:

- `torchvision`, `torchaudio`, `torch-c-dlpack-ext`, `xgrammar`, `flashinfer-python`,
  `tilelang`, `tokenspeed-mla`, `tokenspeed-triton`, `pynvvideocodec`, `vllm`
- `torchcodec` also links ffmpeg and libheif, and ships one core/custom_ops module
  pair per ffmpeg major; only the pair matching the ffmpeg provided here is kept
- `tilelang` bundles `libtvm.so`, which links Z3
- `nvidia-cutlass-dsl-libs-base` and `-libs-cu13` ship the same tree with different
  contents (pip lets the cu13 wheel overwrite); the base copy keeps only the files
  that differ
- `vllm`'s vendored pynvml loads NVML by bare soname, which NixOS keeps out of the
  loader's search path. That failure hides well: the CUDA platform plugin swallows
  it, reports no platform, and the engine dies later with "Device string must not be
  empty"
- `flashinfer-python`'s JIT links with `$CUDA_HOME/lib64`, but nixpkgs ships `lib/`
  and `lib/stubs`, so any JIT-compiled op (vllm's sampler is one) failed to link
  with `cannot find -lcuda`

### Project-side prerequisite

vllm JIT-compiles a few small kernels the first time it runs them, so the *shell*
needs a compiler and a toolkit, not just the packages:

```nix
uvpart.extraPackages = [ pkgs.cudaPackages_13.cudatoolkit pkgs.ninja pkgs.gcc ];
```

Without it the engine dies with `Could not find nvcc and default
cuda_home='/usr/local/cuda' doesn't exist`. The first generation pays a one-off
compile into `~/.cache/flashinfer`, after which the result is reused.

Precompiling those kernels at build time, the way `gptqmodel-ops` does for the marlin
kernels, is possible but awkward here: which ops are needed depends on the model and
the attention backend, and the build sandbox has no GPU to detect an architecture
from.

### Worth knowing before adding vllm to an existing project

vllm pins torch exactly and trails it by a couple of releases, while gptqmodel
requires `protobuf>=7.34.0`; vllm 0.26+ depends on an `nvidia-cutlass-dsl` whose
`libs-base` sidecar caps protobuf below 7. The two cannot coexist, so in one
environment the newest usable combination is vllm 0.25.1 with torch 2.11. A project
that wants newer vllm has to give up gptqmodel (vllm ships its own GPTQ/AWQ/Marlin
kernels) or split it into its own environment.
