# flakepart-fixups: python package fixes for use with uvpart

Most entries in `fixup-overlay.nix` are applied unconditionally by `flake-module.nix`
as `uvpart.pythonOverlays`. The optional ones are guarded with
`lib.optionalAttrs (prev ? <name>)`, so a lock that does not contain that package is
unaffected. The CUDA toolkit is unfree, so `allowUnfree` has to be enabled.

Every entry relies on the package existing in the *lock*: `prev.<name>` is only
defined for packages the lock resolved.

## sdists that forget setuptools

Some packages run `setup.py` or a cffi build without declaring setuptools as a build
dependency, so their build environment has nothing to import. They are matched by
presence rather than by a guard each, so a lock only needs to contain the ones it
actually uses: `defuser`, `device-smi`, `logbar`, `tokenicer` (which get setuptools),
`pluggy`, `pyyaml_ft`, `trove-classifiers` (and `trove_classifiers`; uv2nix exposes the
PyPI spelling, so both are listed) and `pypcre` (which also builds against PCRE2).
`torchao` links torch, so it goes through the same `withCudaLibs` helper as the CUDA
wheels below.

## sdists that forget flit_core

The same failure with a flit backend instead of setuptools: `editables` and
`pathspec` are built by `flit_core.buildapi` but declare it neither in their build
requirements nor anywhere else, so uv2nix — which builds with `--no-build-isolation` — calls a
backend it cannot import and dies with `ModuleNotFoundError: No module named
'flit_core'`.

There is one extra step here that the setuptools list above does not need. Those
packages get `final.setuptools`, and setuptools is present in any lock because
something always depends on it. **flit-core is not**, so there is no `final.flit-core`
to attach unless the project resolves it first:

```toml
dependencies = [
  # ...the framework...
  # Build backend for the editables sdist. Add it here rather than under
  # [tool.uv.extra-build-dependencies]: uv never writes those entries into uv.lock,
  # and the overlay can only use packages the lock resolved.
  "flit-core",
]
```

With that entry present, the fixup is automatic — no `pythonOverlays` stanza of your
own. Any project that hits this needs the same line; more package names can be added
to the list as they turn up. `editables` and `pathspec` are there today.

## sdists whose backend is poetry-core

`tomlkit` builds through `poetry.core.masonry.api` and does not declare it; the import
failure names `poetry`, the parent package. Same presence-matching, and the same
requirement as flit-core above — poetry-core has to be resolved into the lock by the
project, with a `"poetry-core"` dependency.

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

## flash-attn (automatic, one pyproject stanza)

`flash-attn` ships a legacy `setup.py` with no `pyproject.toml`, so nothing declares
what its build needs. Without this stanza, uv2nix builds it in an empty environment
and `setup.py` fails on `import setuptools`:

```toml
[tool.uv.extra-build-dependencies]
"flash-attn" = ["torch", "setuptools", "wheel", "packaging", "psutil", "ninja"]
```

The names in that stanza are resolved against the environment's *build-host* python
package set — the `pyproject-build-systems` input — and not against your lock. That is
why `wheel`, `ninja`, `packaging`, `psutil` and `torch` are available to it without
any of them being locked: this is uv2nix's `resolveBuildSystem`, picked up
automatically from `[tool.uv.extra-build-dependencies]`.

The overlay's own idiom is a different source: `nativeBuildInputs ++ [ final.setuptools ]`
takes `setuptools` from *your lock*, which is why the flit-core and poetry-core
sections above have to ask you to put those in `dependencies` first. Both are right for
what they do — a package needs the stanza when what its build imports is not in your
lock, and the fixup when what it forgot to declare is.

The overlay then supplies the toolkit, forces a source build (its prebuilt-wheel
download cannot work in a sandbox), and pins `-std=c++20`, which torch 2.14's headers
require. Arch selection is a one-line change in `fixup-overlay.nix`
(`FLASH_ATTN_CUDA_ARCHS` / `TORCH_CUDA_ARCH_LIST`), currently `80` — Ampere cubins
run on Ada via CUDA's minor-version binary compatibility.

## causal-conv1d (automatic, one pyproject stanza)

`causal-conv1d` publishes no wheels, so uv2nix builds it from the sdist and its
`setup.py` wants a CUDA toolchain in the build sandbox. Like flash-attn it has no
`pyproject.toml`, so nothing declares what its build needs:

```toml
[tool.uv.extra-build-dependencies]
"causal-conv1d" = ["torch", "setuptools", "wheel", "packaging", "ninja"]
```

With that stanza the fixup is automatic — no `pythonOverlays` stanza of your own. The
overlay supplies the toolkit, forces the source build (the prebuilt-wheel download
cannot work in a sandbox) and sets the cxx11 ABI, which is what PyPI's torch uses.
Arch selection follows the flash-attn entry: `TORCH_CUDA_ARCH_LIST` is `8.0`, whose
cubins run on Ada via CUDA's minor-version binary compatibility.

## llama-cpp-python (automatic, one pyproject stanza)

`llama-cpp-python` compiles the llama.cpp it vendors, through `scikit-build-core`.
That sdist *does* declare its backend, but a lock which never resolved it contains no
such package, and uv2nix then builds the package in an environment without one, so it
has to be declared too:

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

## JIT at run time (when the lock carries a CUDA toolkit)

Some packages compile CUDA C++ on first use instead of linking a prebuilt kernel:
vllm's sampler and its other `cpp_extension` ops, flashinfer's JIT kernels, sglang's
own sampler and attention backends, tilelang's DSL, and inductor's nvcc path.
Triton-based kernels do not — unsloth, liger-kernel, flash-linear-attention and plain
triton compile through `ptxas`, which the triton wheel ships, and need only the driver,
which the preloader provides.

There is nothing to opt into. The toolchain is applied whenever the environment carries
the CUDA components, so a project that compiles CUDA C++ at run time gets it by depending
on them, and a project that does not is left alone entirely — no toolkit in its closure,
no unfree reference. Which components those are, and why they have to be pinned together,
is further down.

A dev shell can hand nvcc, ninja and a host compiler to a JIT by putting them on PATH,
but a deployed program gets neither that PATH nor CUDA_HOME: environment variables belong
to the invocation, not to the artifact. The toolchain is therefore written into the
packages' own lookups as absolute store paths:

| package | lookups repointed |
| --- | --- |
| `torch` | the `/usr/local/cuda` fallback in `_find_cuda_home`, the ninja command, the ninja availability probe, `get_cxx_compiler`, and inductor's `_cuda_compiler`, whose last resort was the bare name `nvcc` |
| `flashinfer-python` | its own copy in `flashinfer/jit/cpp_ext.py`: `get_cuda_path`, `cxx`, and its ninja invocation |
| `sglang` | its own JIT in `sglang/kernels/jit/utils/compile/`: `cuda_home`, `cxx`, the ninja invocation, and the CUDA include path and `lib/` link flag that nothing else supplies to that module |
| `tilelang` | the CUDA_HOME detection in `tilelang/env.py`, which otherwise lands on the PyPI `nvidia-cuda-nvcc` wheel |

`sglang` is the odd one here: its JIT writes its own `build.ninja` instead of going
through `cpp_extension`, so none of the flags tvm-ffi would have supplied reach the
compiler unless `kernels/jit/utils/compile/toolchain.py` states them — which is why that
file also needs the CUDA include directory, its link flag moved off the NVIDIA `lib64`
layout, and the `ninja`/`CXX` names made absolute. Without them a program that inherits
the artifact but not a dev shell's PATH fails at the first build, which is the failure
this section is about.

Those entries also ask their linker for `-Wl,-z,lazy`, which is a hardening flag being
relaxed on purpose. The linker Nix wraps every JIT build with puts hardening's `-z now`
before the caller's arguments, and `-z now` marks the module `DF_1_NOW`: every PLT entry
resolved at load, which is what a JIT module cannot survive. flashinfer leaves its
`CTA_TILE_Q=32` prefill dispatch arm undefined deliberately — `nm -D --undefined-only` on
the built module lists `BatchPrefillWithRaggedKVCacheDispatched<32u, 256u, 256u, …>` — and
binds it through the PLT at first use, so `readelf -d` showing `BIND_NOW` and
`FLAGS_1: NOW` means the import dies with "undefined symbol". The flag the module asks for
comes last on its own command line, so it wins over the wrapper's.

`vllm` needs no entry of its own, its JIT compiling through cpp_extension, and `triton`
needs none either: it ships its own ptxas and only wants the driver, which the
preloader already handles. Every environment variable is still read first, so
CUDA_HOME and CXX still override deliberately.

The tree is a merge of `cuda_nvcc`, `cuda_cudart`, `cuda_crt` and the `include` outputs
of `libcurand` and `libcusparse`, about 683 MiB together, rather than
`cudaPackages_13.cudatoolkit` at about 2.6 GiB. The merged toolkit's `bin/nvcc`
is a symlink to `cuda_nvcc`'s, so the compiler is the same one, `cuda_cudart` brings
`include/cuda_runtime.h` and `cuda_crt` the `crt/` headers it includes. What the merged
toolkit adds is cublas, cufft, nvrtc and friends, which nothing here links, because torch
takes its runtime libraries from the `nvidia-*` wheels. Those two `include` outputs are
merged because a CUDA library package keeps its headers in one of its own — named
`include`, not `dev` — and nothing depends on it implicitly: without them the tree had
`cuda_runtime.h` and the CCCL headers and no `<curand.h>`, so vllm's flashinfer sampler
JIT died on `fatal error: curand.h: No such file or directory`.

The merge is plain: no copied binary, no rewritten profile. A compiler here needs nothing
rewritten, because it keeps its own installation-relative lookups and the tree's `include/`
reaches it through the `-isystem $CUDA_HOME/include` that the callers pass — torch,
flashinfer and tilelang do that themselves, sglang's own JIT gets it from
`base_include_paths`, and that is what makes the merge sufficient. A bare `nvcc` from this
tree does *not* find `cuda_runtime.h`, since its own profile points at the `cuda_nvcc`
package's include; the callers are the interface.

Building the tree from the CUDA wheels instead was tried, and it does not work — worth
recording, because it looks like it should. The wheels do ship a whole compiler
(`bin/nvcc`, `bin/ptxas`, `bin/nvlink`, `cudafe++`, `fatbinary`, `nvvm/bin/cicc`) and
`NVCC_CCBIN` supplies the host compiler they lack. What they cannot supply is coherence: a
compiler and *every* header it consumes have to come from one CTK release, and the lock
cannot arrange that. torch pins `nvidia-cuda-runtime` to 13.0.96, so a wheel-assembled tree
pairs a 13.2.86 nvcc and cccl with 13.0.96 runtime headers, and cccl refuses the
combination at compile time:

```console
…cuda-jit/bin/../include/cccl/cuda/std/__cccl/cuda_toolkit.h:41:8: error: #error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
```

That is the version-skew failure NVIDIA documents for CUDA wheel installs, reproduced here
even with `nvidia-cuda-nvcc`, `nvidia-cuda-crt`, `nvidia-cuda-cccl` and `nvidia-nvvm` all
pinned to 13.2.86: the runtime headers are the layer that cannot be moved, and pinning the
runtime to 13.2.86 instead is unsatisfiable against torch's own pin. The component pin is
still worth keeping, because it removes the *other* skew — where cccl floats releases ahead
of the compiler, `humming-kernels`' `cu13` extra pinning nvcc and leaving cccl, crt and
nvrtc unpinned.

A nixpkgs component set is coherent by construction and independent of torch's pins, at
683 MiB against the wheels' 318 MiB installed. Because the compile never reads the
`nvidia-*` include paths, the versions of those wheels cannot affect it: a project may carry
them drifted, or not at all, and still compile — which is what makes a component pin
unnecessary. The scope of that separation is everything compiling through the patched
lookups, i.e. through `CUDA_HOME`. A consumer that reads the wheel tree directly —
`cuda-python`'s `cu13` extra, `cuda.core`/`cuda.cccl` style helpers — is not patched, and for
those the wheel versions still have to agree.

nixpkgs' `cuda_nvcc` also needs no host compiler variable: measured, it resolves one with an
empty environment *and* with its profile disabled (`-noprof`), so the default is in the binary
— nvcc here is a redistribution, so that is a patch nixpkgs applies. The wheels are the
better route for a project that can pin one whole toolkit release, which is not a project
whose torch pins the runtime. Two details if that route is ever taken: the unversioned
`lib/libcudart.so` has to be linked beside `libcudart.so.13` (`lib/stubs` is not needed, a
torch extension asks only for `-lcudart`), and since nvcc resolves its installation relative
to its real executable, the tree needs `nvvm`, `include` and `lib` beside a *copy* of the
binary rather than a symlink to it.

Because a store path occurring in a file counts as a reference, the toolkit, ninja and
the compiler enter those derivations' closures, so whatever deploys the environment
deploys the tools with it:

```console
$ nix-store -q --references /nix/store/…-torch-2.11.0 | grep -E 'cuda-merged|ninja|gcc-wrapper'
/nix/store/893x2zqx2kjd3rypfr2pb890cngz7l1j-cuda-merged-13.3
/nix/store/r8a159fqvj0mpczq0dq8d3dwdd2rsz8c-ninja-1.13.2
/nix/store/z4c6k0mrlkwl3s4w9ysxc8vq1wylm3ms-gcc-wrapper-15.3.0
```

One entry is not a lookup. GCC 15 rejects a line of torch 2.11's
`ATen/core/List_inl.h` inside nvcc's host pass — `[-Wtemplate-body]`, a "need
`typename`" error on a `typename` that is already written — which fails every CUDA JIT
compile; plain `g++` accepts the same line, and neither `-fpermissive` nor
`-Wno-error=template-body` helps. The equivalent `std::ptrdiff_t` cast is substituted
instead.

Two requirements remain, neither of them a toolchain one. ninja runs its build
commands through a shell, so some `sh` has to be on PATH — `/bin` is enough. And the
JIT writes into `~/.cache/torch_extensions` or `FLASHINFER_JIT_DIR`, so that directory
has to be writable.

Verified by compiling and loading a CUDA extension with the environment alone,
`PATH=/bin`, and CUDA_HOME, CXX and CC unset:

```console
$ env -i PATH=/bin CUDA_HOME= CXX= CC= …-editable-env/bin/python -c '…load_inline(…)'
cpp_extension._find_cuda_home() -> /nix/store/…-cuda-merged-13.3
cpp_extension.get_cxx_compiler() -> /nix/store/…-gcc-wrapper-15.3.0/bin/c++
flashinfer get_cuda_path() -> /nix/store/…-cuda-merged-13.3
inductor _cuda_compiler() -> /nix/store/…-cuda-merged-13.3/bin/nvcc
tilelang CUDA_HOME -> /nix/store/…-cuda-merged-13.3
load_inline -> …/nix_jit_probe.so
forty_two() -> 42
```

The lookups themselves can be checked without a GPU, from the same environment. The probe
loads sglang's and flashinfer's modules out of `site-packages` with the imports they make
stubbed, then asks them where CUDA is, what they include and what they link — which is
also what makes it fail on an unpatched tree rather than pass quietly:

```console
$ env -i PATH=/bin …-editable-env/bin/python tests/jit-toolchain-probe.py
sglang    cuda_home()          -> /nix/store/…-cuda-jit
sglang    host_compiler_path   -> /nix/store/…-gcc-wrapper-15.3.0/bin/c++
sglang    base_include_paths   -> ['…', '/nix/store/…-cuda-jit/include']
sglang    base_link_flags      -> ['-shared', '…', '-Wl,-z,lazy', '-L/nix/store/…-cuda-jit/lib', '-lcudart']
flashinfer ldflags = -shared -L$cuda_home/lib -L$cuda_home/lib/stubs -lcudart -lcuda -Wl,-z,lazy
all checks passed
```

## vllm (automatic)

`vllm` drags in a large CUDA dependency tree, and most of the work is telling
autoPatchelf where the sibling packages keep their libraries: torch's (under
`site-packages/torch/lib`), the CUDA runtime libraries inside the nvidia-* wheels,
and the cutlass DSL and TVM FFI runtimes. One `withCudaLibs` helper in
`fixup-overlay.nix` adds those search paths and leaves the `libcuda.so.1` reference
unresolved, so the driver is preloaded at run time as everywhere else here. The
wheels that need more than that get their own entry:

- `torchvision`, `torchaudio`, `torch-c-dlpack-ext`, `xgrammar`, `flashinfer-python`,
  `tilelang`, `tokenspeed-mla`, `tokenspeed-triton`, `pynvvideocodec`, `vllm`,
  `xformers`
- `sglang`, `sglang-kernel`, `sgl-deep-ep`, `sgl-deep-gemm` and `torch-memory-saver`
  are compiled CUDA wheels from the same stack, and get the same search paths; the two
  `deep_*` ones also resolve a CUDA_HOME at import, because sglang imports `deep_ep`
  eagerly from its scheduler path
- `torchcodec` also links ffmpeg and libheif, and ships one core/custom_ops module
  pair per ffmpeg major; only the pair matching the ffmpeg provided here is kept
- `tilelang` bundles `libtvm.so`, which links Z3 under the versioned soname its build
  used (`libz3.so.4.15` for 0.1.12); the library comes from the `z3-solver` wheel,
  which bundles exactly that name, with nixpkgs' z3 as the fallback
- `nvidia-cutlass-dsl-libs-base` and `-libs-cu13` ship the same tree with different
  contents (pip lets the cu13 wheel overwrite); the base copy keeps only the files
  that differ
- `humming-kernels` and `nvidia-deepstream-videodecode-cu13` go through the helper
- `nvidia-cufile` is matched by name prefix, because the wheels come in cu12 and cu13
  spellings, and gets rdma-core on the search path for `libcufile_rdma`
- `vllm`'s vendored pynvml loads NVML by bare soname, which NixOS keeps out of the
  loader's search path. That failure hides well: the CUDA platform plugin swallows
  it, reports no platform, and the engine dies later with "Device string must not be
  empty"
- `flashinfer-python`'s JIT links with `$CUDA_HOME/lib64`, but nixpkgs ships `lib/`
  and `lib/stubs`, so any JIT-compiled op (vllm's sampler is one) failed to link
  with `cannot find -lcuda`; it and `sglang` also ask for `-Wl,-z,lazy`, for the
  reason in "JIT at run time"

### Shell entries (optional)

vllm JIT-compiles a few small kernels the first time it runs them. Nothing has to be
on PATH for that — see "JIT at run time" above — so no entry is required here; the
example below is only for calling nvcc or ninja by hand:

```nix
uvpart.extraPackages = [ pkgs.cudaPackages_13.cudatoolkit pkgs.ninja pkgs.gcc ];
```

The first generation pays a one-off compile into `~/.cache/flashinfer`, after which
the result is reused.

Precompiling those kernels at build time, the way `gptqmodel-ops` does for the marlin
kernels, is possible but awkward here: which ops are needed depends on the model and
the attention backend, and the build sandbox has no GPU to detect an architecture
from.

### vllm and gptqmodel cannot share an environment

vllm pins torch exactly and trails it by a couple of releases, while gptqmodel
requires `protobuf>=7.34.0`; vllm 0.26+ depends on an `nvidia-cutlass-dsl` whose
`libs-base` sidecar caps protobuf below 7. The two cannot coexist, so the newest
usable combination alongside gptqmodel is vllm 0.25.1 with torch 2.11. Newer vllm
means dropping gptqmodel (vllm ships its own GPTQ/AWQ/Marlin kernels) or giving it an
environment of its own.

Without gptqmodel there is no such constraint: vllm 0.29.0 with torch 2.13.0 builds
and runs from this overlay set unchanged (verified on an RTX 4080 with Qwen3-4B-AWQ).
Two notes:

- Regenerate the lock instead of extending an old one. uv keeps locked versions, and
  vllm pins `openai >= 2.0.0` with no upper bound; a stale `openai` pin is enough to
  break vllm at import (`cannot import name 'NamespaceTool'`).
- No shell entries are needed, for the reasons in "JIT at run time": nothing has to be
  on PATH but a `sh`, and a writable `~/.cache/flashinfer`.
