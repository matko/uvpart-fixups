{
  lib,
  config,
  callPackage,
  tbb_2022,
  stdenv,
  rdma-core,
  file,
  libfabric,
  pmix,
  mpi,
  # flash-attn's and llama-cpp-python's builds need a real nvcc; see the two
  # guarded entries at the bottom of this file.
  cmake,
  ninja,
  patchelf,
  ffmpeg,
  libheif,
  pcre2,
  z3,
  symlinkJoin,
  cudaPackages_13,
}:
final: prev:
let
  # Wheels with compiled CUDA extensions keep much of what they link in sibling
  # packages: torch's libraries (site-packages/torch/lib), the CUDA runtime libraries
  # inside the nvidia-* wheels (nvidia/<component>/lib) and, in a few cases, the TVM
  # FFI library. None of those are directories autoPatchelf searches, and without them
  # the build fails outright rather than shipping a broken .so.
  #
  # libcuda.so.1 is the driver: no sandbox can provide it, so the reference is left
  # unresolved and satisfied at run time by the preloader every CUDA package here
  # relies on (cuda-loader-helper/).
  #
  # It takes the attribute name along with the package, and an nvidia-* package gets
  # nothing but the driver reference below: such wheels do not link libtorch, and the
  # sibling lists create cycles that do not evaluate. Two nvidia packages list each
  # other (nvidia-cufile and the cutlass libs both do), and torch is itself patched
  # with a list of nvidia packages, so an nvidia package pointing back at torch closes
  # the loop. The paths those packages do need are added by their own entries.
  withCudaLibs = name: pkg:
    let
      isNvidia = lib.hasPrefix "nvidia-" name;
    in
      pkg.overrideAttrs (old: {
        cudaDependencies =
          if isNvidia then
            [ ]
          else
            map (n: final.${n}) (
              builtins.filter (n: lib.hasPrefix "nvidia-" n) (builtins.attrNames prev)
            );
        autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [ "libcuda.so.1" ];
        preFixup =
          (old.preFixup or "")
          + lib.optionalString (!isNvidia) ''
            addAutoPatchelfSearchPath ${final.torch}/lib/python*/site-packages/torch/lib
          ''
          + lib.optionalString (!isNvidia && prev ? apache-tvm-ffi) ''
            addAutoPatchelfSearchPath ${final.apache-tvm-ffi}/lib/python*/site-packages/tvm_ffi/lib
          ''
          + lib.optionalString (!isNvidia) ''
            for dep in $cudaDependencies; do
              addAutoPatchelfSearchPath $(find $dep/lib/python*/site-packages -type d -name lib)
            done
          '';
      });
  # The runtime JIT toolchain. Only a project that compiles CUDA C++ at run time wants
  # it: vllm's cpp_extension ops, flashinfer's kernels, tilelang's DSL, inductor's nvcc
  # path. Triton-based kernels need ptxas and the driver instead, which the triton wheel
  # and the preloader already provide.
  #
  # There is no opt-in flag. The toolchain applies when the environment carries these
  # CUDA components, which is the same condition as "this project compiles CUDA C++": a
  # project that does not carry them is left entirely alone, with no toolkit and no
  # unfree reference in its closure.
  #
  # The tree comes from nixpkgs rather than from those wheels. A compiler and every
  # header it consumes have to come from one CTK release, and the wheels cannot supply
  # that here: the lock pins nvidia-cuda-runtime to what torch wants (13.0.96), so a
  # wheel-assembled tree pairs a 13.2.86 nvcc and cccl with 13.0.96 runtime headers, and
  # cccl rejects the combination at compile time -- "CUDA compiler and CUDA toolkit
  # headers are incompatible", the version-skew failure NVIDIA documents for CUDA wheel
  # installs. Measured, not assumed: the guard fires even when cccl, crt and nvvm are
  # pinned to the compiler's own release, because the runtime headers are the part that
  # cannot be moved. A nixpkgs component set is coherent by construction and independent
  # of torch's pins, which makes 683 MiB of cuda_nvcc + cuda_cudart + cuda_crt both
  # cheaper than 2.6 GiB of merged cudatoolkit and more useful than a wheel tree that
  # cannot compile.
  #
  # The merge is a plain symlinkJoin: nvcc keeps its own profile and its own
  # installation-relative lookups, and the tree's include/ reaches it through the
  # -isystem $CUDA_HOME/include that the callers pass (torch, flashinfer, tilelang all
  # do). That is why no copied binary and no rewritten profile are needed here, unlike
  # the wheel tree, where the profile had to be redirected.
  cudaJitPresent = builtins.all (n: prev ? ${n}) [
    "nvidia-cuda-nvcc"
    "nvidia-cuda-crt"
    "nvidia-cuda-cccl"
    "nvidia-nvvm"
    "nvidia-cuda-runtime"
  ];
  # uvpart.cudaJitToolchain can override that proxy, since a proxy is wrong in both
  # directions: true for a project that compiles CUDA C++ without carrying the components,
  # false for one that carries them without compiling, and null -- the default -- for the
  # presence test above, which is how the common case needs no configuration at all.
  cudaJitMode = final.__uvpart-cuda-jit-toolchain or null;
  cudaJitEnabled =
    if cudaJitMode == false then
      false
    else if cudaJitMode == true then
      true
    else
      cudaJitPresent;
  cudaJit = symlinkJoin {
    name = "cuda-jit";
    paths = [
      cudaPackages_13.cuda_nvcc
      cudaPackages_13.cuda_cudart
      # cuda_runtime.h includes crt/host_config.h, which lives in cuda_crt: a merge
      # without it compiles the host pass and then dies in nvcc's include of the runtime
      # header. cuda_cccl needs no merging, since nvcc's own profile reaches it.
      cudaPackages_13.cuda_crt
    ];
  };
in
{
  nvidia-cuda-runtime-cu12 =
    let
      cuda-loader-helper = callPackage ./cuda-loader-helper { };
    in
      prev.nvidia-cuda-runtime-cu12.overrideAttrs (old: {
        patchelfFlags = [
          "--add-needed ${cuda-loader-helper}/lib/cuda_loader_helper.so"
        ];
        appendRunpaths = (old.appendRunpaths or [ ]) ++ [ "$ORIGIN" ];
      });
  nvidia-cusparse-cu12 = prev.nvidia-cusparse-cu12.overrideAttrs (old: {
    preFixup =
      (old.preFixup or "")
      + ''
        addAutoPatchelfSearchPath ${final.nvidia-nvjitlink-cu12}/lib/python*/site-packages/nvidia/nvjitlink/lib/
      '';
  });
  nvidia-cusolver-cu12 = prev.nvidia-cusolver-cu12.overrideAttrs (old: {
    cudaDependencies = with final; [
      nvidia-nvjitlink-cu12
      nvidia-cublas-cu12
      nvidia-cusparse-cu12
    ];
    preFixup =
      (old.preFixup or "")
      + ''
        for dep in $cudaDependencies;do
          addAutoPatchelfSearchPath $dep/lib/python*/site-packages/nvidia/*/lib/
        done
      '';
  });
  nvidia-cudnn-cu12 = prev.nvidia-cudnn-cu12.overrideAttrs (old: {
    appendRunpaths = (old.appendRunpaths or [ ]) ++ [ "$ORIGIN" ];
  });
  torch = prev.torch.overrideAttrs (
    old:
    let
      cudaEnabled = stdenv.isLinux;
    in
      {
        cudaDependencies = map (name: final.${name}) (
          builtins.filter (name: cudaEnabled && lib.hasPrefix "nvidia-" name) (builtins.attrNames prev)
        );
        autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
        preFixup =
          (old.preFixup or "")
          + ''
          for dep in $cudaDependencies;do
            addAutoPatchelfSearchPath $(find $dep/lib/python*/site-packages -type d -name lib)
          done
        ''
          + lib.optionalString cudaJitEnabled ''
            # Every lookup here is a JIT-time one. --replace-fail throughout, so an
            # upstream change is a loud build failure rather than a silent return to PATH.
            substituteInPlace $out/lib/python*/site-packages/torch/utils/cpp_extension.py \
              --replace-fail "'/usr/local/cuda'" "'${cudaJit}'" \
              --replace-fail "['ninja', '-v']" "['${ninja}/bin/ninja', '-v']" \
              --replace-fail "['ninja', '--version']" "['${ninja}/bin/ninja', '--version']" \
              --replace-fail "os.environ.get('CXX', 'c++')" "os.environ.get('CXX', '${stdenv.cc}/bin/c++')"
            # Inductor has its own compiler picker, whose last resort is the bare name "nvcc".
            substituteInPlace $out/lib/python*/site-packages/torch/_inductor/codegen/cuda/compile_utils.py \
              --replace-fail '    return "nvcc"' '    return "${cudaJit}/bin/nvcc"'
            # GCC 15 rejects torch 2.11's decltype form of this line in nvcc's host pass
            # ([-Wtemplate-body]: a "need typename" error on a typename that is already
            # written), which fails every CUDA JIT compile. A vector's difference_type is
            # ptrdiff_t, so the cast keeps its meaning. 2.13 replaced the decltype with
            # c10::detail::ListImpl::list_type, which GCC 15 takes, so the workaround is
            # skipped there; any third form fails the build rather than passing through.
            if grep -q "static_cast<typename decltype(impl_->list)::difference_type>(pos)" \
                $out/lib/python*/site-packages/torch/include/ATen/core/List_inl.h; then
              substituteInPlace $out/lib/python*/site-packages/torch/include/ATen/core/List_inl.h \
                --replace-fail 'static_cast<typename decltype(impl_->list)::difference_type>(pos)' 'static_cast<std::ptrdiff_t>(pos)'
            elif ! grep -q "c10::detail::ListImpl::list_type::difference_type>(pos)" \
                $out/lib/python*/site-packages/torch/include/ATen/core/List_inl.h; then
              echo "List_inl.h has neither the GCC 15 workaround's target nor the form torch 2.13 uses: check upstream, then update fixup-overlay.nix" >&2
              exit 1
            fi
          '';
      }
  );
  pybars3 = prev.pybars3.overrideAttrs (p: {
    nativeBuildInputs = p.nativeBuildInputs ++ [ final.setuptools ];
  });
  pymeta3 = prev.pymeta3.overrideAttrs (p: {
    nativeBuildInputs = p.nativeBuildInputs ++ [ final.setuptools ];
  });
  numba = prev.numba.overrideAttrs (p: {
    buildInputs = p.buildInputs ++ [ tbb_2022 ];
  });

  pyperclip = prev.pyperclip.overrideAttrs (old: {
    buildInputs = (old.buildInputs or [ ]) ++ [ prev.setuptools ];
  });
  triton = prev.triton.overrideAttrs (old: {
    postInstall = ''
      pushd $out/lib/python*/site-packages/
      patch -p1 < ${./triton-find-nixos-driver.patch}
      popd
    '';
  });
  bitsandbytes = prev.bitsandbytes.overrideAttrs (old: {
    # bitsandbytes dynamically loads dependencies, and always after torch.
    # That should take care of all dynamic linking.
    dontAutoPatchelf = true;
  });

  # fastapi has a fastapi binary both in fastapi and fastapi-cli. When merging environments this causes a collision.
  # Since fastapi pulls in fastapi-cli as a dependency to have a functional cli, delete the one in fastapi.
  fastapi = prev.fastapi.overrideAttrs (old: {

    preFixup = ''
      rm -f $out/bin/fastapi
    '';
  });
  fastcoref = prev.fastcoref.overrideAttrs (old: {
    buildInputs = (old.buildInputs or [ ]) ++ [
      prev.setuptools
    ];
  });
  nats-py = prev.nats-py.overrideAttrs (old: {
    buildInputs = (old.buildInputs or [ ]) ++ [ prev.setuptools ];
  });
  nkeys = prev.nkeys.overrideAttrs (old: {
    buildInputs = (old.buildInputs or [ ]) ++ [ prev.setuptools ];
  });
  langdetect = prev.langdetect.overrideAttrs (old: {
    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ final.setuptools ];
  });
  python-magic = prev.python-magic.overrideAttrs (old: {
    # we need to patch the shared object loader to hold an exact location
    preFixup = ''
      sed -i "s|yield 'libmagic.so.1'|yield '${file}/lib/libmagic.so.1'|" $out/lib/python3.13/site-packages/magic/loader.py
    '';
  });
  # TODO - maybe it's not great to only fixup paddle for cu12 but that's what I use
  paddlepaddle-gpu = prev.paddlepaddle-gpu.overrideAttrs (old: {
    buildInputs = (old.buildInputs or [ ]) ++ [ rdma-core ];
    autoPatchelfIgnoreMissingDeps = [
      "libcuda.so.1"
    ];
    cudaDependencies = with final; [
      nvidia-cublas-cu12
      nvidia-cudnn-cu12
      nvidia-cusolver-cu12
      nvidia-curand-cu12
      nvidia-cuda-nvrtc-cu12
      nvidia-cuda-runtime-cu12
    ];

    preFixup =
      (old.preFixup or "")
      + ''
        for dep in $cudaDependencies;do
          addAutoPatchelfSearchPath $dep/lib/python*/site-packages/nvidia/*/lib/
        done

        cat ${./cuda_preloader.py} > core.py
        cat $out/lib/python*/site-packages/paddle/base/core.py >> core.py
        mv core.py $out/lib/python*/site-packages/paddle/base/core.py
      '';
  });
  antlr4-python3-runtime = prev.antlr4-python3-runtime.overrideAttrs (old: {
    buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
  });
  nvidia-nvshmem-cu12 = prev.nvidia-nvshmem-cu12.overrideAttrs (old: {
    buildInputs = (old.buildInputs or [ ]) ++ [
      libfabric
      pmix
      mpi
      rdma-core
    ];
  });
  torchvision = withCudaLibs "torchvision" prev.torchvision;

  nvidia-nvshmem-cu13 = prev.nvidia-nvshmem-cu13.overrideAttrs (old: {
    buildInputs = (old.buildInputs or [ ]) ++ [
      libfabric
      pmix
      mpi
      rdma-core
    ];
  });
  nvidia-cusparse = prev.nvidia-cusparse.overrideAttrs (old: {
    preFixup =
      (old.preFixup or "")
      + ''
        addAutoPatchelfSearchPath ${final.nvidia-nvjitlink}/lib/python*/site-packages/nvidia/cu*/lib/
      '';
  });
  nvidia-cusolver = prev.nvidia-cusolver.overrideAttrs (old: {
    cudaDependencies = with final; [
      nvidia-nvjitlink
      nvidia-cublas
      nvidia-cusparse
    ];
    preFixup =
      (old.preFixup or "")
      + ''
        for dep in $cudaDependencies;do
          addAutoPatchelfSearchPath $dep/lib/python*/site-packages/nvidia/*/lib/
        done
      '';
  });
  nvidia-cuda-runtime =
    let
      cuda-loader-helper = callPackage ./cuda-loader-helper { };
    in
      prev.nvidia-cuda-runtime.overrideAttrs (old: {
        patchelfFlags = [
          "--add-needed ${cuda-loader-helper}/lib/cuda_loader_helper.so"
        ];
        appendRunpaths = (old.appendRunpaths or [ ]) ++ [ "$ORIGIN" ];
      });
  nvidia-cudnn-cu13 = prev.nvidia-cudnn-cu13.overrideAttrs (old: {
    appendRunpaths = (old.appendRunpaths or [ ]) ++ [ "$ORIGIN" ];
  });
  sam2 = prev.sam2.overrideAttrs (old: {
    nativeBuildInputs = (old.nativeBuildInputs or []) ++ [final.setuptools];
  });
  iopath = prev.iopath.overrideAttrs (old: {
    nativeBuildInputs = (old.nativeBuildInputs or []) ++ [final.setuptools];
  });
  pyvips = prev.pyvips.overrideAttrs (old: {
    nativeBuildInputs = (old.nativeBuildInputs or []) ++ [final.setuptools];
  });
}
// builtins.listToAttrs (
  map
    (name: {
      inherit name;
      value = (withCudaLibs name prev.${name}).overrideAttrs (old: {
        # libcufile_rdma links the RDMA userspace libraries, which live in rdma-core.
        # Matched by name prefix rather than by exact attribute: the cufile wheels
        # come in cu12 and cu13 variants, and which spelling reaches the environment
        # depends on the lock.
        preFixup =
          (old.preFixup or "")
          + ''
            addAutoPatchelfSearchPath ${rdma-core}/lib
          '';
      });
    })
    (builtins.filter (name: lib.hasPrefix "nvidia-cufile" name) (builtins.attrNames prev))
)
// lib.optionalAttrs (prev ? flash-attn) {
  # flash-attn's legacy setup.py builds CUDA kernels against torch at build time.
  # Its prebuilt-wheel download path cannot work in a sandbox and would otherwise
  # silently produce a binary for a foreign toolchain, so it is forced off. setup.py
  # only emits gencode for 80/90/100/120 -- there is no 89 -- and Ampere cubins run
  # on Ada through CUDA's minor-version binary compatibility, so 80 is correct for
  # an RTX 4080 and is the cheapest to build; widen FLASH_ATTN_CUDA_ARCHS (matching
  # TORCH_CUDA_ARCH_LIST) for other GPUs.
  #
  # The Python build dependencies (torch, setuptools, wheel, packaging, psutil,
  # ninja) have to be declared, since setup.py has no pyproject.toml to declare them
  # in:
  #   [tool.uv.extra-build-dependencies]
  #   "flash-attn" = ["torch", "setuptools", "wheel", "packaging", "psutil", "ninja"]
  flash-attn = prev.flash-attn.overrideAttrs (old: {
    nativeBuildInputs = (old.nativeBuildInputs or []) ++ [
      ninja
      cudaPackages_13.cudatoolkit
    ];
    # torch 2.14's headers require C++20 (std::strong_ordering in c10), but setup.py
    # pins -std=c++17, which also stops torch from adding its own -std=c++20. If a
    # future release drops the flag this becomes a no-op and torch supplies c++20.
    postPatch = (old.postPatch or "") + ''
      substituteInPlace setup.py --replace "-std=c++17" "-std=c++20"
    '';
    CUDA_HOME = "${cudaPackages_13.cudatoolkit}";
    TORCH_CUDA_ARCH_LIST = "8.0";
    FLASH_ATTENTION_FORCE_BUILD = "TRUE";
    FLASH_ATTN_CUDA_ARCHS = "80";
    NVCC_THREADS = "4";
  });
}
// lib.optionalAttrs (prev ? causal-conv1d) {
  # causal-conv1d publishes no wheels, so uv2nix builds it from the sdist and its
  # setup.py wants a CUDA toolchain in the build sandbox. CUDA_HOME and the arch
  # follow the flash-attn entry above: setup.py emits gencode from
  # TORCH_CUDA_ARCH_LIST, and 8.0 cubins run on Ada through CUDA's minor-version
  # binary compatibility, so those and NVCC_THREADS are the lines to widen for other
  # GPUs.
  #
  # The Python build dependencies (torch, setuptools, wheel, packaging, ninja) have
  # to be declared, since setup.py has no pyproject.toml to declare them in:
  #   [tool.uv.extra-build-dependencies]
  #   "causal-conv1d" = ["torch", "setuptools", "wheel", "packaging", "ninja"]
  causal-conv1d = prev.causal-conv1d.overrideAttrs (old: {
    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
      cudaPackages_13.cudatoolkit
    ];
    CUDA_HOME = "${cudaPackages_13.cudatoolkit}";
    TORCH_CUDA_ARCH_LIST = "8.0";
    NVCC_THREADS = "4";
    # Otherwise setup.py tries to fetch a prebuilt wheel from GitHub, which cannot
    # work in the sandbox. PyPI's torch is cxx11-ABI.
    CAUSAL_CONV1D_FORCE_BUILD = "TRUE";
    CAUSAL_CONV1D_FORCE_CXX11_ABI = "TRUE";
  });
}
// lib.optionalAttrs (prev ? llama-cpp-python) {
  # llama-cpp-python compiles the llama.cpp it vendors, through scikit-build-core.
  # ggml defaults GGML_CUDA to off, and CMake cannot detect an architecture when the
  # build sandbox has no GPU, so both are pinned. GGML_NATIVE=OFF keeps -march=native
  # out of the store, since the build host need not be the run host.
  #
  # scikit-build-core has to be resolvable in the *build* environment. It is the
  # sdist's declared backend, but a lock which never resolved it does not contain it,
  # so it has to be declared as well (see README.md):
  #   [tool.uv.extra-build-dependencies]
  #   "llama-cpp-python" = ["scikit-build-core"]
  #
  # CMAKE_CUDA_ARCHITECTURES is 89 (Ada); widen it for other GPUs.
  llama-cpp-python =
    let
      cuda-loader-helper = callPackage ./cuda-loader-helper { };
    in
      prev.llama-cpp-python.overrideAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or []) ++ [
          cmake
          ninja
          patchelf
          cudaPackages_13.cudatoolkit
        ];
        CMAKE_ARGS = lib.concatStringsSep " " [
          "-DGGML_CUDA=on"
          "-DGGML_NATIVE=OFF"
          "-DCMAKE_CUDA_ARCHITECTURES=89"
          "-DCUDAToolkit_ROOT=${cudaPackages_13.cudatoolkit}"
          # ggml links libcuda only for its VMM API, and autoPatchelf resolves that
          # from the toolkit's lib/stubs -- a stub then loads at run time, ggml's CUDA
          # init fails with "CUDA driver is a stub library" and inference silently
          # falls back to the CPU. Without VMM cudart is the only CUDA runtime
          # dependency, which is the situation the loader helper below already
          # handles for every other package here. Verify a real GPU load with
          # llama_supports_gpu_offload().
          "-DGGML_CUDA_NO_VMM=ON"
        ];
        preFixup =
          (old.preFixup or "")
          + ''
            # Runs before autoPatchelf sets rpaths. The driver itself is not linked:
            # the helper constructor preopens it (see cuda-loader-helper/), so the
            # reference stays unresolved rather than pointing at the toolkit's stub.
            for lib in $out/lib/python*/site-packages/llama_cpp/lib/*.so*; do
              if [ -L "$lib" ]; then continue; fi
              patchelf --add-needed ${cuda-loader-helper}/lib/cuda_loader_helper.so "$lib"
            done
          '';
      });
}
// lib.optionalAttrs (prev ? tokenspeed-mla) {
  # tokenspeed-mla (a vllm dependency on linux) ships prebuilt kernels. The objects
  # in the wheel are sm_100a/sm_103a, so they are never dlopened on Ada, but the
  # wheel still has to patch cleanly, which it does now that the sibling-package
  # libraries it links (cutlass DSL, TVM FFI) are on the search path.
  tokenspeed-mla = withCudaLibs "tokenspeed-mla" prev.tokenspeed-mla;
}
// lib.optionalAttrs (prev ? torchaudio) {
  torchaudio = withCudaLibs "torchaudio" prev.torchaudio;
}
// lib.optionalAttrs (prev ? pynvvideocodec) {
  pynvvideocodec = withCudaLibs "pynvvideocodec" prev.pynvvideocodec;
}
// lib.optionalAttrs (prev ? torchcodec) {
  # torchcodec ships one core/custom_ops module pair per ffmpeg major (4..9) and at
  # import tries them newest first, so only the pair matching the ffmpeg provided
  # here is ever loaded. autoPatchelf is all-or-nothing, and the other pairs link
  # ffmpeg libraries this environment does not have, so drop them.
  torchcodec = (withCudaLibs "torchcodec" prev.torchcodec).overrideAttrs (old: {
    buildInputs = (old.buildInputs or [ ]) ++ [ ffmpeg libheif ];
    preFixup =
      (old.preFixup or "")
      + ''
        ffmpegMajor=${lib.versions.major ffmpeg.version}
        find $out/lib/python*/site-packages/torchcodec -name 'libtorchcodec_*[0-9].so' \
          ! -name "libtorchcodec_core$ffmpegMajor.so" \
          ! -name "libtorchcodec_custom_ops$ffmpegMajor.so" \
          -delete
      '';
  });
}
// lib.optionalAttrs (prev ? torch-c-dlpack-ext) {
  # Its wheel installs a top-level build_backend.py, a helper for building the
  # per-torch addons, which collides with flashinfer-python's module of the same
  # name when the environment is assembled. Nothing imports it at run time -- the
  # addons ship prebuilt -- so drop it and keep flashinfer's, which its JIT uses.
  torch-c-dlpack-ext = (withCudaLibs "torch-c-dlpack-ext" prev.torch-c-dlpack-ext).overrideAttrs (old: {
    preFixup =
      (old.preFixup or "")
      + ''
        rm -f $out/lib/python*/site-packages/build_backend.py
      '';
  });
}
// lib.optionalAttrs (prev ? nvidia-cutlass-dsl-libs-base && prev ? nvidia-cutlass-dsl-libs-cu13) {
  # The cutlass DSL splits into a base wheel and a cu13 wheel which ship the same
  # nvidia_cutlass_dsl tree with different contents for some files; pip installs them
  # in sequence and lets the cu13 one overwrite, while the environment assembly here
  # refuses to merge differing files. Drop exactly the files the cu13 wheel also
  # provides, leaving base's unique ones and letting cu13 win where they disagree.
  #
  # Some of the shipped profiler libraries also link the driver, hence withCudaLibs.
  nvidia-cutlass-dsl-libs-base =
    (withCudaLibs "nvidia-cutlass-dsl-libs-base" prev.nvidia-cutlass-dsl-libs-base).overrideAttrs (old: {
      preFixup =
        (old.preFixup or "")
        + ''
          sp=$(echo $out/lib/python*/site-packages)
          cu13=$(echo ${final.nvidia-cutlass-dsl-libs-cu13}/lib/python*/site-packages)
          if [ -d "$sp/nvidia_cutlass_dsl" ] && [ -d "$cu13/nvidia_cutlass_dsl" ]; then
            cd "$sp"
            find nvidia_cutlass_dsl -type f | while read -r f; do
              if [ -e "$cu13/$f" ] && ! cmp -s "$f" "$cu13/$f"; then
                rm -f "$f"
              fi
            done
          fi
        '';
    });
  nvidia-cutlass-dsl-libs-cu13 = withCudaLibs "nvidia-cutlass-dsl-libs-cu13" prev.nvidia-cutlass-dsl-libs-cu13;
}
// lib.optionalAttrs (prev ? flashinfer-python) {
  # flashinfer JIT-compiles a few small ops on first use (vllm's sampler is one) and
  # links them with -L$CUDA_HOME/lib64 and -L$CUDA_HOME/lib64/stubs -- the NVIDIA
  # toolkit layout. nixpkgs ships the same libraries as lib/ and lib/stubs, so the
  # link dies with "cannot find -lcuda" and the engine never finishes starting.
  #
  # The CUDA_HOME those flags are relative to is a JIT concern and is patched below,
  # gated on the CUDA components being present; this entry stays because it is about
  # linking flags, not lookups.
  flashinfer-python = (withCudaLibs "flashinfer-python" prev.flashinfer-python).overrideAttrs (old: {
    preFixup =
      (old.preFixup or "")
      + ''
        substituteInPlace $out/lib/python*/site-packages/flashinfer/jit/cpp_ext.py \
          --replace-fail '"-L$cuda_home/lib64",' '"-L$cuda_home/lib",' \
          --replace-fail '"-L$cuda_home/lib64/stubs",' '"-L$cuda_home/lib/stubs",'
      ''
      + lib.optionalString cudaJitEnabled ''
        substituteInPlace $out/lib/python*/site-packages/flashinfer/jit/cpp_ext.py \
          --replace-fail 'os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")' 'os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "${cudaJit}"' \
          --replace-fail 'os.environ.get("CXX", "c++")' 'os.environ.get("CXX", "${stdenv.cc}/bin/c++")' \
          --replace-fail '"ninja",' '"${ninja}/bin/ninja",'
      '';
  });
}
// lib.optionalAttrs (prev ? xgrammar) {
  xgrammar = withCudaLibs "xgrammar" prev.xgrammar;
}
// lib.optionalAttrs (prev ? tilelang) {
  # tilelang bundles libtvm.so, which links Z3 under the versioned soname its build
  # used -- libz3.so.4.15 for 0.1.12. nixpkgs' z3 is a different minor with a
  # different soname, so the library that matches comes from the z3-solver wheel in
  # the lock, which bundles exactly that name. The nixpkgs z3 stays as a fallback for
  # releases whose libtvm links the unversioned name instead.
  tilelang = (withCudaLibs "tilelang" prev.tilelang).overrideAttrs (old: {
    preFixup =
      (old.preFixup or "")
      + lib.optionalString (prev ? z3-solver) ''
        addAutoPatchelfSearchPath ${final.z3-solver}/lib/python*/site-packages/z3/lib
      ''
      + ''
        addAutoPatchelfSearchPath ${z3}/lib
      ''
      + lib.optionalString cudaJitEnabled ''
        substituteInPlace $out/lib/python*/site-packages/tilelang/env.py \
          --replace-fail 'os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")' 'os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "${cudaJit}"'
      '';
  });
}
// lib.optionalAttrs (prev ? tokenspeed-triton) {
  tokenspeed-triton = withCudaLibs "tokenspeed-triton" prev.tokenspeed-triton;
}
// lib.optionalAttrs (prev ? humming-kernels) {
  humming-kernels = withCudaLibs "humming-kernels" prev.humming-kernels;
}
// lib.optionalAttrs (prev ? nvidia-deepstream-videodecode-cu13) {
  nvidia-deepstream-videodecode-cu13 = withCudaLibs "nvidia-deepstream-videodecode-cu13" prev.nvidia-deepstream-videodecode-cu13;
}
// builtins.listToAttrs (
  # Sdists which run setup.py or a cffi build without declaring setuptools as a build
  # dependency, so their build environment has nothing to import. Matched by presence
  # rather than by a guard per package, since different locks carry different subsets.
  # setuptools-rust is here for its own build, not for what it provides: it imports
  # setuptools while declaring only the requirements of the crates it builds.
  # The names below were each added when a build failed on the same import. They are
  # deliberately individual rather than matched by a prefix: a package that fails this
  # way should be named, so a genuinely different problem is not swept up with it.
  map
    (name: {
      inherit name;
      value = prev.${name}.overrideAttrs (p: {
        nativeBuildInputs = (p.nativeBuildInputs or [ ]) ++ [ final.setuptools ];
      });
    })
    (builtins.filter (name: builtins.hasAttr name prev) [
      "defuser"
      "libcst"
      "pyyaml-ft"
      "trove_classifiers"
      "trove-classifiers"
      "pyyaml_ft"
      "device-smi"
      "logbar"
      "tokenicer"
      "pluggy"
      "setuptools-rust"
      "axolotl-contribs-lgpl"
      "axolotl-contribs-mit"
      # Legacy sdists with no pyproject.toml at all, found by auditing a lock's
      # sdist-only packages and by the builds that failed on them.
      "rouge-score"
      "sqlitedict"
      "word2number"
      "zstandard"
    ])
)
// lib.optionalAttrs (prev ? pypcre) {
  # Same, and its module builds against PCRE2.
  pypcre = prev.pypcre.overrideAttrs (p: {
    nativeBuildInputs = (p.nativeBuildInputs or [ ]) ++ [
      final.setuptools
      pcre2
    ];
  });
}
// builtins.listToAttrs (
  # Sdists built by flit_core which do not declare it, so uv reaches for a backend
  # and finds nothing importable (uv2nix builds with --no-build-isolation). Unlike
  # setuptools above, flit-core is not dragged into locks by anything else, so the
  # project has to resolve it: add "flit-core" to its dependencies. That is a real
  # runtime dependency rather than a build one, which is untidy, but
  # [tool.uv.extra-build-dependencies] is not enough on its own - uv never writes
  # those into uv.lock, so uv2nix sees no such package in the set.
  map
    (name: {
      inherit name;
      value = prev.${name}.overrideAttrs (p: {
        nativeBuildInputs = (p.nativeBuildInputs or [ ]) ++ [ final."flit-core" ];
      });
    })
    (builtins.filter (name: builtins.hasAttr name prev) [
      "editables"
      "pathspec"
    ])
)
// builtins.listToAttrs (
  # Sdists whose backend is poetry-core.masonry.api, which they also do not declare -
  # the import failure names `poetry`, the parent package. Same presence-matching, and
  # the same caveat as flit-core above: poetry-core is not pulled in by anything else,
  # so the project has to resolve it into its own lock.
  map
    (name: {
      inherit name;
      value = prev.${name}.overrideAttrs (p: {
        nativeBuildInputs = (p.nativeBuildInputs or [ ]) ++ [ final."poetry-core" ];
      });
    })
    (builtins.filter (name: builtins.hasAttr name prev) [
      "tomlkit"
    ])
)
// lib.optionalAttrs (prev ? xformers) {
  # Links torch's libraries and a CUDA runtime, like the other compiled wheels.
  xformers = withCudaLibs "xformers" prev.xformers;
}
// builtins.listToAttrs (
  # Sdists that build a Rust extension and forget setuptools-rust, which is the module
  # they import; libcst needs setuptools as well, so that list is omitted for it here.
  #
  # A package must appear in exactly ONE list below: each list is its own
  # overrideAttrs on prev.<name>, and attrset // makes the later one win outright,
  # silently discarding the earlier one's nativeBuildInputs.
  map
    (name: {
      inherit name;
      value = prev.${name}.overrideAttrs (p: {
        nativeBuildInputs = (p.nativeBuildInputs or [ ]) ++ [
          final.setuptools
          final."setuptools-rust"
          final."semantic-version"
        ];
      });
    })
    (builtins.filter (name: builtins.hasAttr name prev) [
      "libcst"
    ])
)
// lib.optionalAttrs (prev ? torchao) {
  torchao = withCudaLibs "torchao" prev.torchao;
}
// lib.optionalAttrs (prev ? vllm) {
  # vllm's compiled extensions link torch and the CUDA runtime libraries from the
  # nvidia-* wheels; the driver itself is reached through the preloading every other
  # CUDA package here uses (cuda-loader-helper, loaded with torch's cudart).
  #
  # Its vendored pynvml, though, loads NVML by bare soname only, and NixOS keeps the
  # driver outside the loader's search path. That failure is well hidden: vllm's CUDA
  # platform plugin catches the exception, returns None, vllm falls back to "no
  # platform" and dies later with "Device string must not be empty". Accept the
  # system driver path, keeping the bare name for everywhere else.
  vllm =
    (withCudaLibs "vllm" prev.vllm).overrideAttrs (old: {
      preFixup =
        (old.preFixup or "")
        + ''
          substituteInPlace $out/lib/python*/site-packages/vllm/third_party/pynvml.py \
            --replace-fail 'nvmlLib = CDLL("libnvidia-ml.so.1")' \
            'nvmlLib = CDLL("/run/opengl-driver/lib/libnvidia-ml.so.1") if os.path.exists("/run/opengl-driver/lib/libnvidia-ml.so.1") else CDLL("libnvidia-ml.so.1")'
        '';
    });
}
