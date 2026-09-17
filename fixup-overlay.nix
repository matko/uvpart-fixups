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
  cudaPackages_13,
}:
final: prev: {
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
  nvidia-cufile-cu12 = prev.nvidia-cufile-cu12.overrideAttrs (old: {
    buildInputs = (old.buildInputs or [ ]) ++ [
      rdma-core
    ];
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
  torchvision = prev.torchvision.overrideAttrs (old: {
    preFixup =
      (old.preFixup or "")
      + ''
        addAutoPatchelfSearchPath ${final.torch}/lib/python*/site-packages/torch/lib
      '';
  });

  nvidia-cufile = prev.nvidia-cufile.overrideAttrs (old: {
    buildInputs = (old.buildInputs or [ ]) ++ [
      rdma-core
    ];
  });
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
  # ninja) are the consuming project's to declare, since setup.py has no
  # pyproject.toml to declare them in:
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
// lib.optionalAttrs (prev ? llama-cpp-python) {
  # llama-cpp-python compiles the llama.cpp it vendors, through scikit-build-core.
  # ggml defaults GGML_CUDA to off, and CMake cannot detect an architecture when the
  # build sandbox has no GPU, so both are pinned. GGML_NATIVE=OFF keeps -march=native
  # out of the store, since the build host need not be the run host.
  #
  # scikit-build-core has to be resolvable in the *build* environment. It is the
  # sdist's declared backend, but a lock which never resolved it does not contain it,
  # so the consuming project declares it (see README.md):
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
