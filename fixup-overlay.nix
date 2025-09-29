{
  lib,
  config,
  callPackage,
  tbb_2022,
  stdenv,
  rdma-core,
  file
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
    buildInputs = (old.buildInputs or []) ++ [prev.setuptools];
  });
  nkeys = prev.nkeys.overrideAttrs (old: {
    buildInputs = (old.buildInputs or []) ++ [prev.setuptools];
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
}
