{
  lib,
  config,
  cudaPackages,
  tbb_2021_11,
  stdenv,
}:
final: prev: {
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
  torch = prev.torch.overrideAttrs (
    old:
    let
      cudaEnabled = (stdenv.isLinux && config.allowUnfree && config.cudaSupport);
    in
    {
      nativeBuildInputs =
        (old.nativeBuildInputs or [ ]) ++ (lib.optional cudaEnabled cudaPackages.cuda_cudart);
      cudaDependencies = map (name: final.${name}) (
        builtins.filter (
          name: (!cudaEnabled || name != "nvidia-cuda-runtime-cu12") && lib.hasPrefix "nvidia-" name
        ) (builtins.attrNames prev)
      );
      autoPatchelfIgnoreMissingDeps = lib.optional (!cudaEnabled) "libcuda.so.1";
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
    buildInputs = p.buildInputs ++ [ tbb_2021_11 ];
  });
}
