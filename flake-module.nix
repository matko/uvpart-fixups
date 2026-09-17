{
  perSystem =
    { pkgs, lib, config, ... }:
    let
      fixup-overlay = pkgs.callPackage ./fixup-overlay.nix { };
    in
    {
      # uvpart declares uvpart.cudaJitToolchain: whether to provide the CUDA JIT toolchain
      # that packages compiling CUDA C++ at run time need. null, the default, defers to the
      # presence test in fixup-overlay.nix; true forces it; false suppresses it. The choice
      # travels as an attribute rather than a package, so it costs nothing and stays
      # invisible to Python.
      config.uvpart.pythonOverlays = [
        fixup-overlay
        (final: prev: {
          __uvpart-cuda-jit-toolchain = config.uvpart.cudaJitToolchain;
        })
      ];
    };
}
