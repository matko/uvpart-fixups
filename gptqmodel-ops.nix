# Precompiles gptqmodel's torch.ops JIT extensions into the package.
#
# gptqmodel builds these extensions on first use, which needs nvcc and, for some
# of them, a writable package directory; a Nix store provides neither. Every
# extension you name here is built at package build time, installed as
# `gptqmodel_ext/prebuilt/<name>.so`, and loaded from there at run time, so the
# running code needs neither a toolchain nor any environment variable.
#
# Each extension is its own derivation, and the package only copies the results in.
# Changing the selection therefore rebuilds the package and the extensions you
# added, not the ones you already had; locks with overlapping selections share
# builds. Each ops derivation also loads its library back after installing it and
# fails unless all of the operators it declares register.
#
# Extensions are named after `gptqmodel.extension.available_extensions()`; pass
# only what you use, since each costs compile time. The selection is required:
# an empty list fails the evaluation with the instructions below rather than
# building a default set.
# `marlin_fp16`/`marlin_bf16` cover the Marlin formats that AWQ and GPTQ
# checkpoints normally resolve to; the others are reachable through other
# quantization formats.
#
# Usage:
#   pythonOverlays = [
#     (import ./nix/gptqmodel-ops.nix {
#       inherit pkgs;
#       extensions = [ "marlin_fp16" "marlin_bf16" ];
#     })
#   ];
{
  pkgs,
  # Which kernels to precompile. Required, and deliberately without a default:
  # building every extension costs minutes, so the choice belongs to the caller.
  extensions ? [ ],
  # gptqmodel's declared runtime dependencies, from its wheel METADATA. They build
  # the interpreter that imports gptqmodel's extension metadata while building the
  # kernels. build-ops.py fails the build if this falls behind the metadata, so
  # bumping gptqmodel reports the missing name instead of breaking subtly.
  specDeps ? [
    "accelerate"
    "datasets"
    "defuser"
    "device-smi"
    "dill"
    "filelock"
    "jinja2"
    "logbar"
    "maturin"
    "ninja"
    "numpy"
    "packaging"
    "pillow"
    "protobuf"
    "pyarrow"
    "pypcre"
    "safetensors"
    "threadpoolctl"
    "tokenicer"
    "torch"
    "torchao"
    "transformers"
  ],
  # Must match torch's CUDA major: cu13x -> cudaPackages_13, cu12x -> cudaPackages_12.
  # build-ops.py enforces the pairing.
  cudaToolkit ? pkgs.cudaPackages_13.cudatoolkit,
  # Target GPU architectures; "+PTX" keeps the libraries usable on newer cards.
  # null keeps this helper on its own default of Ada alone; the list comes from
  # uvpart.cudaArch, and "+PTX" lands on the last entry.
  cudaArch ? null,
}:
let
  # Rendering for the capability list uvpart.cudaArch supplies; null keeps the default.
  cudaArchRender = import ./cuda-arch.nix { lib = pkgs.lib; };

  # "none" is the explicit opt-out: the loader hook is still patched in (it simply
  # finds no library, matching upstream behaviour), but nothing is compiled, so no
  # CUDA toolkit, compiler or unfree package is pulled in.
  buildNothing = extensions == [ "none" ];
  selected = if buildNothing then [ ] else extensions;
  selectionInvalid =
    !pkgs.lib.isList extensions
    || extensions == [ ]
    || (builtins.elem "none" extensions && builtins.length extensions > 1);
in
if selectionInvalid then
  throw ''
    gptqmodel-ops: select the kernels to precompile.

    This overlay builds the listed gptqmodel torch.ops extensions into the package,
    so running the code needs neither nvcc nor a writable Nix store. Selecting them
    is explicit on purpose: building all of them costs minutes, and most models only
    ever reach a few.

    Pass a non-empty list, e.g.

      pythonOverlays = [
        (import ./nix/gptqmodel-ops.nix {
          inherit pkgs;
          extensions = [ "marlin_fp16" "marlin_bf16" ];
        })
      ];

    AWQ and GPTQ checkpoints normally resolve to the Marlin kernels, so
    "marlin_fp16" and "marlin_bf16" cover them; the remaining names are the ones
    printed by `from gptqmodel import extension; extension.available_extensions()`
    (awq, qqq, exllamav2, exllamav2_awq, exllamav3, paroquant, hadamard,
    pack_block_cpu, floatx_cpu, and the Hopper/Blackwell-only machete and
    swordfish).

    If you deliberately want none of them — gptqmodel will then compile on first
    use, which needs a CUDA toolchain and a writable package directory — pass
    `extensions = [ "none" ];` alone.
  ''
else
  final: prev:
  let
    specEnv = prev.mkVirtualEnv "gptqmodel-ops-spec-env" (
      builtins.listToAttrs (
        map (name: {
          inherit name;
          value = [ ];
        }) specDeps
      )
    );
    # One derivation per extension. It reads the metadata from gptqmodel's own
    # sources rather than the installed package, which is what keeps this from
    # depending on the very derivation that installs the result. The patch makes
    # the source tree look for packaged libraries exactly like the installed copy
    # does, so the self-check exercises the real lookup.
    opsFor =
      name:
      pkgs.stdenv.mkDerivation {
        pname = "gptqmodel-ops-${name}";
        inherit (prev.gptqmodel) version src;
        patches = [ ./gptqmodel-prebuilt.patch ];
        nativeBuildInputs = [
          cudaToolkit
          specEnv
        ];
        CUDA_HOME = "${cudaToolkit}";
        TORCH_PREFIX = "${final.torch}";
        TORCH_CUDA_ARCH_LIST = if cudaArch == null then "8.9+PTX" else cudaArchRender.toTorchCudaArchList cudaArch;
        dontConfigure = true;
        dontBuild = true;
        installPhase = ''
          runHook preInstall
          mkdir -p "$out"
          PYTHONPATH="$PWD" ${specEnv}/bin/python ${./build-ops.py} --extensions ${name}
          install -m644 "$PWD"/gptqmodel_ext/prebuilt/*.so "$out/"
          runHook postInstall
        '';
      };
    ops = map opsFor selected;
  in
  {
    gptqmodel = prev.gptqmodel.overrideAttrs (
      p:
      {
        # The wheel build needs setuptools; precompiling below needs nothing from
        # this derivation's environment.
        nativeBuildInputs = (p.nativeBuildInputs or [ ]) ++ [ final.setuptools ];
        postPatch = (p.postPatch or "") + ''
          patch -p1 < ${./gptqmodel-prebuilt.patch}
        '';
      }
      // pkgs.lib.optionalAttrs (ops != [ ]) {
        postInstall = (p.postInstall or "") + ''
          # Build the path from a glob that always matches: the hooks leave
          # nullglob set, so an unmatched glob would expand to nothing.
          site_packages="$(echo "$out"/lib/python*/site-packages)"
          prebuilt="$site_packages/gptqmodel_ext/prebuilt"
          mkdir -p "$prebuilt"
          for library in ${pkgs.lib.concatStringsSep " " (map (o: "${o}/*.so") ops)}; do
            install -m644 "$library" "$prebuilt/"
          done
          # Every requested extension must have landed, otherwise the loader would
          # fall back to compiling at run time without saying so.
          expected=${builtins.toString (builtins.length ops)}
          found="$(find "$prebuilt" -maxdepth 1 -name '*.so' | wc -l)"
          if [ "$found" -ne "$expected" ]; then
            echo "expected $expected prebuilt libraries, found $found" >&2
            exit 1
          fi
        '';
      }
    );
  }
