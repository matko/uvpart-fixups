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
