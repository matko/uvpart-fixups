#!/usr/bin/env python3
"""Precompile gptqmodel's torch.ops extensions into the installed package.

gptqmodel JIT-compiles these extensions on first use, which needs nvcc and, for
some of them, a writable package directory; a Nix store provides neither. This
runs as a build step instead: for every requested extension it asks gptqmodel for
that extension's own source list and compiler flags, builds the library, installs
it as `gptqmodel_ext/prebuilt/<name>.so`, then loads it back to confirm the
required operators register. `utils/cpp.py` is patched to prefer such a packaged
library, so nothing is needed at run time.

Runs under an interpreter that can import gptqmodel (PYTHONPATH pointing at the
installed package), and needs:
    CUDA_HOME             CUDA toolkit providing nvcc, headers and libcudart
    TORCH_PREFIX          installed torch package, providing include/ and lib/
    TORCH_CUDA_ARCH_LIST  numeric targets, e.g. "8.9" or "8.9+PTX"
    --extensions          extension names to build, e.g. marlin_bf16 qqq
"""

import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gptqmodel import extension as extension_api

# torch's own NVCC flags. The half/bfloat16 macros are part of the API contract:
# the kernels are written against CUDA's explicit-conversion types, exactly as the
# upstream projects (vLLM, exllamav2) build these same sources.
NVCC_COMMON_FLAGS = [
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
]
# nvcc defaults to emitting *static* stubs for template __global__ kernels, but
# some dispatchers take their address from another translation unit. Requires
# nvcc >= 12.8, which the toolkit this fixup pins provides.
TEMPLATE_STUB_FLAG = "-static-global-template-stub=false"
# torch is a cxx11-ABI build (torch._C._GLIBCXX_USE_CXX11_ABI); extension objects
# are linked against its C++ objects, so the ABI must match.
CXX11_ABI_FLAG = "-D_GLIBCXX_USE_CXX11_ABI=1"
TORCH_LINK_LIBRARIES = ["-lc10", "-lc10_cuda", "-ltorch_cpu", "-ltorch_cuda", "-ltorch"]
TORCH_CPU_LINK_LIBRARIES = ["-lc10", "-ltorch_cpu", "-ltorch"]


def _arch_flags(arch_list: str) -> list[str]:
    """Translate TORCH_CUDA_ARCH_LIST into nvcc gencode flags."""

    flags = []
    for entry in arch_list.replace(",", ";").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        arch, _, modifier = entry.partition("+")
        sm = arch.replace(".", "")
        flags.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")
        if modifier.strip().upper() == "PTX":
            flags.append(f"-gencode=arch=compute_{sm},code=compute_{sm}")
    return flags


def _string_constant(module_path: Path, name: str) -> str:
    """Read one literal string constant from an installed module, without importing it."""

    for node in ast.parse(module_path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            targets = [item.id for item in node.targets if isinstance(item, ast.Name)]
            if name in targets and isinstance(node.value, ast.Constant):
                return str(node.value.value)
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == name
                and isinstance(node.value, ast.Constant)
            ):
                return str(node.value.value)
    raise SystemExit(f"`{name}` not found in `{module_path}`")


def _guard_cuda_major(torch_dir: Path, cuda_home: Path) -> None:
    """Fail early when torch and the toolkit disagree on the CUDA major version.

    Their libcudart libraries share a `libcudart.so.<major>` SONAME, so a mismatch
    links and then breaks at run time.
    """

    torch_cuda = _string_constant(torch_dir / "version.py", "cuda")
    nvcc = subprocess.run(
        [str(cuda_home / "bin" / "nvcc"), "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    match = re.search(r"release (\d+\.\d+)", nvcc)
    if match is None:
        raise SystemExit(f"cannot read the nvcc version from `{cuda_home}/bin/nvcc`")
    toolkit_cuda = match.group(1)
    if torch_cuda.split(".")[0] != toolkit_cuda.split(".")[0]:
        raise SystemExit(
            f"torch is built for CUDA {torch_cuda} but `{cuda_home}` provides nvcc "
            f"{toolkit_cuda}; pair the toolkit with torch's CUDA major "
            f"(cudaPackages_13 for cu13x, cudaPackages_12 for cu12x)."
        )


def _torch_dir(torch_prefix: Path) -> Path:
    candidates = sorted(torch_prefix.glob("lib/python*/site-packages/torch"))
    if len(candidates) != 1:
        raise SystemExit(
            f"expected one torch package under `{torch_prefix}`, got {candidates}"
        )
    return candidates[0]


def _cuda_home_include(cuda_home: Path) -> Path:
    include = cuda_home / "include"
    if not include.is_dir():
        raise SystemExit(
            f"`{cuda_home}/bin/nvcc` was not found next to an include/ dir"
        )
    return include


def _build(
    extension,
    *,
    cuda_home: Path,
    torch_dir: Path,
    build_dir: Path,
    arch_flags: list[str],
    jobs: int,
) -> Path:
    """Compile one extension with the flags gptqmodel declares for it."""

    # Resolving the sources is what makes gptqmodel generate kernel translation
    # units for the extensions that need it (marlin); they land next to the
    # sources, so they ship with the package.
    sources = [Path(path) for path in extension._resolve_sequence(extension.sources)]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise SystemExit(f"{extension.name}: missing sources: {', '.join(missing)}")
    # Flatten each source's path below its common root: extensions such as
    # exllamav3 ship same-named files in different directories, and a plain
    # basename would have them overwrite each other's object.
    source_root = Path(os.path.commonpath([str(path.parent) for path in sources]))

    needs_cuda = bool(extension.requires_cuda)
    include_flags = [
        f"-I{torch_dir / 'include'}",
        # The C++ frontend headers (torch/types.h and friends) live here.
        f"-I{torch_dir / 'include/torch/csrc/api/include'}",
        # torch's builder always adds the interpreter's include dir: torch's own
        # headers pull in Python.h.
        f"-I{sysconfig.get_path('include')}",
    ]
    if needs_cuda:
        include_flags += ["-isystem", str(_cuda_home_include(cuda_home))]
    include_flags.extend(
        f"-I{path}" for path in extension._resolved_extra_include_paths()
    )

    # torch's builder prepends the same defines and include paths to both the host
    # and the CUDA compile lines.
    common_flags = [
        f"-DTORCH_EXTENSION_NAME={extension.name}",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        *include_flags,
    ]
    cflags = [
        *common_flags,
        "-fPIC",
        "-std=c++20",
        *extension._resolved_extra_cflags(),
    ]
    cuda_flags = [
        *common_flags,
        *NVCC_COMMON_FLAGS,
        *arch_flags,
        "--compiler-options",
        "-fPIC",
        *extension._resolved_extra_cuda_cflags(),
    ]
    if needs_cuda and os.environ.get("CC"):
        cuda_flags = ["-ccbin", os.environ["CC"], *cuda_flags]
    if needs_cuda:
        # Appended last so it wins over what the extension declares: gptqmodel asks
        # nvcc for c++17, but torch's headers rely on the C++20 relaxation of the
        # typename rule in dependent scopes, the same way the host compile line above
        # does. Without this the AWQ/GPTQ kernel builds fail inside torch's headers.
        cuda_flags.append("-std=c++20")

    def compile_source(source: Path) -> Path:
        flattened = (
            source.relative_to(source_root)
            .as_posix()
            .replace("/", "_")
            .replace(".", "_")
        )
        object_path = build_dir / f"{flattened}.o"
        if source.suffix == ".cu" and needs_cuda:
            command = [
                str(cuda_home / "bin" / "nvcc"),
                "-c",
                "-o",
                str(object_path),
                TEMPLATE_STUB_FLAG,
                *cuda_flags,
                "-Xptxas",
                "-O3,-dlcm=ca",
                str(source),
            ]
        else:
            # Host C++ files go to the host compiler, as torch's build does.
            command = [
                os.environ.get("CXX", "g++"),
                "-c",
                "-o",
                str(object_path),
                *cflags,
                str(source),
            ]
        subprocess.run(command, check=True)
        return object_path

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        objects = list(pool.map(compile_source, sources))

    library_path = build_dir / f"{extension.name}.so"
    link_libraries = TORCH_LINK_LIBRARIES if needs_cuda else TORCH_CPU_LINK_LIBRARIES
    link_flags = [
        f"-L{torch_dir / 'lib'}",
        *link_libraries,
    ]
    link_flags.append("-ltorch_python")
    if needs_cuda:
        link_flags += [f"-L{cuda_home / 'lib'}", "-lcudart"]
    link_flags += list(extension._resolve_sequence(extension.extra_ldflags))
    # Store paths are absolute; recording them keeps the library loadable without
    # depending on the host's loader configuration.
    link_flags.append(f"-Wl,-rpath,{torch_dir / 'lib'}")
    if needs_cuda:
        link_flags.append(f"-Wl,-rpath,{cuda_home / 'lib'}")

    subprocess.run(
        [
            # torch links its extensions with the host compiler; nvcc does not
            # understand `-Wl,` linker flags.
            os.environ.get("CXX", "g++"),
            "-shared",
            "-o",
            str(library_path),
            *objects,
            *link_flags,
        ],
        check=True,
    )
    return library_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extensions",
        nargs="+",
        required=True,
        help="extension names from gptqmodel.extension.available_extensions()",
    )
    args = parser.parse_args()

    cuda_home = Path(os.environ["CUDA_HOME"])
    torch_dir = _torch_dir(Path(os.environ["TORCH_PREFIX"]))

    # Name the mistakes that cost nothing to find before doing any work.
    import gptqmodel

    known = set(extension_api.available_extensions())
    unknown = [name for name in args.extensions if name not in known]
    if unknown:
        raise SystemExit(
            f"unknown extension name(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(known))}."
        )

    _guard_cuda_major(torch_dir, cuda_home)
    arch_flags = _arch_flags(os.environ["TORCH_CUDA_ARCH_LIST"])
    jobs = int(os.environ.get("NIX_BUILD_CORES") or 0) or (os.cpu_count() or 1)

    # Exactly where the patched loader looks for a packaged library.
    install_dir = (
        Path(gptqmodel.__file__).resolve().parent.parent / "gptqmodel_ext" / "prebuilt"
    )
    install_dir.mkdir(parents=True, exist_ok=True)

    for name in args.extensions:
        spec = extension_api._EXTENSION_SPECS_BY_NAME[name]
        extension = spec.resolve()
        build_dir = Path("build-ops") / name
        build_dir.mkdir(parents=True, exist_ok=True)
        library_path = _build(
            extension,
            cuda_home=cuda_home,
            torch_dir=torch_dir,
            build_dir=build_dir,
            arch_flags=arch_flags,
            jobs=jobs,
        )
        installed_path = install_dir / library_path.name
        shutil.copy2(library_path, installed_path)
        # Load it back the way the runtime will, proving the packaged library is
        # found, loadable, and registers every operator the extension declares.
        # `load()` is used rather than `is_available()` because availability also
        # consults the host GPU, which a sandbox does not have.
        if not extension.load():
            raise SystemExit(
                f"{name}: installed `{installed_path}` but it did not load: "
                f"{extension.last_error_message()}"
            )
        print(f"{name}: installed and verified `{installed_path}`", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
