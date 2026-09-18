#!/usr/bin/env python3
"""Acceptance test for the JIT lookups that do not need a GPU: sglang's and flashinfer's.

The fixups overlay writes absolute store paths into the packages that compile CUDA C++ at
run time, so an environment that carries the CUDA components resolves the toolchain with
nothing on PATH and no CUDA_HOME. That part is testable without a device: the two modules
below are asked where CUDA is, what they include and what they link, and flashinfer's own
resolution runs `nvcc --version` through it. Nothing here compiles a kernel or opens a
device; the runtime half — a module that actually builds and dlopens — needs a GPU and is
what tests/cuda-jit-probe.py and the projects' own probes cover.

Run it from a project whose environment carries sglang and flashinfer-python, with an
empty environment so that nothing can accidentally resolve through PATH:

    env -i PATH=/bin /path/to/environment/bin/python tests/jit-toolchain-probe.py

Both modules are located under the interpreter's own site-packages, or named explicitly:

    tests/jit-toolchain-probe.py <sglang toolchain.py> <flashinfer cpp_ext.py>

Expect "all checks passed" and exit status 0. A pass prints the toolkit each module
resolved, the include directories and the link flags it will hand the compiler, so the
distinction between "resolved the store toolkit" and "fell back to /usr/local/cuda" is
visible rather than inferred.
"""

import importlib.util
import pathlib
import sys
import types

RELPATHS = (
    ("sglang", "sglang/kernels/jit/utils/compile/toolchain.py"),
    ("flashinfer", "flashinfer/jit/cpp_ext.py"),
)


class Version:
    """The slice of packaging.version that flashinfer's cpp_ext uses."""

    def __init__(self, text):
        self.text = text
        self.parts = tuple(int(part) for part in text.split("."))

    def __ge__(self, other):
        return self.parts >= other.parts

    def __repr__(self):
        return f"Version({self.text!r})"


def module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def locate(paths):
    """The two files to load: the arguments, or the interpreter's own site-packages."""
    if paths:
        if len(paths) != 2:
            print(f"expected two paths, got {len(paths)}", file=sys.stderr)
            raise SystemExit(2)
        return dict(zip(("sglang", "flashinfer"), (pathlib.Path(path) for path in paths)))

    found = {}
    for prefix in {sys.prefix, sys.base_prefix, *(p for p in sys.path if p.endswith("site-packages"))}:
        for name, relpath in RELPATHS:
            version = f"python{sys.version_info.major}.{sys.version_info.minor}"
            candidate = pathlib.Path(prefix) / "lib" / version / "site-packages" / relpath
            if candidate.is_file():
                found.setdefault(name, candidate)
        if len(found) == len(RELPATHS):
            break
    return found


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stub_dependencies():
    """The imports both modules make, which this probe has no reason to build.

    Only torch is stubbed in a way that a check can see: `_GLIBCXX_USE_CXX11_ABI` reaches
    the compiler flags, and a wrong value here would be quoted in the output.
    """
    module("torch", _C=types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True), version=types.SimpleNamespace(cuda="13.0"))
    module("packaging")
    module("packaging.version", Version=Version)
    module(
        "tvm_ffi",
        libinfo=module(
            "tvm_ffi.libinfo",
            find_include_path=lambda: "/stub/tvm-ffi/include",
            find_dlpack_include_path=lambda: "/stub/tvm-ffi/include/dlpack",
            find_libtvm_ffi=lambda: "/stub/tvm-ffi/lib/libtvm_ffi.so",
        ),
    )


def load_sglang_toolchain(path):
    for name in ("sglang", "sglang.kernels", "sglang.kernels.jit", "sglang.kernels.jit.utils"):
        module(name).__path__ = []
    module(
        "sglang.kernels.jit.utils.arch",
        get_jit_cuda_arch=lambda: types.SimpleNamespace(major=8, minor=9, suffix="", target_name="sm_89"),
    )
    module("sglang.kernels.jit.utils.common", cache_once=lambda f: f, is_hip_runtime=lambda: False)
    return load("uvpart_sglang_toolchain", path)


def load_flashinfer_cpp_ext(path):
    # `from . import env as jit_env` and `from ..compilation_context import …` need a
    # package around the module; the directory layout is irrelevant to the probe.
    module("uvpart_flashinfer").__path__ = []
    module("uvpart_flashinfer.jit").__path__ = []
    module(
        "uvpart_flashinfer.compilation_context",
        CompilationContext=lambda: types.SimpleNamespace(get_nvcc_flags_list=lambda **kwargs: []),
    )
    module(
        "uvpart_flashinfer.jit.env",
        CCCL_INCLUDE_DIRS=[],
        FLASHINFER_INCLUDE_DIR=pathlib.Path("/stub/flashinfer/include"),
        FLASHINFER_CSRC_DIR=pathlib.Path("/stub/flashinfer/csrc"),
        CUTLASS_INCLUDE_DIRS=[],
        SPDLOG_INCLUDE_DIR=pathlib.Path("/stub/flashinfer/spdlog"),
        FLASHINFER_JIT_DIR=pathlib.Path("/stub/jit"),
    )
    return load("uvpart_flashinfer.jit.cpp_ext", path)


def unfold(ninja_text, variables):
    """The generated file wraps long values with " $\\n    " continuations."""
    unfolded, current = [], None
    for line in ninja_text.splitlines():
        if current is not None:
            if line.startswith("    "):
                current = current.rstrip(" $") + " " + line.strip()
                continue
            unfolded.append(current)
            current = None
        if line.startswith(tuple(f"{name} =" for name in variables)):
            current = line[:-1] if line.endswith("$") else line
            if not line.endswith("$"):
                unfolded.append(current)
                current = None
    if current is not None:
        unfolded.append(current)
    return unfolded


failures = []


def check(label, condition, detail):
    print(f"{'ok  ' if condition else 'FAIL'} {label}: {detail}")
    if not condition:
        failures.append(label)


def main():
    wanted = locate(sys.argv[1:])
    missing = [path for name, path in RELPATHS if name not in wanted]
    if missing:
        print(
            "not installed, so there is nothing to check: "
            + ", ".join(missing)
            + "\nrun this from an environment that carries sglang and flashinfer-python, "
            "or name both files on the command line",
            file=sys.stderr,
        )
        return 2

    stub_dependencies()

    toolchain = load_sglang_toolchain(wanted["sglang"])
    toolkit = toolchain.cuda_home()
    includes = toolchain.base_include_paths()
    link = toolchain.base_link_flags(with_device=True)
    print(f"sglang    cuda_home()          -> {toolkit}")
    print(f"sglang    device_compiler_path -> {toolchain.device_compiler_path()}")
    print(f"sglang    host_compiler_path   -> {toolchain.host_compiler_path()}")
    print(f"sglang    base_include_paths   -> {includes}")
    print(f"sglang    base_link_flags      -> {link}")

    check("sglang resolves the store toolkit", toolkit.startswith("/nix/store/"), toolkit)
    check("sglang nvcc", pathlib.Path(toolkit, "bin/nvcc").is_file(), toolchain.device_compiler_path())
    check("sglang host compiler", "/nix/store/" in toolchain.host_compiler_path(), toolchain.host_compiler_path())
    check("sglang CUDA include", pathlib.Path(toolkit, "include/cuda_runtime.h").is_file(), f"{toolkit}/include")
    check("sglang passes it to nvcc", f"{toolkit}/include" in includes, includes)
    check("sglang CUDA_home/lib", f"-L{toolkit}/lib" in link, link)
    check("sglang -lcudart", "-lcudart" in link, link)
    check("sglang lazy binding", "-Wl,-z,lazy" in link, link)

    ninja_py = wanted["sglang"].with_name("ninja.py").read_text()
    command = next(line.strip() for line in ninja_py.splitlines() if "_BUILD_FILE]" in line)
    check("sglang runs ninja by store path", command.startswith('command = ["/nix/store/'), command)

    # An unpatched flashinfer shells out to `which` before it reaches its own fallback, so
    # on the artifact this is meant to catch the section raises rather than fails a check.
    try:
        cpp_ext = load_flashinfer_cpp_ext(wanted["flashinfer"])
        version = cpp_ext.get_cuda_version()
        ninja_text = cpp_ext.generate_ninja_build_for_op(
            name="uvpart-jit-toolchain-probe",
            sources=[pathlib.Path("/stub/probe.cu")],
            extra_cflags=None,
            extra_cuda_cflags=None,
            extra_ldflags=None,
            extra_include_dirs=None,
        )
        unfolded = unfold(ninja_text, ("cuda_home", "cxx", "nvcc", "ldflags"))
        for line in unfolded:
            print(f"flashinfer {line}")
        ldflags = next(line for line in unfolded if line.startswith("ldflags"))
        cxx = next(line for line in unfolded if line.startswith("cxx"))

        check(
            "flashinfer resolves the store toolkit",
            cpp_ext.get_cuda_path().startswith("/nix/store/"),
            cpp_ext.get_cuda_path(),
        )
        check("flashinfer nvcc runs", version.parts[0] == 13, version)
        check("flashinfer cxx", "/nix/store/" in cxx, cxx)
        check("flashinfer lib dir", "-L$cuda_home/lib" in ldflags and "lib64" not in ldflags, ldflags)
        check("flashinfer lazy binding", "-Wl,-z,lazy" in ldflags, ldflags)
    except Exception as error:  # noqa: BLE001 - the report is the point
        check("flashinfer resolves its toolchain", False, f"{type(error).__name__}: {error}")

    print()
    if failures:
        print("FAILURES:", ", ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
