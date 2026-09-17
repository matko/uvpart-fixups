#!/usr/bin/env python3
"""Acceptance test: CUDA C++ JIT compilation works with nothing on PATH and no CUDA_HOME.

The fixups overlay writes absolute store paths into torch, flashinfer and tilelang, so an
environment that carries the CUDA components can compile CUDA at run time without a dev
shell's PATH and without CUDA_HOME being set. This is the test that proves it end to end: it
hands torch a real CUDA source, which forces nvcc to run, then loads the extension and calls
the kernel.

Run it from a project whose environment carries the CUDA components, with an empty
environment so that nothing can accidentally resolve through PATH:

    env -i PATH=/bin HOME=/tmp TMPDIR=/tmp TORCH_EXTENSIONS_DIR=$(mktemp -d) \\
        /path/to/environment/bin/python tests/cuda-jit-probe.py

where the interpreter is the environment's own python. Expect "RESULT: [1.0, 1.0, 1.0, 1.0]"
and exit status 0.

Two things worth knowing before trusting a pass:

  - with_cuda=True on a .cpp does NOT reach nvcc. torch compiles .cpp with the host compiler
    and only links libcudart, so a probe without a CUDA source passes against an environment
    that cannot compile CUDA at all. cuda_sources is what makes this a real test.
  - give it a fresh TORCH_EXTENSIONS_DIR, or an empty HOME. torch caches built extensions by
    name, so a cache populated by another environment's torch gets loaded instead of
    compiled -- which can look like a pass while returning wrong values.
"""
import os

import torch
from torch.utils.cpp_extension import load_inline

print("torch        :", torch.__version__, "cuda", torch.version.cuda)
print("CUDA_HOME env:", os.environ.get("CUDA_HOME", "<unset>"))
print("PATH         :", os.environ.get("PATH"))

cuda_source = r"""
#include <torch/extension.h>

__global__ void fill_one(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = 1.0f;
}

torch::Tensor ones_cuda(torch::Tensor x) {
  auto y = torch::empty_like(x);
  int n = x.numel();
  fill_one<<<(n + 255) / 256, 256>>>(y.data_ptr<float>(), n);
  return y;
}
"""

mod = load_inline(
    name="uvpart_jit_probe",
    cpp_sources="torch::Tensor ones_cuda(torch::Tensor x);",
    cuda_sources=[cuda_source],
    functions=["ones_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["-O0"],
    verbose=True,
)

# load_inline is typed as returning str; at run time it returns the loaded module.
result = mod.ones_cuda(torch.zeros(4, device="cuda")).tolist()  # pyright: ignore[reportAttributeAccessIssue]
print("RESULT:", result)
assert result == [1.0, 1.0, 1.0, 1.0], result
