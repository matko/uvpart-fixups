## PATCH - preload cuda so loading libpaddle later finds it
def _preload_cuda():
    import ctypes
    from pathlib import Path
    from nvidia import cuda_runtime

    cudart_path = Path(cuda_runtime.__file__).parent / "lib/libcudart.so.12"
    ctypes.CDLL(cudart_path, mode=ctypes.RTLD_GLOBAL)


_preload_cuda()

## END PATCH
