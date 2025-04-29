#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_debug_enabled() {
    const char *debug_env = getenv("CUDA_LOADER_DEBUG");
    return debug_env && !strcmp(debug_env, "1");
}

__attribute__((constructor))
static void preopen_libcuda() {
    int debug = is_debug_enabled();
    const char *candidates[] = {
        "libcuda.so", // unqualified first: respect LD_LIBRARY_PATH
	"/run/opengl-driver/lib/libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/lib64/libcuda.so",
        "/usr/lib/libcuda.so",
	"/usr/local/cuda/lib64/libcuda.so",
	"/opt/nvidia/lib64/libcuda.so",
        NULL
    };

    void *handle = NULL;
    const char **path = candidates;
    while (*path) {
        handle = dlopen(*path, RTLD_NOW | RTLD_GLOBAL);
        if (handle) {
            if (debug) {
                fprintf(stderr, "[cuda-loader-helper] Successfully preloaded %s\n", *path);
            }
            break;
        }
        path++;
    }

    if (!handle && debug) {
      fprintf(stderr, "[cuda-loader-helper] ERROR: Failed to preload libcuda.so: %s\n", dlerror());
    }
}
