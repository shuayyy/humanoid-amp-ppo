"""Make headless EGL work when ctypes.util.find_library is broken.

On Turing's compute nodes, the NVIDIA driver libraries exist (libEGL.so.1 in
the default loader path) but `ctypes.util.find_library("EGL")` returns None —
there is no dev symlink and the gcc/ld fallbacks are unavailable after
`module purge`. PyOpenGL (which mujoco's EGL context uses) relies on
find_library to resolve the NAME, then dlopens the result. dlopen by soname
works fine; only the name->soname step is broken.

Import this module BEFORE mujoco/OpenGL to patch the lookup.
"""

import ctypes.util

_ORIG_FIND_LIBRARY = ctypes.util.find_library

_SONAMES = {
    "EGL": "libEGL.so.1",
    "OpenGL": "libOpenGL.so.0",
    "GLdispatch": "libGLdispatch.so.0",
    "GL": "libGL.so.1",
    "GLU": "libGLU.so.1",
}


def _find_library(name: str):
    resolved = _ORIG_FIND_LIBRARY(name)
    if resolved is not None:
        return resolved
    return _SONAMES.get(name)


ctypes.util.find_library = _find_library
