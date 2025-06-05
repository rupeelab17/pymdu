# from pybind11.setup_helpers import Pybind11Extension
import sys
from typing import Any, Dict

from setuptools_cpp import ExtensionBuilder, Pybind11Extension

print("Building rasterize for lidar")

compile_args = []
if sys.platform == "win32":
    compile_args = ['/O2', '/std:c++17']
else:
    compile_args = ['-O3', '-pthread', '-Wall', '-shared', '-std=c++11', '-undefined','dynamic_lookup']

ext_modules = [
    Pybind11Extension("pymdu.image.rasterize", ["pymdu/image/lib.cpp"],
                      extra_compile_args=compile_args,
                      language='c++'),  # mypycify([
    #     '--disallow-untyped-defs',  # Pass a mypy flag
    #     'pymdu/__init__.py',
    # ]),
]


def build(setup_kwargs: Dict[str, Any]) -> None:
    setup_kwargs.update({
        "ext_modules": ext_modules,
        # "cmd_class": {"build_ext": build_ext},
        "cmdclass": dict(build_ext=ExtensionBuilder),
        "zip_safe": False,
    })
