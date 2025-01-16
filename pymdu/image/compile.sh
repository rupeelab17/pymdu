#!/bin/bash
c++-14 -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup $(python3 -m pybind11 --includes) lib.cpp -o rasterize$(python3-config --extension-suffix)
