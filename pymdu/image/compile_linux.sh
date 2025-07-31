#!/bin/bash

# Obtenir les flags d’inclusion pour pybind11
PYTHON_INCLUDE=$(python3 -m pybind11 --includes)

# Suffixe pour le nom du module compilé
PYTHON_EXT_SUFFIX=$(python3-config --extension-suffix)

# Compilation
g++ -O3 -Wall -std=c++14 -shared -fPIC \
    $PYTHON_INCLUDE \
    lib.cpp -o lib$PYTHON_EXT_SUFFIX