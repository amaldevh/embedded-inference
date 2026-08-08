#! /usr/bin/bash

g++ -I$(python -c "import pybind11; print(pybind11.get_include())") $(pkg-config --cflags eigen3) -I $(pwd)/../controllers binding.cc $(pwd)/../controllers/smc_controller.cc $(python3-config --cflags --libs --ldflags) -fPIC -shared -o bindings$(python3-config --extension-suffix) -O2
