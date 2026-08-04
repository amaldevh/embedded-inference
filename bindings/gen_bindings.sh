#! /bin/bash
g++ bindings.cc -I$(python -c "import pybind11 as py; print(py.get_include())") $(python3-config --cflags --libs --ldflags) -shared -o bindings$(python3-config --extension-suffix) -I/usr/include/eigen3 -fPIC -O2
