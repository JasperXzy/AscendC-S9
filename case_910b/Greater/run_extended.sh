#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${ASCEND_OPP_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${ASCEND_OPP_PATH}/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH:-}"
fi

if [[ -x "/home/ma-user/gcc/bin/gcc" && -x "/home/ma-user/gcc/bin/g++" ]]; then
    export PATH="/home/ma-user/gcc/bin:${PATH}"
    export CC="/home/ma-user/gcc/bin/gcc"
    export CXX="/home/ma-user/gcc/bin/g++"
fi

if [[ "${1:-}" == "--install" ]]; then
    python3 setup.py build bdist_wheel
    pip3 install dist/custom_ops*.whl --force-reinstall
    shift
elif [[ ! -d "./dist" ]] || [[ -z "$(find ./dist -name 'custom_ops*.whl' -print -quit)" ]]; then
    python3 setup.py build bdist_wheel
    pip3 install dist/custom_ops*.whl --force-reinstall
fi

python3 test_op_extended.py "$@"
