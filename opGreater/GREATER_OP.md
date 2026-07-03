# Greater Operator

## Scope

This project is the generated Ascend C skeleton for the S9 operator challenge `Greater` task.

Reference semantics:

```python
torch.gt(self, other)
```

Inputs:

- `self`: tensor, dtype in `float16`, `bfloat16`, `float32`, `int32`, `int8`
- `other`: tensor, same dtype as `self`

Output:

- `out`: bool tensor

Required behavior:

- Infer output shape with PyTorch broadcasting rules.
- Return `self > other` element-wise.
- For floating-point inputs, comparisons involving `nan` must return `False`.
- Support non-32-aligned element counts and broadcasted shapes.

## Generated Files

The project was generated from:

```text
msOpGen/greater.json
```

Remote generation command:

```bash
cd /home/ma-user/work/AscendC-S9
/home/ma-user/Ascend/cann-8.5.0/python/site-packages/bin/msopgen gen \
  -i msOpGen/greater.json \
  -c ai_core-Ascend910B4 \
  -lan cpp \
  -out opGreater
```

## Implementation Notes

The current implementation is a correctness-first baseline:

- `op_host/greater.cpp` infers the broadcast output shape and serializes tiling data for output shape and input broadcast strides.
- `op_host/greater_tiling.h` stores total output elements, rank, blockDim, output shape, and input strides.
- `op_kernel/greater.cpp` maps each output linear index to `self` and `other` offsets and writes a bool result.

Testing helpers:

```bash
cd case_910b/Greater
bash run_extended.sh --install
```

Remote CANN build and runtime verification have been completed on `modelart-s9`. The next optimization step should add vectorized fast paths for contiguous same-shape tensors and common broadcast forms.
