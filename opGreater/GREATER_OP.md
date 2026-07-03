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

The current generated code is only a skeleton:

- `op_host/greater.cpp` must be updated for broadcast shape inference and richer tiling.
- `op_host/greater_tiling.h` currently stores only `size`; it needs shape, stride, dtype, and fast-path metadata.
- `op_kernel/greater.cpp` currently contains only the generated kernel entry and TODO.

Suggested implementation order:

1. Implement correct host-side broadcast shape inference.
2. Add tiling fields for total output elements, rank, input/output shapes, broadcast strides, dtype, and fast-path kind.
3. Implement a generic kernel fallback that maps each output linear index to `self` and `other`.
4. Add fast paths for same-shape contiguous input and common last-dimension broadcasting.
5. Verify dtype support on `modelart-s9`, especially `bfloat16`, `int32`, and `int8` comparison paths.
