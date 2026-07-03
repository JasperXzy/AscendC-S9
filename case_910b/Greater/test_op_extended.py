import sys

import torch
import torch_npu
import custom_ops_lib

torch.npu.config.allow_internal_format = False


CASES = [
    ("same_2d_fp16", torch.float16, (32, 64), (32, 64), True),
    ("tail_fp32", torch.float32, (3, 7, 65), (3, 7, 65), True),
    ("broadcast_mid_fp16", torch.float16, (4, 1, 33), (1, 5, 33), True),
    ("broadcast_last_int32", torch.int32, (2, 3, 65), (65,), False),
    ("scalar_other_int8", torch.int8, (2, 5, 31), (), False),
    ("scalar_self_fp32", torch.float32, (), (3, 4, 17), True),
    ("bf16_broadcast", torch.bfloat16, (2, 1, 7), (1, 3, 7), True),
    ("same_1d_int8_tail", torch.int8, (257,), (257,), False),
]


def _make_tensor(shape, dtype, seed, with_specials):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        data = torch.randn(shape, generator=generator, dtype=torch.float32) * 16.0
        data = data.to(dtype)
        if with_specials and data.numel() > 0:
            flat = data.reshape(-1)
            flat[0] = torch.tensor(float("inf"), dtype=dtype)
            if flat.numel() > 1:
                flat[1] = torch.tensor(float("-inf"), dtype=dtype)
            if flat.numel() > 2:
                flat[2] = torch.tensor(float("nan"), dtype=dtype)
        return data

    if dtype == torch.int8:
        return torch.randint(-64, 64, shape, generator=generator, dtype=dtype)

    if dtype == torch.int32:
        return torch.randint(-1000, 1000, shape, generator=generator, dtype=dtype)

    raise TypeError(f"Unsupported dtype: {dtype}")


def _format_mismatch(actual, expected):
    diff = actual ^ expected
    diff_indices = torch.nonzero(diff.reshape(-1), as_tuple=False).reshape(-1)
    sample = diff_indices[:10].tolist()
    return f"{int(diff.sum().item())} mismatches, first flat indices: {sample}"


def run_case(name, dtype, self_shape, other_shape, with_specials):
    self_cpu = _make_tensor(self_shape, dtype, seed=17, with_specials=with_specials)
    other_cpu = _make_tensor(other_shape, dtype, seed=29, with_specials=with_specials)
    expected = torch.gt(self_cpu, other_cpu)

    actual = custom_ops_lib.custom_op(self_cpu.npu(), other_cpu.npu()).cpu()

    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{name}: shape mismatch, actual={tuple(actual.shape)}, expected={tuple(expected.shape)}")
    if actual.dtype != torch.bool:
        raise AssertionError(f"{name}: dtype mismatch, actual={actual.dtype}, expected=torch.bool")
    if not torch.equal(actual, expected):
        raise AssertionError(f"{name}: {_format_mismatch(actual, expected)}")

    print(f"[PASS] {name}: dtype={dtype}, self={self_shape}, other={other_shape}, out={tuple(expected.shape)}")


def main():
    selected = set(sys.argv[1:])
    ran = 0
    for case in CASES:
        if selected and case[0] not in selected:
            continue
        run_case(*case)
        ran += 1

    if ran == 0:
        names = ", ".join(case[0] for case in CASES)
        raise SystemExit(f"No matching cases. Available cases: {names}")


if __name__ == "__main__":
    main()
