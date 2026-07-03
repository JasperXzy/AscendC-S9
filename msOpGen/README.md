# msOpGen Configs

This directory stores operator prototype files used by CANN `msopgen`.

## Greater

Prototype file:

```text
msOpGen/greater.json
```

Expected generated project directory:

```text
opGreater/
```

Remote generation command, run on `modelart-s9` after syncing the local tree:

```bash
cd /home/ma-user/work/AscendC-S9
/home/ma-user/Ascend/cann-8.5.0/python/site-packages/bin/msopgen gen \
  -i msOpGen/greater.json \
  -c ai_core-Ascend910B4 \
  -lan cpp \
  -out opGreater
```

The `modelart-s9` environment reports the NPU name as `910B4`, so the msOpGen compute unit is `ai_core-Ascend910B4`.
This msOpGen build does not accept `-f aclnn`; generate the standard Ascend C project first and handle PyTorch/aclnn integration separately.
