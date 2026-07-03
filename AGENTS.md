# Project Instructions

This project targets the Ascend AI Innovation Competition operator challenge.

## Remote CANN Workflow

The local Mac environment does not have CANN. Use the remote SSH target `modelart-s9` for CANN builds, runtime tests, and profiling.

- Local project path: `/Users/jasperxzy/Projects/AscendC-S9`
- Remote workspace root: `/home/ma-user/work`
- Remote project path: `modelart-s9:/home/ma-user/work/AscendC-S9`

Use the local source tree as the primary source of truth. After local source edits, sync the changed project files to the remote project before building or testing. If debugging requires source edits directly on the remote machine, sync those edits back to local immediately so both trees stay consistent.

Do not sync transient artifacts back by default, including:

- `build_out/`
- `dist/`
- `build/`
- `PROF*/`
- cache directories
- temporary logs

Sync final submission artifacts only when needed, such as `custom_opp_*.run` and final `.zip` packages.

Prefer `rsync` over manual `scp` for tree synchronization. A typical local-to-remote sync command is:

```bash
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.DS_Store' \
  --exclude 'build_out/' \
  --exclude 'dist/' \
  --exclude 'build/' \
  --exclude 'PROF*/' \
  /Users/jasperxzy/Projects/AscendC-S9/ \
  modelart-s9:/home/ma-user/work/AscendC-S9/
```

Before running remote build or tests, make sure the remote tree has the latest local source. Before finalizing a task, make sure any remote source edits have been copied back to the local tree.
