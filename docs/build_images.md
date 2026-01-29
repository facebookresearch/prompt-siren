# Building Dataset Images

The `prompt-siren-build-images` CLI builds all Docker images required by datasets that
implement `ImageBuildableDataset`. It uses each dataset's image tag helpers to ensure
tags are consistent between build time and runtime.

Default registry for pushes: `ghcr.io/ethz-spylab/prompt-siren-images`. Use
`--registry ""` to disable pushes or override with your own registry.

## Usage

```bash
# Build images for a single dataset
prompt-siren-build-images --dataset swebench

# Build images for all datasets that support Docker
prompt-siren-build-images --all-datasets

# Rebuild existing images (delete if already present)
prompt-siren-build-images --dataset swebench --rebuild-existing

# Push tags to a registry
prompt-siren-build-images --dataset swebench --registry my-registry.com/myrepo

# Disable registry pushes
prompt-siren-build-images --dataset swebench --registry ""
```

Notes:
- The build command always uses the local Docker client.
- `--registry` tags and pushes `registry/<tag>` but also keeps the local tag.
- When `--rebuild-existing` is not set, the builder will skip pushing tags that
  already exist in the registry.
- Runtime pulls can be configured with a dataset registry prefix (e.g., SWE-bench config).

## Image Tag Scheme

All dataset images are expected to follow a common pattern:

```
siren-<dataset>-<role>:<id>
```

Where:
- `<dataset>` is a dataset-specific prefix (e.g., `swebench`).
- `<role>` is one of: `benign`, `pair`, `service`, `agent`, `base`, `env`.
- `<id>` is a normalized identifier (lowercase, `/ :` and spaces replaced with `__`).

### SWE-bench examples

Benign task image (per SWE-bench instance):
```
siren-swebench-benign:django__django-11179
```

Malicious service container (per malicious task/service):
```
siren-swebench-service:env_direct_exfil_task
```

Benign × malicious pair (when a malicious task modifies the benign image):
```
siren-swebench-pair:django__django-11179__env_direct_exfil_task
```

Agent image used by malicious tasks:
```
siren-swebench-agent:basic
```

Base/env cache layers (internal, hashed):
```
siren-swebench-base:<16-char-hash>
siren-swebench-env:<16-char-hash>
```

These tags are generated in `src/prompt_siren/datasets/swebench_dataset/image_tags.py`
and used consistently for both build specs and runtime pull specs.
