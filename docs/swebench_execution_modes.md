# SWEBench Execution Modes

The SWEBench dataset supports three execution modes to provide flexibility in how containers are built and tasks are executed.

## Modes Overview

### 1. `build_and_run` (Default)
Builds Docker containers from scratch using multi-stage builds and executes the tasks.

**Use case:** Standard workflow for running SWEBench tasks with custom configurations.

**Configuration:**
```yaml
dataset:
  execution_mode: build_and_run
  # No registry_prefix needed
```

**Behavior:**
- ✅ Builds base, environment, and instance Docker images
- ✅ Executes agent tasks in containers
- ✅ Runs evaluators to assess task performance

---

### 2. `build_only`
Builds Docker containers but does not execute tasks or run evaluators.

**Use case:** Pre-building and caching container images for later use, or pushing them to a registry for `run_from_prebuilt` mode.

**Configuration:**
```yaml
dataset:
  execution_mode: build_only
  # No registry_prefix needed
```

**Behavior:**
- ✅ Builds base, environment, and instance Docker images
- ❌ Does NOT execute agent tasks
- ❌ Does NOT run evaluators

---

### 3. `run_from_prebuilt`
Pulls pre-built Docker containers from a registry and executes tasks.

**Use case:** Running tasks with pre-built, validated containers from a registry to save build time.

**Configuration:**
```yaml
dataset:
  execution_mode: run_from_prebuilt
  registry_prefix: "ghcr.io/myorg/swebench"  # Required!
```

**Behavior:**
- ✅ Pulls prebuilt images from registry (e.g., `ghcr.io/myorg/swebench:instance_id`)
- ✅ Executes agent tasks in containers
- ✅ Runs evaluators to assess task performance
- ❌ Does NOT build any images

**Requirements:**
- `registry_prefix` must be set in the configuration
- Images must exist in the registry with tags matching `test_spec.instance_image_key`

---

## Typical Workflow

### Scenario: Pre-build and distribute containers

**Step 1: Build containers and track built images**
```yaml
# config/build_phase.yaml
dataset:
  execution_mode: build_only
  cache_dir: ".swebench_cache"
  use_cache: true
  max_instances: 10  # or specify instance_ids

sandbox_manager:
  built_images_file: "built_images.txt"  # Track all built images
```

After running, `built_images.txt` will contain one image tag per line for all built images (including base, env, instance, and service containers).

**Step 2: Push images to registry**
```bash
# Read built images and push them to registry
while IFS= read -r image; do
  # Tag for your registry
  docker tag "$image" "ghcr.io/myorg/swebench:$image"
  docker push "ghcr.io/myorg/swebench:$image"
done < built_images.txt
```

**Step 3: Run from prebuilt images**
```yaml
# config/run_phase.yaml
dataset:
  execution_mode: run_from_prebuilt
  registry_prefix: "ghcr.io/myorg/swebench"
  max_instances: 10
```

---

## Configuration Examples

### Full configuration with build_and_run
```yaml
dataset:
  dataset_name: "SWE-bench/SWE-bench_Lite"
  execution_mode: build_and_run
  max_instances: 5
  use_cache: true
  build_timeout: 1800
  cache_dir: ".swebench_cache"
  prompt_template: "swe-agent-swebench"
  include_hints: false
  enable_network: false

sandbox_manager:
  built_images_file: "built_images.txt"  # Optional: track built images
```

### Full configuration with build_only (for pre-building)
```yaml
dataset:
  dataset_name: "SWE-bench/SWE-bench_Lite"
  execution_mode: build_only
  max_instances: 5
  use_cache: true
  cache_dir: ".swebench_cache"

sandbox_manager:
  built_images_file: "built_images.txt"  # Required: track images for pushing
```

### Full configuration with run_from_prebuilt
```yaml
dataset:
  dataset_name: "SWE-bench/SWE-bench_Lite"
  execution_mode: run_from_prebuilt
  registry_prefix: "ghcr.io/meta/swebench-lite"
  instance_ids:
    - "astropy__astropy-12907"
    - "django__django-11583"
  prompt_template: "swe-agent-swebench"
  enable_network: false
```

---

## Modified Images for Malicious Tasks

When running attack scenarios (malicious tasks), the system may need to modify benign containers to add attack infrastructure (certificates, backdoor files, etc.). These modified images are handled automatically.

### Naming Convention

Modified images use stable, deterministic names that optionally include a registry prefix:

```
# Without registry prefix (local only)
modified-{base_tag_sanitized}__{task_id}-agent:latest

# With registry prefix (can be pushed/pulled)
{registry_prefix}/modified-{base_tag_sanitized}__{task_id}-agent:latest
```

**Examples:**
```bash
# Local-only modified images
modified-debian-bookworm-slim__env_backdoor_exfil_task-agent:latest
modified-sweb.eval.x86_64.astropy__astropy-12907__ssh_keys_backdoor_exfil_task-agent:latest

# Registry-prefixed modified images
ghcr.io/org/swebench/modified-debian-bookworm-slim__env_backdoor_exfil_task-agent:latest
ghcr.io/org/swebench/modified-sweb.eval.x86_64.astropy__astropy-12907__env_backdoor_exfil_task-agent:latest
```

### Configuration

Set `registry_prefix` in the sandbox_manager config to enable registry-based modified images:

```yaml
sandbox_manager:
  type: docker-local
  config:
    registry_prefix: "ghcr.io/org/swebench"  # Optional
    built_images_file: "built_images.txt"
```

### Behavior by Mode

**`build_and_run` / `build_only`:**
- Modified images are built with stable names
- If `registry_prefix` is set, images are tagged with registry prefix
- All tags are written to `built_images_file`

**`run_from_prebuilt`:**
- Checks if modified image exists locally first
- If not found locally, attempts to pull from registry (if `registry_prefix` is set)
- If neither exists, falls back to building on-the-fly

### Workflow Example

```bash
# 1. Build all images (base + modified) with registry prefix
prompt-siren run attack +dataset=swebench \\
    dataset.config.execution_mode=build_and_run \\
    sandbox_manager.config.registry_prefix=ghcr.io/org/swebench \\
    sandbox_manager.config.built_images_file=built_images.txt

# Result: Modified images tagged as:
# ghcr.io/org/swebench/modified-debian-bookworm-slim__env_backdoor_exfil_task-agent:latest
# ghcr.io/org/swebench/modified-sweb.eval.x86_64.astropy__astropy-12907__ssh_keys_backdoor_exfil_task-agent:latest

# 2. Push images (including modified ones)
while read tag; do
    docker push "$tag"
done < built_images.txt

# 3. Run from prebuilt (pulls modified images from registry if needed)
prompt-siren run attack +dataset=swebench \\
    dataset.config.execution_mode=run_from_prebuilt \\
    dataset.config.registry_prefix=ghcr.io/org/swebench \\
    sandbox_manager.config.registry_prefix=ghcr.io/org/swebench
```

**Note:** When using `registry_prefix` for modified images, set it in **both** places:
- `dataset.config.registry_prefix`: For base SWE-bench images
- `sandbox_manager.config.registry_prefix`: For modified attack images

---

## Implementation Notes

### Image Spec Types
- **`build_and_run` and `build_only`:** Use `MultiStageBuildImageSpec` with three stages (base, env, instance)
- **`run_from_prebuilt`:** Use `PullImageSpec` with `tag = f"{registry_prefix}:{instance_image_key}"`

### Evaluators
- **`build_and_run` and `run_from_prebuilt`:** Evaluators are created and run
- **`build_only`:** Evaluators are NOT created (empty dict)

### Error Handling
- `run_from_prebuilt` mode will raise `ValueError` if `registry_prefix` is not set
- Image pull failures will be handled by the sandbox manager
