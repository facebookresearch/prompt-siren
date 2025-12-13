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
