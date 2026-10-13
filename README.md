# inference

source for redqueen's inference runtime. the checkout on redqueen is `/srv/ai/inference`.

`hosts/redqueen` tracks host-owned code, configuration, pinned versions, model metadata and integrity manifests, and the ComfyUI workflows used by the deployment. run `hosts/redqueen/link-runtime.sh` as `ai` to make live user-owned host paths symlinks into this checkout and reload user units; it does not restart services. root-owned firewall files can be linked separately with `sudo hosts/redqueen/link-system-runtime.sh`.

- `hosts/redqueen/adapters/` provides the System One/Decisions and OpenAI-compatible image adapters.
- `hosts/redqueen/model-router/` contains llama-swap configuration, runtime helpers, and pinned build metadata.
- `hosts/redqueen/comfyui/` tracks ComfyUI configuration, dependency pins, and workflow assets. the upstream ComfyUI checkout, virtual environment, and user state are separate host data.
- `hosts/redqueen/models/` contains download scripts, version/checksum metadata, and the non-weight model metadata linked from `/srv/ai/models`.
- `hosts/redqueen/systemd/` contains user service units. `systemd-system/` and `nftables.d/` contain root-owned firewall configuration.
- `hosts/redqueen/archive/` contains the encrypted backup setup.
- `hosts/redqueen/tests/` contains host-side verification and stress scripts.

Downloaded model weights, credentials, the private archive environment, virtual environments, caches, generated input/output, and compiled upstream binaries stay on the host. historical rollback snapshots stay local.

the memory scheduler owns port `8081`; llama-swap listens privately on `127.0.0.1:8083`. it reserves weights, configured context/KV cache and compute memory using llama.cpp's allocation-free planner, plus headroom, against both host RAM and the GPU's shared memory pool. idle backends are evicted in least-recently-requested order; active requests hold leases through completion. capacity conflicts return `503` with `Retry-After: 1`, and a busy model returns `429`. image and decision adapters stay listening while their backends are unloaded. the legacy GLM/ComfyUI pause helper is disabled; scheduling now coordinates all backends.

deploy as `ai` from `/srv/ai/inference`: save `previous=$(git rev-parse HEAD)`, run `git pull --ff-only`, then `bash hosts/redqueen/model-router/deploy-scheduler.sh --rollback-ref "$previous"`. the deploy command creates `/srv/ai/model-router/venv` with Python 3.13 and pinned dependencies, rebuilds host-local memory profiles, validates units/config, obtains an idle admission window, relinks the existing runtime paths and restarts the API services. it gives up after 60 seconds if active work cannot drain. activation failure restores the previous source revision, profiles, unit enablement and active services; that rollback leaves the checkout detached at the saved revision. add `--check` to validate/profile without restarting services. rebuild profiles whenever weights, runtime binaries, context settings or scheduler policy change; stale profiles fail closed.

the scheduler, memory profiler and deployment tooling use the dedicated model-router venv. image/decision adapters use `/srv/ai/adapters/venv`; ComfyUI and CLM retain their own venvs. llama.cpp runtimes are compiled binaries isolated by their versioned directories and pinned ROCm library paths.

deployment reuses an existing model-router venv only when its dependency pins match; changed pins require a separately built replacement so failed deployments cannot silently change live dependencies. the first migration from the old adapters needs a quiet maintenance window because those processes do not hold scheduler leases.

install the uncensored image runtime once with `bash hosts/redqueen/comfyui/install-qwen-image-gguf.sh`; it installs the pinned custom node and GGUF dependency in ComfyUI's venv. deployment drains and stops resident backends so updated unit settings and plugins take effect on the next lazy start. image admission verifies the required workflow nodes are registered.
