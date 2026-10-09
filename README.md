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
