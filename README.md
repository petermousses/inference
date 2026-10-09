# inference

source for redqueen's inference runtime. the checkout on redqueen is `/srv/ai/inference`.

`hosts/redqueen` tracks host-owned code, configuration, pinned versions, model integrity manifests, and the ComfyUI workflows used by the deployment. the host installers copy files from this checkout into the runtime paths below; pulling a commit alone does not change running services.

- `hosts/redqueen/adapters/` installs to `/srv/ai/adapters` and contains the System One/Decisions and OpenAI-compatible image adapters.
- `hosts/redqueen/model-router/` installs to `/srv/ai/model-router` and contains llama-swap configuration and runtime helpers.
- `hosts/redqueen/comfyui/` tracks ComfyUI configuration, dependency pins, workflow preparation, and tracked workflow assets for `/srv/ai/comfyui`. the upstream ComfyUI checkout, virtual environment, and user state are separate host data.
- `hosts/redqueen/models/` contains download scripts and version/checksum metadata for files under `/srv/ai/models`.
- `hosts/redqueen/systemd/` contains user service units installed under `/home/ai/.config/systemd/user`; `hosts/redqueen/systemd-system/` and `hosts/redqueen/nftables.d/` contain host firewall configuration.
- `hosts/redqueen/archive/` contains the encrypted backup setup for `/srv/ai`.
- `hosts/redqueen/tests/` contains host-side verification and stress scripts.

Downloaded model weights, credentials, virtual environments, caches, generated input/output, and compiled upstream runtimes stay on the host. the repository keeps their pins and integrity metadata where available.
