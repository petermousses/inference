# glm kv cache on omv smb and redqueen k3s placement plan

## tainted-node onboarding gate

redqueen stays outside k3s until the host/runtime, reboot-recovery, and mixed-load acceptance gates in the [AI inference plan](https://github.com/petermousses/kubernetes/blob/d4d5ed1ccc353359f4f8a10b1be8836d724146f2/AI_INFERENCE_PLAN.md) pass. when admitted, it joins as a **k3s agent** for host-managed inference support; do not deploy GPU Operator, claim GPU resources, or schedule inference engines in Pods.

do not join or enable redqueen or other additional nodes during NAS-backed volume migration stages 1–3. finish those stages and obtain the NAS plan's explicit approval before making nodes available. the `litellm` backup/restore row is still pending: complete and verify its backup and isolated restore before moving the gateway. after stage 3, moving each workload still needs its own explicit approval; this plan proposes LiteLLM as a follow-on and does not itself grant that approval. see the [NAS migration plan](https://github.com/petermousses/kubernetes/blob/d4d5ed1ccc353359f4f8a10b1be8836d724146f2/nas-backed-volume-migration-plan.md).

after those gates and approval, label and taint the joined agent as defined by the AI inference plan:

```sh
kubectl label node <redqueen-node> \
  ai.mousses.xyz/accelerator=strix-halo \
  node-role.kubernetes.io/ai=true
kubectl taint node <redqueen-node> \
  ai.mousses.xyz/inference=true:NoSchedule
```

only explicitly approved monitoring agents and, after its separate move approval, the LiteLLM gateway may tolerate this taint. tolerations permit scheduling but do not select redqueen, so LiteLLM also needs required node affinity or a node selector for the AI label. keep ordinary workloads repelled.

## status and scope

this is a proposal, not a deployment record. it does not authorize joining redqueen, creating an smb share, changing host services, moving the litellm pod, or changing the database.

goals:

1. reuse a validated GLM-5.3-Flash prompt state from an SMB share hosted by OpenMediaVault (OMV), without making inference depend on the share being available;
2. after the existing migration gates pass, join redqueen as a tainted k3s agent and move only the LiteLLM gateway pod there;
3. keep llama.cpp, llama-swap, model weights, and GPU admission host-managed. keep LiteLLM PostgreSQL on OMV until a separately approved database move.

## existing plan integration

the source plans were checked in the sibling `../kubernetes` checkout at commit [`d4d5ed1`](https://github.com/petermousses/kubernetes/tree/d4d5ed1ccc353359f4f8a10b1be8836d724146f2):

- [`AI_INFERENCE_PLAN.md`](https://github.com/petermousses/kubernetes/blob/d4d5ed1ccc353359f4f8a10b1be8836d724146f2/AI_INFERENCE_PLAN.md) defines redqueen's labels and `ai.mousses.xyz/inference=true:NoSchedule` taint. it keeps inference engines host-managed and describes joining redqueen only after host, mixed-load, and reboot gates. its current v1 architecture pins LiteLLM and PostgreSQL to OMV.
- [`nas-backed-volume-migration-plan.md`](https://github.com/petermousses/kubernetes/blob/d4d5ed1ccc353359f4f8a10b1be8836d724146f2/nas-backed-volume-migration-plan.md) forbids joining or enabling other nodes during backup stages 1–3. after stage 3, node availability and each workload move still require explicit approval. the `litellm` backup/restore row is still pending in that plan.
- the existing [`litellm` Deployment](https://github.com/petermousses/kubernetes/blob/d4d5ed1ccc353359f4f8a10b1be8836d724146f2/apps/litellm/deployment.yaml) is one `Recreate` replica requesting/limited to 1 CPU and 4 GiB, currently pinned to `openmediavault`. the PostgreSQL StatefulSet and migration Job are also pinned there; the database uses a retained local PV. the gateway's cache, migrations, and `/tmp` are disposable `emptyDir`s.

**placement change:** neither existing plan currently specifies moving LiteLLM to redqueen. treat that as a post-v1 follow-on proposal. this plan adds an exact LiteLLM toleration and node placement only after the NAS and Redqueen gates; it does not relax the taint for general workloads. Kubernetes tolerations permit placement but do not select a node, so the gateway also needs required node affinity or a node selector. [Kubernetes taints and tolerations](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/)

## kv sizing and expected benefit

redqueen's tracked GLM launch uses a 220,000-token context and `q8_0` K/V cache types in [`config.yaml`](model-router/config.yaml). GLM-5.3-Flash has 45 text layers: 34 linear-attention layers with fixed-size recurrent state and 11 sparse-attention layers with a 512-value compressed KV latent. its indexer uses a 128-value key pooled by four. [GLM config](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/04c4e9e95c5da8862dced7e5056455116f83a7e0/config.json)

using q8_0's 34-byte blocks for 32 values, the token-dependent payload is approximately 5,984 bytes/token for the MLA latent alone, or 6,358 bytes/token if the pooled 32-value index key is also q8_0. [q8_0 block layout](https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h). this gives:

| saved prompt state | estimated token-dependent bytes |
| --- | ---: |
| 1,000 tokens | 6.0–6.4 MB (5.7–6.1 MiB) |
| 220,000 tokens | 1.32–1.40 GB (1.23–1.30 GiB) |
| 1,000 full 220,000-token sessions | 1.32–1.40 TB (1.20–1.27 TiB) |

these are architecture-derived estimates, not measured slot-file sizes. add any fixed KDA recurrent state serialized for each saved session, token/state metadata, and filesystem overhead. measure actual files on the pinned runtime before setting a quota. do not plan for 1,000 full-context sessions without explicit OMV capacity and retention approval.

cache reuse only avoids recomputing a matching prefix. it does not speed the first cold prefill, new suffix tokens, model loading, or token generation. restoring a state can resume directly at the next token only when the next request adds no prompt tokens; a normal chat turn must prefill its newly appended user message and chat-template suffix.

## phase 1 — prove the pinned runtime can restore hybrid state

do this on redqueen with the exact deployed GLM GGUF, tokenizer/template, and `llama.cpp-d812350-moe-v13` runtime before provisioning the SMB cache. the general `llama-server` slot API supports `--slot-save-path` in documented versions, but the live redqueen binary and its custom patches must be checked directly. [llama-server slot API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md?plain=1)

1. check the pinned server's `--help` and local `/slots` API for save/restore support; record the exact runtime revision and patch-manifest hash.
2. with a disposable local directory, run a long GLM prompt, save its slot, erase/unload it, restore it, then resend the exact token prefix plus a short suffix.
3. verify real prefix reuse from processed/cached token counts and TTFT. `n_restored` or HTTP 200 alone is insufficient.
4. repeat after a server restart and with the longest expected prompt. exercise failed, truncated, incompatible, and missing state files.

**hard gate:** upstream llama.cpp has reported that slot save/restore on hybrid/recurrent models can return success but reprocess the full prompt because server prompt checkpoints are not persisted. GLM's linear-attention layers make this directly relevant. the redqueen source currently has no configured `--slot-save-path`; do not enable disk persistence until the exact pinned build proves a cache hit. if it fails, either track and test a compatible checkpoint-persistence fix in the pinned runtime or stop this project at the in-memory cache. do not solve this by silently upgrading the production runtime. [upstream hybrid restore issue](https://github.com/ggml-org/llama.cpp/issues/25913)

## phase 2 — provision an isolated OMV SMB cache share

after phase 1 passes **and** NAS backup stages 1–3 and their approval gates are complete:

1. create a dedicated OMV directory and SMB share for GLM cache files, separate from PostgreSQL, user files, model weights, and Borg/Restic backup repositories. proposed filesystem path: `/filesystem/k3s/data/redqueen-kv-cache/glm-5.3-flash/`; verify the actual OMV filesystem, free space, quota support, and snapshot/backup policy first.
2. create one non-admin SMB identity for redqueen cache access. grant it read/write only on this share; never commit its credential or put it in a command line, environment dump, or model-router config.
3. restrict TCP 445 to redqueen's reserved host address (`10.9.20.242`) on OMV and redqueen firewalls. require SMB 3.1.1 and SMB-layer encryption if the OMV server supports it. store a root-owned, mode-0600 credentials file outside the checkout. [mount.cifs options](https://www.man7.org/linux/man-pages/man8/mount.cifs.8.html)
4. mount the share on redqueen at `/srv/ai/model-router/cache/kv/glm-5.3-flash`, owned by the `ai` service identity with restrictive modes. use a root-managed systemd mount/automount with network-mount dependencies; verify mount identity before every cache operation so an absent share can never write into the local mountpoint. [systemd mount units](https://freedesktop.org/software/systemd/man/latest/systemd.mount.html)
5. test create/read/rename/fsync, a multi-gigabyte sequential transfer, reconnect after OMV restart, and read-only/full-share failure. gate every save/restore call on a verified CIFS mount; when SMB is down, skip cache I/O and run the ordinary inference path without persistent caching. never let a cache operation fall through to the local mountpoint directory.
6. exclude this disposable cache share from routine backups and snapshots unless the owner explicitly chooses otherwise. enforce a byte budget, maximum sessions, and age/LRU eviction before production use. record that eviction removes performance state, not recoverable model or user data.

## phase 3 — add safe session-aware cache orchestration

`--slot-save-path` exposes state save/restore operations; it does not by itself provide a durable conversation-to-slot policy. LiteLLM must not be assumed to provide the required stable session identity or lifecycle callbacks without a prototype.

1. prototype a host-local cache broker adjacent to the existing scheduler/llama-swap path. it must acquire the existing GLM worker lease, serialize access to the single GLM slot, restore before dispatch, and save only after successful generation completion and backend drain. a client disconnect must not release the lease before the backend stops using the slot.
2. accept a stable conversation/session identifier from the trusted caller when available. otherwise key entries from the canonical token prefix after the actual GLM chat template. always verify the saved token hash and model identity before reuse; never use a caller-controlled filename.
3. namespace entries by model-file manifest/checksum, tokenizer and chat-template revision, llama.cpp revision/patches, cache quantization, and cache-format version. invalidate incompatible entries and fall back to full prefill.
4. keep slot save/restore endpoints private to the host broker. do not expose them through LiteLLM, the public API, or a cluster Service. avoid logging prompts, tokens, cache contents, or raw session IDs.
5. use one writer per session; write a temporary file on the same SMB share, verify it, then atomically rename it. incomplete or corrupt files are misses. a save failure must not turn a completed inference into a failed response.
6. ensure a restored response is bound to the same authenticated LiteLLM virtual-key/session scope. do not share KV state across users merely because their system prompts match.

benchmark cold prefill, warm in-memory prefix reuse, and SMB restore separately at representative 16k, 64k, and 220k token contexts. record restore/save duration, bytes transferred, prefix tokens reused, suffix tokens processed, TTFT, and decode rate. require SMB restore to beat cold prefill by a useful margin on the actual network. test interruption during save, missing share, out-of-space, client cancellation, concurrent sessions, and server restart. a restored slot must never produce a different continuation from a cold run on the same exact prompt and decoding settings.

## phase 4 — satisfy cluster and migration gates

the onboarding and NAS sequencing gates at the beginning of this plan are mandatory. before joining, validate the K3s agent version, Redqueen node IP/hostname, cluster join path, Pod-CIDR routing, firewall rules, node allocatable resources, and monitoring agents. transfer the agent join token through a private owner-managed path; never put it in Git. reserve the LiteLLM Pod's actual CPU and memory use in Redqueen's host budget: its current Kubernetes request and limit are 1 CPU and 4 GiB, in addition to the K3s agent and host scheduler. after approval, join redqueen as an agent and apply the labels and taint defined in the opening section. keep raw host API ports private to the cluster/host firewall allowlist.

## phase 5 — move the LiteLLM gateway, not its database

make the placement change in the sibling Kubernetes repository, not in this inference repository. keep the current one-replica `Recreate` strategy and the existing service, public IngressRoute, virtual keys, and model configuration.

1. change only the LiteLLM gateway Deployment from the `openmediavault` node selector to required redqueen node placement, and add the exact inference-taint toleration. keep UID/GID, read-only root filesystem, resource limits, probes, service account policy, Secrets, and security controls unchanged.
2. keep LiteLLM PostgreSQL and its retained local PV on OMV. keep the short-lived LiteLLM migration Job on OMV unless a separately reviewed change requires otherwise. do not copy or relocate PostgreSQL data as part of this gateway move.
3. preserve the LiteLLM ClusterIP and Traefik routes. prove from a Pod on redqueen that PostgreSQL, Authentik/OIDC, Redqueen inference adapters, image endpoints, and required DNS/policy paths work. verify actual source addresses at redqueen's firewall before modifying any allowlist.
4. use `./apps/litellm/deploy.sh` for any LiteLLM release/schema change; never deploy this application with plain `kubectl apply -k apps/litellm`. a placement-only cutover must not run an unreviewed schema migration.
5. accept the planned `Recreate` downtime explicitly. validate authenticated models, streaming, long GLM first-token behavior, OIDC/Admin UI, metrics scraping, database connectivity, network policies, and the existing 30-minute mixed workload. check Redqueen memory pressure, scheduler admission, GPU resets, OOM counters, and service restarts.
6. confirm only the LiteLLM gateway Pod is on redqueen; PostgreSQL and the migration Job remain on OMV. confirm no ordinary cluster workload can schedule on the tainted node.

this placement is not HA: OMV remains the k3s control plane, public ingress host, and LiteLLM database host. a later database/cluster-HA project must follow the separate approval and restore-based migration sequence in `nas-backed-volume-migration-plan.md`.

## rollback and acceptance

**kv rollback:** disable broker restore/save and the disk-cache flag; inference falls back to normal prefill. keep cache files until the owner-approved retention window expires. do not restore files across changed weights, tokenizer/template, runtime, or cache format.

**LiteLLM rollback:** restore the gateway's OMV node selector and remove its redqueen toleration, then roll out the existing Deployment and verify the same service/API path. the database never moved, so no database restore is part of this rollback. retain the Redqueen node taint and approved monitoring tolerations.

accept only when the pinned hybrid runtime demonstrates real prefix-token reuse after SMB restore, cache failures degrade to misses without interrupting inference, privacy/capacity limits are enforced, and the LiteLLM relocation passes every phase-5 check. until then, disk KV caching remains disabled and LiteLLM remains on OMV.
