---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
  - gguf
  - deepseek
  - metal
---

# DeepSeek V4.1 Flash GGUF for DwarfStar

Calibrated weights for the Metal implementation in
[DwarfStar](https://github.com/antirez/ds4), developed on the `ds4.1flash` branch.
These are not interchangeable with DeepSeek V4 Flash weights or its DSpark drafter.

| Download target | File size | Main weights |
| --- | ---: | ---: |
| `ds41f-q2` | 340.60 GiB | 151.77 GiB |
| `ds41f-q4` | 482.98 GiB | 294.15 GiB |
| `ds41f-vision` | 0.90 GiB | Separate vision encoder |

The language GGUFs each include 188.83 GiB of native FP8 Engram tables.
DwarfStar reads the needed rows directly from disk; it never makes the whole
table resident. Keep the GGUF on a fast local SSD. The main-weight sizes above
do not include context or runtime buffers.

## Download and Run

From a DwarfStar checkout with V4.1 support:

```sh
./download_model.sh ds41f-q2
./ds4 -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ssd-streaming --ctx 32768
```

Use SSD streaming on a single 128 GB Mac. Q2 also fits resident tensor
parallelism across two 128 GB Macs with RDMA, or full residency on a larger
Mac. Q4 needs SSD streaming on smaller Macs; the resident target is a 512 GB Mac.
It does not fit resident TP across two 128 GB Macs.

```sh
./download_model.sh ds41f-q4
./download_model.sh ds41f-vision
```

Q4 exceeds Hugging Face's single-file limit, so it is stored in two binary
parts. The downloader joins them into `DeepSeek-V4.1-Flash-Q4.gguf`, verifies
the complete file and removes the temporary parts. Allow another 37 GiB of
free disk space while joining. Rerun the command after an interruption.
The individual parts are not runnable GGUFs.

For images, add `--vision gguf/DeepSeek-V4.1-Flash-Vision.gguf` to the language
model command. The same model and memory options work with `ds4-agent` and
`ds4-server`.

## Quantization

Q2 uses IQ2_XXS routed gate/up experts and Q2_K down experts. Q4 uses Q4_K for
all routed experts. Both retain Q8 attention projections, shared experts and
output, with F16/F32 tensors elsewhere and unchanged native Engram data.
They use the same 8,192-token activation imatrix, not a requantization of one
GGUF into the other.

Source: `deepseek-ai/DeepSeek-V4.1-Flash` at
`df42c109f1defefcbfcedbe7d905718a12266e40`. The separate encoder comes from the
same checkpoint. See the included upstream MIT license.
