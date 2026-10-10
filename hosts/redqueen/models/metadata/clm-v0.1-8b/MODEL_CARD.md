---
license: apache-2.0
pipeline_tag: text-ranking
tags:
- decision-model
- contrastive-learning
- verifier
- reranker
base_model:
- Qwen/Qwen3-8B
---

# CLM v0.1-8B

Upstream: [Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)

This release is a 75.6 MB PyTorch checkpoint containing state and action projection heads. It is not a standalone language model and cannot run under `llama-server` or llama-swap. Serving it requires the `contrastive-lm` API and the matching Qwen3-8B encoder, exposed through an embeddings endpoint.

The reference deployment uses Qwen3-8B with last-token pooling and a 2048-token input limit. Redqueen serves the pinned full-precision Qwen3-8B GGUF through its existing llama.cpp runtime and runs the small CLM heads on CPU. This keeps the encoder weights and pooling method aligned with the reference; the different inference kernels may still introduce numeric drift, so treat published benchmark scores as reference results until this backend is evaluated locally.

CLM scores only supplied candidates: it does not generate text, and its probabilities are relative to the candidates in each question. The published DeepSWE and Terminal-Bench verifier results use separately fine-tuned heads; they do not describe this zero-shot reference checkpoint. CLM v0.1-8B has no image or video input support.

Runtime: `contrastive-lm==0.1.0`, `torch==2.8.0+cpu`, and the Redqueen llama.cpp build. Expect about 16.4 GB of encoder storage in addition to the 75.6 MB head.

The checkpoint is served behind Redqueen's authenticated `/v1/systemone` and `/v1/decisions` adapter as `clm-v0.1-8b` (or `clm-latest`).
