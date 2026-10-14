# Experimental parallel MTP

`pmtp` tests whether a native serial MTP head can be fine-tuned into a causal, one-pass parallel drafter. It reuses the native MTP projection, normalization, attention and feed-forward weights, plus the frozen verifier embedding and output head. New training defaults to projecting the verifier's anchor hidden state into separate future-position inputs, with a learned shared token placeholder. A shared hidden placeholder remains available as a baseline. The prototype supports text-only packed training, including Qwen3.5-MoE's native full-attention MTP layer.

This is an experiment, not a measured speedup or a drop-in vLLM speculative decoding implementation. A PMTP checkpoint cannot be deployed through serial MTP's `stitch-mtp` path; that command rejects it. `draft_block()` provides an uncached correctness reference, not a serving implementation.

## Training alignment

For a prefix ending with known token `x[a+1]`, the first MTP input combines verifier state `h[a]` and `embedding(x[a+1])`. It predicts `x[a+2]`. For a four-token proposal, three future rows predict `x[a+3]`, `x[a+4]`, and `x[a+5]`, with RoPE positions `a+1`, `a+2`, and `a+3`. The first row retains the existing MTP implementation's position `a`.

Select the future hidden inputs with `--pmtp-hidden-init`:

- `projection` (new-training default): `h_proj = Linear(H, (K-1)*H, bias=True)` maps each anchor's verifier hidden state into `K-1` distinct position inputs. Each H-wide output slice has independent weights and bias. The projection uses PyTorch's standard linear initialization and has no residual connection, GRU, or hidden-state regression loss. Its parameters learn through the existing token cross entropy. K=1 creates no projection parameters.
- `shared`: all future positions use the original learned `mask_hidden` vector, independent of the input before attention. This retains the original PMTP baseline.

Both modes use the same learned `mask_embedding` for unknown future tokens. The projection consumes the already available verifier state `h[a]`, not the first MTP output or any future verifier state, so all rows still run in one MTP decoder call. Projected vectors are learned draft inputs; they are not constrained to equal the verifier's actual future hidden states. Start each comparison from the same native MTP checkpoint; training a shared-mode checkpoint first is unnecessary.

Each prediction group sees base context rows up to its anchor, plus its own earlier future rows and itself. Groups cannot read other groups or future teacher hidden states. Packed documents are isolated using the collator's `document_ids`; labels crossing document boundaries or padding are excluded. The implementation uses a dense SDPA mask and a single decoder call. It does not yet implement SWA, multiple draft layers, iterative refinement, or KV caching.

Training randomly selects at most `--pmtp-max-anchors` groups per packed sequence; validation selects evenly spaced anchors deterministically. Cross entropy uses the existing MTP step weights controlled by `--step-weight-beta`. Vocabulary logits are computed only for selected groups, in checkpointed chunks controlled by `--pmtp-logit-chunk-size`. Dense attention memory still grows quadratically with sequence length; start with 2048 tokens.

`token_agreement_N` compares each output with the reference dataset token. `reference_prefix_length` counts leading correct predictions (0 through K) for complete groups only, with no bonus token. Neither metric is a measured serving acceptance rate. Online extraction computes teacher hidden states for the provided token sequence; it does not automatically regenerate assistant replies. Prefer target-generated replies for initial speculative-decoding experiments.

## Qwen3.5-35B-A3B with online hidden states

Use separate Linux/CUDA environments for vLLM and training, with the local `hs_connectors` installed in both. The vLLM environment must support Qwen3.5-MoE and hidden-state extraction. Allocate separate GPUs to the verifier and drafter; the verifier still needs the complete 35B weights despite its sparse activation. GPU count, tensor parallelism, and memory limits must match your hardware.

From the repository root, set the following values to your actual paths/devices:

```bash
export VLLM_PYTHON=/path/to/vllm-env/bin/python
export TRAIN_PYTHON=/path/to/train-env/bin/python
export VLLM_GPUS=0,1
export TRAIN_GPU=2
export DATA=/path/to/target-generated-conversations.jsonl
bash examples/train/pmtp_qwen3_5_35b_a3b_online.sh
```

The GPU IDs above are an example, not a tested memory configuration. The recipe derives the final text layer index from the model config (40 for the default checkpoint), launches extraction with provenance enabled, prepares the dataset, then trains with `--on-missing generate`. Both processes use the same absolute file-transfer directory. Files are temporary transport for online extraction; there is no separate offline extraction stage.

Defaults are a 50-step smoke run, up to 1000 examples, K=4, direct projection, 32 anchors, and 2048-token sequences. Set `HIDDEN_INIT=shared` for the shared-hidden baseline, or explicitly use `HIDDEN_INIT=projection`. Override `MAX_STEPS`, `MAX_SAMPLES`, `EPOCHS`, `MAX_ANCHORS`, `SEQ_LENGTH`, `LR`, `TP`, `PORT`, or `OUTPUT_DIR` as needed. Extra script arguments are forwarded to training. Use a fresh output directory for each experiment. For multi-GPU training, adapt the final command to `torchrun` and optionally `--fsdp-shard`; this smoke recipe deliberately uses one training GPU.

Saved checkpoints carry the `pmtp` config, hidden-input mode, projection or hidden-placeholder parameters, token placeholder, and updated MTP weights. As with native MTP, verifier embeddings/output weights are restored from the verifier when loading. Resume a PMTP checkpoint with `--speculator-type pmtp --from-pretrained /path/to/checkpoint`; its saved hidden-input mode, K, and anchor/chunk settings are authoritative. The CLI does not convert a resumed checkpoint between modes. Older model configs without `pmtp_hidden_init` load as `shared` to preserve their existing weights and behavior. For the same reason, direct `PMTPSpeculatorConfig(...)` construction defaults to `shared`; pass `pmtp_hidden_init="projection"` explicitly when constructing a new projection config in Python. Replaying an older training YAML from native weights uses the new training default unless `pmtp_hidden_init: shared` is added to its `pmtp` block.

Before evaluating the hypothesis, run the CPU correctness tests:

```bash
python -m pytest tests/unit/models/test_pmtp.py
```

On a CPU-only Windows checkout, use the dedicated runner:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/test_pmtp_cpu.py
```

It runs the same tiny Qwen3 and Qwen3.5-MoE tests without downloading verifier weights, and disables CUDA in its process. Because the package imports Unix `fcntl` unconditionally, the Windows runner supplies an import-only shim that raises if file locking is actually invoked. These tests validate the model and configuration, not online file transport. A Linux environment can run pytest directly without this shim.

Then run the GPU smoke recipe and compare validation loss and reference agreement against a serial MTP baseline on identical data. Actual speculative acceptance and throughput comparisons require implementing and validating a parallel proposer, verifier cache rollback, and rejection/correction handling first.
