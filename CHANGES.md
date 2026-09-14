# What was changed and why

## Bugs that explain the 0.31 PESQ reproduction gap (training recipe; the model code was already a faithful GTCRN)

| File | Change | Why |
|---|---|---|
| `losses.py` | `CompressedComplexLoss` now sums separate real and imaginary MSEs | Official: `30*(real_loss + imag_loss)`. Old code took one MSE over the stacked tensor = `(real+imag)/2`, halving the complex weight. |
| `losses.py` | `mag_floor` default 1e-5 -> 0 (off) | The official loss has no floor; the gradient on near-silent bins is what pushes pause suppression. Kept as an explicit ablation option. |
| `train.py` | `--w_time` default 1.0 -> **0.1** | Official SI-SNR term is `-log10(ratio)` = bels = dB/10. With `NegSISDRLoss` in dB the equivalent weight is 0.1. The old 1.0 made SI-SDR dominate (pause under-suppression, high LSD) - exactly the symptom in §5.2. |
| `train.py` | `--lr` 1e-3, `--min_lr` 1e-6, new `--warmup_frac 0.1`; cosine is now linear-warmup -> cosine (per step), `T0/Tmult` removed | SEtrain: 25k warmup / 250k steps, 1e-3 -> 1e-6. The paper's 2e-3 with no warmup is a plausible cause of the high-loss basin (Lcomp 0.076). |
| `train.py` | warns if loss weights differ from the official ones | So a stray flag can't silently reintroduce the bug. |
| `run_all.ps1` | full utterances (`--segment 0`), batch 16, 200 epochs | ~135k steps vs the paper's 25k; 3 s crops truncated all recurrent context and zero-padded most VoiceBank files. |

## Bugs in the paper's numbers (evaluation)

| File | Change | Why |
|---|---|---|
| `metrics.py` (new) | one implementation of pause residual (Eq. 16), LSD (Eq. 17), SI-SDR, STOI, PESQ, bootstrap, paired stats | `evaluate.py` and `evaluate_any.py` used **different** definitions (mean of per-bin dB vs energy ratio; log vs linear frame masks; sqrt-Hann vs Hann). Table 6's in-domain row was not comparable with rows A/B/C, so the "step" in pause residual may be an artifact. Re-run all four conditions. |
| `evaluate.py` | win counts respect metric direction | Old `wins = (d>0)` for every key, including pause/LSD where lower is better. |
| `evaluate.py` | MACs = thop + analytic attention matmuls, `--macs_seconds` | thop does not hook `torch.matmul`, so the T×T attention was counted as **zero**. Real overhead: ~+1.3 % at 1 s, ~+12 % at 10 s (full context). Report the input length. `--attn_window` bounds it. |
| `evaluate.py`, `evaluate_any.py` | `--official` flag; `model.load_official_gtcrn()` | The old loader could not load the released `.tar` (no `cfg`, different key names). This is the reproducible "identical evaluation code" claim. |
| both evaluators | NaN-PESQ count logged and stored | Means were silently over fewer than 824 files if PESQ failed. |
| `gate_stats.py` | temperatures + gate stats + attention entropy for **all six** blocks, CSV output | Old script printed one block's temperatures and the paper reported them as the model's. |

## New experiment support

| File | Change |
|---|---|
| `model.py` | `zero_init_out` config / `--zero_init_out` flag: W_o = 0 at init (your proposed mode-B remedy, §6.2) |
| `model.py` | `attention_macs(T)` for honest complexity |
| `run_all.ps1` | resumable sequential runner: official reference, baseline ×5, H=1 ×5, H=4 ×5, zero-init ×5, SE-only/attn-only/H=2 ×3, windowed ×3, eval with same-seed paired comparison, gate stats, OOD on all seeds, summary |
| `collect_results.py` | one table per run and per configuration: mean ± std over seeds, mode-A/B count, expected and mode-A-conditional gain |

## Paper text that must change regardless of the new results

* §5.2: drop the "trained for joint denoising and dereverberation" candidate - the released **VCTK** checkpoint is a VCTK-only denoiser (the DNS3 checkpoint is the joint-task one).
* §3.6 / abstract: "+0.3 % MACs" -> quote the number at a stated length, or use `--attn_window`.
* Table 6 / Fig. 8: recompute with `metrics.py`.
* Abstract / §6: "1.5–4.0 dB" LSD -> max in Table 6 is 3.7 dB.
* §5.3: "d = 1.41" -> "√d = 1.41"; temperatures are per block (report all six or the mean).
* Lead with the seed-averaged gain, not the seed-0 gain with the utterance-level CI.
