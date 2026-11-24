# Methods report: causal and offline decoding of finger flexion from human ECoG

*The full account of the project: what was done, why each choice was made, what the literature and the published
code show, and where every number comes from. Every results table below is printed by
`python scripts/report_tables.py` from the JSON files in `results/`, so each value can be re-derived. Section 12 lists
what a reviewer should check first.*

Author: Andreea Luiza Marin. LLM coding agents were used to write the code, under her direction.

---

## 0. Contents of the archive

| Path | What it is |
|---|---|
| `src/ecogdecode/` | the library: data loading, causal and offline features, every decoder, training, protocol, streaming (`online.py`: the packaged `OnlineDecoder`), offline replicas, FingerFlex (`fingerflex.py`), provenance (`provenance.py`) |
| `scripts/` | every experiment: `run_benchmark.py` (the blind test), `explore_validation.py` (validation studies), `kalman_study.py`, `offline_study.py`, `robustness.py`, `summarise.py`, `report_tables.py` (prints every table below), `verify_scores.py` (recomputes every stored score from the saved predictions), figures and animations |
| `tests/` | 84 tests on synthetic data (causality, streaming equals offline, exact scans, offline replicas use the future, test labels stay out of training arrays, display scaling, ...) |
| `results/runs/*.json` | per approach: validation and test scores per subject and seed, every chosen setting |
| `results/preds/` | test predictions of the seed-0 model of each approach (25 Hz steps), and the recorded flexion at the same steps |
| `results/validation_study*.json` | every validation-only configuration (rounds 4 and 5) |
| `results/kalman_study.json` | the Kalman variants (validation only) |
| `results/offline/` | offline replicas, the Wiener live/offline comparison, epoch selection on test, non-causal smoothing, diagnostics; `offline/preds/` holds the replicas' test predictions |
| `results/robustness.json`, `results/robustness_curve.json` | one electrode lost; electrodes lost best first and at random (post hoc) |
| `results/posthoc.json` | bootstrap intervals, paired differences, lag, cross-finger correlations, movement envelope, masking (post hoc) |
| `results/calibration.json`, `results/calibration/` | test r against minutes of training data (post hoc) |
| `results/offline/fingerflex.json`, `results/offline/lstm_offline.json` | FingerFlex as published (both epoch rules); approach 5b given every offline liberty |
| `results/test_looks.jsonl` | one line per script run that read the test labels: time, purpose, git commit, uncommitted changes (from round 8) |
| `results/figures/` | figures and the README GIFs (16 s excerpts; the three whole-segment videos are assets of release v0.1.0) |
| `results/logs/` | console logs of the long runs (index in `results/logs/README.md`), including `verify_scores.log` |
| `results/report_tables.md` | the output of `report_tables.py` |
| `README.md` | the public summary; `METHODS.md` (this file) is the full account |

The competition data are not included; they are free to download from the competition site (section 1). The test
labels (`true_labels.zip`) were published by the organisers after the competition.

### 0.1 Commands and run times

| Task | Command | Time |
|---|---|---|
| Tests (synthetic data, no download) | `pytest` | about 10 min |
| Lint and format | `ruff check src scripts tests` and `ruff format src scripts tests` | seconds |
| Recompute every stored score | `python scripts/verify_scores.py --data-dir data` | minutes |
| Every table of this report from the JSON files | `python scripts/report_tables.py > results/report_tables.md` | seconds |
| Validation-only study | `python scripts/explore_validation.py --data-dir data --seeds 0 1 2` | hours |
| Blind-test benchmark | `python scripts/run_benchmark.py --data-dir data --streaming` | about 3.5 h on one core, plus FingerFlex (one to several hours per subject and seed on 4 threads) |
| Join parts of a split run | `python scripts/merge_runs.py <approach>` | seconds |
| Summary and `metrics.json` | `python scripts/summarise.py` | seconds |
| Offline replicas, live versus offline | `python scripts/offline_study.py --data-dir data` (part `fingerflex`: several hours per subject and seed); `python scripts/offline_lstm.py --data-dir data` | hours |
| Post-hoc analyses (bootstrap, lag, fingers, masking) | `python scripts/posthoc.py --data-dir data` | minutes |
| Calibration curve | `python scripts/calibration.py --data-dir data --subject S --seed N` per subject and seed, then `--summarise` | about 40 min per process |
| Electrode-loss curve | `python scripts/robustness_curve.py --data-dir data` | about 30 min |
| Figures | `python scripts/make_figures.py`; `python scripts/make_posthoc_figures.py` | seconds |
| Videos, whole test segment | `python scripts/make_animation.py --data-dir data`; `python scripts/make_offline_animation.py`; the same with `--offline lstm_selected_offline --name lstm_live_vs_offline_sub3` | about 6 min each on 2 Linux cores; longer on Windows (one process) |
| One video frame as PNG | add `--still 70` to either video script | seconds |

Small changes are best tested with `pytest` or a validation-only run, not with the full benchmark. Use at most about
six PyTorch threads per process (`--threads`): on hybrid CPUs the default of one per logical core is several times
slower. Videos need `ffmpeg` on the path.

---

## 1. Data and task

* **BCI Competition IV, data set 4** (Miller and Schalk; recorded at Harborview Hospital, Seattle). Three epilepsy
  patients with subdural grids of 62, 48 and 64 electrodes, ECoG at 1 kHz (band-passed 0.15-200 Hz by the
  amplifier, already re-referenced to the common average by the organisers), finger flexion from a data glove at
  25 Hz, "super-sampled" to 1 kHz. Cue protocol: a finger name for 2 s, then 2 s rest; 30 cues per finger.
  Source: the data-set description, <https://bbci.de/competition/iv/desc_4.pdf>.
* **Split (fixed by the competition):** first 400 s of each recording for training, last 200 s for testing.
* **Metric (fixed by the competition):** Pearson r between predicted and recorded flexion, per subject and finger, at
  1 kHz; the score is the mean of the 12 values for thumb, index, middle and little finger (ring finger excluded
  because it moves with its neighbours). Same source.
* **The description gives a 37 ms (± 3 ms) lag between glove and amplifier, direction not stated.** The live
  pipeline here does not shift either stream: every decoder predicts the glove value at step t from ECoG up to
  step t and learns any lag from data (LSTM memory, Wiener lags). The second-place replica applies the shift as its
  paper did (section 8).
* **Competition results** (<https://www.bbci.de/competition/iv/results/>): 1. Liang and Bougrain 0.46;
  2. Flamary and Rakotomamonjy 0.42; 3. Salvaris 0.27; 4. Popescu 0.10; 5. Yoon and Shahabi 0.05. Only the first two
  have published papers; no paper or code was found for the third (wavelet-packet band power, CFS feature selection,
  SVR, per the results page).

---

## 2. Design decisions at a glance

| Decision | Why | Where |
|---|---|---|
| Every decoder is causal (pseudo-online): one output every 40 ms from past signal only | An implanted decoder has to run this way; the competition did not require it | 3.1 |
| Every choice is made on the last 20 % of the training segment | The test segment must stay blind; random splits of a time series leak | 3.2 |
| One model per number, never an average of models | A deployed system runs one model; averaging hides its variability | 3.5 |
| Classical, recurrent, state-space, convolutional and gated decoders under one protocol | To separate the effect of the architecture from the effect of the protocol | 5 |
| The best competition entries rebuilt twice: as published (offline) and causally | To measure what each offline liberty is worth | 7 |
| Every look at the test set is logged | Repeated looks at one test set make results optimistic | 10 |

---

## 3. The live protocol (shared by every approach)

### 3.1 What "live" means here
*Live* means pseudo-online: recorded data replayed causally, as a live system would process them, without a
closed loop (no patient saw the decoder's output).
Every decoder outputs a prediction every 40 ms (the glove's native rate) using only ECoG samples up to that moment:
forward-only filters, trailing windows, causal models. This is enforced by tests that change future samples and
check that no earlier output changes (`tests/test_features.py`, `tests/test_models.py`). Predictions are held
until the next one (zero-order hold) when scored at 1 kHz. *Why:* an implanted BCI has to run this way; the
original competition did not require it (section 7).

### 3.2 Chronological split and choices on validation only
Training segment (400 s) = first 1 s discarded (filters settling) + fit block (first 80 %) + validation block (last
20 %, 80 s, the part closest in time to the test segment). Every setting (features, penalties, network size,
stopping epoch, smoother, electrode count, ...) is chosen on the validation block; the model is then refitted on
the whole training segment and scored once on the test segment. *Why:* random splits of time series leak
information between neighbouring samples; the latest block best resembles the test conditions.

### 3.3 Features (`src/ecogdecode/features.py`, `config.py`)
Sections 3.3 and 3.4 apply to every approach except 14 (FingerFlex), which has its own features and training
(section 9); the rest of section 3 applies to all.

* Re-referencing to the **median** across electrodes at each sample (*why:* one bad electrode cannot shift the
  reference, unlike the mean; added after round 1, section 10).
* Notch filters at 60, 120, 180 Hz (US mains), forward-only IIR.
* Per electrode, every 40 ms over a trailing 100 ms window: local motor potential (mean of the signal) and log
  power in 8-12, 18-24, 75-115, 125-159 and 159-175 Hz (4th-order Butterworth band-passes, forward-only). *Why:*
  mu/beta desynchronise and high gamma rises with movement (Kubánek et al., 2009; Miller et al.); the high-gamma
  bands avoid the 120 Hz and 180 Hz harmonics. Window 100 ms beat 200 and 300 ms on validation (subject 1, round 1;
  200 ms re-tested in round 4: 0.528 vs 0.554).
* Standardisation with training statistics; **artefact masking**: any electrode whose features leave ±6 training SDs
  at a step is set to the training mean for that step (online rule; the threshold was set once, not tuned).

### 3.4 Training of the networks (`models/training.py`)
Random 256-step windows (about 10 s) from the fit block, first 50 steps excluded from the loss (state warm-up),
batch 16, 128 windows per epoch, AdamW (lr 1e-3, weight decay 1e-2), MSE on standardised flexion, early stopping on
validation (patience 10, at most 60 epochs). The validation block is decoded causally after 30 s of warm-up, as it
would be online. After selection the network is retrained from the same initial weights on the whole training
segment for the chosen number of epochs (no early stopping in the refit, so no test or validation information
enters it).

### 3.5 One model, no averaging (from round 4)
Each network is trained with 3 seeds; each seed's model is scored on its own; tables give the mean and the SD over
seeds of the three-subject score. Outputs are never averaged across seeds or approaches. Figures and animations show
the seed-0 model, fixed in advance. *Why:* averaging hides the variability a single deployed decoder would have.

### 3.6 Output smoother chosen on validation
Each approach may add a causal exponential smoother to its output, time constant from {0, 40, 80, 120, 160, 240,
320} ms, chosen on validation (0 = none). It becomes part of the decoder. Most networks chose 0; the second-place
replica chose 120-320 ms, which removes jumps between its state models.

### 3.7 Display (videos and trace figures only)
Nothing is smoothed for display since round 7: figures and videos show the raw glove and the raw decoder output,
one value every 40 ms (earlier rounds used a causal 100 ms display smoother). Scores are always computed
on the raw decoder output against the raw glove.

In the videos each finger is drawn relative to its own resting level (`handviz.display_flexion`). The glove's resting
reading drifts slowly: for patient 3 the middle finger's resting level (10th percentile per 5 s block) ranges from
-1.29 to +0.72 over the test segment, about half the finger's flexion range (2nd-98th percentile span 3.7), and sits
near its top between about 60 and 75 s. With one fixed scale for the whole segment, as in earlier rounds, the recorded
middle finger therefore looked half-flexed at rest. The recorded trace now has a running 10th percentile removed
(centred 8 s window, smoothed over 2 s). The decoders' resting outputs drift less (span of the 10th percentile per
10 s block, as a fraction of the recorded flexion range: glove up to 0.51, live decoder at most 0.09, offline decoder
at most 0.28), so each decoded trace has its own running 10th percentile removed over a trailing (causal) 20 s
window, which a live display could also do. Decoded traces share the recorded trace's scale (99th
percentile after baseline removal), so a decoder that under-shoots is drawn under-shooting.

### 3.8 Live replay
For approaches 5, 6, 9 and 15 the trained model is exported to ONNX and the whole recording is replayed in 40-sample
chunks through a streaming feature extractor; outputs match the offline (batch) causal outputs to floating-point
rounding; one step takes 0.7-0.9 ms on one CPU thread (`streaming.py`, `export.py`,
`tests/test_pipeline.py::test_live_replay_matches_offline`). `online.OnlineDecoder` packages the same steps for use
on new data (`fit`, `step`, `decode`, ONNX export); chunk-by-chunk output equals the batch prediction
(`tests/test_online.py`).

---

## 4. Results of the live benchmark (blind test)

| # | Approach | Validation r | S1 | S2 | S3 | Test r | SD |
|---|---|---|---|---|---|---|---|
| 5b | LSTM on the most movement-related electrodes (number chosen per patient) | 0.680 | 0.519 | 0.485 | 0.659 | 0.554 | 0.010 |
| 5 | LSTM (256 units, dropout 0.4, augmentation; tuned on validation) | 0.566 | 0.431 | 0.398 | 0.651 | 0.493 | 0.020 |
| 13 | Manifold-gated decoder, CEBRA-style manifold | 0.534 | 0.448 | 0.383 | 0.613 | 0.481 | 0.014 |
| 11b | Multi-patient supervised pretraining + fine-tuned LSTM | 0.489 | 0.434 | 0.378 | 0.625 | 0.479 | 0.006 |
| 12 | Manifold-gated decoder, PCA manifold | 0.556 | 0.448 | 0.368 | 0.616 | 0.477 | 0.016 |
| 11 | Contrastive pretraining + fine-tuned LSTM | 0.461 | 0.437 | 0.352 | 0.612 | 0.467 | 0.006 |
| 8b | Causal temporal convolutional network (TCN) | 0.471 | 0.432 | 0.330 | 0.630 | 0.464 | 0.018 |
| 7 | Mamba (selective state space) | 0.442 | 0.469 | 0.284 | 0.614 | 0.456 | 0.010 |
| 10 | Transient-gated decoder (Dekleva-inspired) | 0.477 | 0.495 | 0.297 | 0.564 | 0.452 | 0.009 |
| 9b | LFP-gated decoder, ablation: gate sees all bands | 0.425 | 0.454 | 0.273 | 0.582 | 0.436 | 0.010 |
| 9 | LFP-gated decoder (gate: LMP, mu, beta) | 0.435 | 0.435 | 0.282 | 0.582 | 0.433 | 0.011 |
| 2 | Flamary & Rakotomamonjy, 2nd place (live re-implementation) | 0.451 | 0.475 | 0.261 | 0.539 | 0.425 | 0.000 |
| 3 | Wiener filter (lagged ridge regression) | 0.424 | 0.418 | 0.296 | 0.532 | 0.415 | 0.000 |
| 6 | S4D state-space model | 0.397 | 0.342 | 0.303 | 0.574 | 0.406 | 0.013 |
| 14 | FingerFlex (Lomtev et al., 2023), made causal | 0.484 | 0.362 | 0.315 | 0.506 | 0.394 | 0.012 |
| 4 | Kalman filter | 0.368 | 0.419 | 0.199 | 0.502 | 0.373 | 0.000 |
| 1 | Liang & Bougrain, 2008 winner (live re-implementation) | 0.485 | 0.464 | 0.278 | 0.347 | 0.363 | 0.000 |
| 8 | Neural ODE (continuous-time dynamics) | 0.341 | 0.182 | 0.120 | 0.489 | 0.264 | 0.014 |
| 15 | POSSM-style electrode-token decoder (after Ryoo et al., 2025) | 0.127 | 0.029 | 0.078 | 0.177 | 0.095 | 0.004 |

`Test r` is the mean over the three subjects of single-model scores (networks: mean over 3 seeds; `SD` is the SD
over seeds of the three-subject score; closed-form decoders are deterministic). `Validation r` is the score on the
validation block that drove every choice; it is optimistic because choices were made on it (section 11).

### 4.1 Uncertainty from the test data (post hoc, `scripts/posthoc.py`)
The SD over seeds measures training randomness, not how much the score would change with other test data. A paired
block bootstrap resamples the 50 four-second blocks of each patient's test segment (about one cue and its rest),
the same blocks for every approach, 2,000 times, and recomputes the official score exactly from per-block sums
(seed-0 models; the lag, envelope and within-movement columns are explained in 4.2).

| Approach | Test r, seed 0 | 95 % interval (test blocks resampled) | Lag at peak (ms) | Envelope r | Within-movement r |
|---|---|---|---|---|---|
| lstm_selected | 0.541 | 0.507 to 0.571 | 77 | 0.622 | 0.435 |
| manifold_pca | 0.500 | 0.461 to 0.535 | 77 | 0.570 | 0.405 |
| manifold_contrastive | 0.490 | 0.452 to 0.525 | 67 | 0.547 | 0.405 |
| lstm | 0.479 | 0.446 to 0.509 | 77 | 0.544 | 0.405 |
| tcn | 0.476 | 0.445 to 0.502 | 103 | 0.545 | 0.346 |
| pretrained_lstm | 0.476 | 0.439 to 0.507 | 63 | 0.528 | 0.396 |
| multipatient_lstm | 0.473 | 0.432 to 0.508 | 63 | 0.534 | 0.384 |
| gated_transient | 0.458 | 0.428 to 0.486 | 83 | 0.542 | 0.381 |
| mamba | 0.456 | 0.424 to 0.480 | 133 | 0.551 | 0.336 |
| gated_lfp | 0.447 | 0.419 to 0.470 | 117 | 0.527 | 0.361 |
| gated_all | 0.429 | 0.399 to 0.451 | 133 | 0.513 | 0.326 |
| switching_linear | 0.425 | 0.394 to 0.449 | 173 | 0.544 | 0.272 |
| wiener | 0.415 | 0.392 to 0.434 | 90 | 0.494 | 0.377 |
| s4d | 0.415 | 0.387 to 0.442 | 83 | 0.499 | 0.335 |
| fingerflex | 0.404 | 0.367 to 0.438 | 80 | 0.459 | 0.307 |
| kalman | 0.373 | 0.348 to 0.395 | 93 | 0.449 | 0.298 |
| liang_bougrain | 0.363 | 0.337 to 0.449 | 93 | 0.446 | 0.335 |
| neural_ode | 0.279 | 0.253 to 0.304 | 180 | 0.329 | 0.203 |
| possm | 0.090 | 0.058 to 0.123 | 243 | 0.120 | 0.102 |

| A | B | A minus B | 95 % interval | Share of resamples with A <= B |
|---|---|---|---|---|
| lstm_selected | lstm | +0.062 | +0.029 to +0.094 | 0.000 |
| lstm_selected | fingerflex | +0.137 | +0.091 to +0.183 | 0.000 |
| lstm_selected | possm | +0.450 | +0.406 to +0.491 | 0.000 |
| lstm | mamba | +0.023 | -0.007 to +0.060 | 0.064 |
| lstm | tcn | +0.003 | -0.027 to +0.035 | 0.392 |
| lstm | s4d | +0.063 | +0.029 to +0.101 | 0.000 |
| lstm | wiener | +0.063 | +0.038 to +0.092 | 0.000 |
| lstm | possm | +0.388 | +0.345 to +0.430 | 0.000 |
| wiener | liang_bougrain | +0.052 | -0.038 to +0.069 | 0.100 |
| wiener | kalman | +0.042 | +0.027 to +0.056 | 0.000 |

### 4.2 What the decoders get right and wrong (post hoc)
* **Lag.** Correlation between decoded and recorded flexion as the decoded trace is shifted by -400 to +400 ms:
  the live decoders peak 60-180 ms late (approach 5b: 70 ms on the mean curve; the POSSM-style decoder, which
  barely decodes, 240 ms), the offline winner replica at +10 ms. Shifted back by its lag, approach 5b would score about 0.58 instead of 0.54 (seed 0).
* **Envelope against shape.** r of the movement envelope (both traces averaged over 1 s) and r within movements
  only (samples where that finger is flexed above 25 % of its range): for approach 5b 0.62 and 0.43.
* **Which finger.** Decoded-against-recorded correlations off the diagonal are about as large as the recorded
  fingers' own correlations (index with middle and ring, ring with little), so the decoders add little cross-talk
  beyond what the hand itself does (`results/figures/fingers_sub3.png`).
* **Masking.** The online artefact rule fired rarely:

| Subject | Steps with any electrode masked | Electrode-steps masked | Electrodes ever masked |
|---|---|---|---|
| 1 | 0.46 % | 0.012 % | 21 |
| 2 | 1.76 % | 0.053 % | 19 |
| 3 | 8.50 % | 0.147 % | 22 |

---

## 5. The approaches, with the reason for each choice

| # | Approach | Why it is here; key choices |
|---|---|---|
| 1 | Liang and Bougrain, live | The competition winner, made causal. Amplitude modulation in 1-60, 60-100, 100-200 Hz per 40 ms, 25-step memory (1 s), per-finger least squares, greedy forward selection of at most 10 electrode-band pairs on a 3/5-2/5 split. Causal Butterworth filters instead of the paper's FIR filters; median re-reference, masking and an intercept as in the shared pipeline. |
| 2 | Flamary and Rakotomamonjy, live | Second place, made causal. State classifier (rest or which finger) switching between per-state ridge models. Changes needed to run live: band-power state features at t and t-200 ms (the paper's AR features, made causal, gave chance-level states on validation); local motor potential at t, t-200, t-400 ms (paper: centred Savitzky-Golay filter and shifts t ± tau); states from a threshold on training flexion (paper: manual segmentation); balanced multinomial logistic regression with group-norm electrode selection (paper: mixed-norm least squares). |
| 3 | Wiener filter | The classical linear baseline: ridge regression on the current and previous feature vectors; history (10-25 steps) and penalty chosen on validation. |
| 4 | Kalman filter | The standard linear-Gaussian BCI decoder. Wu et al. (2006) form: state = positions and velocities of the five fingers, linear dynamics; observations = leading principal components of the features (8-64, chosen on validation). Section 6 explains its low score. |
| 5 | LSTM | The standard recurrent decoder. Tuned on validation in round 4: 256 units, input dropout 0.4, augmentation (a random per-electrode offset on all its standardised features in every training window, SD 0.3, plus feature noise SD 0.3, mimicking slow gain and impedance changes). |
| 5b | LSTM on selected electrodes | Round 5: approach 5 fed only by the k electrodes most related to movement, ranked on training rows by the largest absolute correlation between any of their features and any finger; k chosen per patient on validation from {4, 8, 16, 24, 32, 48, all}: 32, 8, 32. Electrodes are re-ranked on the whole training segment at refit. |
| 6 | S4D | Long memory at linear cost. Diagonal state-space model (Gu et al., 2022), trained as a convolution, run as a recurrence (tested equal). |
| 7 | Mamba | Input-dependent memory. Selective state space (Gu and Dao, 2023), pure PyTorch with an exact chunked scan (tested against the step-by-step recurrence). |
| 8 | Neural ODE | Continuous-time latent dynamics. dh/dt = -h/tau + tanh(W[h, u]), RK4 once per 40 ms. |
| 8b | TCN | A causal convolutional baseline (dilations 1-16, receptive field 2.5 s), normalised per time step (a group norm over time was found to leak the future by the causality test and was replaced). |
| 9 | LFP-gated decoder | Own design: a gate reads only the slow, LFP-band features (local motor potential, mu, beta) and weights six experts that read high-gamma power through leaky integrators (the membrane equation of neuromorphic neurons). 9b: ablation whose gate sees all bands. |
| 10 | Transient-gated decoder | As 9, plus onset/offset transients (step-to-step change of high gamma) in the gate, after Dekleva et al. (2021), who found grasp-onset transients larger and more reliable than sustained responses. No 2024/2025 Dekleva gated decoder was found; the 2021 paper is the reference used. |
| 11 | Contrastive pretraining | A shared recurrent core pretrained without labels on all three patients (InfoNCE: steps 40-200 ms apart are positives), per-patient input layers, then fine-tuned per patient (in the spirit of CEBRA). |
| 11b | Multi-patient supervised pretraining | As 11, pretrained with labels. |
| 12 | Manifold-gated decoder, PCA | Own design: features projected on an 8-D PCA manifold (fitted on training rows); a gate integrates the manifold trajectory and weights six read-outs of a shared 128-unit LSTM (dropout 0.4, same augmentation as 5). |
| 13 | Manifold-gated decoder, CEBRA-style | As 12, but the manifold is learned contrastively with finger kinematics as the auxiliary variable (CEBRA-behaviour; Schneider et al., 2023), fitted on training rows and frozen. |
| 14 | FingerFlex, made causal | The best published result on these data, re-implemented (section 9) and made causal: causal filters, Morlet wavelets delayed by half their length, left-padded convolutions, pooling windows that end at the current sample, hold upsampling; statistics from training rows; epoch (at most 20, stop after 5 without improvement) and output smoother chosen on validation. The published 200 ms ECoG-to-glove shift is dropped (a causal decoder that kept it would ignore the latest 200 ms of signal; the paper reports the shift is not critical). |
| 15 | POSSM-style decoder | After POSSM (Ryoo et al., 2025; Mila), a real-time decoder built for transfer across sessions and subjects: one token per electrode (shared projection of its six features plus a learned electrode embedding), four latent queries cross-attending to the tokens of the current step only, a 256-unit GRU and a linear read-out; trained with approach 5's recipe. **It did not learn on these data** (validation 0.13, test 0.09): its validation score peaks within a few epochs. A validation-only check (`scripts/possm_validation.py`, `results/validation_study_r8_possm_*.json`) of unit-scale electrode embeddings (0.12) and eight latents (0.13) did not help, so the approach was left as run. Routing information through attention over electrode tokens needs more data or training than 6.5 minutes of one patient give; POSSM itself is pretrained on many sessions. A multi-patient pretrained variant (15b) was stopped before it produced a score and removed. |

Left out, with reasons: **LFADS** infers each moment from the whole trial, past and future, so it cannot run live as
published; **seed or model averaging** (section 3.5).

### 5.1 How approach 5 and 5b were chosen (validation only)

| Config key | File | Validation r | S1 | S2 | S3 | Options |
|---|---|---|---|---|---|---|
| t_top8_warp | validation_study_5e.json | 0.667 | 0.533 | 0.713 | 0.756 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'top_electrodes': 8, 'time_warp': 0.2} |
| t_top8 | validation_study_5d.json | 0.664 | 0.544 | 0.693 | 0.755 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'top_electrodes': 8} |
| t_top4 | validation_study_5e.json | 0.635 | 0.588 | 0.586 | 0.732 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'top_electrodes': 4} |
| t_top24 | validation_study_5d.json | 0.622 | 0.589 | 0.550 | 0.728 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'top_electrodes': 24} |
| t_top16 | validation_study_5d.json | 0.622 | 0.598 | 0.542 | 0.725 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'top_electrodes': 16} |
| t_top32 | validation_study_5c.json | 0.618 | 0.624 | 0.462 | 0.767 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'top_electrodes': 32} |
| t_top32_warp | validation_study_5f.json | 0.612 | 0.573 | 0.507 | 0.757 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'top_electrodes': 32, 'time_warp': 0.2} |
| t_top16_cd | validation_study_5d.json | 0.602 | 0.562 | 0.536 | 0.708 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'top_electrodes': 16, 'channel_dropout': 0.1} |
| t_top32_cd | validation_study_5c.json | 0.591 | 0.552 | 0.485 | 0.737 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'top_electrodes': 32, 'channel_dropout': 0.1} |
| t_top48 | validation_study_5c.json | 0.580 | 0.569 | 0.406 | 0.765 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'top_electrodes': 48} |
| t_warp02 | validation_study_5c.json | 0.578 | 0.552 | 0.433 | 0.749 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'time_warp': 0.2} |
| t_wpe256 | validation_study_5b.json | 0.572 | 0.568 | 0.400 | 0.749 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'windows_per_epoch': 256} |
| h256_drop04_aug05 | validation_study_4.json | 0.572 | 0.522 | 0.455 | 0.739 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.5, 'feature_noise': 0.5} |
| h256_drop04_aug | validation_study_3.json | 0.567 | 0.542 | 0.428 | 0.731 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3} |
| t_wd1e-1 | validation_study_5a.json | 0.564 | 0.557 | 0.393 | 0.742 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'weight_decay': 0.1} |
| h256_drop04 | validation_study_3.json | 0.554 | 0.542 | 0.437 | 0.684 | {'hidden': 256, 'dropout': 0.4} |
| t_lr2e-3 | validation_study_5a.json | 0.554 | 0.530 | 0.382 | 0.751 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'lr': 0.002} |
| t_drop05 | validation_study_5b.json | 0.554 | 0.536 | 0.389 | 0.736 | {'hidden': 256, 'dropout': 0.5, 'gain_jitter': 0.3, 'feature_noise': 0.3} |
| manifold_pca_lstm256_aug | validation_study_4.json | 0.550 | 0.560 | 0.372 | 0.718 | {'gain_jitter': 0.3, 'feature_noise': 0.3} |
| h256_drop04_adapt180 | validation_study_3.json | 0.549 | 0.554 | 0.380 | 0.714 | {'hidden': 256, 'dropout': 0.4, 'adaptive_norm_s': 180.0} |
| t_lr5e-4 | validation_study_5a.json | 0.548 | 0.548 | 0.378 | 0.717 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'lr': 0.0005, 'max_epochs': 120, 'patience': 20} |
| h384_drop04 | validation_study_3.json | 0.544 | 0.552 | 0.346 | 0.735 | {'hidden': 384, 'dropout': 0.4} |
| h128_drop04_aug | validation_study_4.json | 0.543 | 0.538 | 0.363 | 0.727 | {'hidden': 128, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3} |
| h256_drop04_aug_cos40 | validation_study_4.json | 0.538 | 0.508 | 0.376 | 0.729 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'schedule': 'cosine', 'max_epochs': 40} |
| t_batch32 | validation_study_5b.json | 0.537 | 0.540 | 0.352 | 0.718 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'batch_size': 32, 'windows_per_epoch': 256} |
| lstm_h256 | validation_study_2.json | 0.534 | 0.533 | 0.351 | 0.719 | {} |
| t_finehg | validation_study_5b.json | 0.531 | 0.519 | 0.357 | 0.718 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3} |
| manifold_contrastive_lstm256_aug | validation_study_4.json | 0.530 | 0.530 | 0.361 | 0.700 | {'gain_jitter': 0.3, 'feature_noise': 0.3} |
| h256_drop04_win200 | validation_study_3.json | 0.528 | 0.535 | 0.363 | 0.684 | {'hidden': 256, 'dropout': 0.4} |
| lstm_drop04 | validation_study_2.json | 0.526 | 0.505 | 0.375 | 0.699 | {} |
| t_seq512 | validation_study_5a.json | 0.520 | 0.487 | 0.336 | 0.738 | {'hidden': 256, 'dropout': 0.4, 'gain_jitter': 0.3, 'feature_noise': 0.3, 'seq_len': 512} |
| h256_drop04_cos80 | validation_study_3.json | 0.519 | 0.490 | 0.382 | 0.686 | {'hidden': 256, 'dropout': 0.4, 'schedule': 'cosine', 'max_epochs': 80} |
| lstm_mse | validation_study.json | 0.516 | 0.492 | 0.367 | 0.689 | {} |
| lstm_ema | validation_study.json | 0.513 | 0.473 | 0.374 | 0.693 | {'weight_ema': 0.99} |
| manifold_contrastive_lstm | validation_study.json | 0.498 | 0.485 | 0.341 | 0.669 | {} |
| lstm_2layer | validation_study_2.json | 0.494 | 0.479 | 0.354 | 0.648 | {} |
| pretrained_multipatient | validation_study_2.json | 0.488 | 0.420 | 0.357 | 0.688 | {} |
| manifold_pca_lstm | validation_study.json | 0.487 | 0.448 | 0.360 | 0.654 | {} |
| pretrained_contrastive | validation_study_2.json | 0.477 | 0.460 | 0.348 | 0.622 | {} |
| tcn | validation_study_2.json | 0.471 | 0.471 | 0.287 | 0.655 | {} |
| lstm_h64 | validation_study_2.json | 0.460 | 0.434 | 0.282 | 0.664 | {} |
| manifold_pca_mse | validation_study.json | 0.406 | 0.411 | 0.231 | 0.576 | {} |

Keys are the names used when each run was made (definitions in `scripts/explore_validation.py`). Two of them are
misleading and kept for traceability: `manifold_pca_lstm256_aug` and `manifold_contrastive_lstm256_aug` used a
128-unit LSTM (the argument meant to set 256 units was ignored by the code at the time; this was found in round 4,
the code was changed to make the size explicit, and approaches 12-13 are described as 128-unit models);
`manifold_*_lstm` also used 128 units. `h128_drop04_aug` is the matching LSTM without a gate.

Round 4 rule: best mean validation r; when two configurations were within 0.01, the milder one was kept (stronger
augmentation 0.5 scored 0.005 above 0.3 with twice the seed spread, so 0.3 was kept). Round 5: one change at a time
to approach 5; electrode selection was the only large gain; k chosen per patient (the per-patient optimum differed by
up to 0.23 between k = 8 and k = 32 for patient 2). Choosing the best of seven k per patient flatters the validation
score (0.68 on validation, 0.55 on test).

### 5.2 Robustness: one electrode lost (post hoc, nothing fed back)

| Approach | Subject | Electrodes used | Lost electrode | Intact | Lost |
|---|---|---|---|---|---|
| lstm | 1 | 62 | 0 | 0.420 | 0.298 |
| lstm | 2 | 48 | 23 | 0.439 | 0.331 |
| lstm | 3 | 64 | 53 | 0.577 | 0.488 |
| lstm | mean | | | 0.479 | 0.372 |
| lstm_selected | 1 | 32 | 0 | 0.511 | 0.345 |
| lstm_selected | 2 | 8 | 23 | 0.473 | 0.333 |
| lstm_selected | 3 | 32 | 53 | 0.638 | 0.561 |
| lstm_selected | mean | | | 0.541 | 0.413 |

Both LSTMs lose 0.11-0.13 when their most informative electrode is silenced for the whole test segment. Training
with electrode dropout would buy robustness at a price in validation accuracy (0.62 to 0.59-0.60), so it was not
used for the headline; for an implant that trade-off should be made deliberately.

Electrode-loss curve (`scripts/robustness_curve.py`; seed-0 models of approach 5 (`lstm`) and 5b
(`lstm_selected`), retrained for this study, and 5b trained with electrode dropout (`lstm_selected_dropout`: each
electrode silenced with probability 0.1 in every training window); mean of the three patients): electrodes
silenced best first (by the training-data ranking of approach 5b), or at random from all of the patient's
electrodes (20 draws).

| Decoder | Intact | 1 lost (best first) | 2 | 4 | 8 | 4 lost at random | 16 lost at random |
|---|---|---|---|---|---|---|---|
| lstm | 0.483 | 0.378 | 0.351 | 0.181 | 0.095 | 0.452 | 0.398 |
| lstm_selected | 0.544 | 0.402 | 0.355 | 0.168 | 0.058 | 0.509 | 0.445 |
| lstm_selected_dropout | 0.544 | 0.449 | 0.404 | 0.211 | 0.101 | 0.517 | 0.469 |

Decoding rests on a few electrodes: losing the best one costs 5b 0.14 and approach 5 0.10, and with the best four
gone every decoder falls to about 0.2. Random failures cost far less (16 random electrodes: 0.07-0.10), because most
electrodes carry little information about the fingers. Electrode dropout, rejected on validation (above), keeps the
intact test score (0.544) and degrades more slowly (0.449 with the best electrode lost, against 0.402). It stays out
of the headline, because every choice is made on validation (section 3.2), but for an implant it is the better
default. The intact scores differ slightly from the table above because the models were retrained on another
machine (section 11).

### 5.3 Calibration data (post hoc, `scripts/calibration.py`)
Test r against minutes of the patient's own training data: the last D minutes before the test segment (so the gap
to the test segment stays the same), split 80/20 for every choice and refitted on all D minutes. `lstm_others`: an
LSTM whose recurrent core was pretrained with labels on the other two patients' whole training segments (as
approach 11b, which pretrains on all three), then fine-tuned on the D minutes with a fresh input layer: a new
patient joining a pretrained model. Mean of three patients; ± SD over seeds (the Wiener filter is deterministic).

| Approach | 1 min | 2 min | 4 min | 6.5 min (all) |
|---|---|---|---|---|
| lstm | 0.290 ± 0.009 | 0.309 ± 0.030 | 0.412 ± 0.013 | 0.493 ± 0.018 |
| lstm_others | 0.216 ± 0.023 | 0.281 ± 0.019 | 0.371 ± 0.010 | 0.444 ± 0.009 |
| lstm_selected | 0.345 ± 0.016 | 0.409 ± 0.017 | 0.504 ± 0.024 | 0.546 ± 0.014 |
| wiener | 0.276 | 0.311 | 0.371 | 0.415 |

* **More data still helps at 6.5 minutes.** No decoder levels off: the last 2.5 minutes add 0.04 to 5b and 0.08 to
  approach 5.
* **Choosing electrodes is worth more than minutes of data.** 5b with 4 minutes (0.504) beats approach 5 with all
  6.5 (0.493); with 1 minute (0.345) it comes close to the live 2008 winner with all of it (0.363).
* **Pretraining on the other two patients does not help.** `lstm_others` is below the LSTM trained from scratch at
  every amount (0.216 against 0.290 with 1 minute, 0.444 against 0.493 with all), just as approach 11b gains
  nothing over approach 5 (section 4). Two patients with different grids are too few to learn a shared code; a
  library with many more patients (Miller, 2019) would test this properly.
* The 6.5-minute column retrains the benchmark's models on another machine: the Wiener filter reproduces section 4
  exactly, the networks differ slightly (section 11).

---

## 6. Why the Kalman filter is not better for live decoding here

Short answer: every decoder in this project is live, not only the Kalman filter. The Kalman filter is the optimal
*live* estimator only when its two linear-Gaussian models are right: finger state evolves linearly with Gaussian
noise, and the neural features are a linear function of the finger state plus Gaussian noise. For ECoG band power
and finger flexion both are poor descriptions:

* **Observation model.** High-gamma power rises when a finger moves, regardless of whether it is flexing or
  extending, and decays as it stops; it is not a linear function of finger *position*. The Kalman filter has to
  explain the features from the state, so a wrong observation model limits it however good the filtering is.
* **Dynamics.** Fingers rest most of the time and flex in short bursts on cue; a linear dynamic model with constant
  noise smooths and delays those bursts.
* **Where Kalman filters do excel:** intracortical spike decoding of cursor velocity, where firing rates are close to
  linear in intended velocity and the user closes the loop; the ReFIT Kalman filter (Gilja et al., 2012, Nature
  Neuroscience) recalibrates on the user's intended movements during closed-loop use, which this offline, open-loop
  dataset cannot provide.
* **The discriminative Kalman filter** (Burkhart et al., 2020, Neural Computation) replaces the generative
  observation model with a regression from features to state, which suits nonlinear observation models.

Validation study (`scripts/kalman_study.py`; test labels never loaded):

| Variant | S1 | S2 | S3 | Mean |
|---|---|---|---|---|
| wiener | 0.423 | 0.281 | 0.567 | 0.424 |
| kalman_pca64 | 0.332 | 0.223 | 0.492 | 0.349 |
| kalman_pca64_lags3 | 0.322 | 0.232 | 0.485 | 0.346 |
| kalman_pca64_lags5 | 0.314 | 0.224 | 0.467 | 0.335 |
| dkf_on_wiener | 0.382 | 0.155 | 0.446 | 0.328 |
| kalman_pca32 | 0.298 | 0.243 | 0.412 | 0.318 |
| kalman_pca32_lags3 | 0.274 | 0.238 | 0.416 | 0.309 |
| kalman_pca32_lags5 | 0.222 | 0.211 | 0.421 | 0.285 |
| kalman_pca16 | 0.216 | 0.179 | 0.393 | 0.263 |
| kalman_pca16_lags3 | 0.152 | 0.146 | 0.310 | 0.203 |
| kalman_pca16_lags5 | 0.163 | 0.124 | 0.261 | 0.182 |
| kalman_pca8 | 0.091 | 0.152 | 0.198 | 0.147 |
| kalman_pca8_lags3 | 0.076 | 0.150 | 0.198 | 0.141 |
| kalman_pca8_lags5 | 0.062 | 0.134 | 0.182 | 0.126 |

Findings: more principal components help the classical filter but it stays below the Wiener filter; stacking recent
feature vectors into the observation (lagged Kalman) does not help; the discriminative Kalman filter on top of the
Wiener filter scores *below* the Wiener filter alone. With an informative regression the DKF should help; here the
regression explains little of the finger variance (validation r 0.28-0.57), so the filter leans on its
dynamics, which smooth and delay the bursts. None of the variants was adopted, so the test set was not opened for
them. The Kalman filter's real strength, principled temporal smoothing, is also available to the other decoders
(LSTM memory; the validation-chosen output smoother), and validation chose little or no smoothing for them.

---

## 7. Offline versus live: what the published numbers measure

### 7.1 What the literature and code show
* **The competition itself was offline.** Participants received the whole test recording and returned predictions
  for all of it (data-set description). Nothing required causality.
* **Liang and Bougrain (winner).** Amplitude modulation is defined over a window that *starts* at the predicted
  moment, x(t_n) = sum of v^2(t_n + t) for t in [0, 40 ms] (their Eq. 1, so up to 40 ms of future), the memory uses
  the present and previous 1 s of AM values, the regression has no bias term and is solved with the pseudo-inverse.
  Their FIR filters' direction and length are not stated (Front. Neurosci. 2012, 6:91; arXiv:0907.5500).
* **Flamary and Rakotomamonjy (second).** Their paper states that the positive shifts (+t_s in the state features,
  t + tau in the regression features) make the method non-causal and that they were kept because they
  considerably improve performance; the AR-coefficient spline interpolation and the Savitzky-Golay filter also use
  later samples; training states were segmented by hand; parameter values (t_s, tau, K, M, lambda) are not reported
  (Front. Neurosci. 2012, 6:29; arXiv:1106.3395). Their reported submission (Table 3C): 0.48 / 0.24 / 0.56 with the
  ring finger; 0.49 / 0.23 / 0.55 without it, which averages to the competition score of 0.42.
* **FingerFlex (Lomtev et al., 2023; best published, 0.67).** Official code read in full
  (<https://github.com/Irautak/FingerFlex>, notebooks `prepare_data.ipynb` and `Lightning_BCI-autoencoder.ipynb`):
  the test recording is passed as the *validation* set, the checkpoint with the best correlation on it is kept
  (`ModelCheckpoint(monitor="corr_mean_val")`), predictions are smoothed with a centred Gaussian (sigma 6 samples at
  100 Hz = 60 ms) before scoring, the encoder-decoder convolves symmetrically over 2.56 s windows, filters are
  zero-phase (MNE), each recording is normalised with its own statistics, and the notch is at 50 Hz harmonics
  although the data were recorded in the USA (60 Hz). Their 0.67 is therefore an offline number with model
  selection on the test set.
* **Independent reproductions** (<https://github.com/longkey-szpt/ecog-diagnostic-benchmark>): the winner
  reproduced at 0.460 (S1 0.447, S2 0.323, S3 0.611; zero-phase linear-phase FIR, plus a bad-electrode screen not in
  the paper); the second place at only 0.226 (S1 0.117, S2 0.187, S3 0.375); FingerFlex at 0.642 (3 seeds). Their
  FingerFlex predictions are shipped in that repository; they were not used here.
* **Other repositories** found: <https://github.com/XiaoYvQiKa/ECoG_Finger_Decoding_Benchmark> (could not be cloned:
  authentication required), <https://github.com/xhatME/FingerFlexion> (a course project, FingerFlex-inspired, scored on
  a different hidden set), <https://github.com/anandawolz/ECoG-finger-flexion> (analysis). None contains code by the
  competition winners; no official code for Liang and Bougrain or Flamary and Rakotomamonjy was found.

### 7.2 Exact offline replicas (`src/ecogdecode/offline.py`)
Both replicas follow their papers as written, including the non-causal steps; every point a paper leaves open is
listed in the docstrings with the choice made.

* **Liang and Bougrain, offline:** windowed linear-phase FIR band-passes (3301 taps for 1-60 Hz, 1001 otherwise;
  equiripple design, as in the paper, diverges numerically at these lengths) applied with the group delay removed
  (zero-phase); AM over [t_n, t_n + 40 ms); 25 AM values of memory; no bias term; pseudo-inverse; forward selection
  per finger on a 3/5-2/5 split of the 400 s training data, at most 10 pairs, stop when validation r stops
  increasing; refit on all training data; 25 Hz predictions linearly interpolated to 1 kHz. No re-referencing or
  artefact handling beyond what the organisers applied. Variants: with a bias term; made causal (same FIR filters
  forward-only, AM over the past 40 ms, hold).
* **Flamary and Rakotomamonjy, offline:** ECoG delayed by 37 ms, 250 Hz; first two coefficients of AR(10) models on
  non-overlapping 300-sample windows, cubic-spline interpolated, at t and t ± t_s; channel ranking from per-finger
  ±1 regressions on the first AR coefficient; six linear classifiers with a row-sparse (l1-l2) penalty, winner
  takes all; per-state ridge models on centred Savitzky-Golay (3rd order, 0.4 s) ECoG at t, t - tau, t + tau, keeping
  the M largest-coefficient features. Tuned on the paper's 75 %/25 % split: flexion models on true states (the
  paper's Table 2 setting), then labels (instants or movement periods), the AR window reading (0.3 s at 1 kHz or
  1.2 s at 250 Hz), t_s, K and lambda_s by the validation correlation of the whole decoder (the paper chose K "to
  maximize validation cross-correlation"). Refit on all training data; linear interpolation to 1 kHz.

| Variant | S1 | S2 | S3 | Mean |
|---|---|---|---|---|
| flamary_offline | 0.407 | 0.275 | 0.315 | 0.333 |
| liang_causal | 0.299 | 0.211 | 0.430 | 0.313 |
| liang_offline | 0.443 | 0.313 | 0.594 | 0.450 |
| liang_offline_intercept | 0.445 | 0.316 | 0.456 | 0.406 |

| Replica | With every electrode (as published) | Without electrode 49 |
|---|---|---|
| liang_offline | 0.594 | 0.594 |
| flamary_offline | 0.315 | 0.523 |

### 7.3 One model under live and offline conditions (`offline_study.py --parts wiener`)
The Wiener filter with ten 40 ms steps of history (fixed here so that only the access to the future changes; the
benchmark's Wiener filter chose 10-25 steps on validation and scored 0.415), ridge penalty chosen on validation in
each condition, scored on test at 1 kHz (hold for the live condition, linear interpolation for the offline ones):

| Condition | S1 | S2 | S3 | Mean |
|---|---|---|---|---|
| live: causal features, past lags | 0.396 | 0.282 | 0.531 | 0.403 |
| zero-phase centred features, past lags | 0.417 | 0.281 | 0.551 | 0.417 |
| zero-phase centred features, past and future lags | 0.467 | 0.307 | 0.572 | 0.449 |

### 7.4 Choosing the stopping epoch on the test set (`--parts selection`)
Approach 5b retrained (same initial weights, seed and data order as its benchmark refit) with the test block
decoded after every epoch, as a checkpoint callback monitoring the test set does. The epoch count the protocol chose
on validation is compared with the best epoch on test. *These scores are computed at the 25 Hz step level, as during
training, so they are higher than the 1 kHz scores of the benchmark; the comparison between the two columns is what
matters.*

| Subject | Seed | Epochs chosen on validation | Test r there | Best test epoch | Test r there |
|---|---|---|---|---|---|
| 1 | 0 | 24 | 0.538 | 8 | 0.603 |
| 1 | 1 | 18 | 0.562 | 10 | 0.594 |
| 1 | 2 | 19 | 0.536 | 11 | 0.568 |
| 2 | 0 | 22 | 0.485 | 23 | 0.514 |
| 2 | 1 | 17 | 0.510 | 12 | 0.540 |
| 2 | 2 | 20 | 0.498 | 15 | 0.518 |
| 3 | 0 | 12 | 0.652 | 34 | 0.715 |
| 3 | 1 | 12 | 0.682 | 25 | 0.741 |
| 3 | 2 | 10 | 0.682 | 29 | 0.730 |
| mean | | | 0.572 | | 0.614 |

### 7.5 Non-causal smoothing of a live decoder's output (`--parts smoothing`)
The seed-0 model of approach 5b: its live output as scored (held between 40 ms steps), the same output linearly
interpolated between steps (which uses the next output, i.e. 40 ms of future), and additionally smoothed with a
centred Gaussian (as FingerFlex does):

| Condition | S1 | S2 | S3 | Mean |
|---|---|---|---|---|
| live (as scored) | 0.511 | 0.473 | 0.638 | 0.541 |
| linear interpolation only (uses the next output) | 0.541 | 0.486 | 0.652 | 0.560 |
| centred Gaussian, sigma 40 ms, interpolated | 0.556 | 0.488 | 0.652 | 0.566 |
| centred Gaussian, sigma 60 ms, interpolated | 0.569 | 0.490 | 0.650 | 0.570 |
| centred Gaussian, sigma 120 ms, interpolated | 0.584 | 0.489 | 0.638 | 0.570 |

### 7.6 What this means
* **The winner replicates offline** (0.450 against the 0.46 awarded; per subject 0.443 / 0.313 / 0.594 against the
  published 0.43 / 0.37 / 0.59 without the ring finger). One detail decided subject 3: the paper's Eq. 2 has no bias
  term; without it the greedy search does not pick the electrode with the test-segment artefact, with a bias term it
  does, and the middle finger falls to r = 0.04.
* **Running the winner's filters forward only does not make it live**: its long linear-phase FIR filters then delay
  each band by 0.5-1.65 s, and the score falls to 0.313. A live system needs filters designed for it; the live
  redesign (approach 1) scores 0.363, still hurt on subject 3 by picking that electrode.
* **The second place does not replicate from its paper**: the faithful attempt here scores 0.333 (0.407 / 0.275 /
  0.315 against the submitted 0.49 / 0.23 / 0.55, computed without the ring finger from the paper's Table 3C) and an independent attempt 0.226. With true states, its flexion
  models reach r = 0.75-0.90 on validation, as in the paper's oracle analysis; the gap lies in the state classifier,
  which depends on hand-segmented labels and unreported parameters, and, on subject 3, in the electrode with the
  test-segment artefact: removing it (post hoc) lifts subject 3 from 0.315 to 0.523 (submitted: 0.55). The live
  redesign with validated choices (approach 2) scores 0.425, above both offline attempts.
* **What each offline liberty is worth for a fixed model** (all measured on the blind test set): zero-phase centred
  features +0.014, and +0.046 with 400 ms of future lags as well, for the Wiener filter; interpolating between steps +0.019 (+0.029
  with a centred 60 ms Gaussian as well), and picking the stopping epoch on the test set +0.042 for approach 5b. Each is
  a few hundredths; together they are of the same order as the gap between the best live decoder here (0.554) and
  the best published offline result (0.67); section 7.7 measures FingerFlex itself.
* None of these analyses changed any live decoder or any reported live score.

### 7.7 FingerFlex, retrained (`offline_study.py --parts fingerflex`, approach 14)
Offline, as published, each seed trained twice from the same initial weights: run A on all training data with the
test recording decoded after every epoch (the published procedure keeps the epoch with the best test
correlation); run B on the first 80 % with the epoch chosen by the same criterion on the last 20 %. Run A trained
for run B's number of epochs is the refit, so its test output there is the offline score without test-set
selection.

| FingerFlex, offline | S1 | S2 | S3 | Mean (seed 0) | Notebook metric | Epochs kept (per subject) |
|---|---|---|---|---|---|---|
| as published: epoch with the best test correlation | 0.676 | 0.549 | 0.656 | 0.627 | 0.626 | S1: 19; S2: 5; S3: 2 |
| epoch chosen on validation | 0.646 | 0.540 | 0.603 | 0.596 | 0.593 | S1: 11; S2: 8; S3: 8 |

Seeds: 0. Competition metric (1 kHz, ring finger excluded); the notebook metric is the authors' own (100 Hz, all five fingers, Gaussian sigma 1 sample).

Retrained with the published procedure, FingerFlex scores 0.627 (seed 0; each replica takes four to six hours, so
one seed was run), below its reported 0.67 and the independent reproduction's 0.642 over three seeds. The authors'
own metric gives the same value (0.626), so the gap is not a matter of scoring. Keeping the epoch chosen on
validation instead of the one with the best test correlation costs 0.031 (0.596): the test-picked epochs (19, 5
and 2) are not the ones validation picks (11, 8 and 8).

### 7.8 Designed offline or designed live
The same comparison from both sides: a method designed offline (the winner) made causal, and the best method
designed live (approach 5b) given every offline liberty (`scripts/offline_lstm.py`: zero-phase centred features, a
bidirectional LSTM over the whole recording, a centred Gaussian smoother chosen on validation, interpolation
between outputs; same electrode counts, three seeds).

| Method | Designed for | Scored offline | Scored live | Cost of running live |
|---|---|---|---|---|
| Liang and Bougrain, 2008 winner | offline | 0.450 (replica) | 0.363 (approach 1) | -0.087 |
| FingerFlex (Lomtev et al., 2023) | offline | 0.596 (epoch chosen on validation; seed 0) | 0.394 ± 0.012 (approach 14) | -0.202 |
| LSTM on selected electrodes | live | 0.617 ± 0.014 (bidirectional, zero-phase) | 0.554 ± 0.010 (approach 5b) | -0.063 |

The two methods designed offline lose 0.09 (the 2008 winner) and 0.20 (FingerFlex) when they must run
causally; the method designed live gains only 0.06 when it is given every offline liberty. FingerFlex looks ahead
in every layer (zero-phase filters, symmetric convolutions over 2.56 s windows, whole-recording inference); made
causal, it keeps its architecture but loses that context, and falls below a plain LSTM (section 4). A decoder that
has to run live is best designed and selected live.

---

## 8. Animations (`make_animation.py`, `make_offline_animation.py`)
* `hand_live_sub3.mp4`: the whole 200 s test segment of patient 3 in real time: high-gamma activity of the 24
  electrodes most related to movement in the training data (standardised with training statistics), the hand driven
  by the glove, the hand driven by approach 5b (seed 0) with the recorded pose as a white outline, and 8 s of scrolling
  traces.
* `live_vs_offline_sub3.mp4`: three hands on the same 200 s - recorded, live (approach 5b, seed 0), offline (the
  winner's method as published) - and their traces. The r values in the labels are for the raw decoder outputs.
* `lstm_live_vs_offline_sub3.mp4`: the same layout with approach 5b live and the same LSTM given every offline
  liberty (section 7.8), both seed 0: the best live and the best offline decoder here, on the same 200 s.
* The GIFs in the README are 16 s excerpts (60-76 s, a window chosen in an earlier round because all five fingers move
  in it, not for decoding quality).
* Display rules: section 3.7. The hand (`handviz.py`) is a kinematic model: three tapered segments per finger
  (MCP, PIP and DIP flex together, up to 80, 100 and 65 degrees), a thumb that swings across the palm, weak
  perspective, one light; it shows the five flexion values, not measured joint angles.
* Frames are independent, so `render.py` draws each video in parallel chunks and joins them without re-encoding.

---

## 9. FingerFlex, re-implemented
From the paper (arXiv:2211.01960 v2) and the authors' notebooks, which have no licence, so no code was copied
(`src/ecogdecode/fingerflex.py`). As published: each recording z-scored with its own statistics and median
re-referenced; MNE-style zero-phase FIR band-pass 40-300 Hz (331 taps, -6 dB at 35 and 337.5 Hz) and notches at the
50 Hz harmonics (6601 taps; the notebooks notch 50 Hz although the recordings are from the USA, kept as
published); 40 Morlet wavelets log-spaced 40-300 Hz (7 cycles, power), kept every 10 ms; robust scaling (10th to
90th percentile) fitted on training data; fingers cubic-interpolated from 25 to 100 Hz and min-max scaled; ECoG
paired with the fingers 200 ms later. Network: the notebooks' `AutoEncoder1D` (660,293 parameters for 64
electrodes; tested), half MSE plus half cosine distance, Adam (lr 8.42e-5, weight decay 1e-6), batch 128, every
2.56 s window at stride 1, 20 epochs; whole-recording inference; centred Gaussian smoothing (sigma 6 samples for
checkpoint selection, 1 sample at test). The first convolution is computed as one matrix product per time step
and a shift-and-add (identical weights and outputs, tested; about 3x faster on a CPU). Deviations: seeds 0-2 (the
notebooks set none); epoch-selection runs stop after 5 epochs without improvement; the network runs in eval mode
at test time (the notebooks' test path does not switch dropout off). Each run took one to six hours on four CPU
threads of a laptop that ran other studies at the same time.

---

## 10. Development history and every look at the test set

1. **Round 1:** first benchmark (common-average reference, early-stopped models). Subject 3 collapsed (r about 0.1)
   because electrode 49 carries a burst of artefact about 130-150 s into the test segment (SD over the test segment
   253 times its training SD; normal throughout training). Median re-referencing and the online masking rule were
   added; the 6 SD threshold was set once.
2. **Round 2:** refit on the whole training segment, seed averaging, a validation-selected ensemble, approach 10.
3. **Round 3:** code review and refactor; the two competition entries implemented live; every approach rerun.
4. **Round 4:** averaging dropped; validation-only study of 22 configurations; test opened once for all 16 approaches.
5. **Round 5:** settings of approach 5 and electrode selection (validation only); test opened once for approach 5b;
   robustness check (post hoc).
6. **Round 6:** Kalman variants (validation only; not adopted, so no test look); offline replicas and
   the offline-vs-live analyses (test opened for published methods and for analyses that feed nothing back). One
   revision followed a test look: the first version of the second-place replica tuned its state classifier by
   balanced state accuracy with a single label definition and scored 0.22 on test (`results/logs/offline1.log`). It
   was then revised to tune the classifier by the whole decoder's validation correlation (as the paper describes
   for K) and to choose two details the paper leaves open (label definition, AR window reading) on validation; the
   revised replica scores 0.33. Its test score is therefore not untouched. The winner's replica was not changed
   after its test result.
7. **Round 7:** display only: full-length videos, no display smoothing, each trace drawn relative to its own
   resting level, a redrawn hand. No model, score or test look changed.

8. **Round 8** (after a critical review of the whole project): new approaches 14 (FingerFlex, causal) and 15 (POSSM-style), each fixed
   and committed before it was scored on the test set (FingerFlex ran as nine processes, one per patient and seed);
   15 failed and was not revised, and a variant 15b was stopped before it produced a score. Also offline FingerFlex
   (both epoch rules, seed 0), offline 5b and post-hoc analyses (bootstrap, lag, cross-finger, masking, calibration
   curve, electrode-loss curve). From this round every look is logged with its git commit in
   `results/test_looks.jsonl`, including runs stopped before they produced a score (memory limits, reordering): one
   FingerFlex process, restarted unchanged; 15b; and four calibration processes. The calibration study was then
   restarted with a different pretrained decoder (an LSTM pretrained on the other patients instead of the
   POSSM-style decoder, which had failed), so no calibration result predates that change. The log records the
   commit at the start of a run, each results file the commit at its end; for long runs the two differ (the seed-0
   FingerFlex runs started at 3dd2a91), and the commits in between change nothing on their path that affects a
   model or a score (compared function by function: formatting, and the test labels kept out of the training
   arrays). Figure, video and verification scripts read the labels only to draw recorded traces or recompute stored
   scores; they log their reads from the end of this round. Once the log itself was committed, each look appended to
   a tracked file, so one look (`make_figures.py` at a32ef0c) is marked dirty although no code had changed; from then
   on the log does not count as an uncommitted change (`provenance.git_state`). No earlier approach or score was
   changed. Code changes in this round that touch every approach were verified not to change any stored score
   (`verify_scores.py`). Before publication the history was rewritten to remove personal and tool metadata (the
   author's e-mail address, a local configuration file, commit-message trailers); code, data and results are
   unchanged. The commits named in this report, in `results/test_looks.jsonl` and in the results files are those of
   the original history; `results/logs/commit_map.txt` maps each to its published commit.

Every choice inside each round was made on validation data, but repeated looks at the same test set across rounds
can make the final live numbers slightly optimistic. The earlier README claim that contrastive pretraining scored
0.53 came from averaging three models; it was withdrawn in round 4.

---

## 11. Known limitations
* Three patients, 6.5 minutes of training data each: differences of a few hundredths are within seed variability.
* Validation scores are optimistic (stopping epoch, smoother and k all chosen on the same 80 s block).
* The 37 ms glove lag is not modelled explicitly in the live pipeline (learned implicitly).
* The second-place replica depends on choices its paper does not report (state labels, AR window, parameters); its
  scores are a faithful attempt, not a reproduction of the submitted 0.42.
* Glove super-sampling to 1 kHz is not documented by the organisers; predictions are held (live) or linearly
  interpolated (offline) between 40 ms steps.
* Pseudo-online, open loop: the user never saw the decoder's output, so closed-loop adaptation is not measured.
* Electrode positions are not part of the data set; electrode maps are by index.
* The bootstrap intervals cover test-sample variability for fixed seed-0 models; they do not cover the choice of
  approach across rounds or the per-patient electrode counts of 5b.
* Two machines: approaches 1-13 were trained on a Linux virtual machine, approaches 14 and 15 and every round-8
  study on a Windows laptop. Retraining with the same seed is not bit-identical across them (floating-point results
  depend on the CPU, the maths library and the thread count). In the 6.5-minute column of section 5.3, which
  retrains the benchmark's models on the laptop, the Wiener filter reproduces section 4 exactly; approach 5 stops at
  the same epoch in all nine patient-seed runs and scores within 0.013; approach 5b, with the same electrodes, stops
  at a different epoch in five of nine and scores within 0.04 (three-patient score 0.546 ± 0.014 against
  0.554 ± 0.010).
* The offline FingerFlex replica was run with one seed (four to six hours per run).

## 12. What a reviewer should check first
1. Causality of live features and models: `tests/test_features.py`, `tests/test_models.py` (future changes must not
   alter past outputs); `extract_features(..., offline=False)`.
2. No test information in choices: `scripts/explore_validation.py` sets `test_flex = None`; `run_network` uses
   `fit_rows`/`val_rows` for every choice and `train_rows` for the refit; `top_electrodes` sees only the rows passed.
3. Scores: `evaluation.score_test` / `pipeline.evaluate` (hold to 1 kHz, four fingers) and `offline.score_1khz`.
   `scripts/verify_scores.py` recomputes every stored seed-0 and replica score from the saved predictions; the run
   in `results/logs/verify_scores.log` reproduces all of them (largest difference 7e-10), apart from flagged
   cases whose saved predictions are coarser than the output that was scored: the causal winner replica (up
   to 0.010), the second-place replica and the round-8 offline runs (up to 0.002).
4. Replica fidelity: compare `offline.py` docstrings with the papers (links in section 13).
5. The claims about FingerFlex's code: the two notebooks in its repository; the re-implementation in
   `fingerflex.py` and its tests (`tests/test_fingerflex.py`).
6. Every test look of round 8: `results/test_looks.jsonl` against the git history (original commits mapped to the
   published ones in `results/logs/commit_map.txt`).

## 13. References
* Ryoo AH, Krishna NH, Mao X, Azabou M, Dyer EL, Perich MG, Lajoie G (2025). Generalizable, real-time neural decoding with hybrid state-space models. NeurIPS. arXiv:2506.05320.
* Azabou M, et al. (2023). A unified, scalable framework for neural population decoding (POYO). NeurIPS.
* Schalk G, et al. (2007). Decoding two-dimensional movement trajectories using electrocorticographic signals in humans. J Neural Eng 4:264-275 (required citation for the data).
* Miller KJ, Schalk G (2008). BCI Competition IV, data set 4 description. <https://bbci.de/competition/iv/desc_4.pdf>
* BCI Competition IV results. <https://www.bbci.de/competition/iv/results/>
* Liang N, Bougrain L (2012). Decoding finger flexion from band-specific ECoG signals in humans. Front Neurosci 6:91. <https://www.frontiersin.org/articles/10.3389/fnins.2012.00091/full>; preprint arXiv:0907.5500.
* Flamary R, Rakotomamonjy A (2012). Decoding finger movements from ECoG signals using switching linear models. Front Neurosci 6:29. <https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2012.00029/full>; preprint arXiv:1106.3395.
* Lomtev V, Kovalev A, Timchenko A (2023). FingerFlex: inferring finger trajectories from ECoG signals. arXiv:2211.01960; code <https://github.com/Irautak/FingerFlex>.
* Independent benchmark with reproductions: <https://github.com/longkey-szpt/ecog-diagnostic-benchmark>.
* Kubánek J, Miller KJ, Ojemann JG, Wolpaw JR, Schalk G (2009). Decoding flexion of individual fingers using electrocorticographic signals in humans. J Neural Eng 6:066001.
* Wu W, Gao Y, Bienenstock E, Donoghue JP, Black MJ (2006). Bayesian population decoding of motor cortical activity using a Kalman filter. Neural Comput 18:80-118.
* Gilja V, et al. (2012). A high-performance neural prosthesis enabled by control algorithm design. Nat Neurosci 15:1752-1757.
* Burkhart MC, Brandman DM, Franco B, Hochberg LR, Harrison MT (2020). The discriminative Kalman filter for Bayesian filtering with nonlinear and nongaussian observation models. Neural Comput 32:969-1017.
* Gu A, Gupta A, Goel K, Ré C (2022). On the parameterization and initialization of diagonal state space models. NeurIPS.
* Gu A, Dao T (2023). Mamba: linear-time sequence modeling with selective state spaces. arXiv:2312.00752.
* Bai S, Kolter JZ, Koltun V (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271.
* Schneider S, Lee JH, Mathis MW (2023). Learnable latent embeddings for joint behavioural and neural analysis. Nature 617:360-368.
* Dekleva BM, Weiss JM, Boninger ML, Collinger JL (2021). Generalizable cursor click decoding using grasp-related neural transients. J Neural Eng 18:0460e9.
* Peng JJ, et al. (2025). Grid cells accurately track movement during path integration-based navigation despite switching reference frames. Nat Neurosci.
* Dvorak D, Radwan B, Sparks FT, Talbot ZN, Fenton AA (2018). Control of recollection by slow gamma dominating mid-frequency gamma in hippocampus CA1. PLoS Biol 16:e2003354.
* Miller KJ (2019). A library of human electrocorticographic data and analyses. Nat Hum Behav 3:1225-1235.
