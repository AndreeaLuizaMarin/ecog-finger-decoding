# Console logs of the long runs

| Log | Run |
|---|---|
| explore4.log | validation study, round 4 (pretraining, LSTM sizes, TCN) -> validation_study_2.json |
| explore5.log | validation study, round 4 (256-unit LSTM with dropout, augmentation, schedule, window) -> validation_study_3.json |
| explore6.log | validation study, round 4 (manifold-gated 128-unit, augmentation strength) -> validation_study_4.json |
| r5a.log ... r5h.log | validation study, round 5 (training settings, electrode selection, time warp) -> validation_study_5*.json |
| final_wiener.log, final_A.log, final_B.log | the one-time blind test of the sixteen round-4 approaches |
| final_5b.log | the one-time blind test of approach 5b |
| robust.log | one electrode lost (post hoc) |
| kalman_study.log | Kalman variants, validation only |
| offline1.log | non-causal smoothing; first version of the replicas (second-place replica later revised) |
| offline2.log | second-place replica (revised), Wiener live vs offline, stopping epoch on test |
| offline3.log | subject-3 diagnostic without electrode 49 |
| r8_fingerflex_causal_s{S}_seed{N}.log | round 8: approach 14 (FingerFlex, causal), one process per patient and seed; a repeated `start` line marks a process stopped before it produced a score and started again |
| r8_fingerflex_offline_s{S}_seed0.log | round 8: FingerFlex as published and with the epoch chosen on validation (seed 0) |
| r8_possm.log | round 8: approach 15 (POSSM-style), blind test |
| r8_possm_pretrained.log | round 8: approach 15b, stopped before it wrote any output (empty) |
| r8_possm_validation_scale1.log, r8_possm_validation_latents8.log | round 8: approach 15 variants, validation only |
| r8_offline_lstm.log | round 8: approach 5b given every offline liberty |
| r8_calibration_s{S}_seed{N}.log, r8_calibration_summary.log | round 8: calibration curve |
| r8_robustness_curve.log | round 8: electrode-loss curve |
| r8_posthoc.log | round 8: bootstrap, lag, which finger, masking |
| verify_scores.log | every stored score recomputed from the saved predictions |
| commit_map.txt | original commit hashes (named in the provenance records) mapped to the published ones |

Warnings were stripped, and the round-8 logs (written on Windows) converted to UTF-8. The first version of the second-place replica (offline1.log: 0.165 / 0.168 / 0.325) tuned
its state classifier by balanced accuracy only and used one label definition; it was revised to tune by the whole
decoder's validation correlation, as the paper describes for K, before its final result (offline2.log) was taken.
