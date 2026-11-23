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
| verify_scores.log | every stored score recomputed from the saved predictions |

Warnings were stripped. The first version of the second-place replica (offline1.log: 0.165 / 0.168 / 0.325) tuned
its state classifier by balanced accuracy only and used one label definition; it was revised to tune by the whole
decoder's validation correlation, as the paper describes for K, before its final result (offline2.log) was taken.
