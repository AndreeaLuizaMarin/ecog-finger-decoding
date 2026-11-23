## BENCHMARK

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
| 4 | Kalman filter | 0.368 | 0.419 | 0.199 | 0.502 | 0.373 | 0.000 |
| 1 | Liang & Bougrain, 2008 winner (live re-implementation) | 0.485 | 0.464 | 0.278 | 0.347 | 0.363 | 0.000 |
| 8 | Neural ODE (continuous-time dynamics) | 0.341 | 0.182 | 0.120 | 0.489 | 0.264 | 0.014 |

## VALIDATION

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

## ROBUSTNESS

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

## KALMAN

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

## REPLICAS

| Variant | S1 | S2 | S3 | Mean |
|---|---|---|---|---|
| flamary_offline | 0.407 | 0.275 | 0.315 | 0.333 |
| liang_causal | 0.299 | 0.211 | 0.430 | 0.313 |
| liang_offline | 0.443 | 0.313 | 0.594 | 0.450 |
| liang_offline_intercept | 0.445 | 0.316 | 0.456 | 0.406 |

## DIAGNOSTICS

| Replica | With every electrode (as published) | Without electrode 49 |
|---|---|---|
| liang_offline | 0.594 | 0.594 |
| flamary_offline | 0.315 | 0.523 |

## WIENER

| Condition | S1 | S2 | S3 | Mean |
|---|---|---|---|---|
| live: causal features, past lags | 0.396 | 0.282 | 0.531 | 0.403 |
| zero-phase centred features, past lags | 0.417 | 0.281 | 0.551 | 0.417 |
| zero-phase centred features, past and future lags | 0.467 | 0.307 | 0.572 | 0.449 |

## SELECTION

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

## SMOOTHING

| Condition | S1 | S2 | S3 | Mean |
|---|---|---|---|---|
| live (as scored) | 0.511 | 0.473 | 0.638 | 0.541 |
| linear interpolation only (uses the next output) | 0.541 | 0.486 | 0.652 | 0.560 |
| centred Gaussian, sigma 40 ms, interpolated | 0.556 | 0.488 | 0.652 | 0.566 |
| centred Gaussian, sigma 60 ms, interpolated | 0.569 | 0.490 | 0.650 | 0.570 |
| centred Gaussian, sigma 120 ms, interpolated | 0.584 | 0.489 | 0.638 | 0.570 |

(generated from the JSON files in results/; numbers rounded to 3 decimals)
