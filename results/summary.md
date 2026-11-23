| # | Approach | Validation r | S1 | S2 | S3 | **Test r** (one model; mean ± SD over seeds) | Live step |
|---|---|---|---|---|---|---|---|
| | *BCI Competition IV entries, re-implemented to run live* | | | | | | |
| 1 | Liang & Bougrain, 2008 winner (live re-implementation) | 0.49 | 0.46 | 0.28 | 0.35 | **0.36** | – |
| 2 | Flamary & Rakotomamonjy, 2nd place (live re-implementation) | 0.45 | 0.48 | 0.26 | 0.54 | **0.43** | – |
| | *Classical decoders* | | | | | | |
| 3 | Wiener filter (lagged ridge regression) | 0.42 | 0.42 | 0.30 | 0.53 | **0.42** | – |
| 4 | Kalman filter | 0.37 | 0.42 | 0.20 | 0.50 | **0.37** | – |
| | *Deep sequence models* | | | | | | |
| 5 | LSTM (256 units, dropout 0.4, augmentation; tuned on validation) | 0.57 | 0.43 | 0.40 | 0.65 | **0.49** ± 0.02 | 0.90 ms |
| 5b | LSTM on the most movement-related electrodes (number chosen per patient) | 0.68 | 0.52 | 0.49 | 0.66 | **0.55** ± 0.01 | – |
| 6 | S4D state-space model | 0.40 | 0.34 | 0.30 | 0.57 | **0.41** ± 0.01 | 0.87 ms |
| 7 | Mamba (selective state space) | 0.44 | 0.47 | 0.28 | 0.61 | **0.46** ± 0.01 | – |
| 8 | Neural ODE (continuous-time dynamics) | 0.34 | 0.18 | 0.12 | 0.49 | **0.26** ± 0.01 | – |
| 8b | Causal temporal convolutional network (TCN) | 0.47 | 0.43 | 0.33 | 0.63 | **0.46** ± 0.02 | – |
| | *New in this project* | | | | | | |
| 9 | LFP-gated decoder (gate: LMP, mu, beta) | 0.44 | 0.44 | 0.28 | 0.58 | **0.43** ± 0.01 | 0.72 ms |
| 9b | LFP-gated decoder, ablation: gate sees all bands | 0.42 | 0.45 | 0.27 | 0.58 | **0.44** ± 0.01 | – |
| 10 | Transient-gated decoder (Dekleva-inspired) | 0.48 | 0.50 | 0.30 | 0.56 | **0.45** ± 0.01 | – |
| 11 | Contrastive pretraining + fine-tuned LSTM | 0.46 | 0.44 | 0.35 | 0.61 | **0.47** ± 0.01 | – |
| 11b | Multi-patient supervised pretraining + fine-tuned LSTM | 0.49 | 0.43 | 0.38 | 0.62 | **0.48** ± 0.01 | – |
| 12 | Manifold-gated decoder, PCA manifold | 0.56 | 0.45 | 0.37 | 0.62 | **0.48** ± 0.02 | – |
| 13 | Manifold-gated decoder, CEBRA-style manifold | 0.53 | 0.45 | 0.38 | 0.61 | **0.48** ± 0.01 | – |
