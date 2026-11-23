"""Decoders. Classical and contest models are closed-form; the rest are PyTorch networks."""
from .contest import LiangBougrainDecoder, SwitchingLinearDecoder, finger_states, select_switching
from .conv import TCNDecoder
from .gated import LeakyIntegrator, StateGatedDecoder, StateGatedStep, TransientGatedDecoder
from .kalman import DiscriminativeKalmanDecoder, KalmanDecoder, LaggedKalmanDecoder
from .manifold import ContrastiveManifold, ManifoldGatedDecoder, PCAManifold, build_manifold_gated
from .linear import LaggedRidge, lag_matrix, select_ridge
from .recurrent import LSTMDecoder, LSTMStep, ODEDecoder
from .ssm import MambaDecoder, S4DDecoder, S4DLayer, S4DStep, selective_scan
from .training import predict_sequence, train_sequence_model

__all__ = ["LiangBougrainDecoder", "SwitchingLinearDecoder", "finger_states", "select_switching",
           "LeakyIntegrator", "StateGatedDecoder", "StateGatedStep", "TransientGatedDecoder", "KalmanDecoder",
           "LaggedKalmanDecoder", "DiscriminativeKalmanDecoder",
           "ContrastiveManifold", "ManifoldGatedDecoder", "PCAManifold", "build_manifold_gated",
           "LaggedRidge", "lag_matrix", "select_ridge", "LSTMDecoder", "LSTMStep", "ODEDecoder", "MambaDecoder",
           "S4DDecoder", "S4DLayer", "S4DStep", "selective_scan", "TCNDecoder", "predict_sequence", "train_sequence_model"]
