"""
pipeline/model.py
─────────────────
Symmetric Autoencoder for anomaly detection.
Learns to reconstruct 'normal' log feature vectors.
High reconstruction error = anomaly.
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64, 32), encoding_dim=16):
        super().__init__()

        # Encoder
        enc_layers, prev = [], input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        enc_layers += [nn.Linear(prev, encoding_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror)
        dec_layers, prev = [], encoding_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))
