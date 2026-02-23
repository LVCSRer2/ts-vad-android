"""
Personal VAD Model Definition & ONNX Export

Based on: "Personal VAD: Speaker-Conditioned Voice Activity Detection"
(Ding et al., Odyssey 2020)

Architecture: 2-layer LSTM + FC1 (hidden) + FC2 (output) → 3-class
Parameters: ~130K
Input: 296-dim (40 log-fbank + 256 speaker embedding) per frame
"""

import torch
import torch.nn as nn


class PersonalVAD(nn.Module):
    def __init__(
        self,
        fbank_dim: int = 40,
        embed_dim: int = 256,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
    ):
        super().__init__()
        self.fbank_dim = fbank_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        input_dim = fbank_dim + embed_dim  # 296

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        # Matches pretrained: fc1 (hidden) + fc2 (output)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, h0, c0):
        """
        Args:
            x: (batch, seq_len, 296) - fbank + speaker embedding
            h0: (num_layers, batch, hidden_dim) - LSTM hidden state
            c0: (num_layers, batch, hidden_dim) - LSTM cell state
        Returns:
            logits: (batch, seq_len, 3)
            hn: (num_layers, batch, hidden_dim)
            cn: (num_layers, batch, hidden_dim)
        """
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc1(out)  # linear activation (identity) for "linear" variant
        logits = self.fc2(out)
        return logits, hn, cn


def export_onnx(
    output_path: str = "personal_vad.onnx",
    weights_path: str = None,
):
    model = PersonalVAD()

    if weights_path:
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        print(f"Loaded pretrained weights from {weights_path}")

    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Fixed shape: batch=1, seq_len=25 (250ms chunk)
    batch = 1
    seq_len = 25
    x = torch.randn(batch, seq_len, 296)
    h0 = torch.zeros(model.num_layers, batch, model.hidden_dim)
    c0 = torch.zeros(model.num_layers, batch, model.hidden_dim)

    torch.onnx.export(
        model,
        (x, h0, c0),
        output_path,
        input_names=["input", "h0", "c0"],
        output_names=["logits", "hn", "cn"],
        dynamo=False,
        opset_version=17,
    )
    print(f"Exported ONNX model to {output_path}")
    print(f"  Input shape: (1, {seq_len}, 296)")
    print(f"  Output shape: (1, {seq_len}, 3)")


if __name__ == "__main__":
    import sys

    weights = sys.argv[1] if len(sys.argv) > 1 else None
    export_onnx(weights_path=weights)
