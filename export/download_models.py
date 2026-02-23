"""
Download pre-trained models for Android TS-VAD app:
1. WeSpeaker ResNet34-LM (speaker embedding, ONNX)
2. Export Personal VAD to ONNX
"""

import os
import subprocess
import sys


def download_wespeaker_onnx(output_dir: str = "models"):
    """Download WeSpeaker ResNet34-LM ONNX model from HuggingFace."""
    os.makedirs(output_dir, exist_ok=True)

    model_id = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
    target_file = os.path.join(output_dir, "speaker_encoder.onnx")

    if os.path.exists(target_file):
        print(f"Speaker encoder already exists: {target_file}")
        return target_file

    print("Downloading WeSpeaker ResNet34-LM ONNX model...")
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=model_id,
            filename="speaker_encoder.onnx",
            local_dir=output_dir,
        )
        print(f"Downloaded speaker encoder to: {path}")
        return path
    except ImportError:
        print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        print(f"Or manually download from: https://huggingface.co/{model_id}")
        sys.exit(1)


def export_personal_vad(output_dir: str = "models"):
    """Export Personal VAD model to ONNX."""
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "personal_vad.onnx")
    if os.path.exists(output_path):
        print(f"Personal VAD model already exists: {output_path}")
        return output_path

    from personal_vad_model import export_onnx

    export_onnx(output_path)
    return output_path


def verify_models(model_dir: str = "models"):
    """Verify ONNX models can be loaded."""
    import onnxruntime as ort
    import numpy as np

    # Verify speaker encoder
    speaker_path = os.path.join(model_dir, "speaker_encoder.onnx")
    if os.path.exists(speaker_path):
        sess = ort.InferenceSession(speaker_path)
        print(f"\nSpeaker Encoder:")
        for inp in sess.get_inputs():
            print(f"  Input: {inp.name}, shape={inp.shape}, dtype={inp.type}")
        for out in sess.get_outputs():
            print(f"  Output: {out.name}, shape={out.shape}, dtype={out.type}")
        print(f"  File size: {os.path.getsize(speaker_path) / 1024 / 1024:.1f} MB")

    # Verify personal VAD
    vad_path = os.path.join(model_dir, "personal_vad.onnx")
    if os.path.exists(vad_path):
        sess = ort.InferenceSession(vad_path)
        print(f"\nPersonal VAD:")
        for inp in sess.get_inputs():
            print(f"  Input: {inp.name}, shape={inp.shape}, dtype={inp.type}")
        for out in sess.get_outputs():
            print(f"  Output: {out.name}, shape={out.shape}, dtype={out.type}")
        print(f"  File size: {os.path.getsize(vad_path) / 1024 / 1024:.3f} MB")

        # Test inference
        x = np.random.randn(1, 10, 296).astype(np.float32)
        h0 = np.zeros((2, 1, 64), dtype=np.float32)
        c0 = np.zeros((2, 1, 64), dtype=np.float32)
        logits, hn, cn = sess.run(None, {"input": x, "h0": h0, "c0": c0})
        print(f"  Test output shape: {logits.shape}")  # (1, 10, 3)
        print(f"  Test passed!")


if __name__ == "__main__":
    export_personal_vad()
    download_wespeaker_onnx()
    verify_models()
