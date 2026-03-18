# -*- coding: utf-8 -*-
"""
export_onnx.py — One-time export of fine-tuned BERT to ONNX format
==================================================================
Run once:  python export_onnx.py
Output  :  G:\Data\job_posting\processed\model\bert_soc_406.onnx

Then use with 10_full_inference.py which loads this ONNX model via
onnxruntime CUDAExecutionProvider for ~3-5x faster inference.
"""

import sys
import time
from pathlib import Path

MODEL_DIR = Path(r"G:\Data\job_posting\processed\model")
ONNX_PATH = MODEL_DIR / "bert_soc_406.onnx"

def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("  ONNX Export: BertForSequenceClassification → ONNX")
    print("=" * 60)

    # ── Load PyTorch model ────────────────────────────────────────
    import torch
    from transformers import BertForSequenceClassification

    print(f"\n  Loading model from {MODEL_DIR} ...")
    model = BertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    num_labels = model.config.num_labels
    print(f"  Model loaded: {num_labels} classes, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # ── Create dummy inputs (batch=2 so ONNX tracer sees batch as dynamic) ─
    # Using batch=1 causes the tracer to sometimes hardcode dim=1 in internal
    # nodes (e.g. LayerNorm), breaking inference at other batch sizes.
    dummy_ids   = torch.zeros(2, 128, dtype=torch.long)
    dummy_mask  = torch.ones(2, 128, dtype=torch.long)
    dummy_types = torch.zeros(2, 128, dtype=torch.long)

    # ── Export ────────────────────────────────────────────────────────
    print(f"\n  Exporting to {ONNX_PATH} ...")
    t0 = time.time()

    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask, dummy_types),
        str(ONNX_PATH),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch", 1: "seq_len"},
            "attention_mask":  {0: "batch", 1: "seq_len"},
            "token_type_ids":  {0: "batch", 1: "seq_len"},
            "logits":          {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    elapsed = time.time() - t0
    mb = ONNX_PATH.stat().st_size / 1e6
    print(f"  Export done in {elapsed:.0f}s  ({mb:.1f} MB)")

    # ── Validate ──────────────────────────────────────────────────────
    import onnx
    print("\n  Validating ONNX model ...")
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("  ONNX model validated ✓")

    # ── Quick test: compare PyTorch vs ONNX on a dummy input ──────────
    import onnxruntime as ort
    import numpy as np

    print("\n  Smoke test: PyTorch vs ONNX on dummy input ...")
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # PyTorch forward
    with torch.no_grad():
        pt_logits = model(dummy_ids, dummy_mask, dummy_types).logits.numpy()

    # ONNX forward
    ort_inputs = {
        "input_ids":      dummy_ids.numpy(),
        "attention_mask":  dummy_mask.numpy(),
        "token_type_ids":  dummy_types.numpy(),
    }
    (ort_logits,) = session.run(None, ort_inputs)

    max_diff = np.abs(pt_logits - ort_logits).max()
    argmax_match = (pt_logits.argmax(axis=-1) == ort_logits.argmax(axis=-1)).all()
    print(f"  Max logit difference : {max_diff:.2e}")
    print(f"  Argmax match         : {argmax_match}")

    if max_diff < 1e-4 and argmax_match:
        print("\n  ✓ Export successful — ONNX model is numerically equivalent")
    else:
        print("\n  ⚠ WARNING: Significant numerical difference detected")

    print(f"\n  Output: {ONNX_PATH}")
    print("  Use with: python 10_full_inference.py --phase b --batch_full 512")


if __name__ == "__main__":
    main()
