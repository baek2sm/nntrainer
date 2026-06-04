from pathlib import Path

import torch
from safetensors import safe_open


HF_PATH = Path(r"C:\Users\sb92.hong\Documents\projects\nntrainer\.claude\worktrees\nntr-siglip2-verify\verify_artifacts\hf_model\model.safetensors")
OUT_PATH = Path("C:/wt/vl/nntr_model/lfm2_vl_450m_connector.bin")

KEYS = [
    "model.multi_modal_projector.layer_norm.weight",
    "model.multi_modal_projector.layer_norm.bias",
    "model.multi_modal_projector.linear_1.weight",
    "model.multi_modal_projector.linear_1.bias",
    "model.multi_modal_projector.linear_2.weight",
    "model.multi_modal_projector.linear_2.bias",
]


def main():
    total = 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with safe_open(str(HF_PATH), framework="pt", device="cpu") as safefile:
        with OUT_PATH.open("wb") as handle:
            for key in KEYS:
                tensor = safefile.get_tensor(key).to(torch.float32).contiguous()
                data = tensor.numpy().tobytes()
                handle.write(data)
                total += len(data)
    print(f"total bytes written: {total}")


if __name__ == "__main__":
    main()
