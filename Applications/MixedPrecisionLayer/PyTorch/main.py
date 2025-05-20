import torch
import torch.nn as nn
import numpy as np
import os

class MixedPrecisionLayerModel(nn.Module):
    def __init__(self):
        super(MixedPrecisionLayerModel, self).__init__()
        self.fc1 = nn.Linear(2, 3, bias=False).half()
        self.fc2 = nn.Linear(3, 2, bias=False)

    def forward(self, x):
        x = x.half()
        x = self.fc1(x)
        x = x.float()
        x = self.fc2(x)
        return x

def save_for_nntrainer(model, file_path):
    params = model.state_dict()
    with open(file_path, "wb") as f:
        # Save fc1 weights (FP16) - transposed
        np.array(params['fc1.weight'].T.float().numpy(), dtype='float16').tofile(f)
        
        # Save fc2 weights (FP32) - transposed
        np.array(params['fc2.weight'].T.numpy(), dtype='float32').tofile(f)

model = MixedPrecisionLayerModel()
PT_WEIGHTS_PATH = "model_weights.pth"
NNTR_WEIGHTS_PATH = "model_weights_nntr.bin"

if os.path.exists(PT_WEIGHTS_PATH):
    state_dict = torch.load(PT_WEIGHTS_PATH, weights_only=True)
    model.load_state_dict(state_dict)
else:
    torch.save(model.state_dict(), PT_WEIGHTS_PATH)

save_for_nntrainer(model, NNTR_WEIGHTS_PATH)
input_data = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
output = model(input_data)
print(output)
