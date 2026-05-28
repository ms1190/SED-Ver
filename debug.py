import torch

from data_util.genderage_classes import as_strong_train_classes, as_strong_eval_classes
from helpers.encode import ManyHotEncoder
from models.prediction_wrapper import PredictionsWrapper
from models.m2d.M2D_wrapper import M2DWrapper

encoder = ManyHotEncoder(as_strong_train_classes)

print("Checking labels and label size")
print(len(encoder.labels))
print(encoder.labels)

base_model = M2DWrapper()

model = PredictionsWrapper(
    base_model,
    checkpoint=None,
    seq_model_type=None,
    n_classes_strong=len(encoder.labels),
    embed_dim=3840
)

x = torch.randn(2, 1, 80, 64000)
with torch.no_grad():
    out, weak = model(x)

print("Checking model output size")
print(out.shape)