import torch

ckpt_path = "checkpoints/epoch=59.ckpt"

ckpt = torch.load(ckpt_path, map_location="cpu")
state_dict = ckpt["state_dict"]

new_state = {}

for k, v in state_dict.items():
    # remove Lightning wrapper prefix
    if k.startswith("model."):
        k = k[len("model."):]
    new_state[k] = v

torch.save(new_state, "resources/my_model.pt")

print("Saved cleaned checkpoint to resources/my_model.pt")