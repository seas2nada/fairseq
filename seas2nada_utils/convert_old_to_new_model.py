import torch
import sys

old_model = sys.argv[1]
new_model = sys.argv[2]
out_model = sys.argv[3]

old_dict = torch.load(old_model)
new_dict = torch.load(new_model)

for name in new_dict["model"]:

    if name not in old_dict["model"].keys():
        raise NotImplementedError(f"{name} not in old_dict")
    new_dict["model"][name] = old_dict["model"][name]

torch.save(new_dict, out_model)