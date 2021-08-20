import timm
import torch, torchvision
from pprint import pprint

model_names = timm.list_models(pretrained=True)
pprint(model_names)

print(torch.cuda_version)
print(torch.tensor(1.0))
print(torchvision.__version__)