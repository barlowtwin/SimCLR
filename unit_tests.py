import torch
from torchvision import utils
import matplotlib.pyplot as plt


from simclr import SimCLR_loss
projections = torch.rand(10,128)
labels = torch.tensor([1,2,3,4,5,1,2,3,4,5])
loss = simclr_loss()
loss(projections, labels = labels)

