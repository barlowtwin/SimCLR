from resnet import resnet18
from simclr import Projection, simclr_loss, train_simclr
from data import custom_data_loader
import torch

batch_size = 256
in_channels = 3
encoder = resnet18(in_channels = in_channels)
in_dim = 512
proj_dim = 128
temperature = 0.1
lr = 0.001
epochs = 100


if torch.cuda.is_available():
	device = torch.device('cuda')
	print("gpu detected for trainig")
else :
	device = torch.device('cpu')
	print("cpu used for training")

model = Projection(encoder, in_dim = in_dim, proj_dim = proj_dim)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = lr)
criterion = simclr_loss(temperature = temperature, device = device)

data_loader = custom_data_loader(batch_size = batch_size)
train_simclr(batch_size, data_loader, model, criterion, optimizer, device, epochs)




