import torch
import torch.nn as nn
import torch.nn.functional as F


class Projection(nn.Module):

	# encoder : resnet18 or resnet34
	# in_dim : encoder output size
	# proj_dim : size of projection

	def __init__(self, encoder, in_dim, proj_dim):
		super(Projection, self).__init__()

		self.encoder = encoder # resnet encoder
		self.proj = nn.Sequential(
			nn.Linear(in_dim, in_dim),
			nn.ReLU(inplace = True),
			nn.Linear(in_dim, proj_dim))

	def forward(self, x):
		out = self.encoder(x)
		return F.normalize(self.proj(out), dim = 1)

		# projections already are normalized hence we only need
		# to do mm to get cosine similarity


class simclr_loss(nn.Module):

	def __init__(self, temperature = 0.1, device = None):
		super(simclr_loss, self).__init__()
		self.temperature = temperature
		self.device = device

	def forward(self, projections, labels):

		# let 128 be number of projected features
		# projections : 2*batch_size x 128
		# labels : 2*batch_size

		projections = F.normalize(projections, dim = 1)

		sim_matrix = torch.matmul(projections, projections.T) / self.temperature
		exp_sim_matrix = torch.exp(sim_matrix).to(self.device)

		mask_similar_classes = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() # 2*batch_size x 2*batch_size
		# similar classes have value 1 and rest are 0
		# we need to discard the diagonal 1s because they are cosine similarity
		# between 1 image considered twice.

		mask_diag = 1 - torch.eye(labels.shape[0])
		mask_diag = mask_diag.to(self.device)
		# all elements except diagonal ones are 1. diagonal elements are 0.

		mask_combined = mask_similar_classes * mask_diag
		# now we have final masks. cosine similarity between exact same images is ignored now

		num = exp_sim_matrix * mask_combined # 2*batch_size x 2*batch_size
		denom = torch.sum(exp_sim_matrix * mask_diag, dim = 1).unsqueeze(1) # 2*batch_size
		loss = (num / denom).sum(dim = 1)
		return loss.mean()



def train_simclr(batch_size, train_loader, model, criterion, optimizer, device, epochs):

	model.train()
	train_loss_list = []

	# only the image and its augmented image belong to the same class
	# rest belong to different class
	labels = torch.tensor([i for i in range(batch_size)])
	labels = labels.repeat(2)
	labels = labels.to(device)

	for epoch in range(epochs):
		running_average_loss = 0 # for batches
		epoch_loss = 0

		for idx, (images, _) in enumerate(train_loader):

			images = torch.cat(images)
			images = images.to(device)

			projections = model(images)
			projections = projections.to(device)
			loss = criterion(projections, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			loss = loss.item()
			epoch_loss += loss
			running_average_loss = epoch_loss / (idx + 1)
			print(" epoch : " + str(epoch) + ", batch : " + str(idx) + " / 194, " + " bl " + str(loss) + ", ral : " + str(running_average_loss))

		train_loss_list.append(epoch_loss)
		print("Epoch " + str(epoch) + "loss : " + str(epoch_loss))
		plot_SimCLR_loss(epoch,  train_loss_list)


def plot_SimCLR_loss(epochs, losses):

	if not os.path.isdir('Plots'):
		os.mkdir('Plots')
	plt.plot(range(epochs), losses)
	plt.xlabel('Epochs')
	plt.ylabel('Supervised Contrastive Loss')
	plt.savefig('Plots/SimCLRLoss.jpeg')





		


		
