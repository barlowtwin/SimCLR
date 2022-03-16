import torch
from torchvision import transforms, datasets
import os



class ImageTransform :

	def __init__(self, transform):
		self.transform = transform

	def __call__(self, x):
		return [self.transform(x), self.transform(x)]

def get_transform(color_distortion, crop_size):

	# mean = (0.4914, 0.4822, 0.4465)
	# std = (0.2023, 0.1994, 0.2010)

	# s is strength of color distortion
	s = color_distortion

	color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
	data_transform = transforms.Compose([
		transforms.RandomResizedCrop(size = crop_size),
		transforms.RandomHorizontalFlip(p = 0.5),
		transforms.RandomApply([color_jitter], p = 0.8),
		transforms.RandomGrayscale(p = 0.2), # keeps channels same
		transforms.GaussianBlur(kernel_size = int(0.1 * crop_size)),
		transforms.ToTensor(),
		#transforms.Normalize(mean = mean, std = std)
		])

	return data_transform


def custom_data_loader(batch_size, color_distortion = 1, crop_size = 32):

	if not os.path.isdir('data'):
		os.mkdir('data')

	data_transform = get_transform(color_distortion = color_distortion, crop_size = crop_size)
	train_dataset = datasets.CIFAR10('data', transform = ImageTransform(data_transform), download = True)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True, drop_last = True)
	return train_loader


