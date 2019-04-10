import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable


######################################################################
def weights_init_kaiming(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
		init.constant_(m.bias.data, 0.0)
	elif classname.find('BatchNorm1d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		init.normal_(m.weight.data, std=0.001)
		init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
	def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
	             return_f=False):
		super(ClassBlock, self).__init__()
		
		self.return_f = return_f
		self.dropout = nn.Dropout(p=0.5)
		add_block = []  # for dimensionality reduction
		if linear:
			add_block += [nn.Linear(input_dim, num_bottleneck)]
		else:
			num_bottleneck = input_dim
		if bnorm:
			add_block += [nn.BatchNorm1d(num_bottleneck)]
		if relu:
			add_block += [nn.LeakyReLU(0.1)]
		if droprate > 0:
			add_block += [nn.Dropout(p=droprate)]
		add_block = nn.Sequential(*add_block)
		add_block.apply(weights_init_kaiming)
		
		classifier = []
		classifier += [nn.Linear(num_bottleneck, class_num)]
		classifier = nn.Sequential(*classifier)
		classifier.apply(weights_init_classifier)
		
		self.add_block = add_block
		self.classifier = classifier
	
	def forward(self, x):
		x = self.add_block(x)
		if self.return_f:
			f = x
			x = self.classifier(x)
			return x, f
		else:
			# x = self.dropout(x)
			# if self.part_detector_weighted:
			# 	weight = self.part_detector_block(x)
			# 	x = weight * x / (x.shape[0] * x.shape[1])
			x = self.classifier(x)
			return x


class weighted_avg_pooling(nn.Module):
	def __init__(self, num_ftrs=2048):
		super().__init__()
		self.num_ftrs = num_ftrs
		part_detector_block = []
		part_detector_block += [nn.Conv2d(self.num_ftrs, self.num_ftrs, 1)]  # 1*1 conv layer
		part_detector_block += [nn.Sigmoid()]
		part_detector_block = nn.Sequential(*part_detector_block)
		part_detector_block.apply(weights_init_kaiming)
		self.part_detector_block = part_detector_block
	
	def forward(self, x):
		mask = self.part_detector_block(x)
		mask = torch.sum(mask * x, dim=(3, 2)) / (x.shape[-2] * x.shape[-1])  # 32 * 2048
		return mask


# Define the ResNet50-based Model
class ft_net(nn.Module):
	# result in stride=2
	# Rank@1:0.864905 Rank@5:0.944477 Rank@10:0.963777 mAP:0.688464
	# after reranking:
	# top1:0.889549 top5:0.936758 top10:0.951306 mAP:0.829203
	
	def __init__(self, class_num, attr_num=30, droprate=0.5, stride=1, weight_avg=False):
		super(ft_net, self).__init__()
		self.attr_num = attr_num
		model_ft = models.resnet50(pretrained=True)
		# avg pooling to global pooling
		if stride == 1:
			# Rank@1: 0.878266 Rank@5: 0.952197 Rank@10: 0.969418 mAP: 0.702516
			model_ft.layer4[0].downsample[0].stride = (1, 1)
			model_ft.layer4[0].conv2.stride = (1, 1)
		self.branch2_layer4 = model_ft.layer4
		model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.model = model_ft
		self.num_ftrs = 2048
		
		# torch.nn.KLDivLoss
		
		# self.classifier = ClassBlock(2048, class_num, droprate)
		
		for c in range(self.attr_num + 1):
			if c == self.attr_num:  # for identity classification
				self.__setattr__('class_%d' % c, ClassBlock(self.num_ftrs, class_num, 0, num_bottleneck=256))
			else:
				self.__setattr__('class_%d' % c, nn.Sequential(weighted_avg_pooling(num_ftrs=self.num_ftrs),
				                                               ClassBlock(self.num_ftrs, 2, droprate=0,
				                                                          num_bottleneck=128,
				                                                          relu=False, bnorm=True)))
	
	def forward(self, x):
		x = self.model.conv1(x)
		x = self.model.bn1(x)
		x = self.model.relu(x)
		x = self.model.maxpool(x)
		x = self.model.layer1(x)
		x = self.model.layer2(x)
		x = self.model.layer3(x)  # batch * 1024 * 18 * 9
		attr_x = x
		x = self.model.layer4(x)  # batch * 2048 * 9 * 5
		attr_x = self.branch2_layer4(attr_x)
		# attr_x = self.model.layer4(attr_x)
		x = self.model.avgpool(x)  # average pooling for id feature
		x = x.view(x.size(0), x.size(1))  # batch * 2048
		# x = self.classifier(x)
		# return list(self.__getattr__('class_%d' % c)(x) for c in range(self.attr_num + 1)), x
		attr_output = list(self.__getattr__('class_%d' % c)(attr_x) for c in range(self.attr_num))
		id_output = self.__getattr__('class_{}'.format(self.attr_num))(x)
		attr_output.append(id_output)  # cat two output
		return attr_output, x


# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):
	
	def __init__(self, class_num, droprate=0.5):
		super().__init__()
		model_ft = models.densenet121(pretrained=True)
		model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		model_ft.fc = nn.Sequential()
		self.model = model_ft
		# For DenseNet, the feature dim is 1024
		self.classifier = ClassBlock(1024, class_num, droprate)
	
	def forward(self, x):
		x = self.model.features(x)
		x = x.view(x.size(0), x.size(1))
		x = self.classifier(x)
		return x


# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle:
# Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):
	
	def __init__(self, class_num, droprate=0.5):
		super(ft_net_middle, self).__init__()
		model_ft = models.resnet50(pretrained=True)
		# avg pooling to global pooling
		model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.model = model_ft
		self.classifier = ClassBlock(2048 + 1024, class_num, droprate)
	
	def forward(self, x):
		x = self.model.conv1(x)
		x = self.model.bn1(x)
		x = self.model.relu(x)
		x = self.model.maxpool(x)
		x = self.model.layer1(x)
		x = self.model.layer2(x)
		x = self.model.layer3(x)
		# x0  n*1024*1*1
		x0 = self.model.avgpool(x)
		x = self.model.layer4(x)
		# x1  n*2048*1*1
		x1 = self.model.avgpool(x)
		x = torch.cat((x0, x1), 1)
		x = x.view(x.size(0), x.size(1))
		x = self.classifier(x)
		return x


# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
	def __init__(self, class_num):
		super(PCB, self).__init__()
		
		self.part = 6  # We cut the pool5 to 6 parts
		model_ft = models.resnet50(pretrained=True)
		self.model = model_ft
		self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
		self.dropout = nn.Dropout(p=0.5)
		# remove the final downsample
		self.model.layer4[0].downsample[0].stride = (1, 1)
		self.model.layer4[0].conv2.stride = (1, 1)
		# define 6 classifiers
		for i in range(self.part):
			name = 'classifier' + str(i)
			setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))
	
	def forward(self, x):
		x = self.model.conv1(x)
		x = self.model.bn1(x)
		x = self.model.relu(x)
		x = self.model.maxpool(x)
		
		x = self.model.layer1(x)
		x = self.model.layer2(x)
		x = self.model.layer3(x)
		x = self.model.layer4(x)
		x = self.avgpool(x)
		x = self.dropout(x)
		part = {}
		predict = {}
		# get six part feature batchsize*2048*6
		for i in range(self.part):
			part[i] = torch.squeeze(x[:, :, i])
			name = 'classifier' + str(i)
			c = getattr(self, name)
			predict[i] = c(part[i])
		
		# sum prediction
		# y = predict[0]
		# for i in range(self.part-1):
		#    y += predict[i+1]
		y = []
		for i in range(self.part):
			y.append(predict[i])
		return y


class PCB_test(nn.Module):
	def __init__(self, model):
		super(PCB_test, self).__init__()
		self.part = 6
		self.model = model.model
		self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
		# remove the final downsample
		self.model.layer4[0].downsample[0].stride = (1, 1)
		self.model.layer4[0].conv2.stride = (1, 1)
	
	def forward(self, x):
		x = self.model.conv1(x)
		x = self.model.bn1(x)
		x = self.model.relu(x)
		x = self.model.maxpool(x)
		
		x = self.model.layer1(x)
		x = self.model.layer2(x)
		x = self.model.layer3(x)
		x = self.model.layer4(x)
		x = self.avgpool(x)
		y = x.view(x.size(0), x.size(1), x.size(2))
		return y


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
	# Here I left a simple forward function.
	# Test the model, before you train it.
	net = ft_net(751, stride=1)
	net.classifier = nn.Sequential()
	print(net)
	input = Variable(torch.FloatTensor(8, 3, 256, 128))
	output = net(input)
	print('net output size:')
	print(output.shape)
