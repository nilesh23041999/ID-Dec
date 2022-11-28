import torch.nn as nn
import torchvision.models as models




class Model(nn.Module):
	
	def __init__(self,num_classes=2,input_channels=3,resnet_layers=18,pretrained_flag=False):
		super(Model,self).__init__()

		if resnet_layers == 152:
			self.resnet18 = models.resnet152(pretrained=pretrained_flag)
		elif resnet_layers == 101:
			self.resnet18 = models.resnet101(pretrained=pretrained_flag)
		elif resnet_layers == 50:
			self.resnet18 = models.resnet50(pretrained=pretrained_flag)
		elif resnet_layers == 34:
			self.resnet18 = models.resnet34(pretrained=pretrained_flag)
		else:
			self.resnet18 = models.resnet18(pretrained=pretrained_flag)
		
		self.resnet18.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

		modules=list(self.resnet18.children())[:-2]
		modules.append(nn.AdaptiveAvgPool2d((1,1)))
		self.resnet18=nn.Sequential(*modules)

		
		if resnet_layers == 152 or resnet_layers == 101 or resnet_layers == 50:
			self.fc1=nn.Sequential(nn.Linear(2048*1*1,1024),nn.ReLU())
		else:
			self.fc1=nn.Sequential(nn.Linear(512*1*1,1024),nn.ReLU())

		self.fc2=nn.Sequential(nn.Linear(1024,num_classes))
		
	def forward(self,x):
		x=self.resnet18(x)
		x=x.view(x.size(0),-1)
		x=self.fc1(x)
		x=self.fc2(x)
		return x
		
def get_model(num_classes=2,input_channels=3,resnet_layers =18,pretrained_flag=False):
	model = Model(num_classes,input_channels,resnet_layers,pretrained_flag)
	return model		
