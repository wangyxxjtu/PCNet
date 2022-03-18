import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .model_util import *
from train_util import *

# define the function includes in import *
__all__ = [
	'SpixelNet1l','SpixelNet1l_bn'
]

class SpixelNet(nn.Module):
	expansion = 1

	def __init__(self, batchNorm=True, Train=True):
		super(SpixelNet,self).__init__()

		self.batchNorm = batchNorm
		self.assign_ch = 9
		self.Train=Train

		self.conv0a = conv(self.batchNorm, 3, 16, kernel_size=3)
		self.conv0b = conv(self.batchNorm, 16, 16, kernel_size=3)

		self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
		self.conv1b = conv(self.batchNorm, 32, 32, kernel_size=3)

		self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
		self.conv2b = conv(self.batchNorm, 64, 64, kernel_size=3)

		self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
		self.conv3b = conv(self.batchNorm, 128, 128, kernel_size=3)

		self.conv4a = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
		self.conv4b = conv(self.batchNorm, 256, 256, kernel_size=3)

		self.deconv3 = deconv(256, 128)
		self.conv3_1 = conv(self.batchNorm, 256, 128)
		self.pred_mask3 = predict_mask(128, self.assign_ch)

		self.deconv2 = deconv(128, 64)
		self.conv2_1 = conv(self.batchNorm, 128, 64)
		self.pred_mask2 = predict_mask(64, self.assign_ch)

		self.deconv1 = deconv(64, 32)
		self.conv1_1 = conv(self.batchNorm, 64, 32)
		self.pred_mask1 = predict_mask(32, self.assign_ch)

		self.deconv0 = deconv(32, 16)
		#===========================================================
		#for scale 1
		self.conv0_1 = conv(self.batchNorm, 32, 16)

		#for scale2
		#encode the scale 1 image
		#self.conv_s10 = conv(self.batchNorm, 3, 16, kernel_size=3)
		#self.conv_s11 = conv(self.batchNorm, 16, 16, kernel_size=3)

		self.SR = nn.Sequential(
		conv(self.batchNorm, 16, 16 * 4**2),
		nn.PixelShuffle(4)
		)
		self.conv_s1 = conv(self.batchNorm, 16, 16)

		self.im_recon = nn.Sequential(
		nn.Conv2d(16,3, 1, bias=False),
		nn.Tanh()
		)
		#========================================================
		self.pred_head1 = nn.Sequential(
		predict_mask(16,self.assign_ch),
		nn.Softmax(1)
		)

		#self.softmax = nn.Softmax(1)

		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				kaiming_normal_(m.weight, 0.1)
				if m.bias is not None:
					constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				constant_(m.weight, 1)
				constant_(m.bias, 0)

	def forward(self, full_input, patch_input=None):
		full_out= self.forward_(full_input, True)
		if self.Train:
		   patch_recon, patch_out = self.forward_(patch_input, False)

		   return full_out[0], full_out[1], patch_recon, full_out[2], patch_out
		else:
		   return full_out

	def forward_(self, x, full):
		b,c,h,w = x.shape
		H = h//4
		W = w // 4
		x_pool1 = F.interpolate(x, size=(H,W))

		out1 = self.conv0b(self.conv0a(x_pool1)) #5*5
		out2 = self.conv1b(self.conv1a(out1)) #11*11
		out3 = self.conv2b(self.conv2a(out2)) #23*23
		out4 = self.conv3b(self.conv3a(out3)) #47*47
		out5 = self.conv4b(self.conv4a(out4)) #95*95

		out_deconv3 = self.deconv3(out5)

		concat3 = torch.cat((out4, out_deconv3), 1)
			
		out_conv3_1 = self.conv3_1(concat3)

		out_deconv2 = self.deconv2(out_conv3_1)
		concat2 = torch.cat((out3, out_deconv2), 1)
		out_conv2_1 = self.conv2_1(concat2)

		out_deconv1 = self.deconv1(out_conv2_1)
		concat1 = torch.cat((out2, out_deconv1), 1)
		out_conv1_1 = self.conv1_1(concat1)

		out_deconv0 = self.deconv0(out_conv1_1)
		concat0 = torch.cat((out1, out_deconv0), 1)
		out_conv0_1 = self.conv0_1(concat0)

		#make the prediction for scale 1
		#encode the high resolution image
		#scale1 = self.conv_s10(x)
		#scale1 = self.conv_s11(scale1)
		#=================================

		#SR_feat = self.SR(out_conv0_1+F.avg_pool2d(scale1, 4))
		SR_feat = self.SR(out_conv0_1)
		up1 = self.conv_s1(SR_feat)

		pred_s1 = self.pred_head1(up1)
		
		if self.Train:
		   #reconstruct the high resolution image
		   SR_Recon = self.im_recon(SR_feat)
		   if full:
			   return up1, SR_Recon, pred_s1
		   else:
			   return SR_Recon, pred_s1
		else:
		   return pred_s1

	def weight_parameters(self):
		return [param for name, param in self.named_parameters() if 'weight' in name]

	def bias_parameters(self):
		return [param for name, param in self.named_parameters() if 'bias' in name]


def SpixelNet1l( data=None, Train=True):
	# Model without  batch normalization
	model = SpixelNet(batchNorm=False, Train=Train)
	if data is not None:
		model.load_state_dict(data['state_dict'])
	return model

def SpixelNet1l_bn(data=None, Train=True):
	# model with batch normalization
	model = SpixelNet(batchNorm=True, Train=Train)
	if data is not None:
		model.load_state_dict(data['state_dict'])
	return model
#
