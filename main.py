import argparse
import os
import shutil
import time

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import flow_transforms
import models
import datasets
from loss import compute_semantic_pos_loss
import datetime
from tensorboardX import SummaryWriter
from train_util import *
import pdb
from datasets.BSD500 import get_dataset
import random
from datasets.data_util import collate_fn

'''
Main code for training for PCNet 

author: Yaxiong Wang 
'''

model_names = sorted(name for name in models.__dict__
					 if not name.startswith("__"))
dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch SpixelFCN Training on BSDS500',
								 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# ================ training setting ====================
#parser.add_argument('--dataset', metavar='DATASET', default='BSD500',	choices=dataset_names,
#					 help='dataset type : ' +  ' | '.join(dataset_names))
parser.add_argument('--dataset',default='', help='specify the dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='SpixelNet1l_bn',  help='model architecture')
parser.add_argument('--data', metavar='DIR',default='', help='path to input dataset')
parser.add_argument('--savepath',default='', help='path to save ckpt')
parser.add_argument('--suffix',default='', help='suffix to differiate')


parser.add_argument('--train_img_height', '-t_imgH', default=512,  type=int, help='img height')
parser.add_argument('--train_img_width', '-t_imgW', default=512, type=int, help='img width')
parser.add_argument('--input_img_height', '-v_imgH', default=320, type=int, help='img height_must be 16*n')  #
parser.add_argument('--input_img_width', '-v_imgW', default=320,  type=int, help='img width must be 16*n')

# ======== learning schedule ================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers')
parser.add_argument('--epochs', default=3000000, type=int, metavar='N', help='number of total epoches, make it big enough to follow the iteration maxmium')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',	help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch_size', default= 6000,	help='choose any value > 408 to use all the train and val data')
parser.add_argument('-b', '--batch-size', default=4, type=int,	 metavar='N', help='mini-batch size')

parser.add_argument('--solver', default='adam',choices=['adam','sgd'], help='solver algorithms, we use adam')
parser.add_argument('--lr', '--learning-rate', default=0.00008, type=float,metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',  help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',	help='beta parameter for adam')
parser.add_argument('--weight_decay', '--wd', default=4e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--bias_decay', default=0, type=float, metavar='B', help='bias decay, we never use it')
parser.add_argument('--milestones', default=[200000], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')
parser.add_argument('--additional_step', default= 100000, help='the additional iteration, after lr decay')

# ============== hyper-param ====================
parser.add_argument('--pos_weight', '-p_w', default=0.003, type=float, help='weight of the pos term')
parser.add_argument('--downsize', default=16, type=float,help='grid cell size for superpixel training ')

# ================= other setting ===================
parser.add_argument('--gpu', default= '0', type=str, help='gpu id')
parser.add_argument('--print_freq', '-p', default=10, type=int,  help='print frequency (step)')
parser.add_argument('--record_freq', '-rf', default=5, type=int,  help='record frequency (epoch)')
parser.add_argument('--label_factor', default=5, type=int, help='constant multiplied to label index for viz.')
parser.add_argument('--pretrained', dest='pretrained', default=None, help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true',  help='don\'t append date timestamp to folder' )
parser.add_argument('--mini', action='store_true',	help='don\'t append date timestamp to folder' )

best_EPE = -1
best_epoch =0
n_iter = 0
args = parser.parse_args()

# !----- NOTE the current code does not support cpu training -----!
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	print('Current code does not support CPU training! Sorry about that.')
	exit(1)

if args.dataset == 'face_human':
	mean1 = np.array([130.38/255., 116.94/255., 108.27/255.])
	mean2 = np.array([124.75/255., 120.83/255., 115.57/255.])
	mean = ((mean1 + mean2) /2).tolist()
	class_num = 24
	print('train on face_human dataset!')
elif args.dataset == 'BSDS500':
	mean = [0.411,0.432,0.45]
	class_num = 50

def random_seed_setting(seed=2021):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)

def main():
	global args, best_EPE, save_path, intrinsic, best_epoch
	random_seed_setting()

	# ============= savor setting ===================
	if not args.no_date:
		timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M")
	else:
		timestamp = ''
	if args.mini:
		args.suffix = 'mini' + args.suffix
	save_path = os.path.abspath(args.savepath) + '/' + args.dataset + args.suffix + '/'

	# ==========  Data loading code ==============
	input_transform = transforms.Compose([
		flow_transforms.ArrayToTensor(),
		transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
		transforms.Normalize(mean=mean, std=[1,1,1])
	])

	val_input_transform = transforms.Compose([
		flow_transforms.ArrayToTensor(),
		transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
		transforms.Normalize(mean=mean, std=[1, 1, 1])
	])

	target_transform = transforms.Compose([
		flow_transforms.ArrayToTensor(),
	])

	co_transform = flow_transforms.Compose([
			flow_transforms.RandomCropBoundary(args.train_img_height, args.train_img_width),
			flow_transforms.RandomVerticalFlip(),
			flow_transforms.RandomHorizontalFlip()
		])

	print("=> loading img pairs from '{}'".format(args.data))
	train_set, val_set = get_dataset(
		#os.path.join(args.data,args.dataset),
		args.data,
		transform=input_transform,
		val_transform = val_input_transform,
		target_transform=target_transform,
		co_transform=co_transform,
		mini=args.mini
	)
	print('{} samples found, {} train samples and {} val samples '.format(len(val_set)+len(train_set),
																		   len(train_set),
																		   len(val_set)))
	train_loader = torch.utils.data.DataLoader(
		train_set, batch_size=args.batch_size,
		num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True, collate_fn=collate_fn)

	val_loader = torch.utils.data.DataLoader(
		val_set, batch_size=args.batch_size,
		num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=True, collate_fn=collate_fn)

	# ============== create model ====================
	if args.pretrained:
		network_data = torch.load(args.pretrained)
		args.arch = network_data['arch']
		print("=> using pre-trained model '{}'".format(args.arch))
	else:
		network_data = None
		print("=> creating model '{}'".format(args.arch))

	model = models.__dict__[args.arch]( data = network_data).cuda()
	model = torch.nn.DataParallel(model).cuda()
	cudnn.benchmark = True

	#=========== creat optimizer, we use adam by default ==================
	assert(args.solver in ['adam', 'sgd'])
	print('=> setting {} solver'.format(args.solver))
	param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': args.bias_decay},
					{'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay}]
	if args.solver == 'adam':
		optimizer = torch.optim.Adam(param_groups, args.lr,
									 betas=(args.momentum, args.beta))
	elif args.solver == 'sgd':
		optimizer = torch.optim.SGD(param_groups, args.lr,
									momentum=args.momentum)

	# for continues training
	if args.pretrained and ('dataset' in network_data):
		if args.pretrained and args.dataset == network_data['dataset'] :
			optimizer.load_state_dict(network_data['optimizer'])
			best_EPE = network_data['best_EPE']
			args.start_epoch = network_data['epoch']
			save_path = os.path.dirname(args.pretrained)

	print('=> will save everything to {}'.format(save_path))
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	train_writer = SummaryWriter(os.path.join(save_path, 'train'))
	val_writer = SummaryWriter(os.path.join(save_path, 'val'))

	# spixelID: superpixel ID for visualization,
	# XY_feat: the coordinate feature for position loss term
	spixelID, XY_feat_stack = init_spixel_grid(args)
	spixelID, XY_feat_stack_d1 = init_spixel_grid(args, scale=2)

	val_spixelID,  val_XY_feat_stack = init_spixel_grid(args, b_train=False)

	for epoch in range(args.start_epoch, args.epochs):
		# train for one epoch
		train_avg_slic, train_avg_sem, iteration = train(train_loader, model, optimizer, epoch,
														 train_writer, spixelID, XY_feat_stack, XY_feat_stack_d1)
		if epoch % args.record_freq == 0:
			train_writer.add_scalar('Mean avg_slic', train_avg_slic, epoch)

		# evaluate on validation set and save the module( and choose the best)
		with torch.no_grad():
			avg_slic, avg_sem  = validate(val_loader, model, epoch, val_writer, spixelID, val_XY_feat_stack, XY_feat_stack_d1)
			if epoch % args.record_freq == 0:
				val_writer.add_scalar('Mean avg_slic', avg_slic, epoch)

		rec_dict = {
				'epoch': epoch + 1,
				'arch': args.arch,
				'state_dict': model.module.state_dict(),
				'best_EPE': best_EPE,
				'optimizer': optimizer.state_dict(),
				'dataset': args.dataset
			}
		'''
		if (iteration) >= (args.milestones[-1] + args.additional_step):
			save_checkpoint(rec_dict, is_best =False, filename='%d_step.tar' % iteration)
			print("Train finished!")
			break
		'''
		if best_EPE < 0:
			best_EPE = avg_sem
		is_best = avg_sem < best_EPE
		best_EPE = min(avg_sem, best_EPE)
		save_checkpoint(rec_dict, is_best)
		if is_best:
			best_epoch = epoch
		val_writer.add_scalar('Best Epoch:', best_epoch, epoch)

def train(train_loader, model, optimizer, epoch, train_writer, init_spixl_map_idx, xy_feat, xy_feat_d1):
	global n_iter, args, intrinsic
	batch_time = AverageMeter()
	data_time = AverageMeter()

	total_loss = AverageMeter()
	losses_sem = AverageMeter()
	losses_pos = AverageMeter()

	epoch_size =  len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

	# switch to train mode
	model.train()
	end = time.time()
	iteration = 0

	for i, (full_input,full_label, full_bd, input, patch_label, bin_label1hot, bd, full_patch_posi, full_patch_label, full_side_indices) in enumerate(train_loader):
		iteration = i + epoch * epoch_size
		full_input = full_input.to(device)
		full_label = full_label.to(device)
		full_bd = full_bd.to(device)
		patch_label = patch_label.to(device)
		bin_label1hot = bin_label1hot.to(device)
		bd = bd.to(device)
		b,c,h,w=full_input.shape

		# ========== adjust lr if necessary  ===============
		if (iteration + 1) in args.milestones:
			state_dict = optimizer.state_dict()
			for param_group in state_dict['param_groups']:
				param_group['lr'] = args.lr * ((0.5) ** (args.milestones.index(iteration + 1) + 1))
			optimizer.load_state_dict(state_dict)

		# ========== complete data loading ================
		full_label1hot = label2one_hot_torch(full_label, C=class_num) # set C=50 as SSN does
		input_gpu = input.to(device)
		patch_LABXY_feat_tensor = build_LABXY_feat(bin_label1hot, xy_feat)	# B* (50+2 )* H * W
		full_LABXY_feat_tensor = build_LABXY_feat(full_label1hot, xy_feat)	# B* (50+2 )* H * W

		torch.cuda.synchronize()
		data_time.update(time.time() - end)

		# ========== predict association map ============
		full_pix_emb, full_recon, patch_recon, full_out, patch_out = model(full_input, input_gpu)
		slic_loss, loss_sem, loss_pos = compute_semantic_pos_loss(full_out, full_LABXY_feat_tensor,
																pos_weight= args.pos_weight, kernel_size=args.downsize, im_recon=full_recon, im_gt=full_input, bd_local=full_bd, patch_posi=full_patch_posi, patch_label=full_patch_label, side_indices=full_side_indices, pix_emb=full_pix_emb)

		slic_loss_pt, loss_sem_pt, loss_pos_pt = compute_semantic_pos_loss(patch_out, patch_LABXY_feat_tensor,
																pos_weight= args.pos_weight, kernel_size=args.downsize, im_recon=patch_recon, im_gt=input_gpu, bd_local=bd)

		slic_loss = slic_loss + 0.5*slic_loss_pt
		#loss_sem = loss_sem_up2
		#loss_pos = loss_pos_up2 #+ loss_pos_up2 * 0.6
		# ========= back propagate ===============
		optimizer.zero_grad()
		slic_loss.backward()
		optimizer.step()

		# ========	measure batch time ===========
		torch.cuda.synchronize()
		batch_time.update(time.time() - end)
		end = time.time()

		# =========== record and display the loss ===========
		# record loss and EPE
		total_loss.update(slic_loss.item(), input_gpu.size(0))
		losses_sem.update(loss_sem.item(), input_gpu.size(0))
		losses_pos.update(loss_pos.item(), input_gpu.size(0))

		if i % args.print_freq == 0:
			print('train Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Total_loss {5}\t Loss_sem {6}\t Loss_pos {7}\t'
				  .format(epoch, i, epoch_size, batch_time, data_time, total_loss, losses_sem, losses_pos))

			train_writer.add_scalar('Total_train_loss', slic_loss.item(), i + epoch*epoch_size)
			train_writer.add_scalar('Total_sem_loss', loss_sem.item(), i + epoch*epoch_size)
			train_writer.add_scalar('Total_pos_loss', loss_pos.item(), i + epoch*epoch_size)

			train_writer.add_scalar('learning rate',optimizer.param_groups[0]['lr'], i + epoch * epoch_size)

		n_iter += 1

	# =========== write information to tensorboard ===========
	if epoch % args.record_freq == 0:
		train_writer.add_scalar('Total_loss_epoch', slic_loss.item(),  epoch )
		train_writer.add_scalar('Total_loss_sem',  loss_sem.item(),  epoch )
		train_writer.add_scalar('Total_loss_pos',  loss_pos.item(), epoch)

		'''train_writer.add_scalar('UP0_total_loss', slic_loss_up1.item(), epoch)
		train_writer.add_scalar('UP0_sem_loss', loss_sem_up1.item(), epoch)
		train_writer.add_scalar('UP0_pos_loss', loss_pos_up1.item(), epoch)
		'''
		input_ = F.avg_pool2d(full_input, kernel_size=2, stride=2)
		label_ = F.avg_pool2d(full_label, kernel_size=2, stride=2).round()
		out_up_ = F.avg_pool2d(full_out, kernel_size=2,stride=2)

		mean_values = torch.tensor(mean, dtype=input_gpu.dtype).view(3, 1, 1).to(device)
		input_l_save = (make_grid((input_ + mean_values).clamp(0, 1), nrow=args.batch_size))
		label_save = make_grid(args.label_factor * label_)

		train_writer.add_image('Input', input_l_save, epoch)
		train_writer.add_image('label', label_save, epoch)

		curr_spixl_map = update_spixl_map(init_spixl_map_idx,out_up_)
		spixel_lab_save = make_grid(curr_spixl_map, nrow=args.batch_size)[0, :, :]
		spixel_viz, _ = get_spixel_image(input_l_save, spixel_lab_save)
		train_writer.add_image('Spixel viz', spixel_viz, epoch)

		#save associ map,  --- for debug only
		# _, prob_idx = torch.max(output, dim=1, keepdim=True)
		# prob_map_save = make_grid(assign2uint8(prob_idx))
		# train_writer.add_image('assigment idx', prob_map_save, epoch)

		print('==> write train step %dth to tensorboard' % i)


	return total_loss.avg, losses_sem.avg, iteration


def validate(val_loader, model, epoch, val_writer, init_spixl_map_idx, xy_feat, xy_feat_d1):
	global n_iter,	 args,	  intrinsic
	batch_time = AverageMeter()
	data_time = AverageMeter()

	total_loss = AverageMeter()
	losses_sem = AverageMeter()
	losses_pos = AverageMeter()

	# set the validation epoch-size, we only randomly val. 400 batches during training to save time
	epoch_size = min(len(val_loader), 400)

	# switch to train mode
	model.eval()
	end = time.time()

	for i, (input, full_label, _,patch_input, patch_label,bin_label_1hot, bd, full_patch_posi, full_patch_label, full_side_indices) in enumerate(val_loader):
		full_label = full_label.to(device)
		patch_label = patch_label.to(device)
		bd = bd.to(device)
		b,c,h,w = full_label.shape
		# measure data loading time
		label_1hot = label2one_hot_torch(full_label, C=class_num)
		input_gpu = input.to(device)
		LABXY_feat_tensor = build_LABXY_feat(label_1hot, xy_feat)  # B* 50+2 * H * W
		
		torch.cuda.synchronize()
		data_time.update(time.time() - end)

		# ========== predict association map ============
		# compute output
		with torch.no_grad():
			_,_,im_recon, full_out, out_up1 = model(input_gpu, patch_input)
			slic_loss_up2, loss_sem_up2, loss_pos_up2 = compute_semantic_pos_loss(full_out, LABXY_feat_tensor,
																	pos_weight= args.pos_weight, kernel_size=args.downsize) #, im_recon=im_recon, im_gt=input_gpu, bd_local=bd)

			slic_loss = slic_loss_up2
			loss_sem =	loss_sem_up2
			loss_pos =	loss_pos_up2
	 
		# measure elapsed time
		torch.cuda.synchronize()
		batch_time.update(time.time() - end)
		end = time.time()

		# record loss and EPE
		total_loss.update(slic_loss.item(), input_gpu.size(0))
		losses_sem.update(loss_sem.item(), input_gpu.size(0))
		losses_pos.update(loss_pos.item(), input_gpu.size(0))

		if i % args.print_freq == 0:
			print('val Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Total_loss {5}\t Loss_sem {6}\t Loss_pos {7}\t'
				  .format(epoch, i, epoch_size, batch_time, data_time, total_loss, losses_sem, losses_pos))


	# =============  write result to tensorboard ======================
	if epoch % args.record_freq == 0:
		val_writer.add_scalar('Total_loss_epoch', slic_loss.item(),  epoch )
		val_writer.add_scalar('Total_loss_sem',  loss_sem.item(),  epoch )
		val_writer.add_scalar('Total_loss_pos',  loss_pos.item(), epoch)

		val_writer.add_scalar('UP1_total_loss', slic_loss_up2.item(), epoch)
		val_writer.add_scalar('UP1_sem_loss', loss_sem_up2.item(), epoch)
		val_writer.add_scalar('UP1_pos_loss', loss_pos_up2.item(), epoch)


		mean_values = torch.tensor(mean, dtype=input_gpu.dtype).view(3, 1, 1)
		input_l_save = (make_grid((input + mean_values).clamp(0, 1), nrow=args.batch_size))

		curr_spixl_map = update_spixl_map(init_spixl_map_idx, out_up1) # get the superixle index of each pixel belongs to
		spixel_lab_save = make_grid(curr_spixl_map, nrow=args.batch_size)[0, :, :]
		spixel_viz, _ = get_spixel_image(input_l_save, spixel_lab_save)

		label_save = make_grid(args.label_factor * full_label)

		val_writer.add_image('Input', input_l_save, epoch)
		val_writer.add_image('label', label_save, epoch)
		val_writer.add_image('Spixel viz', spixel_viz, epoch)

		# --- for debug
		#	  _, prob_idx = torch.max(assign, dim=1, keepdim=True)
		#	  prob_map_save = make_grid(assign2uint8(prob_idx))
		#	  val_writer.add_image('assigment idx level %d' % j, prob_map_save, epoch)

		print('==> write val step %dth to tensorboard' % i)

	return total_loss.avg, losses_sem.avg


def save_checkpoint(state, is_best, filename='checkpoint.tar'):
	torch.save(state, os.path.join(save_path,filename))
	if is_best:
		shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.tar'))

if __name__ == '__main__':
	main()
