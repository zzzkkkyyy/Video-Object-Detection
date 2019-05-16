import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torch.nn import init
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np

import visdom
import logging

import os, sys, time, io, datetime
import random
import argparse

from network.RefineDet import RefineDetArm, RefineDetObm
from network.FlowNet import FlowNetS
from network.network_utils import EmbeddingNetwork, L2Norm
from network.MultiBoxLoss import RefineMultiBoxLoss
from dataset.bosch import BoschTrainDetection, BoschTestDetection, detection_collate
from dataset.data_preprocessing import TrainAugmentation, TestTransform
from network.prior_box import PriorBox

def reindex_tensor(input_tensor, input_index):
     if isinstance(input_index, list):
         input_index = torch.Tensor(input_index)
     elif isinstance(input_index, np.array):
         input_index = torch.from_numpy(input_index)
     input_index = input_index.long()
     per_batch_length = np.prod(np.array(input_tensor.size())[1: ])
     expand_index = input_index.unsqueeze(-1).repeat(1, per_batch_length).view(input_tensor.size())
     return torch.gather(input_tensor, 0, expand_index)

def reindex(array, index):
    return [array[item] for item in index]

def str2bool(v):
    return v.lower() in ("yes", "true", "1", "t")

"""
def cosine_similarity(preceed, current):
    preceed_vector = EmbeddingNetwork(preceed)
    current_vector = EmbeddingNetwork(current)
    preceed_vector_sum_sqrt = torch.sqrt(torch.sum(torch.pow(preceed_vector, 2), 1))
    current_vector_sum_sqrt = torch.sqrt(torch.sum(torch.pow(current_vector, 2), 1))

    return torch.exp(preceed_vector * current_vector / (preceed_vector_sum_sqrt * current_vector_sum_sqrt))
"""

# input: [batch_size, channels, height, width], gather_index: [batch_index, 2, height, width]
def gather_nd(input, gather_index):
    input.cuda()
    gather_index.cuda()
    base_index_x, base_index_y = torch.meshgrid([torch.arange(input.size()[2]), torch.arange(input.size()[3])])
    base_index = torch.stack([base_index_x, base_index_y], -1).view(input.size()[2], input.size()[3], 2)
    base_index = torch.stack([base_index for _ in range(input.size()[0])]).double()

    input = input.permute(0, 2, 3, 1).contiguous().double()
    gather_index = gather_index.permute(0, 2, 3, 1).contiguous().double()
    gather_index = base_index + gather_index
    gather_index = gather_index.view(-1, 2).double()
    clamp_gather_index = torch.DoubleTensor(gather_index.size()).cuda()
    clamp_gather_index[:, 0] = torch.clamp(gather_index[:, 0], 0., float(input.size()[1] - 1)).double()
    clamp_gather_index[:, 1] = torch.clamp(gather_index[:, 1], 0., float(input.size()[2] - 1)).double()
    gather_index_ceil = torch.ceil(clamp_gather_index).double()
    gather_index_floor = torch.floor(clamp_gather_index).double()

    output = []
    for i in range(gather_index.size()[0]):
        batch_index = i // (input.size()[1] * input.size()[1])

        cor_x, cor_y = clamp_gather_index[i][0], clamp_gather_index[i][1]
        cor_x_ceil, cor_y_ceil = gather_index_ceil[i][0], gather_index_ceil[i][1]
        cor_x_floor, cor_y_floor = gather_index_floor[i][0], gather_index_floor[i][1]
        weight_ceil_x, weight_ceil_y = cor_x - cor_x_floor, cor_y - cor_y_floor
        weight_floor_x, weight_floor_y = cor_x_ceil - cor_x, cor_y_ceil - cor_y

        output_ceil = input[batch_index, cor_x_ceil.int(), cor_y_ceil.int()]
        output_floor = input[batch_index, cor_x_floor.int(), cor_y_floor.int()]
        output_y_ceil = weight_ceil_x * input[batch_index, cor_x_ceil.int(), cor_y_ceil.int()] + weight_floor_x * input[batch_index, cor_x_floor.int(), cor_y_ceil.int()]
        output_y_floor = weight_ceil_x * input[batch_index, cor_x_ceil.int(), cor_y_floor.int()] + weight_floor_x * input[batch_index, cor_x_floor.int(), cor_y_floor.int()]
        output.append(weight_ceil_y * output_y_ceil + weight_floor_y * output_y_floor)

    result = torch.stack(output, 0).view(tuple(input.size())).permute(0, 3, 1, 2).contiguous().float()

    return result

# TODO: set batch_size=1 may cause error from cosine similarity part, check unsqueeze afterwards
parser = argparse.ArgumentParser()

parser.add_argument('-v', '--version', default='RefineDet', help='feature network')
parser.add_argument('-s', '--size', default=320, help='320 or 512 input size')
parser.add_argument('-d', '--dataset', default='Bosch', help='Cityscapes, ImageNet VID, Bosch or Sensetime')
parser.add_argument('-b', '--batch_size', default=4, type=int, help='batch size')

parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='min jaccard index for matching')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers in dataloading')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda')
parser.add_argument('--gpu_id', default=0, type=int, help='gpu list')
parser.add_argument('--lr', '--learning_rate ', default=1e-4, help='initial learning rate')
parser.add_argument('--base_lr', default=1e-8, help='base feature network learning rate')
parser.add_argument('--flownet_lr', default=1e-8, help='flownet learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--feature_basenet', default="models/vgg16_reducedfc.pth", help='feature network pretrained model')
parser.add_argument('--flow_basenet', default="models/FlowNet2-S_checkpoint.pth.tar", help='flow network pretrained model')
parser.add_argument('--dataset_path', default="/mnt/lustre/zhoukeyang/Bosch_Dataset")

parser.add_argument('--debug_step', default=100, help='debug step')
parser.add_argument('--q_threshold', default=0., help='q threshold')
parser.add_argument('--use_aggr', default=True, help='use aggregation')
parser.add_argument('--use_partial', default=True, help='use partial feature updating')
parser.add_argument('--mask_loss_weight', default=1., type=float, help='update mask loss weight')

parser.add_argument('--resume', default=False, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iteration for retraining')

parser.add_argument('--max_epoch', default=300, type=int, help='max epoch for retraining')
parser.add_argument('--valid_iter', default=1, type=int, help='epoch that print the loss')
parser.add_argument('--save_folder', default='models', type=str, help='location to save checkpoint')
parser.add_argument('--save_iter', default=3, type=int, help='epoch that save the model')
parser.add_argument('--visdom', default=False, help='use visualization')
parser.add_argument('--num_classes', default=14, help='num classes')

parser.add_argument('--is_training', default=True, type=bool, help='training or validating')

args = parser.parse_args()
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

VOC_320 = {
    'feature_maps': [40, 20, 10, 5],
    'min_dim': 320,
    'steps': [8, 16, 32, 64],
    'min_sizes': [32, 64, 128, 256],
    'max_sizes': [],
    'aspect_ratios': [[2], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
}


def test(dataloader, refinedet_arm, refinedet_obm, flownetS, arm_criterion, obm_criterion, device):
    refinedet_arm.eval(True)
    refinedet_obm.eval(True)
    flownetS.eval(True)


def train(dataloader, refinedet_arm, refinedet_obm, flownetS, arm_criterion, obm_criterion, optimizer, device, epoch):
    refinedet_arm.train(True)
    refinedet_obm.train(True)
    flownetS.train(True)

    previous_images, previous_arm_conf, previous_arm_loc = None, None, None
    preceed_out, preceed_featuremap, preceed_conf, preceed_loc, preceed_input = [], [], [], [], []

    running_loss = 0.
    running_arm_regression_loss = 0.
    running_arm_classification_loss = 0.
    running_obm_regression_loss = 0.
    running_obm_classification_loss = 0.
    running_updating_mask_loss = 0.

    for i, data in enumerate(dataloader):
        previous_images, previous_arm_conf, previous_arm_loc = None, None, None
        preceed_out, preceed_featuremap, preceed_conf, preceed_loc, preceed_input = [], [], [], [], []

        corresponding_key = []
        for batch_index in range(args.batch_size):
            corresponding_key.append(random.randint(0, batch_index))
        images, targets = data[0], data[1]
        images = images.to(device)
        #boxes = boxes.to(device)
        #labels = labels.to(device)
        preceed_input = images

        #preceed_images = torch.stack((torch.unbind(preceed_images, 0)[corresponding_key]), 0).to(device)
        #preceed_images = torch.stack(reindex(torch.unbind(preceed_input, 0), corresponding_key), 0).to(device)
        preceed_images = reindex_tensor(preceed_input, corresponding_key)

        optimizer.zero_grad()

        # arm_sources 512, 512, 1024, 512
        out, arm_sources, arm_conf, arm_loc = refinedet_arm(images)
        images_stack = torch.cat((images, preceed_images), 1)
        images_stack = F.interpolate(images_stack, size=(256, 256), mode='bilinear')
        flow_result, flow_list, q_propagate = flownetS(images_stack)

        preceed_out.append(out)
        preceed_featuremap.append(arm_sources)
        preceed_conf = arm_conf
        preceed_loc = arm_loc
        arm_out = out

        updating_mask = torch.zeros(size=(1, ))

        # enforce q_propagate equals to 1 and 0 with 1/3 probability add later...
        if args.use_partial:
            updating_mask = torch.clamp(q_propagate - args.q_threshold + 0.5, 0., 1.)
            prop_condition = 1 - updating_mask
            prop_condition = prop_condition.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            new_arm_sources = []

            for j, arm_source_item in enumerate(arm_sources):
                flow_map = F.interpolate(flow_result, size=tuple(arm_source_item.size())[2:], mode='bilinear') * float(arm_source_item.size()[2] / flow_result.size()[2])
                #flow_map *= float(arm_source_item.size()[2] / flow_result.size()[2])
                new_preceed = gather_nd(reindex_tensor(arm_source_item, corresponding_key), flow_map)
                new_arm_sources.append(prop_condition * new_preceed + (1. - prop_condition) * arm_source_item)
                #new_arm_sources.append(prop_condition * gather_nd(torch.stack(reindex(torch.unbind(arm_source_item, 0), corresponding_key), 0), flow_map) + (1. - prop_condition) * arm_source_item)
            """
            arm_conf_original_shape = arm_conf.size()
            arm_loc_original_shape = arm_loc.size()

            arm_conf_shape = list(flow_result.size())
            arm_conf_shape[1] = 2
            arm_loc_shape = list(flow_result.size())
            arm_loc_shape[1] = 2

            arm_conf = arm_conf.view(arm_conf_shape)
            arm_loc = arm_loc.view(arm_loc_shape)
            arm_conf = prop_condition * gather_nd(torch.stack(reindex(torch.unbind(arm_conf, 0), corresponding_key), 0), flow_result) + (1. - prop_condition) * arm_conf
            arm_loc = prop_condition * gather_nd(torch.stack(reindex(torch.unbind(arm_loc, 0), corresponding_key), 0), flow_result) + (1. - prop_condition) * arm_loc

            arm_conf = arm_conf.view(arm_conf_original_shape)
            arm_loc = arm_loc.view(arm_loc_original_shape)
            """
            arm_out = new_arm_sources[-1]
            arm_sources = new_arm_sources

        # warp operation add later...
        if args.use_aggr:
            def cosine_similarity(preceed, current, Embedding):
                preceed_vector = Embedding(preceed)
                current_vector = Embedding(current)
                preceed_vector_sum_sqrt = torch.sqrt(torch.sum(torch.pow(preceed_vector, 2), -1))
                current_vector_sum_sqrt = torch.sqrt(torch.sum(torch.pow(current_vector, 2), -1))

                return torch.exp(torch.sum(preceed_vector * current_vector, 1) / (preceed_vector_sum_sqrt * current_vector_sum_sqrt)).float()

            preceed = []
            new_arm_sources = []
            new_preceed = []
            for item in arm_sources:
                #preceed.append(torch.stack(reindex(torch.unbind(item, 0), corresponding_key), 0))
                preceed.append(reindex_tensor(item, corresponding_key))
            succeed = arm_sources

            # wrap key frame using flow result
            for j, preceed_map in enumerate(preceed):
                flow_map = F.interpolate(flow_result, size=tuple(preceed_map.size())[2: ], mode='bilinear')
                flow_map *= (preceed_map.size()[2] / flow_result.size()[2])

                propagate_similarity = cosine_similarity(preceed[j].float(), succeed[j].float(), EmbeddingNetwork[j])
                propagate_similarity = propagate_similarity / (1. + propagate_similarity)
                self_similarity = 1. - propagate_similarity
                propagate_similarity, self_similarity = torch.mean(propagate_similarity), torch.mean(self_similarity)

                new_preceed.append(gather_nd(preceed_map, flow_map))
                new_arm_sources.append(self_similarity * succeed[j] + propagate_similarity * preceed[j])
            arm_out = new_arm_sources[-1]
            arm_sources = new_arm_sources

        obm_out = refinedet_obm(arm_out, arm_sources, arm_conf, arm_loc, is_training=True)
        feature_layer, arm_conf, arm_loc, obm_conf, obm_loc = obm_out

        #labels = torch.unsqueeze(labels, -1)
        #arm_targets = torch.cat((boxes, torch.gt(labels, 0).float()), -1)
        #obm_targets = torch.cat((boxes, labels.float()), -1)
        arm_regression_loss, arm_classification_loss = arm_criterion((arm_loc, arm_conf), priors, targets)
        obm_regression_loss, obm_classification_loss = obm_criterion((obm_loc, obm_conf), priors, targets, (arm_loc, arm_conf), False)

        arm_detection_loss = (arm_regression_loss + arm_classification_loss).double()
        obm_detection_loss = (obm_regression_loss + obm_classification_loss).double()
        update_mask_loss = torch.sum(updating_mask).double()
        loss = arm_detection_loss + obm_detection_loss + update_mask_loss

        #loss.backward()
        #optimizer.step()

        running_loss += loss.item()
        running_arm_regression_loss += arm_regression_loss.item()
        running_arm_classification_loss += arm_classification_loss.item()
        running_obm_regression_loss += obm_regression_loss.item()
        running_obm_classification_loss += obm_classification_loss.item()
        running_updating_mask_loss += update_mask_loss.item()

        if i and i % args.debug_step == 0:
            avg_loss = running_loss / args.debug_step
            avg_arm_reg_loss = running_arm_regression_loss / args.debug_step
            avg_arm_clf_loss = running_arm_classification_loss / args.debug_step
            avg_obm_reg_loss = running_obm_regression_loss / args.debug_step
            avg_obm_clf_loss = running_obm_classification_loss / args.debug_step
            avg_update_mask_loss = running_updating_mask_loss / args.debug_step
            print("Epoch: {}, Step: {}, Avg loss: {}, Avg arm loss: {}, Avg obm loss: {}, Update mask loss: {}"
                  .format(epoch, i, avg_loss, avg_arm_reg_loss + avg_arm_clf_loss, avg_obm_reg_loss + avg_obm_clf_loss, avg_update_mask_loss))
            running_loss, running_arm_regression_loss, running_arm_classification_loss, running_obm_regression_loss, running_obm_classification_loss, running_updating_mask_loss = 0., 0., 0., 0.


if __name__ == "__main__":
    save_folder = args.save_folder + "_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    if args.visdom:
        viz = visdom.Visdom()

    cfg = VOC_320
    priorbox = PriorBox(cfg)
    priors = priorbox.forward()
    #priors = Variable(priorbox.forward(), volatile=True)

    train_transform = TrainAugmentation(args.size, np.array([123, 117, 104]), 128.)
    train_dataset = BoschTrainDetection(root_dir=args.dataset_path, \
                                  yaml_file="train.yaml", transform=train_transform, target_transform=False)

    print(len(train_dataset))

    label_file = os.path.join(save_folder, "bosch-dataset-labels.txt")
    with open(label_file, "w") as f:
        f.write("\n".join(train_dataset.class_names))
    num_classes = len(train_dataset.class_names)

    refinedet_arm = RefineDetArm(vgg_type='300', in_channels=3, batch_norm=args.is_training)
    refinedet_obm = RefineDetObm(num_classes=num_classes, batch_norm=args.is_training)
    flownetS = FlowNetS(in_channels=6, is_training=True)

    EmbeddingCosine1 = EmbeddingNetwork(512)
    EmbeddingCosine2 = EmbeddingNetwork(512)
    EmbeddingCosine3 = EmbeddingNetwork(1024)
    EmbeddingCosine4 = EmbeddingNetwork(512)

    EmbeddingNetwork = [EmbeddingCosine1, EmbeddingCosine2, EmbeddingCosine3, EmbeddingCosine4]

    total_net = nn.ModuleList(
        [
            refinedet_arm,
            refinedet_obm,
            flownetS,
        ]
    )

    train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=detection_collate)

    if not args.resume:
        def xavier(param):
            init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0

        refinedet_arm.apply(weights_init)
        refinedet_obm.apply(weights_init)
        flownetS.apply(weights_init)
        for item in EmbeddingNetwork:
            item.apply(weights_init)

        feature_weight = torch.load(args.feature_basenet)
        flow_weight = torch.load(args.flow_basenet)['state_dict']

        refinedet_arm.vgg_list.load_state_dict(feature_weight)

        # select and restore parameters partly
        flownet_dict = {}
        for k, v in flow_weight.items():
            if 'conv' in k:
                new_k = k.split('.')[0] + '.' + k.split('.')[-1]
                if 'deconv' not in k and 'bias' in k:
                    continue
            elif 'upsample' in k:
                new_k = k.split('_')[0] + '_' + k.split('_')[1] + '_' + k.split('_')[3]
            else:
                new_k = k
            flownet_dict[new_k] = v

        flownet_dict['q_propagate.weight'] = flownetS.state_dict()['q_propagate.weight']
        flownet_stat_dict = flownetS.state_dict()
        flownet_stat_dict.update(flownet_dict)
        flownetS.load_state_dict(flownet_stat_dict)

        vgg_pretrained_list = []
        flownet_pretrained_list = []
        random_list = []
        for name, param in list(total_net.named_parameters()):
            if 'vgg_list' in name:
                vgg_pretrained_list.append(param)
            elif 'predict_flow' in name:
                flownet_pretrained_list.append(param)
            else:
                random_list.append(param)

    else:
        resume_path = os.path.join(save_folder, args.resume_epoch + ".pth")
        state_dict = torch.load(resume_path)
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[: 7] == "module.":
                name = k[7: ]
            else:
                name = k
            new_state_dict[name] = v
        total_net.load_state_dict(new_state_dict)

    if args.gpu_id:
        refinedet_arm = torch.nn.DataParallel(refinedet_arm, device_ids=args.gpu_id)
        refinedet_obm = torch.nn.DataParallel(refinedet_obm, device_ids=args.gpu_id)
        flownetS = torch.nn.DataParallel(flownetS, device_ids=args.gpu_id)

    if args.cuda:
        refinedet_arm.cuda()
        refinedet_obm.cuda()
        flownetS.cuda()
        cudnn.benchmark = True

    optimizer = optim.Adam(
        [
            {"params": vgg_pretrained_list, "lr": args.base_lr},
            {"params": flownet_pretrained_list, "lr": args.flownet_lr},
            {"params": random_list}
        ], lr=args.lr)
    arm_criterion = RefineMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5, False)
    obm_criterion = RefineMultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)

    priors = torch.Tensor(priors.astype(np.float32)).cpu()

    for epoch in range(args.resume_epoch, args.max_epoch):
        train(train_dataloader, refinedet_arm, refinedet_obm, flownetS, arm_criterion, obm_criterion, optimizer, DEVICE, epoch)
        torch.save(total_net.state_dict(), os.path.join(save_folder, "Epoch_" + str(epoch) + ".pth"))

