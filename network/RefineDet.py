import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_utils import L2Norm


def vgg_layer_init(vgg_type='300', in_channels=3, batch_norm=True):
    vgg_cfg = {
        '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                512, 512, 512],
        '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                512, 512, 512],
    }

    vgg_list = []
    for layer in vgg_cfg[vgg_type]:
        if layer == 'M':
            vgg_list += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif layer == 'C':
            vgg_list += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        elif batch_norm:
            vgg_list += [nn.Conv2d(in_channels, layer, kernel_size=3, padding=1, stride=1)]
            vgg_list += [nn.BatchNorm2d(layer)]
            vgg_list += [nn.ReLU(inplace=True)]
            in_channels = layer
        else:
            vgg_list += [nn.Conv2d(in_channels, layer, kernel_size=3, padding=1, stride=1)]
            vgg_list += [nn.ReLU(inplace=True)]
            in_channels = layer

    vgg_list += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
    vgg_list += [nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)]
    vgg_list += [nn.ReLU(inplace=True)]
    vgg_list += [nn.Conv2d(1024, 1024, kernel_size=1)]
    vgg_list += [nn.ReLU(inplace=True)]

    return vgg_list


class RefineDetArm(nn.Module):

    def __init__(self, vgg_type, in_channels, batch_norm=True):
        super(RefineDetArm, self).__init__()

        self.vgg_type = vgg_type
        self.in_channels = in_channels
        self.batch_norm = batch_norm

        self.vgg_list = nn.ModuleList(
            vgg_layer_init(in_channels=self.in_channels, batch_norm=False)
        )

        self.extras = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.arm_sources = []
        self.arm_confidences = []
        self.arm_locations = []
        self.arm_conf = None
        self.arm_loc = None

        self.arm_location_layers = nn.ModuleList(
            [
                nn.Conv2d(512, 12, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 12, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(1024, 12, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 12, kernel_size=3, stride=1, padding=1),
            ]
        )
        self.arm_confidence_layers = nn.ModuleList(
            [
                nn.Conv2d(512, 6, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 6, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(1024, 6, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 6, kernel_size=3, stride=1, padding=1),
            ]
        )

    def forward(self, x):
        self.arm_sources.clear()
        self.arm_locations.clear()
        self.arm_confidences.clear()

        for layer in self.vgg_list[: 23]:
            x = layer(x)
        self.arm_sources.append(x)

        for layer in self.vgg_list[23: 30]:
            x = layer(x)
        self.arm_sources.append(x)

        for layer in self.vgg_list[30:]:
            x = layer(x)
        self.arm_sources.append(x)

        x = self.extras(x)
        self.arm_sources.append(x)

        for arm_source, arm_confidence_layer, arm_location_layer in zip(self.arm_sources, self.arm_confidence_layers,
                                                                        self.arm_location_layers):
            self.arm_confidences.append(arm_confidence_layer(arm_source).permute(0, 2, 3, 1).contiguous())
            self.arm_locations.append(arm_location_layer(arm_source).permute(0, 2, 3, 1).contiguous())
        # why use transpose and reshape?
        arm_conf = torch.cat([item.view(item.size(0), -1) for item in self.arm_confidences], 1)
        arm_loc = torch.cat([item.view(item.size(0), -1) for item in self.arm_locations], 1)

        return x, self.arm_sources, arm_conf, arm_loc


class RefineDetObm(nn.Module):

    def __init__(self, num_classes, batch_norm=True):
        super(RefineDetObm, self).__init__()

        self.last_layer_trans = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        )

        self.L2Norm_4_3 = L2Norm(512, 10)
        self.L2Norm_5_3 = L2Norm(512, 8)

        self.obm_sources = []
        self.obm_confidences = []
        self.obm_locations = []
        self.transfer_list = []

        self.arm_sources = None
        self.arm_conf = None
        self.arm_loc = None
        self.obm_conf = None
        self.obm_loc = None
        self.trans_result = None

        self.num_classes = num_classes

        self.obm_location_layers = nn.ModuleList(
            [
                nn.Conv2d(256, 12, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 12, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 12, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 12, kernel_size=3, stride=1, padding=1),
            ]
        )
        self.obm_confidence_layers = nn.ModuleList(
            [
                nn.Conv2d(256, 3 * self.num_classes, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 3 * self.num_classes, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 3 * self.num_classes, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 3 * self.num_classes, kernel_size=3, stride=1, padding=1),
            ]
        )
        self.transfer_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                ),
                nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                ),
            ]
        )
        self.upconv_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
            ]
        )
        self.latent_layers = nn.ModuleList(
            [
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            ]
        )
        self.softmax = nn.Softmax()

    # if "is_propagate" is set to True, then "previous_layer" is needed.
    # currently simply use previous feature maps
    def forward(self, x, arm_sources, arm_conf, arm_loc, is_training=True):

        self.obm_sources.clear()
        self.obm_confidences.clear()
        self.obm_locations.clear()
        self.transfer_list.clear()

        self.arm_sources = arm_sources
        self.arm_conf = arm_conf
        self.arm_loc = arm_loc

        x = self.last_layer_trans(x)
        self.obm_sources.append(x)

        for arm_source, transfer_layer in zip(self.arm_sources, self.transfer_layers):
            self.transfer_list.append(transfer_layer(arm_source))

        self.transfer_list.reverse()
        self.arm_sources.reverse()

        for transfer_item, upconv_item, latent_item in zip(self.transfer_list, self.upconv_layers, self.latent_layers):
            x = F.relu(latent_item(F.relu(upconv_item(x) + transfer_item, inplace=True)), inplace=True)
            self.obm_sources.append(x)

        for obm_source, obm_confidence_layer, obm_location_layer in zip(self.obm_sources, self.obm_confidence_layers, self.obm_location_layers):
            self.obm_confidences.append(obm_confidence_layer(obm_source).permute(0, 2, 3, 1).contiguous())
            self.obm_locations.append(obm_location_layer(obm_source).permute(0, 2, 3, 1).contiguous())
        self.obm_conf = torch.cat([item.view(item.size(0), -1) for item in self.obm_confidences], 1)
        self.obm_loc = torch.cat([item.view(item.size(0), -1) for item in self.obm_locations], 1)

        output = (
            self.arm_sources[-1],
            self.softmax(self.arm_conf.view(self.arm_conf.size(0), -1, 2)),
            self.arm_loc.view(self.arm_loc.size(0), -1, 4),
            self.softmax(self.obm_conf.view(self.obm_conf.size(0), -1, self.num_classes)),
            self.obm_loc.view(self.obm_loc.size(0), -1, 4),
            #None, # updating mask
        )

        return output

