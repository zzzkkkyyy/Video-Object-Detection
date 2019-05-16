import torch
import torch.nn as nn
import numpy as np
from torch.nn import init


def conv(in_channels, out_channels, kernel_size=3, stride=1, is_training=True):
    if is_training:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )


class FlowNetS(nn.Module):
    def __init__(self, in_channels=6, is_training=True):
        super(FlowNetS, self).__init__()

        self.is_training = is_training
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=(not is_training))
        self.batch_norm_1 = nn.BatchNorm2d(64)
        self.leaky_relu_1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=(not is_training))
        self.batch_norm_2 = nn.BatchNorm2d(128)
        self.leaky_relu_2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=(not is_training))
        self.batch_norm_3 = nn.BatchNorm2d(256)
        self.leaky_relu_3 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=(not is_training))
        self.batch_norm_3_1 = nn.BatchNorm2d(256)
        self.leaky_relu_3_1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=(not is_training))
        self.batch_norm_4 = nn.BatchNorm2d(512)
        self.leaky_relu_4 = nn.LeakyReLU(0.1, inplace=True)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=(not is_training))
        self.batch_norm_4_1 = nn.BatchNorm2d(512)
        self.leaky_relu_4_1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=(not is_training))
        self.batch_norm_5 = nn.BatchNorm2d(512)
        self.leaky_relu_5 = nn.LeakyReLU(0.1, inplace=True)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=(not is_training))
        self.batch_norm_5_1 = nn.BatchNorm2d(512)
        self.leaky_relu_5_1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=(not is_training))
        self.batch_norm_6 = nn.BatchNorm2d(1024)
        self.leaky_relu_6 = nn.LeakyReLU(0.1, inplace=True)
        self.conv6_1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=(not is_training))
        self.batch_norm_6_1 = nn.BatchNorm2d(1024)
        self.leaky_relu_6_1 = nn.LeakyReLU(0.1, inplace=True)

        self.deconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=True)
        self.deconv4 = nn.ConvTranspose2d(1026, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.deconv3 = nn.ConvTranspose2d(770, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.deconv2 = nn.ConvTranspose2d(386, 64, kernel_size=4, stride=2, padding=1, bias=True)

        self.predict_flow6 = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.predict_flow5 = nn.Conv2d(1026, 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.predict_flow4 = nn.Conv2d(770, 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.predict_flow3 = nn.Conv2d(386, 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.predict_flow2 = nn.Conv2d(194, 2, kernel_size=3, stride=1, padding=1, bias=True)

        self.upsampled_flow6_5 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow5_4 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow4_3 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow3_2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

        self.q_propagate = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.upsampled_q6_1 = nn.Upsample(scale_factor=64, mode='bilinear')
        self.upsampled_q5_1 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.upsampled_q4_1 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsampled_q3_1 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsampled_q2_1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsampled_flow2_1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        #input = nn.functional.interpolate(x, size=(256, 256), mode='bilinear')

        out_conv1 = self.conv1(x)
        if self.is_training:
            out_conv1 = self.leaky_relu_1(self.batch_norm_1(out_conv1))
        else:
            out_conv1 = self.leaky_relu_1(out_conv1)

        out_conv2 = self.conv2(out_conv1)
        if self.is_training:
            out_conv2 = self.leaky_relu_2(self.batch_norm_2(out_conv2))
        else:
            out_conv2 = self.leaky_relu_2(out_conv2)

        out_conv3 = self.conv3(out_conv2)
        if self.is_training:
            out_conv3 = self.leaky_relu_3(self.batch_norm_3(out_conv3))
        else:
            out_conv3 = self.leaky_relu_3(out_conv3)

        out_conv3 = self.conv3_1(out_conv3)
        if self.is_training:
            out_conv3 = self.leaky_relu_3_1(self.batch_norm_3_1(out_conv3))
        else:
            out_conv3 = self.leaky_relu_3_1(out_conv3)

        out_conv4 = self.conv4(out_conv3)
        if self.is_training:
            out_conv4 = self.leaky_relu_4(self.batch_norm_4(out_conv4))
        else:
            out_conv4 = self.leaky_relu_4(out_conv4)

        out_conv4 = self.conv4_1(out_conv4)
        if self.is_training:
            out_conv4 = self.leaky_relu_4_1(self.batch_norm_4_1(out_conv4))
        else:
            out_conv4 = self.leaky_relu_4_1(out_conv4)

        out_conv5 = self.conv5(out_conv4)
        if self.is_training:
            out_conv5 = self.leaky_relu_5(self.batch_norm_5(out_conv5))
        else:
            out_conv5 = self.leaky_relu_5(out_conv5)

        out_conv5 = self.conv5_1(out_conv5)
        if self.is_training:
            out_conv5 = self.leaky_relu_5_1(self.batch_norm_5_1(out_conv5))
        else:
            out_conv5 = self.leaky_relu_5_1(out_conv5)

        out_conv6 = self.conv6(out_conv5)
        if self.is_training:
            out_conv6 = self.leaky_relu_6(self.batch_norm_6(out_conv6))
        else:
            out_conv6 = self.leaky_relu_6(out_conv6)

        out_conv6 = self.conv6_1(out_conv6)
        if self.is_training:
            out_conv6 = self.leaky_relu_6_1(self.batch_norm_6_1(out_conv6))
        else:
            out_conv6 = self.leaky_relu_6_1(out_conv6)


        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        q_propagate6 = self.upsampled_q6_1(self.q_propagate(flow6))

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        q_propagate5 = self.upsampled_q5_1(self.q_propagate(flow5))

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        q_propagate4 = self.upsampled_q4_1(self.q_propagate(flow4))

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        q_propagate3 = self.upsampled_q3_1(self.q_propagate(flow3))

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        q_propagate2 = self.upsampled_q2_1(self.q_propagate(flow2))

        q_propagate = torch.cat((q_propagate2 * 4, q_propagate3 * 8, q_propagate4 * 16, q_propagate5 * 32, q_propagate6 * 64), 1)
        q_propagate = torch.mean(torch.abs(q_propagate), dim=3, keepdim=False)
        q_propagate = torch.mean(torch.abs(q_propagate), dim=2, keepdim=False)
        q_propagate = torch.mean(torch.abs(q_propagate), dim=1, keepdim=False)

        flow_result = self.upsampled_flow2_1(flow2)

        #flow_result = nn.functional.interpolate(flow_result, size=list(x.size())[2: ], mode='bilinear')

        if self.training:
            return flow_result, [flow2, flow3, flow4, flow5, flow6], q_propagate
        else:
            return flow_result, q_propagate
