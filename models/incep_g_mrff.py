from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from operator import add

__all__ = ['IncepGMRFF']


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization + relu.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Inception_base(nn.Module):
    def __init__(self, depth_dim, input_size, config):
        super(Inception_base, self).__init__()
        self.depth_dim = depth_dim

        self.conv1      = nn.Sequential( nn.Conv2d(in_channels=input_size, out_channels=config[0][0], kernel_size=1, stride=1, padding=0),
                          nn.BatchNorm2d(num_features = config[0][0]), nn.ReLU() )
        self.conv3_1    = nn.Sequential( nn.Conv2d(in_channels=input_size, out_channels=config[1][0], kernel_size=1, stride=1, padding=0),
                          nn.BatchNorm2d(num_features = config[1][0]), nn.ReLU() )
        self.conv3_3    = nn.Sequential( nn.Conv2d(in_channels=config[1][0], out_channels=config[1][1], kernel_size=3, stride=1, padding=1),
                          nn.BatchNorm2d(num_features = config[1][1]), nn.ReLU() )
        self.conv5_1    = nn.Sequential( nn.Conv2d(in_channels=input_size, out_channels=config[2][0], kernel_size=1, stride=1, padding=0),
                          nn.BatchNorm2d(num_features = config[2][0]), nn.ReLU() )
        self.conv5_5    = nn.Sequential( nn.Conv2d(in_channels=config[2][0], out_channels=config[2][1],kernel_size=5, stride=1, padding=2),
                          nn.BatchNorm2d(num_features = config[2][1]), nn.ReLU() )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=config[3][0], stride=1, padding=1)
        self.conv_max_1 = nn.Sequential( nn.Conv2d(in_channels=input_size, out_channels=config[3][1], kernel_size=1, stride=1, padding=0),
                     nn.BatchNorm2d(num_features = config[3][1]), nn.ReLU() )

    def forward(self, input):
        output1 = self.conv1(input)
        output2 = self.conv3_1(input)
        output2 = self.conv3_3(output2)
        output3 = self.conv5_1(input)
        output3 = self.conv5_5(output3)
        output4 = self.conv_max_1(self.max_pool_1(input))
        return torch.cat([output1, output2, output3, output4], dim=self.depth_dim)

class Conv1(nn.Module):
    def __init__(self):
        super(Conv1, self).__init__()
        self.conv1      = nn.Sequential( nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                                         nn.BatchNorm2d(num_features = 64), nn.ReLU() )
        self.max_pool1  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2      = nn.Sequential( nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                                         nn.BatchNorm2d(num_features = 64), nn.ReLU() )
        self.conv3      = nn.Sequential( nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(num_features = 192), nn.ReLU() )
        self.max_pool3  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, input):

        output = self.max_pool1(self.conv1(input))
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.max_pool3(output)

        return output


class Inception1(nn.Module):
    def __init__(self):
        super(Inception1, self).__init__()
        self.inception_3a = Inception_base(1, 192, [[64], [96, 128], [16, 32], [3, 32]])  # 3a
        self.inception_3b = Inception_base(1, 256, [[128], [128, 192], [32, 96], [3, 64]])  # 3b
        self.max_pool_inc3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

    def forward(self, input):

        output = self.inception_3a(input)
        output = self.inception_3b(output)
        output = self.max_pool_inc3(output)

        return output


class Inception2(nn.Module):
    def __init__(self):
        super(Inception2, self).__init__()
        self.inception_4a = Inception_base(1, 480, [[192], [96, 204], [16, 48], [3, 64]])  # 4a
        self.inception_4b = Inception_base(1, 508, [[160], [112, 224], [24, 64], [3, 64]])  # 4b
        self.inception_4c = Inception_base(1, 512, [[128], [128, 256], [24, 64], [3, 64]])  # 4c
        self.inception_4d = Inception_base(1, 512, [[112], [144, 288], [32, 64], [3, 64]])  # 4d
        self.inception_4e = Inception_base(1, 528, [[256], [160, 320], [32, 128], [3, 128]])  # 4e
        self.max_pool_inc4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        output = self.inception_4a(input)
        output = self.inception_4b(output)
        output = self.inception_4c(output)
        output = self.inception_4d(output)
        output = self.inception_4e(output)
        output = self.max_pool_inc4(output)

        return output


class Inception3(nn.Module):
    def __init__(self):
        super(Inception3, self).__init__()
        self.inception_5a = Inception_base(1, 832, [[256], [160, 320], [48, 128], [3, 128]])  # 5a
        self.inception_5b = Inception_base(1, 832, [[384], [192, 384], [48, 128], [3, 128]])  # 5b

    def forward(self, input):
        output = self.inception_5a(input)
        output = self.inception_5b(output)

        return output


class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""

    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # bilinear resizing
        x = F.upsample(x, (x.size(2) * 2, x.size(3) * 2), mode='bilinear', align_corners=True)
        # scaling conv
        x = self.conv2(x)
        return x


class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""

    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)
    Aim: Spatial Attention + Channel Attention
    Output: attention maps with shape identical to input.
    """

    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = F.sigmoid(self.conv(y))
        return y


class LSTMAttn1(nn.Module):
    def __init__(self, in_channels):
        super(LSTMAttn1, self).__init__()
        self.num_layers = 1
        self.hidden_size = in_channels
        # self.norm = nn.BatchNorm2d(in_channels)
        # self.relu = nn.ReLU(inplace=False)
        self.lstm = nn.LSTM(in_channels, hidden_size=in_channels, num_layers=self.num_layers, batch_first=True,
                            bidirectional=True)
        self.conv = ConvBlock(in_channels * 2, in_channels, 1)

    def forward(self, x):
        [batch, channel, row, col] = x.size()
        # x = self.norm(x)
        # x = self.relu(x)
        # x = self.conv(x)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        x = x.view(batch, channel, -1)
        h_out, (h_t, c_t) = self.lstm(x.transpose(1, 2), (h0, c0))  # NxSeqx(2*Feature)
        h_out = h_out.transpose(1, 2).view(batch, -1, row, col)
        y = F.sigmoid(self.conv(h_out))
        return y


class HardAttn(nn.Module):
    """Hard Attention (Sec. 3.1.II)"""

    def __init__(self, in_channels):
        super(HardAttn, self).__init__()
        self.fc = nn.Linear(in_channels, 4 * 2)
        self.init_params()

    def init_params(self):
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([0, -0.75, 0, -0.25, 0, 0.25, 0, 0.75], dtype=torch.float))

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1))
        # predict transformation parameters
        theta = F.tanh(self.fc(x))
        theta = theta.view(-1, 4, 2)
        return theta


class HarmAttn(nn.Module):
    """Harmonious Attention (Sec. 3.1)"""

    def __init__(self, in_channels):
        super(HarmAttn, self).__init__()
        self.soft_attn = SoftAttn(in_channels)
        self.hard_attn = HardAttn(in_channels)

    def forward(self, x):
        y_soft_attn = self.soft_attn(x)
        theta = self.hard_attn(x)
        return y_soft_attn, theta


class _dim_change(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_dim_change, self).__init__()
        # self.add_module('norm', nn.BatchNorm2d(in_channels))
        # self.add_module('relu1', nn.ReLU())
        # self.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
        #                                   kernel_size=1, stride=1, bias=True))
        self.add_module('conv', ConvBlock(in_channels, out_channels, k=1, s=1))
        # self.add_module('dropout', nn.Dropout(p=0.5))
        # self.add_module('relu2', nn.ReLU())


class _part_detector(nn.Module):
    def __init__(self, in_channels, num_parts, sample_height, sample_width, factor_of_scale_factors=1):
        super(_part_detector, self).__init__()
        self.num_parts = num_parts
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.factor_of_scale_factors = factor_of_scale_factors
        # self.norm = nn.BatchNorm2d(in_channels)
        # self.relu = nn.ReLU()
        self.fc = nn.Linear(in_channels, 2 * num_parts, bias=True)
        self.init_params(num_parts)

    def init_params(self, num_parts):
        bias_x = torch.zeros(1, num_parts).view(-1)
        bias_y_temp = torch.linspace(start=-1, end=1, steps=2 * num_parts + 1)
        idx = torch.arange(1, num_parts + 1) * 2 - 1
        idx = idx.long()
        bias_y = bias_y_temp[idx]
        # scale_x = torch.ones(num_parts) * 0.5
        # scale_y = torch.ones(num_parts) * 0.25
        # trans_param = torch.stack([scale_x, scale_y, bias_x, bias_y], dim=0)
        trans_param = torch.stack([bias_x, bias_y], dim=0)
        trans_param = trans_param.view(-1)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(trans_param)

    def forward(self, x):
        # squeeze operation (global average pooling)
        # x = self.norm(x)
        # x = self.relu(x)
        x_pool = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1))
        # predict transformation parameters
        trans_param_xy = self.fc(x_pool).view(-1, 2, self.num_parts)  # Nx2xParts
        trans_param_xy = F.tanh(trans_param_xy)
        trans_param_scale_x = torch.ones([trans_param_xy.size(0), 1, trans_param_xy.size(2)],
                                         dtype=trans_param_xy.dtype, device=trans_param_xy.device) \
                              * 1.0 * self.factor_of_scale_factors
        trans_param_scale_y = torch.ones([trans_param_xy.size(0), 1, trans_param_xy.size(2)],
                                         dtype=trans_param_xy.dtype, device=trans_param_xy.device) \
                              * 0.25 * self.factor_of_scale_factors
        trans_param = torch.cat([trans_param_scale_x, trans_param_scale_y, trans_param_xy], dim=1)
        theta_t = trans_param[:, -2:, :]  # Nx2xParts
        # theta_t = theta_t.view(-1, 2, self.num_parts) #Nx2xParts
        # x_sample_out_list = []
        theta_list = []
        # scale_factors = torch.tensor([[0.5, 0], [0, 0.25]], dtype=x.dtype,
        #                              device=x.device) * self.factor_of_scale_factors
        scale_factors = trans_param[:, :2, :]  # Nx2xParts
        for i in range(self.num_parts):
            theta = torch.zeros(theta_t.size(0), 2, 3, dtype=theta_t.dtype, device=theta_t.device)
            theta[:, 0, 0] = scale_factors[:, 0, i]
            theta[:, 1, 1] = scale_factors[:, 1, i]
            # theta[:, 0, 0] = scale_factors[0, 0]
            # theta[:, 1, 1] = scale_factors[1, 1]
            theta[:, :, -1] = theta_t[:, :, i]
            theta_list.append(theta)

        return theta_list, trans_param  # x_sample_out_list


class _foregroundAtten(nn.Module):
    def __init__(self, in_channels):
        super(_foregroundAtten, self).__init__()
        # self.norm = nn.BatchNorm2d(in_channels)
        # self.relu = nn.ReLU()
        # self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, stride=2, padding=1, bias=False)
        self.lstmAttn = LSTMAttn1(in_channels)
        # self.sigmoid = nn.Sigmoid()
        # self.spatialAttn = SpatialAttn()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.norm(x)
        # x = self.relu(x)
        # x = self.conv(x)
        x = self.lstmAttn(x)
        # x = self.sigmoid(x)
        # x = F.upsample(x, (x.size(2) * 2, x.size(3) * 2), mode='bilinear', align_corners=True)
        # x = self.spatialAttn(x)
        # x = self.sigmoid(x)
        return x



class BilinearModule(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(BilinearModule, self).__init__()
        midChannels = inChannels // 8

        self.dimReduceLayer1 = _dim_change(inChannels, midChannels)
        self.dimReduceLayer2 = _dim_change(midChannels*(midChannels+1)//2, outChannels)


    def forward(self, x):
        x = self.dimReduceLayer1(x)

        [batches, channels, rows, cols] = x.size()
        # x = x.view(batches, channels, -1)
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, channels)
        x = torch.bmm(x.unsqueeze(2), x.unsqueeze(1)) #BRowColxCxC
        x_list = []
        for i in range(channels):
            x_temp = torch.diagonal(x, offset=-i, dim1=1, dim2=2)
            x_list.append(x_temp)

        x = torch.cat(x_list, dim=1) #BRowColxC'
        x = x.view(batches, rows, cols, -1) #BxRowxColxC'
        x = x.permute(0, 3, 1, 2).contiguous() #BxC'xRowxCol

        output = self.dimReduceLayer2(x)

        # x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.channelSize)
        # x = torch.bmm(x.unsqueeze(2), x.unsqueeze(1))
        # out = x.view(self.batchSize, self.heightSize, self.widthSize, self.channelSize*self.channelSize).permute(0, 3, 1, 2).contiguous()
        # out = self.dimReduceLayer(out)
        return output
        
class LinearAttention(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(LinearAttention, self).__init__()
        midChannels = inChannels // 4


        self.dimReduceLayer1 = _dim_change(inChannels, midChannels)
        self.dimIncreaseLayer1 = _dim_change(midChannels, outChannels)
        self.dimIncreaseLayer2 = _dim_change(midChannels, outChannels)
        #self.dimKeep = _dim_change(1, 1)


    def forward(self, x):
        input_x = x

        x = self.dimReduceLayer1(x)

        mCha = F.avg_pool2d(x, x.size()[2:])
        mCha = self.dimIncreaseLayer1(mCha)

        output = self.dimIncreaseLayer2(x)

        output = output * mCha

        output =  F.sigmoid(output) * input_x

        return output    









class RFFConv(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization + relu.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(RFFConv, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        torch.nn.init.normal_(self.conv.weight)
        torch.nn.init.uniform_(self.conv.bias, a = 0.0, b = 6.28)

        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return torch.cos(self.bn(self.conv(x)))
        #return F.relu(self.bn(self.conv(x)))




class RffAttention(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(RffAttention, self).__init__()
        midChannels = inChannels // 4 ##120
        factor = 4


        self.dimReduceLayer1 = _dim_change(inChannels, midChannels)
        self.rffFeatureLayer = RFFConv(midChannels, midChannels*factor, k=1)
        self.dimIncreaseLayer1 = _dim_change(midChannels*factor, outChannels)
        self.dimIncreaseLayer2 = _dim_change(midChannels*factor, outChannels)


    def forward(self, x):
        input_x = x

        x = self.dimReduceLayer1(x)
        x = self.rffFeatureLayer(x)

        mCha = F.avg_pool2d(x, x.size()[2:])
        mCha = self.dimIncreaseLayer1(mCha)

        output = self.dimIncreaseLayer2(x)

        output = output * mCha

        output =  F.sigmoid(output) * input_x

        return output        









class Features(nn.Module):
    def __init__(self, height=256, width=128, nchannels=(480, 832, 1024), num_focused_parts=4, drop_rate=0.1, factor_of_scale_factors=1,
                 pretrained_dict=None):
        super(Features, self).__init__()


        self.conv = Conv1()
        self.conv = self.load_dict(self.conv, pretrained_dict)

        # ============== Block 1 ==============
        self.inception1 = Inception1()
        self.inception1 = self.load_dict(self.inception1, pretrained_dict)
        self.rffatt_global1 = RffAttention(nchannels[0], nchannels[0])


        # ============== Block 2 ==============
        self.inception2 = Inception2()
        self.inception2 = self.load_dict(self.inception2, pretrained_dict)
        self.rffatt_global2 = RffAttention(nchannels[1], nchannels[1])

        # ============== Block 3 ==============
        self.inception3 = Inception3()
        self.inception3 = self.load_dict(self.inception3, pretrained_dict)
        self.rffatt_global3 = RffAttention(nchannels[2], nchannels[2])


        self.norm1_1 = nn.BatchNorm2d(nchannels[2])
        # self.norm1_2 = nn.BatchNorm2d(nchannels[2] * num_focused_parts)

    def load_dict(self, model, pretrained_dict=None):
        if pretrained_dict is not None:
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            pass
        return model


    def get_part_features(self, theta_list, features_in):
        features_out_list = []
        for theta in theta_list:
            grid = F.affine_grid(theta, size=torch.Size(features_in.size()))
            features_out = F.grid_sample(features_in, grid)  # Nx128x8x8
            features_out_list.append(features_out)
        return features_out_list

    def coord_convert(self, in_loc, rows, cols):
        out_loc = (in_loc + 1) / 2
        out_loc[:, 0, :] = out_loc[:, 0, :] * (cols - 1)
        out_loc[:, 1, :] = out_loc[:, 1, :] * (rows - 1)
        out_loc = torch.round(out_loc).int()  # Nx2xPart

        return out_loc

    def get_g_loc(self, x, trans_param):
        loc = torch.tensor([[-1, 1, 1, -1, 0],
                            [-1, -1, 1, 1, 0]], dtype=trans_param.dtype, device=trans_param.device)
        loc = loc.reshape(1, 2, 5, 1).expand((trans_param.size(0), 2, 5, trans_param.size(2)))
        [rows, cols] = x.size()[-2:]

        g_loc_list = []
        for i in range(loc.size(2)):
            g_loc_temp_x = trans_param[:, 0, :] * loc[:, 0, i, :] + trans_param[:, 2, :]  # NxPart
            g_loc_temp_y = trans_param[:, 1, :] * loc[:, 1, i, :] + trans_param[:, 3, :]  # NxPart
            g_loc_temp = torch.cat([g_loc_temp_x.unsqueeze(1), g_loc_temp_y.unsqueeze(1)], dim=1)  # Nx2xPart
            g_loc_temp = self.coord_convert(g_loc_temp, rows, cols)
            g_loc_list.append(g_loc_temp.unsqueeze(2))  # Nx2x1xPart

        g_loc = torch.cat(g_loc_list, dim=2)  # Nx2x5xPart
        return g_loc

    def forward(self, x):

        x0 = self.conv(x)
        x1 = self.inception1(x0)
        x1 = self.rffatt_global1(x1)
        x2 = self.inception2(x1)
        x2 = self.rffatt_global2(x2)
        x3 = self.inception3(x2)
        x3 = self.rffatt_global3(x3)

        features_global = x3  # Nx384x16x16

        features_global = self.norm1_1(features_global)
        

        return features_global






class IncepGMRFF(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, height=256, width=128, feat_dim=512,
                 num_focused_parts=4, factor_of_scale_factors=1, drop_rate=0.1,
                 use_gpu=True, pretrained_dict=None, **kwargs):
        super(IncepGMRFF, self).__init__()
        self.loss = loss
        self.feat_dim = feat_dim
        self.num_parts = num_focused_parts

        self.nchannels = (480, 832, 1024)

        self.features = Features(height=height, width=width, nchannels=self.nchannels,
                                 num_focused_parts=num_focused_parts,
                                 drop_rate=drop_rate,
                                 factor_of_scale_factors=factor_of_scale_factors,
                                 pretrained_dict = pretrained_dict)

        self.fc_global = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.nchannels[2], feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )

        self.fc_local = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.nchannels[2] * num_focused_parts, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )

        self.classifier_global = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes))

        self.classifier_local = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes))


    def forward(self, x):
        features_global = self.features(x)  # Nx384x16x16 Nx(384*Parts)x16x16

        features_global = F.relu(features_global, inplace=True)
        f_global = F.avg_pool2d(features_global, features_global.size()[2:]).view(features_global.size(0), -1)
        #f_local = F.avg_pool2d(features_local, features_local.size()[2:]).view(features_local.size(0), -1)

        f_global = self.fc_global(f_global)
        #f_local = self.fc_local(f_local)

        if not self.training:
            f_global = f_global / f_global.norm(p=2, dim=1, keepdim=True)
            #f_local = f_local / f_local.norm(p=2, dim=1, keepdim=True)
            #f = torch.cat([f_global, f_local], 1)
            f = f_global
            return f

        y_global = self.classifier_global(f_global)
        #y_local = self.classifier_local(f_local)

        if self.loss == {'xent'}:
            return y_global, y_local
        elif self.loss == {'xent', 'htri'}:
            return (y_global), (f_global)
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
