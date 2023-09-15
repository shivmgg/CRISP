#builder
import math

import torch
import torch.nn as nn

from utils.options import args
import utils.conv_type
# from devkit.sparse_ops import SparseConv

class Builder(object):
    def __init__(self, conv_layer, bn_layer, N=2, M=4, first_layer=None):
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer
        self.first_layer = first_layer or conv_layer

    def conv(self, kernel_size, in_planes, out_planes, stride=1, first_layer=False, bias=False):
        conv_layer = self.first_layer if first_layer else self.conv_layer

        if first_layer:
            print(f"==> Building first layer with {args.first_layer_type}")

        if kernel_size == 3:
            conv = conv_layer(
                args.N, 
                args.M,
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias,
            )
        elif kernel_size == 1:
            conv = conv_layer(
                args.N, 
                args.M,
                in_planes, out_planes, kernel_size=1, stride=stride, bias=bias
            )
        elif kernel_size == 5:
            conv = conv_layer(
                args.N, 
                args.M,
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=bias,
            )
        elif kernel_size == 7:
            conv = conv_layer(
                args.N, 
                args.M,
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=bias,
            )
        else:
            return None

        #self._init_conv(conv)

        return conv

    def conv2d(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        return self.conv_layer(
                args.N, 
                args.M,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

    def conv3x3(self, in_planes, out_planes, stride=1, first_layer=False, bias=False):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride, first_layer=first_layer, bias=bias)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1, first_layer=False):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c
    
    def conv1x1_fc(self, in_planes, out_planes, stride=1, first_layer=False):
        """full connect layer"""
        c = self.conv(1, in_planes, out_planes, stride=stride, first_layer=first_layer, bias=False)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1, first_layer=False):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1, first_layer=False):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def batchnorm(self, planes, last_bn=False, first_layer=False):
        return self.bn_layer(planes)

    def activation(self):
        if args.nonlinearity == "relu":
            return (lambda: nn.ReLU(inplace=True))()
        else:
            raise ValueError(f"{args.nonlinearity} is not an initialization option!")

    def _init_conv(self, conv):
        if args.init == "signed_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
            if args.scale_fan:
                fan = fan * (1 - args.prune_rate)
            gain = nn.init.calculate_gain(args.nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = conv.weight.data.sign() * std

        elif args.init == "unsigned_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
            if args.scale_fan:
                fan = fan * (1 - args.prune_rate)

            gain = nn.init.calculate_gain(args.nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = torch.ones_like(conv.weight.data) * std

        elif args.init == "kaiming_normal":

            if args.scale_fan:
                fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
                fan = fan * (1 - args.prune_rate)
                gain = nn.init.calculate_gain(args.nonlinearity)
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    conv.weight.data.normal_(0, std)
            else:
                nn.init.kaiming_normal_(
                    conv.weight, mode=args.mode, nonlinearity=args.nonlinearity
                )

        elif args.init == "standard":
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
        else:
            raise ValueError(f"{args.init} is not an initialization option!")
        
        if conv.bias.data is not None:
            conv.data.zero_()
            import pdb
            pdb.set_trace()

def get_builder():

    print("==> Conv Type: {}".format(args.conv_type))
    print("==> BN Type: {}".format(args.bn_type))

    conv_layer = getattr(utils.conv_type, args.conv_type)
    bn_layer = getattr(utils.conv_type, args.bn_type)

    if args.first_layer_type is not None:
        first_layer = getattr(utils.conv_type, args.first_layer_type)
        print(f"==> First Layer Type {args.first_layer_type}")
    else:
        first_layer = None

    builder = Builder(conv_layer=conv_layer, bn_layer=bn_layer, first_layer=first_layer)

    return builder
