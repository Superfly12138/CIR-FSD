# -*- coding: utf-8 -*-
import torch
from torch import dropout, nn, strided
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
# from torch.nn import Conv2d
import numpy as np
import cv2
from fsdet.layers import (
    Conv2d,
    DeformConv,
    FrozenBatchNorm2d,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)



class AttentionBase(nn.Module):
    def __init__(self,inchannel):
        super(AttentionBase,self).__init__()
        self.stride=1
        self.inchannel = inchannel
        norm="BN"

        self.conv1 = Conv2d(
            in_channels=self.inchannel,
            out_channels=256,
            kernel_size=(3,3),
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, 256),
        )

        self.conv2 = Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, 256),
        )

        self.conv3 = Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, 256),
        )
        self.conv4 = Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, 256),
        )
        self.conv5 = Conv2d(
            in_channels=256,
            out_channels=2,
            kernel_size=(3,3),
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, 2),
        )

       

        for layer in [self.conv1, self.conv2, self.conv3,self.conv4,self.conv5]:
            if layer is not None:  # shortcut can be None
                # weight_init.c2_msra_fill(layer)
                weight_init.c2_xavier_fill(layer)


    def forward(self, x):
        #x:(b,c,w,h) 

        # print(type(x))
        # x2 = torch.randn(x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3],device='cuda:0')
        # print(type(x2))
        out = self.conv1(x)
        # print(out.shape)
        out = F.relu_(out)

        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = F.relu_(out)

        out = self.conv3(out)
        out = F.relu_(out)
        
        out = self.conv4(out)
        out = F.relu_(out)

        out = self.conv5(out)
        out = F.relu_(out)
        # print(out.shape)
        # return out


        # return("调用一次")
        return out
        # return self.layers(x)





class AttentionMultiply(nn.Module):
    def __init__(self):
        super(AttentionMultiply,self).__init__()
        self.m = nn.Softmax(dim=1)


    def forward(self, attention_layer,c4):
        # c4(b,1024,w,h) attention_layer(b,2,w,h)
        # out(b,2,w,h)
        out = self.m(attention_layer)
        # out(b,w,h) 取切片，第一个通道
        out = out[:, 0, :, :]
        # out(b,1,w,h)
        out = torch.unsqueeze(out,1)
        # 按元素相乘
        out = torch.mul(out,c4)
        # out = out * input
        return out

def conver_mask(targets):
# 对batch内的图片，放入数组
    gt_boxes = [x.gt_boxes for x in targets]
    height = [x.image_size[0] for x in targets]
    width = [x.image_size[1] for x in targets]

    # h, w =  targets[0].image_size[0],targets[0].image_size[1]
    # 遍历每个实例
    batch_mask = []
    for i in range(0, len(gt_boxes)):
        # 取第0个图片 shape(n,4) n是实例的个数
        boxes = gt_boxes[i].tensor.cpu().numpy()
        mask = np.zeros([height[i], width[i]])
        for b in boxes:
            b2 = [b[0], b[3]]
            b4 = [b[2], b[1]]
            b = np.insert(b, 2, b2)
            b = np.insert(b, 6, b4)
            b = np.reshape(b, [4, 2])
            rect = np.array(b, np.int32)
            cv2.fillConvexPoly(mask, rect, 1)
        mask = np.array(np.expand_dims(mask, axis=-1), np.float32)
        batch_mask.append(mask)
    return batch_mask

def computer_attention_loss(c4_attention_layer,batch_mask):
    
    c4_attention_loss = 0
    for i in range(len(batch_mask)):

        mask1 = batch_mask[i]
        # c4_attention_layer（b,2,h,w）->(2,h,w)
        c4_attention_layer_m = c4_attention_layer[i,:,:,:]
        c4_attention_layer_m = torch.unsqueeze(c4_attention_layer_m,dim=0)
        c4_attention_layer_m = F.interpolate(c4_attention_layer_m, size=[mask1.shape[0],mask1.shape[1]], mode='bilinear',align_corners=True)
        mask = torch.from_numpy(mask1).cuda()
        # mask.requires_grad_()
        mask = mask.view(-1)
        # c4_attention_layer(2,h,w)
        c4_attention_layer_m = c4_attention_layer_m.view(2,-1)
        
        c4_t = c4_attention_layer_m.t()
        # criterion = nn.CrossEntropyLoss()
    
        # c4_attention_loss += criterion(c4_t,mask.long())
        c4_attention_loss += F.cross_entropy(c4_t,mask.long(),reduction="mean")
        del c4_attention_layer_m
    c4_attention_loss = c4_attention_loss / len(batch_mask)
    return c4_attention_loss
        
   
class DenseASPP(nn.Module):
    def __init__(self,num_features=2048):
        super(DenseASPP,self).__init__()
        dropout0 = 0.1
        d_feature0 = 512
        d_feature1 = 256
        dim_in = num_features
        blob_out = "input"

        # self.stride=1
        # self.inchannel = inchannel
        norm="BN"

        self.aspp3 = DenseAsppBlock(input_num=dim_in,num1=d_feature0,num2=d_feature1,dilation_rate=3,drop_out=dropout0)

        self.aspp6 = DenseAsppBlock(input_num=dim_in + d_feature1 * 1,num1=d_feature0,num2=d_feature1,dilation_rate=6,drop_out=dropout0)

        self.aspp12 = DenseAsppBlock(input_num=dim_in + d_feature1 * 2,num1=d_feature0,num2=d_feature1,dilation_rate=12,drop_out=dropout0)

        self.aspp18 = DenseAsppBlock(input_num=dim_in + d_feature1 * 3,num1=d_feature0,num2=d_feature1,dilation_rate=18,drop_out=dropout0)

        self.aspp24 = DenseAsppBlock(input_num=dim_in + d_feature1 * 4,num1=d_feature0,num2=d_feature1,dilation_rate=24,drop_out=dropout0)

        self.conv1 = Conv2d(
            in_channels=5 * d_feature1,
            out_channels=256,
            kernel_size=(1,1),
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            norm=get_norm(norm, 256),   
        )
       
        for layer in [self.conv1]:
            if layer is not None:  # shortcut can be None
                # weight_init.c2_msra_fill(layer)
                weight_init.c2_xavier_fill(layer)


    def forward(self, feature):
        aspp3 = self.aspp3(feature)
        feature = torch.cat((feature,aspp3),dim=1)
        aspp6 = self.aspp6(feature)
        feature = torch.cat((feature,aspp6),dim=1)
        aspp12 = self.aspp12(feature)
        feature = torch.cat((feature,aspp12),dim=1)
        aspp18= self.aspp18(feature)
        feature = torch.cat((feature,aspp18),dim=1)
        aspp24 = self.aspp24(feature)
        feature = torch.cat((aspp3, aspp6, aspp12, aspp18, aspp24),dim=1)
        return self.conv1(feature)
        

class DenseAsppBlock(nn.Module):
    """ ConvNet block for building DenseASPP. """
    def __init__(self, input_num, num1, num2, dilation_rate, drop_out):
        super(DenseAsppBlock,self).__init__()
        self.drop_out = drop_out
        norm="BN"
        self.conv1 = Conv2d(
            in_channels=input_num,
            out_channels=num1,
            kernel_size=(1,1),
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            norm=get_norm(norm, num1),
        )
        self.conv2 = Conv2d(
            in_channels=num1,
            out_channels=num2,
            kernel_size=(3,3),
            stride=1,
            padding=1 * dilation_rate,
            dilation=dilation_rate,
            bias=False,
        )
        for layer in [self.conv1, self.conv2]:
            if layer is not None:  # shortcut can be None
                # weight_init.c2_msra_fill(layer)
                weight_init.c2_xavier_fill(layer)
    def forward(self, x): 
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)
        out = F.relu_(out)
        return F.dropout(out,p=self.drop_out)
        
