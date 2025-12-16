import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import json
from collections import defaultdict
import re
import ast


class ChestAnatomySegmenter(nn.Module):
    def __init__(self,num_regions=4):
        super().__init__()
        try:
            from torchvision.models import ResNet34_Weights
            weights = ResNet34_Weights.DEFAULT
            resnet = models.resnet34(weights=weights)
        except:
            resnet = models.resnet34(pretrained=True)
        
        #Encoder
        self.enc1 = nn.Sequential(*list(resnet.children())[:3])
        self.enc2 = nn.Sequential(*list(resnet.children())[3:5])
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4

        #Decoder
        self.dec5 = self._decoder_block(512,512)
        self.dec4 = self._decoder_block(512+256,256)
        self.dec3 = self._decoder_block(256+128,128)
        self.dec2 = self._decoder_block(128+64,64)
        self.dec1 = self._decoder_block(64+64,64)

        self.seg_head = nn.Conv2d(64,num_regions,kernel_size=1)

                                        
    def _decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d5 = F.interpolate(e5,scale_factor=2,mode='bilinear',align_corners=False)
        d5 = self.dec5(d5)

        d4 = torch.cat([d4,e4],dim=1)
        d4 = F.interpolate(d4,scale_factor=2,mode='bilinear',align_corners=False)
        d4 = self.dec4(d4)

        d3 = torch.cat([d4,e3],dim=1)
        d3 = F.interpolate(d3,scale_factor=2,mode='bilinear',align_corners=False)
        d3 = self.d3(d3)
        
        d2 = torch.cat([d3, e2], dim=1)
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.dec2(d2)
        
        d1 = torch.cat([d2, e1], dim=1)
        d1 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.dec1(d1)
        
        masks = torch.sigmoid(self.seg_head(d1))
        return masks
    
class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT
            resnet = models.resnet50(weights=weights)
        except:
            resnet = models.resnet50(pretrained=True)
            
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_dim = 2048
    
    def forward(self, x):
        return self.backbone(x)
    
class RegionTokenConstructor(nn.Module):
    def __init__(self,visual_dim=2048,token_dim=768):
        super().__init__()
        self.projection = nn.Linear(visual_dim,token_dim)
    
    def forward(self,visual_features,masks):
        B,C,H,W = visual_features.shape
        masks_resized = F.interpolate(masks,size=(H,W),mode='bilinear',align_corners=False)

        region_tokens = []
        for i in range(masks.shape[1]):
            mask = masks_resized[:,i:i+1]
            masked_features = visual_features*mask
            sum_features = masked_features.sum(dim=[2,3])
            sum_mask = mask.sum(dim=[2, 3]) + 1e-8
            avg_features = sum_features / sum_mask
            token = self.projection(avg_features)
            region_tokens.append(token)
        
        return torch.stack(region_tokens, dim=1)
