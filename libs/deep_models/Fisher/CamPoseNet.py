import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class CamPoseNet(nn.Module):
    def __init__(self, base='ResNet', pretrained=False):
        super(CamPoseNet).__init__()

        self.base = base
        if self.base == 'ResNet':
            self.model = models.resnet34(pretrained=pretrained)
            fe_out_planes = self.model.fc.in_features
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)
            self.model.fc = nn.Linear(fe_out_planes, 2048)
        elif self.base == 'Inception':
            self.model = models.inception_v3(pretrained=pretrained, aux_logits=False)
            self.model.fc = nn.Linear(2048,2048)
        else:
            raise NotImplemented

        self.fc_pose = nn.Sequential(
            nn.Linear(2048, 9)   '''要改'''
        )

        if pretrained:
            init_modules = [self.fc_pose]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.model(x)
        A = self.fc_pose(x)

        return A