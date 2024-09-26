import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

### PointNet

class PointNet(nn.Module):
    """
    PointNet for Classification
        input : [N, C, L]
        output: [N, 40]
    """
    def __init__(self, input_dim, output_dim):
        super(PointNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.bn1 = nn.BatchNorm1d(64)
        #self.bn2 = nn.BatchNorm1d(128)
        #self.bn3 = nn.BatchNorm1d(1024)
        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(1024)

        self.classifier = nn.Linear(1024, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return self.classifier(x)

    def embed(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x



### PointNet Segment

class PointNetSeg(nn.Module):
    """
    PointNet for Classification
        input : [N, C, L]
        output: [N, 50, L]
    """
    def __init__(self, input_dim, output_dim):
        super(PointNetSeg, self).__init__()

        self.output_dim = output_dim
        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        # self.bn1 = nn.InstanceNorm1d(64)
        # self.bn2 = nn.InstanceNorm1d(128)
        # self.bn3 = nn.InstanceNorm1d(1024)

        self.classifier = nn.Linear(1024+16+64+128, output_dim)

    def forward(self, x, label):
        B, N, d = x.shape
        label_idx = F.one_hot(label, num_classes=16).float().unsqueeze(2)  # [B, 16, 1]

        x = x.permute(0, 2, 1)                       # [B, 3, N]
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = self.bn3(self.conv3(x2))                # [B, 1024, N]

        g = torch.max(x3, 2, keepdim=True)[0]        # [B, 1024, 1]
        g = torch.cat([g, label_idx], dim=1).repeat(1, 1, N)
        x = torch.cat([g, x1, x2], dim=1)            # [B, 1024+..., N]

        x = x.permute(0, 2, 1)                       # [B, N, 1024+...]

        return self.classifier(x)

