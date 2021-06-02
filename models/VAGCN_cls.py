import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
import torch


def Norm(name, c, channels_per_group=16, momentum=0.1, md=1):
    if name == 'bn':
        return eval(f'nn.BatchNorm{md}d')(c, momentum=momentum)
    elif name == 'gn':
        num_group = c // channels_per_group
        if num_group * channels_per_group != c:
            num_group = 1
        return nn.GroupNorm(num_group, c)


class MLP_md(nn.Module):
    def __init__(self, channels, norm='bn', resi=False, md=1, **kwargs):
        super(MLP_md, self).__init__()
        self.linears = nn.ModuleList([eval(f'nn.Conv{md}d')(i, o, 1) for i, o in zip(channels, channels[1:])])
        self.norms = nn.ModuleList([Norm(norm, o, md=md, **kwargs) for o in channels[1:]])
        self.actis = nn.ModuleList([nn.ReLU(inplace=True) for _ in channels[1:]])
        self.resi = resi

    def forward(self, x, feature_last=True):  # [B, ..., C]
        twoD = len(x.shape) == 2
        if twoD:
            feature_last = False
            x = x.unsqueeze(-1)
        if feature_last: x = x.transpose(1, -1)  # [B, C, ...]

        for linear, norm, acti in zip(self.linears, self.norms, self.actis):
            inp = x if self.resi else None
            x = linear(x)
            x = norm(x)
            x = acti(x)
            if inp is not None and x.shape == inp.shape:
                x = x + inp

        if feature_last:
            x = x.transpose(1, -1)  # [B, ..., C]
            x = x.contiguous()
        if twoD: x = x.squeeze(-1)

        return x


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.lin1 = MLP_md([3, 16])
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channels=16+3, hidden_channels=64, out_channels=64)
        self.lin2 = MLP_md([64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.4, nsample=64, in_channels=128+3, hidden_channels=64, out_channels=256)
        self.lin3 = MLP_md([256, 512])
        self.sa3 = PointNetSetAbstraction(npoint=128, radius=0.6, nsample=64, in_channels=512+3, hidden_channels=64, out_channels=1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        norm = norm.permute(0, 2, 1)
        norm = self.lin1(norm).permute(0, 2, 1)
        l1_xyz, l1_points = self.sa1(xyz, norm)

        l1_points = l1_points.permute(0, 2, 1)
        l1_points = self.lin2(l1_points).permute(0, 2, 1)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = l2_points.permute(0, 2, 1)
        l2_points = self.lin3(l2_points).permute(0, 2, 1)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = torch.max(l3_points, dim=2)[0]
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
