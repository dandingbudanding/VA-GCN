import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, knn


def sphg(pos, r, B, flow='source_to_target', max_num_neighbors=48, fpsi=None, resetFpsi=False,
         random_replace=True):
    # Make sure "batch" is ascending
    assert flow in ['source_to_target', 'target_to_source']

    N = int(len(pos) / B)
    n = N if fpsi is None else int(len(fpsi) / B)
    C = pos.size(-1)
    k = max_num_neighbors
    dev = pos.device

    with torch.no_grad():
        pos_i = pos.view(B, N, C) if fpsi is None else pos[fpsi].view(B, n, C)
        pos_j = pos.view(B, N, C)
        dis = pos_i.unsqueeze(-2).expand(B, n, N, C) - pos_j.unsqueeze(-2).transpose(1, 2)
        dis = dis.norm(dim=-1)  # [B, n, N]
        max_valid_neighbors = max(k, (dis <= r).sum(dim=-1).max())
        dis, sid = dis.topk(max_valid_neighbors, largest=False)  # [B, n, max_valid_neighbors]
        sid += torch.arange(B, device=dev, dtype=sid.dtype).view(B, 1, 1) * N
        invalid_mask = dis > r  # [B, n, max_valid_neighbors]
        # For those have too many valid neighbors, randomly shuffle and choose without repetition
        shuffle_order = torch.rand(B, n, max_valid_neighbors, device=dev)
        shuffle_order[invalid_mask] = -1
        _, shuffle_order = shuffle_order.topk(k, largest=True)
        sid = sid.gather(-1, shuffle_order)[..., :k]  # [B, n, k]
        # Invalid neighbors are clustered at the end, so we can intercept the mask directly
        invalid_mask = invalid_mask[..., :k]  # [B, n, k]
        # For those have less valid neighbors, randomly replace all invalid neighbors
        if random_replace:
            replacement = torch.rand(B, n, k, device=dev) * (k - invalid_mask.float().sum(dim=-1, keepdim=True))
            replacement.floor_()
            replacement.clamp_(max=k - 1)
            replacement = sid.gather(-1, replacement.long())
            sid[invalid_mask] = replacement[invalid_mask]
        else:
            sid[invalid_mask] = sid[..., 0:1].expand(B, n, k)[invalid_mask]
        sid = sid.view(-1)
        if fpsi is None or resetFpsi: fpsi = torch.arange(B * n, device=dev)
        tid = fpsi.view(-1, 1).repeat(1, k).view(-1)

    return (sid, tid) if flow == 'source_to_target' else (tid, sid)


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


class EAConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, norm='bn'):
        super(EAConv, self).__init__()
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.lins = nn.ModuleList([nn.Linear(in_channels, hidden_channels) for _ in range(6)])
        self.norm1 = Norm(norm, hidden_channels, md=1)
        self.acti1 = nn.ReLU(inplace=True)
        self.norm2 = Norm(norm, out_channels, md=1)
        self.acti2 = nn.ReLU(inplace=True)
        self.mlp = nn.Linear(4, 1)

    def forward(self, x, p, B, n, id_euc):
        # x[B*N, C] p[B*N, 3] sid/tid[B*n*k]
        sid_euc, tid_euc = id_euc
        k = int(len(sid_euc) / B / n)

        euc_i, euc_j = x[tid_euc], x[sid_euc]  # [B*n*k, C]
        edge = euc_j - euc_i

        p_diff = p[sid_euc] - p[tid_euc]  # [B*n*k, 3] 算法相

        p_dis = p_diff.norm(dim=-1, keepdim=True).clamp(min=1e-16)  # [B*n*k, 1] 算距离

        global_pool = torch.cat([p[tid_euc], p_dis], dim=1)  # [B*n*k, 1]
        global_pool = torch.max(self.mlp(global_pool).reshape(B*n, k, 1), dim=1)[0]

        z = p_diff[:, 2].unsqueeze(dim=1)
        p_elevation = (z / p_dis).cos()  # [B*n*k, 1]
        p_elevation = p_elevation.reshape(B, n, k, -1)  # [B, n, k, 1]

        p_azimuth = (p_diff[:, 0].unsqueeze(dim=1) / (p_diff[:, :-1].norm(dim=-1, keepdim=True).clamp(
            min=1e-16))).cos()  # [B*n*k, 1]
        p_azimuth = p_azimuth.reshape(B, n, k, -1)  # [B, n, k, 1]

        edge = torch.stack([lin(edge) for lin in self.lins]).reshape(6, B, n, k, -1)  # [bases, B*n*k, C]
        edge = torch.sum(edge, dim=0).flatten(0, 1).flatten(0, 1)

        edge = edge * (torch.mul(p_elevation, p_azimuth).flatten(0, 1).flatten(0, 1))  # [B, npoint, nsample, C+D]
        edge = edge.reshape(B, n, k, -1)

        p_dis = p_dis.view(B, n, k, 1)
        p_r = p_dis.max(dim=2, keepdim=True)[0] * 1.1  # [B, n, 1, 1]
        p_d = (p_r - p_dis) ** 2  # [B, n, k, 1]
        edge = edge * p_d / p_d.sum(dim=2, keepdim=True)  # [B, n, k, C]

        y = edge.sum(dim=2).transpose(1, -1)  # [B, C, n]
        y = self.acti1(self.norm1(y)).transpose(1, -1)  # [B, n, C]
        #x = self.lin1(x[tid_euc[::k]]).view(B, n, -1)  # [B, n, C]
        x = self.lin1(x[tid_euc[::k]] * global_pool).view(B, n, -1)
        y = x + self.lin2(y)  # [B, n, C]
        y = y.transpose(1, -1)  # [B, C, n]
        y = self.acti2(self.norm2(y))
        y = y.transpose(1, -1)  # [B, n, C]
        y = y.flatten(0, 1)  # [B*n, C]

        return y


class PointPlus(nn.Module):  # PointNet++
    def __init__(self, in_channels, out_channels, norm='bn', first_layer=False):
        super(PointPlus, self).__init__()
        self.first_layer = first_layer
        self.fc1 = MLP_md([in_channels, out_channels], norm=norm, md=2)

    def forward(self, x, B, n, id_euc):
        # x[B*N, C] sid/tid[B*n*k]
        sid_euc, tid_euc = id_euc
        k = int(sid_euc.size(0) / B / n)

        if self.first_layer:
            x, norm = x[:, :3], x[:, 3:]
            x_i, x_j = x[tid_euc], x[sid_euc]  # [B*n*k, C]
            norm_j = norm[sid_euc]  # [B*n*k, C]
            edge = torch.cat([x_j - x_i, norm_j], dim=-1)  # [B*n*k, C]
        else:
            x_i, x_j = x[sid_euc], x[tid_euc]  # [B*n*k, C]
            edge = x_j - x_i
        edge = edge.view(B, n, k, -1)  # [B, n, k, C]
        edge = self.fc1(edge)  # [B, n, k, C]
        y = edge.max(2)[0]  # [B, n, C]
        y = y.view(B * n, -1)  # [B*n, C]

        return y


class get_model(nn.Module):
    def __init__(self, num_classes, norm='bn'):
        super(get_model, self).__init__()
        self.pp1 = PointPlus(6, 64, norm, first_layer=True)
        self.pp2 = PointPlus(64, 128, norm)
        self.pp3 = PointPlus(128, 384, norm)

        self.lin1 = MLP_md([6, 16])
        self.conv1 = EAConv(16, 32, 32, norm=norm)
        self.lin2 = MLP_md([64, 128])
        self.conv2 = EAConv(128, 64, 256, norm=norm)
        self.res_conv = MLP_md([64, 512])

        # master
        self.conv3 = EAConv(896, 64, 768, norm=norm)
        self.lin3 = MLP_md([2496, 1024], norm=norm)
        self.lin4 = MLP_md([1024, 512], norm=norm)
        self.dp1 = nn.Dropout(p=0.5)
        self.lin5 = MLP_md([512, 256], norm=norm)
        self.dp2 = nn.Dropout(p=0.5)
        self.lin6 = MLP_md([256, num_classes], norm=norm)

    def forward(self, xyz):
        print(xyz.size())

        B = xyz.size()[0]  # B,N,C
        xyz = xyz.flatten(0, 1)  # B*N,C
        N = int(xyz.size(0) / B)
        pos = xyz[:, :3]
        norm = xyz[:, 3:]
        x = torch.cat([pos, norm], dim=-1)

        # branch1
        id_euc = knn(pos.view(B, N, -1), 16)
        x1 = self.pp1(x, B, N, id_euc)  # [B*N, C]
        x2 = self.pp2(x1, B, N, id_euc)
        x3 = self.pp3(x2, B, N, id_euc)

        # branch2
        x4 = self.lin1(x.view(B, N, -1)).view(B * N, -1)  # [B*N, C]
        id_euc = sphg(pos, 0.15, B, max_num_neighbors=16)
        x5 = self.conv1(x4, pos, B, N, id_euc)  # [B*N, C]
        id_euc = sphg(pos, 0.3, B, max_num_neighbors=16)
        x5_2 = self.conv1(x4, pos, B, N, id_euc)  # [B*N, C]
        x5 = torch.cat([x5, x5_2], dim=1)  # [B*N, C]

        x6 = self.lin2(x5.view(B, N, -1)).view(B * N, -1)
        x7 = self.conv2(x6, pos, B, N, id_euc)  # [B*N, C]
        id_euc = sphg(pos, 0.6, B, max_num_neighbors=16)
        x7_2 = self.conv2(x6, pos, B, N, id_euc)  # [B*N, C]
        x7 = torch.cat([x7, x7_2], dim=1)  # [B*N, C]
        x7 = x7 + self.res_conv(x5)

        # master
        x8 = torch.cat([x3, x5, x7], dim=-1)  # [B*N, 960]

        x9 = self.conv3(x8, pos, B, N, id_euc)
        id_euc = sphg(pos, 0.8, B, max_num_neighbors=16)
        x9_2 = self.conv3(x8, pos, B, N, id_euc)
        x9 = torch.cat([x9, x9_2], dim=1).reshape(B, N, -1).permute(2, 1)  # [B, 1536, N]
        x9 = x9.max(dim=-1, keepdim=True)[0]  # [B, 1536, 1]
        x9 = x9.repeat(1, 1, N).permute(0, 2, 1).flatten(0, 1)  # [B*N, 1536]
        x10 = torch.cat([x9, x3, x5, x7])  # [B*N, 2496]
        x10 = self.lin3(x10.view(B, N, -1)).view(B * N, -1)
        x10 = self.lin4(x10.view(B, N, -1)).view(B * N, -1)

        x10 = self.dp1(x10)
        x10 = self.lin5(x10.view(B, N, -1)).view(B * N, -1)

        x10 = self.dp2(x10)
        x = self.lin6(x10.view(B, N, -1)).view(B * N, -1)

        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss


if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))