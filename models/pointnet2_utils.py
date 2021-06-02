import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn(x, k, d=1, f=None): # x[B, N, C] f[B*n]
    B, N, _ = x.size()
    n = N if f is None else int(f.size(0) / B)
    dev = x.device
    inner = -2 * torch.matmul(x, x.transpose(2, 1)) # [B, N, N]
    xx = torch.sum(x**2, dim=-1, keepdim=True) # [B, N, 1]
    dis = -xx.transpose(2, 1) - inner - xx # [B, N, N]
    if f is not None: dis = dis.view(B*N, N)[f].view(B, n, N)
    sid = dis.topk(k=k*d-d+1, dim=-1)[1][..., ::d] # (B, n, k)
    sid += torch.arange(B, device=dev).view(B, 1, 1) * N
    sid = sid.reshape(-1) # [B*n*k]
    tid = torch.arange(B * N, device=dev) if f is None else f # [B*n]
    tid = tid.view(-1, 1).repeat(1, k).view(-1) # [B*n*k]
    return sid, tid # [B*n*k]


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points, grouped_xyz_norm


def Norm(name, c, channels_per_group=16, momentum=0.1, md=1):
    if name == 'bn':
        return eval(f'nn.BatchNorm{md}d')(c, momentum=momentum)
    elif name == 'gn':
        num_group = c // channels_per_group
        if num_group * channels_per_group != c:
            num_group = 1
        return nn.GroupNorm(num_group, c)


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channels, hidden_channels, out_channels, norm='bn'):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.lins = nn.ModuleList([nn.Linear(in_channels, hidden_channels) for _ in range(6)])
        self.norm1 = Norm(norm, hidden_channels, md=1)
        self.acti1 = nn.ReLU(inplace=True)
        self.norm2 = Norm(norm, out_channels, md=1)
        self.acti2 = nn.ReLU(inplace=True)
        self.mlp = nn.Linear(4, 1)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]

        xyz = xyz.permute(0, 2, 1)
        B, _, _ = xyz.size()
        if points is not None:
            points = points.permute(0, 2, 1)

        new_xyz, new_points, p_diff = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        p_diff = p_diff.flatten(0, 1).flatten(0, 1)
        new_xyz_temp = new_xyz.unsqueeze(dim=2).expand(B, self.npoint, self.nsample, 3).flatten(0, 1).flatten(0, 1)  # [B*n*k, 3]
        p_dis = p_diff.norm(dim=-1, keepdim=True).clamp(min=1e-16)  # [B*n*k, 1] 算距离
        global_pool = torch.cat([new_xyz_temp, p_dis], dim=1)  # [B*n*k, 1]
        global_pool = torch.max(self.mlp(global_pool).reshape(B * self.npoint, self.nsample, 1), dim=1)[0]

        z = p_diff[:, 2].unsqueeze(dim=1)
        p_elevation = (z / p_dis).asin().cos()  # [B*n*k, 1]
        p_elevation = p_elevation.reshape(B, self.npoint, self.nsample, -1)  # [B, n, k, 1]

        p_azimuth = (p_diff[:, 0].unsqueeze(dim=1) / (p_diff[:, :-1].norm(dim=-1, keepdim=True).clamp(
                min=1e-16))).atan().cos()  # [B*n*k, 1]
        p_azimuth = p_azimuth.reshape(B, self.npoint, self.nsample, -1)  # [B, n, k, 1]
        new_points = new_points.flatten(0, 1).flatten(0, 1)  # [B*n*k, C]

        edge = torch.stack([lin(new_points) for lin in self.lins]).reshape(6, B, self.npoint, self.nsample, -1)  # [bases, B*n*k, C]
        edge = torch.sum(edge, dim=0).flatten(0, 1).flatten(0, 1)

        edge = edge * (torch.mul(p_elevation, p_azimuth).flatten(0, 1).flatten(0, 1))  # [B, npoint, nsample, C+D]
        edge = edge.reshape(B, self.npoint, self.nsample, -1)

        p_dis = p_dis.view(B, self.npoint, self.nsample, 1)
        p_r = p_dis.max(dim=2, keepdim=True)[0] * 1.1  # [B, n, 1, 1]
        p_d = (p_r - p_dis) ** 2  # [B, n, k, 1]
        edge = edge * p_d / p_d.sum(dim=2, keepdim=True)  # [B, n, k, C]

        y = edge.sum(dim=2).transpose(1, -1)  # [B, C, n]
        y = self.acti1(self.norm1(y)).transpose(1, -1)  # [B, n, C]
        # x = self.lin1(x[tid_euc[::k]]).view(B, n, -1)  # [B, n, C]
        points = torch.max(new_points.reshape(B, self.npoint, self.nsample, -1), dim=2)[0].flatten(0, 1)  # [B*n, C]
        x = self.lin1(points * global_pool).view(B, self.npoint, -1)
        y = x + self.lin2(y)  # [B, n, C]
        y = y.transpose(1, -1)  # [B, C, n]
        y = self.acti2(self.norm2(y))
        new_points = y.transpose(1, -1)  # [B, n, C]

        new_points = new_points.permute(0, 2, 1)  # [B, C, npoint]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points