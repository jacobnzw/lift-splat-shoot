"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from efficientnet_pytorch import EfficientNet
from torch import nn
from torchvision.models.resnet import resnet18

from .tools import QuickCumsum, cumsum_trick, gen_dx_bx


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320 + 112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, : self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(
                    self.trunk._blocks
                )  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints["reduction_{}".format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        endpoints["reduction_{}".format(len(endpoints) + 1)] = x
        x = self.up1(endpoints["reduction_5"], endpoints["reduction_4"])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(
            self.grid_conf["xbound"],
            self.grid_conf["ybound"],
            self.grid_conf["zbound"],
        )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16  # gap in pixels between frustum grid points in horizontal/vertical directions
        self.camC = 64  # number of channels in the camera encoder output
        
        # a template of frustum grid points to be later transformed by each cam's extrinsics/intrisics
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape

        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self):
        """
        Create a point grid in the camera view frustum.

        This method generates a frustum grid by stacking image plane grids at different depths, based on the depth bounds specified in the `grid_conf`.

        Returns:
            nn.Parameter: A tensor representing the frustum with shape (D, H, W, 3),
                          where D is the number of depth bins, H and W are the height
                          and width of the frustum grid, and 3 represents the x, y, z coordinates.
        """
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf["final_dim"]  # (128, 352)
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # (8, 22) for downsample=16
        ds = (
            torch.arange(*self.grid_conf["dbound"], dtype=torch.float)
            .view(-1, 1, 1)  # look at the tensor as if it had 2 additional axes
            .expand(-1, fH, fW)  # broadcast the arange into the height and width dimensions w/o additional memory allocation
        )
        D, _, _ = ds.shape
        xs = (
            torch.linspace(0, ogfW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, ogfH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        # (D, H, W, 3)
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x, y, z) locations of the points in the point cloud in the ego frame.

        Args:
            rots (torch.Tensor): Rotation matrices, part of the extrinsic camera parameters.
            trans (torch.Tensor): Translation vectors, part of the extrinsic camera parameters.
            intrins (torch.Tensor): Intrinsic camera parameters.
            post_rots (torch.Tensor): Post-rotation matrices.
            post_trans (torch.Tensor): Post-translation vectors.

        Returns:
            torch.Tensor: Point cloud locations with shape (B, N, D, H/downsample, W/downsample, 3).
        """
        B, N, _ = trans.shape

        # undo post-transformation  TODO: WTF is post-transformation?
        # (B, N, D, H, W, 3)
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)  # un-translate the points
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)  # one rotation matrix for each camera and batch sample
            .matmul(points.unsqueeze(-1))  # un-rotate the points
        )

        # cam_to_ego
        # [x*z, y*z, z]: points in the XY plane drift apart as we go further from the camera
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        # Inverse intrinsics transform from homogeneous to cartesian coordinates, while staying in the camera frame
        # Extrinsic rotation of each camera in the ego frame; ego_SO3_camera
        combine = rots.matmul(torch.inverse(intrins))  # (B, N, 3, 3)
        # Maps points from camera homogenous frame to ego cartesian frame?
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """
        Extract camera features from input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C, H, W).

        Returns:
            torch.Tensor: Camera features of shape (B, N, D, H/downsample, W/downsample, camC).
                B: batch size
                N: number of cameras
                D: depth bins
                H: height
                W: width
                camC: number of camera encoder output channels
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B * N, C, imH, imW)
        # Run images through efficientnet
        x = self.camencode(x)
        x = x.view(
            B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample
        )
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        """
        Pool features from the same voxel together.
        TODO: is this the "frustum pooling" from the paper?

        Args:
            geom_feats (torch.Tensor): Frustum grid points in the ego frame of shape (B, N, D, H/downsample, W/downsample, 3).
            x (torch.Tensor): Input features from camera encoder of shape (B, N, D, H/downsample, W/downsample, C).

        Returns:
            torch.Tensor: Pooled features of shape (B, C, Z, X, Y).
        """
        B, N, D, H, W, C = x.shape  # C: number of camera encoder output channels
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # Convert geometric features from world coordinates to voxel grid indices
        # Subtract the minimum bound (self.bx) and half the voxel size (self.dx / 2.0) to center voxels
        # Divide by voxel size (self.dx) to get indices, and convert to integer (long)
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)

        # Create a batch index tensor (N * D * H * W, B)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        # Append the batch index to geom_feats
        # This allows us to distinguish between voxels from different batches
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # Filter out points that are outside the 3D voxel grid bounds
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # Compute unique voxel ranks for each frustum grid point
        ranks = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)  # X coordinate contribution
            + geom_feats[:, 1] * (self.nx[2] * B)  # Y coordinate contribution
            + geom_feats[:, 2] * B  # Z coordinate contribution
            + geom_feats[:, 3]  # Batch index contribution
        )
        # Sort tensors based on ranks achieving the effect that features from the same voxel follow each other
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # Aggregate features from the same voxel using cumulative sum
        # x: (Nvox, C), geom_feats: (Nvox, 4), where Nvox is the number of non-empty voxels <= prod(self.nx)
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # Griddify (B, C, Z, X, Y)
        # geom_feats by now should be called "voxel coordinates"
        # Fill in non-empty voxels, leave empty voxels as zeros
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[
            geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]
        ] = x

        # Create 2D BEV grid by simply dropping the Z dimension: (B, C, Z, X, Y) -> (B, C, X, Y)
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        # Let Hd = H // downsample and Wd = W // downsample be sub-sampled cam image height and width
        # The output shapes are as follows:
        # Frustum grid points of each cam in the ego frame
        # (B, N, D, Hd, Wd, 3) <=> (4, 6, 41, 8, 22, 3)
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        # (B, N, D, Hd, Wd, camC) <=> (4, 6, 41, 8, 22, 64); camC = cam encoder output channels
        x = self.get_cam_feats(x)

        # For each 3D point of the frustum grid (geom) we now have a camC-dimensional feature vector (x)
        # We can now project them into BEV grid using voxel pooling
        # (4, 64, 200, 200)
        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        """
        Args:
            x: (B, N, C, H, W)
            rots: (B, N, 3, 3)
            trans: (B, N, 3)
            intrins: (B, N, 3, 3)
            post_rots: (B, N, 3, 3)
            post_trans: (B, N, 3)
        """
        # (4, 64, 200, 200)
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        # (4, 1, 200, 200)
        x = self.bevencode(x)
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
