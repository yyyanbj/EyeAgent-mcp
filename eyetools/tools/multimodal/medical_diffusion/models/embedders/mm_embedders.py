import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class ChamLoss(nn.Module):
    def __init__(self, num_vertex=1448):
        super().__init__()
        self.num_vertex = num_vertex

        left_mu = np.load(
            f"/home/bowen/MeshFusion/mm_params/{self.num_vertex}pts/left_shape_mu.npy"
        )
        left_delta = np.load(
            f"/home/bowen/MeshFusion/mm_params/{self.num_vertex}pts/left_shape_delta.npy"
        )
        right_mu = np.load(
            f"/home/bowen/MeshFusion/mm_params/{self.num_vertex}pts/right_shape_mu.npy"
        )
        right_delta = np.load(
            f"/home/bowen/MeshFusion/mm_params/{self.num_vertex}pts/right_shape_delta.npy"
        )

        self.left_mu = torch.tensor(left_mu, dtype=torch.float32)
        self.left_delta = torch.tensor(left_delta, dtype=torch.float32)
        self.right_mu = torch.tensor(right_mu, dtype=torch.float32)
        self.right_delta = torch.tensor(right_delta, dtype=torch.float32)

    def decode(self, latent, eye_labels):
        latent = torch.mean(latent, dim=-1)
        latent = torch.mean(latent, dim=-1)

        B, C = latent.shape
        points = []
        for b in range(B):
            latent_vector = latent[b].view(-1, 1)
            eye_info = eye_labels[b].item()
            if eye_info == 1:
                shape_recons = self.right_mu + torch.mm(self.right_delta, latent_vector)
            else:
                shape_recons = self.left_mu + torch.mm(self.left_delta, latent_vector)

            shape_recons = shape_recons.view(-1, 3)
            points.append(shape_recons)

        return torch.stack(points, dim=0)

    def forward(self, inputs, targets, eye_labels):
        inputs = self.decode(inputs, eye_labels)
        targets = self.decode(targets, eye_labels)

        dist_matrix = torch.cdist(inputs, targets, p=2)
        dist_1 = torch.mean(torch.min(dist_matrix, dim=-1))
        dist_2 = torch.mean(torch.min(dist_matrix, dim=-2))

        return dist_1 + dist_2
