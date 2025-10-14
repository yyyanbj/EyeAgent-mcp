import torch.nn as nn
import torch
from monai.networks.layers.utils import get_act_layer

import torch.nn.functional as F


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class LabelEmbedder(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_classes, emb_dim)

        # self.embedding = nn.Embedding(num_classes, emb_dim//4)
        # self.emb_net = nn.Sequential(
        #     nn.Linear(1, emb_dim),
        #     get_act_layer(act_name),
        #     nn.Linear(emb_dim, emb_dim)
        # )

    def forward(self, condition):
        c = self.embedding(condition)  # [B,] -> [B, C]
        # c = self.emb_net(c)
        # c = self.emb_net(condition[:,None].float())
        # c = (2*condition-1)[:, None].expand(-1, self.emb_dim).type(torch.float32)
        return c


class MultiLabelEmbedder(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim
        # self.embedding = nn.Embedding(num_classes, emb_dim)

        # eye
        # 0:None, 1:Left, 2:Right
        self.embedding1 = nn.Embedding(3, emb_dim)

        # biomarker
        # -1:None, 0-N:categories
        self.embedding2 = MLP(
            input_dim=1, hidden_dim=int(emb_dim / 16), output_dim=emb_dim, num_layers=3
        )

        # age
        # -1:None
        self.embedding3 = MLP(
            input_dim=1, hidden_dim=int(emb_dim / 16), output_dim=emb_dim, num_layers=1
        )

        # gender
        # 0:None, 1:Female, 2:Male
        self.embedding4 = nn.Embedding(3, emb_dim)

        # # hospital ID
        # # 0, 1, 2
        # embedding5 = nn.Embedding(5, emb_dim)

        # self.embeddings = nn.ModuleList([embedding1,
        #                                  embedding2,
        #                                  embedding3,
        #                                  embedding4,])
        #                                  # embedding5])

    def forward(self, condition):
        # c = self.embedding(condition) #[B,] -> [B, C]
        # B = img.shape[0]

        # batch_c = []
        # # for each sample
        # for batch in range(B):
        #     c = torch.zeros((1, self.emb_dim)).to(img.device)  # [1, C]
        #     # for each not-None condition
        #     for i, embed in enumerate(self.embeddings):
        #         cond = condition[i][batch:batch+1]
        #         if cond.numel() != 0:
        #             c_tmp = embed(cond)
        #             if i == 4:
        #                 c_tmp = torch.mean(c_tmp, dim=1)
        #             c += c_tmp

        #     batch_c.append(c)

        # batch_c = torch.cat(batch_c, dim=0)  # [B, C]

        c1 = self.embedding1(condition[0])
        c2 = self.embedding2(condition[1])
        c3 = self.embedding3(condition[2])
        c4 = self.embedding4(condition[3])

        # c5 = self.embedding5(condition[4])
        # c5 = torch.mean(c5, dim=1)

        c = c1 + c2 + c3 + c4

        # c = []
        # for i, cond in enumerate(condition):
        #     c_tmp = self.embeddings[i](cond)
        #     # if i == 4:
        #     #     c_tmp = torch.mean(c_tmp, dim=1)
        #     c.append(c_tmp)
        # c = torch.stack(c, dim=1)
        # c = torch.sum(c, dim=1)

        # c = self.emb_net(c)
        # c = self.emb_net(condition[:,None].float())
        # c = (2*condition-1)[:, None].expand(-1, self.emb_dim).type(torch.float32)
        return c


class BiomarkerEmbedder(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim
        # self.embedding = nn.Embedding(num_classes, emb_dim)
        # biomarker
        self.embedding = MLP(
            input_dim=38, hidden_dim=int(emb_dim / 16), output_dim=emb_dim, num_layers=2
        )

    def forward(self, condition):
        c = self.embedding(condition)
        return c


class MeshFusionEmbedder(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim

        # eye
        # 0:Left, 1:Right
        self.embedding1 = nn.Embedding(2, emb_dim)

        # fundus embedding

    def forward(self, condition):
        c1 = self.embedding1(condition[0])
        c2 = condition[1]

        c = c1 + c2
        return c


class MeshFusionEmbedder_CFP_META(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim

        # eye
        # 0:Left, 1:Right
        self.embedding1 = nn.Embedding(2, emb_dim)

        # age
        # self.embedding2 = MLP(1, int(emb_dim//4), emb_dim, 1)

        # gender
        # self.embedding3 = nn.Embedding(2, emb_dim)

        # sph
        # self.embedding_meta = nn.Linear(3, emb_dim)
        # self.norm_meta = nn.LayerNorm(emb_dim)

        # al
        # self.embedding5 = MLP(1, int(emb_dim//4), emb_dim, 1)

        # fundus embedding
        # self.embedding0 = MLP(emb_dim, int(emb_dim//4), emb_dim, 2)

    def forward(self, condition):
        c0 = condition[0]
        c = c0

        if None in [condition[1], condition[4], condition[5]]:

            return c
        else:
            c1 = self.embedding1(condition[1])
            c += c1

            c_meta = torch.cat([condition[4], condition[5]], dim=1)
            assert c_meta.shape[1] == 1024
            c += c_meta

            return c


# c_meta = self.embedding_meta(c_meta)

# c2 = self.embedding2(condition[2])
# c3 = self.embedding3(condition[3])
# c4 = self.embedding4(condition[4])
# c5 = self.embedding5(condition[5])

# c = [c0, c1, c2, c3, c4, c5]
# c = c0 + c1 + c_meta


class MeshFusionEmbedder_CFP_META_MLP(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim

        # eye
        # 0:Left, 1:Right
        self.embedding1 = nn.Embedding(2, emb_dim)

        # age
        # self.embedding2 = MLP(1, int(emb_dim//4), emb_dim, 1)

        # gender
        # self.embedding3 = nn.Embedding(2, emb_dim)

        # sph
        self.embedding_meta = nn.Linear(2, emb_dim)
        self.norm_meta = nn.LayerNorm(emb_dim)

        # al
        # self.embedding5 = MLP(1, int(emb_dim//4), emb_dim, 1)

        # fundus embedding
        # self.embedding0 = MLP(emb_dim, int(emb_dim//4), emb_dim, 2)

    def forward(self, condition):
        c0 = condition[0]
        c1 = self.embedding1(condition[1])

        c_meta = torch.cat([condition[4], condition[5]], dim=1)
        c_meta = self.embedding_meta(c_meta)
        c_meta = self.norm_meta(c_meta)

        # c2 = self.embedding2(condition[2])
        # c3 = self.embedding3(condition[3])
        # c4 = self.embedding4(condition[4])
        # c5 = self.embedding5(condition[5])

        # c = [c0, c1, c2, c3, c4, c5]
        c = c0 + c1 + c_meta
        return c


class MeshFusionEmbedder_CFP_META_KMeans(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {}), num_clusters=5):
        super().__init__()
        self.emb_dim = emb_dim

        # eye
        # 0:Left, 1:Right
        self.embedding1 = nn.Embedding(2, emb_dim)

        # age
        # self.embedding2 = MLP(1, int(emb_dim//4), emb_dim, 1)

        # gender
        # self.embedding3 = nn.Embedding(2, emb_dim)

        # sph
        self.embedding_meta = nn.Embedding(num_clusters, emb_dim)
        # self.norm_meta = nn.LayerNorm(emb_dim)

        # al
        # self.embedding5 = MLP(1, int(emb_dim//4), emb_dim, 1)

        # fundus embedding
        # self.embedding0 = MLP(emb_dim, int(emb_dim//4), emb_dim, 2)

    def forward(self, condition):
        c0 = condition[0]
        c1 = self.embedding1(condition[1])

        # c_meta = torch.cat([condition[4], condition[5]], dim=1)
        # assert c_meta.shape[1] == 1024
        c_meta = self.embedding_meta(condition[5])

        # c2 = self.embedding2(condition[2])
        # c3 = self.embedding3(condition[3])
        # c4 = self.embedding4(condition[4])
        # c5 = self.embedding5(condition[5])

        # c = [c0, c1, c2, c3, c4, c5]
        c = c0 + c1 + c_meta
        return c


class CFPLatentEmbedder(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, condition):
        # c = self.embedding(condition) #[B,] -> [B, C]
        # c = self.emb_net(c)
        # c = self.emb_net(condition[:,None].float())
        # c = (2*condition-1)[:, None].expand(-1, self.emb_dim).type(torch.float32)
        return condition
