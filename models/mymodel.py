import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from models.common import knn, rigid_transform_3d
from utils.SE3 import transform
from utils.timer import Timer
import math

def distance(x):  # bs, channel, num_points
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    distance = xx + inner + xx.transpose(2, 1).contiguous()  # bs, num_points, num_points
    return distance


def feature_knn(x, k):
    dis = -distance(x)
    idx = dis.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def mask_score(score, H0, iter, num_layer, choice='topk'):
    bs, num, _ = H0.size()
    # score: row->V, col->E
    if choice == 'auto':
        W = score * H0
        H = torch.where(torch.greater(W, 0), H0, torch.zeros_like(H0))

    elif choice == 'topk':
        k = round(num * 0.1 * (num_layer - 1 - iter))
        topk, _ = torch.topk(score, k=k, dim=-1)  # 每个节点选出topk概率的超边，该超边包含该节点
        a_min = torch.min(topk, dim=-1).values.unsqueeze(-1).repeat(1, 1, num)
        W = torch.where(torch.greater_equal(score, a_min), score, torch.zeros_like(score))
        H = torch.where(torch.greater(W, 0), H0, torch.zeros_like(H0))

    else:
        raise NotImplementedError

    return H, W


class GraphUpdate(nn.Module):
    def __init__(self, num_channels, num_heads, num_layer):
        super(GraphUpdate, self).__init__()
        self.projection_q = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_k = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_v = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.num_channels = num_channels
        self.head = num_heads
        self.num_layer = num_layer
        self.make_score_choice = 'topk'

    def forward(self, H0, vertex_feat, edge_feat, iter):
        # 根据 vertex_feat 和 edge_feat 更新 H （attention）, 计算相关参数: 决定下一轮中边的特征从哪些点得到
        bs, num_vertices, _ = H0.size()
        Q = self.projection_q(vertex_feat).view([bs, self.head, self.num_channels // self.head, num_vertices])  # row
        K = self.projection_k(edge_feat).view([bs, self.head, self.num_channels // self.head, num_vertices])  # col
        # if self.make_score_choice != 'topk':
        #     V = None
        # else:
        #     V = self.projection_v(edge_feat).view([bs, self.head, self.num_channels // self.head, num_vertices])

        attention = torch.einsum('bhco, bhci->bhoi', Q, K) / (self.num_channels // self.head) ** 0.5  # [bs,
        # head, num_corr, num_corr]

        # mask attention using SC2 prior
        attention_mask = 1 - H0  # 固定图结构，验证主网络
        attention_mask = attention_mask.masked_fill(attention_mask.bool(), -1e9)
        score = torch.sigmoid(attention + attention_mask[:, None, :,
                                          :])  # [bs, head, num_corr, num_corr] row->V, col->E, 考察每个节点分别属于每个边的概率(col上的softmax,因为每个节点经过mask后属于的边数不同)
        # if V is not None:
        #     edge_message = torch.einsum('bhoi, bhci-> bhco', score, V).reshape([bs, -1, num_vertices])  # [bs, dim, num_corr]

        score = (torch.sum(score, dim=1) / self.head).view(bs, num_vertices, num_vertices)  # 合并多个head的score
        H, W = mask_score(score, H0, iter, self.num_layer, choice=self.make_score_choice)

        # update D_n_1, W_edge according to new H
        degree_E = H.sum(dim=1)
        De = torch.diag_embed(degree_E)  # torch.sparse_matrix
        De_n_1 = De ** -1
        De_n_1[torch.isinf(De_n_1)] = 0

        degree_V = H.sum(dim=2)
        Dv = torch.diag_embed(degree_V)
        Dv_n_1 = Dv ** -1
        Dv_n_1[torch.isinf(Dv_n_1)] = 0

        W_edge = W.sum(dim=1)  # [bs, num]
        W_edge = F.normalize(W_edge, dim=1)
        W_edge = W_edge.view([bs, num_vertices, 1])
        return H, W, De_n_1, Dv_n_1, W_edge


class NonLocalBlock(nn.Module):
    def __init__(self, num_channels=128, num_heads=1):
        super(NonLocalBlock, self).__init__()
        self.fc_message = nn.Sequential(
            nn.Conv1d(num_channels, num_channels // 2, kernel_size=1),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels // 2, num_channels // 2, kernel_size=1),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels // 2, num_channels, kernel_size=1),
        )
        self.projection_q = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_k = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_v = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.num_channels = num_channels
        self.head = num_heads

    def forward(self, feat, attention, H):
        """
        Input:
            - feat:     [bs, num_channels, num_corr]  input feature
            - attention [bs, num_corr, num_corr]      spatial consistency matrix
        Output:
            - res:      [bs, num_channels, num_corr]  updated feature
        """
        bs, num_corr = feat.shape[0], feat.shape[-1]
        Q = self.projection_q(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        K = self.projection_k(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        V = self.projection_v(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        feat_attention = torch.einsum('bhco, bhci->bhoi', Q, K) / (self.num_channels // self.head) ** 0.5  # [bs,
        # head, num_corr, num_corr]
        attention_mask = 1 - H
        attention_mask = attention_mask.masked_fill(attention_mask.bool(), -1e9)

        # combine the feature similarity with spatial consistency
        weight = torch.softmax(attention[:, None, :, :] * feat_attention + attention_mask[:, None, :, :],
                               dim=-1)  #[bs, head, num_corr, num_corr]
        message = torch.einsum('bhoi, bhci-> bhco', weight, V).reshape([bs, -1, num_corr])  # [bs, dim, num_corr]
        message = self.fc_message(message)
        res = feat + message
        return res


class feature_aggregation_layer(nn.Module):
    def __init__(self,
                 k=20,
                 num_channels=128,
                 head=1,
                 use_knn=False,
                 aggr='mean'):
        super(feature_aggregation_layer, self).__init__()
        self.k = k
        self.use_knn = use_knn
        self.head = head
        self.num_channels = num_channels
        self.aggr = aggr
        self.mlp = nn.Sequential(
            nn.Conv1d(2 * num_channels, num_channels, kernel_size=1),
            nn.BatchNorm1d(num_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
        )
        if self.aggr == 'self_attention':
            self.fc = nn.Sequential(
                nn.Conv1d(num_channels, num_channels // 4, kernel_size=1),
                nn.BatchNorm1d(num_channels // 4),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                nn.Conv1d(num_channels // 4, 1, kernel_size=1),
            )

    def forward(self, x, y, W, H, De_n_1, Dv_n_1, W_edge, k):
        bs, num_dims, num_points = x.size()
        edge_feature = None
        if self.use_knn:
            corr_pos = x  # (bs, dim, num_corr)
            idx = feature_knn(corr_pos, k=k + 1)  # (bs, num_corr, k)
            idx = idx[:, :, 1:]  # ignore the center point

            # bs, num_points, _ = idx.size()
            idx_base = torch.arange(bs).to(x.device).view(-1, 1, 1) * num_points  #
            idx = idx + idx_base  #

            idx = idx.view(-1)
            # _, num_dims, _ = corr_pos.size()

            corr_pos = corr_pos.transpose(2, 1).contiguous()
            neighbor_corr_pose = corr_pos.view(bs * num_points, -1)[idx, :]
            neighbor_corr_pose = neighbor_corr_pose.view(bs, num_points, k, num_dims)
            corr_pos = corr_pos.view(bs, num_points, 1, num_dims)

            neighbor_corr_pose = (neighbor_corr_pose - corr_pos).permute(0, 3, 1, 2)

            neighbor_corr_pose = neighbor_corr_pose.max(dim=-1, keepdim=True)[
                0]  # attention here [bs, num_dim, num_point, k] -> [bs, num_dim, num_point, 1]

            neighbor_corr_pose = neighbor_corr_pose.view(bs, num_dims, num_points)
            feature = x + neighbor_corr_pose
            # feature = torch.cat((x, neighbor_corr_pose), dim=1)  # bs, dim*2, num_corr, k

        else:
            # feature aggregation using softmax or more complicate ways (attention)
            if self.aggr == 'mean':
                # v->e
                # aggregation message from v to e
                feature = x.permute(0, 2, 1)  # [bs, num, dim]
                feature = torch.bmm(H.permute(0, 2, 1), feature)
                edge_feature = torch.bmm(De_n_1, feature)

                if y is not None:
                    edge_feature = self.mlp(torch.cat((y, edge_feature.permute(0, 2, 1)), dim=1)).permute(0, 2,
                                                                                                          1)  # [bs, num, dim]

                # update message of e
                feature = W_edge * edge_feature  # [bs, num, 1] * [bs, num, dim]

                # aggregation message from e to v
                feature = torch.bmm(H, feature)
                feature = torch.bmm(Dv_n_1, feature)

                feature = feature.permute(0, 2, 1)  # [bs, dim, num]
                edge_feature = edge_feature.permute(0, 2, 1)

            elif self.aggr == 'self_attention':
                # e->v nonlocal net using weight matrix W
                feature = x.permute(0, 2, 1)
                feature = torch.bmm(H.permute(0, 2, 1), feature)  # H^T=H
                edge_feature = torch.bmm(De_n_1, feature)  # [bs, num, dim]

                feat = edge_feature.permute(0, 2, 1)  # [bs, dim, num]
                score = torch.softmax(self.fc(feat).permute(0, 2, 1), dim=1)  # [bs, num, 1]
                feature = score * W_edge * feature

                feature = torch.bmm(H, feature)
                feature = torch.bmm(Dv_n_1, feature)

                feature = feature.permute(0, 2, 1)
                edge_feature = edge_feature.permute(0, 2, 1)

            else:
                raise NotImplementedError

        return feature, edge_feature


class HGNN_layer(nn.Module):
    def __init__(self, in_channels, out_channels, residual_connection=False, use_edge_feature=True):
        super(HGNN_layer, self).__init__()
        self.con = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.fg = feature_aggregation_layer(k=20)
        self.residual_connection = residual_connection
        self.use_edge_feature = use_edge_feature

    def forward(self, x, y, W, H, De_n_1, Dv_n_1, W_edge, alpha, thete, x0, k):
        bs, num_vertex, num_edge = H.size()
        x1, y = self.fg(x, y, W, H, De_n_1, Dv_n_1, W_edge,
                        k)  # x1: feature of vertex in n+1 layer; y: feature of edge in n layer
        # update feature of vertex
        if self.residual_connection:
            x = x1 * (1 - alpha) + alpha * x0
            x = (1 - thete) * x + thete * self.bn(self.con(x)) 
        else:  
            x = self.bn(self.con(x1)) + x
        x = F.leaky_relu(x, negative_slope=0.2)
        if not self.use_edge_feature:
            y = None
        return x, y


class HGNN(nn.Module):
    def __init__(self, in_channel=6,
                 n_emb_dims=128,
                 k=20,
                 num_layers=6,
                 lamda=0.5,
                 alpha=0.1):
        super(HGNN, self).__init__()
        self.k = k
        self.lamda = lamda
        self.alpha = alpha
        self.num_layers = num_layers
        self.change_H0 = False
        dim = [n_emb_dims, n_emb_dims, n_emb_dims, n_emb_dims, n_emb_dims, n_emb_dims,
               n_emb_dims]  #  deeper network is not equal to better performance
        self.layer0 = nn.Conv1d(in_channel, dim[0], kernel_size=1, bias=True)
        self.blocks = nn.ModuleDict()
        for i in range(num_layers):
            self.blocks[f'GNN_layer_{i}'] = HGNN_layer(in_channels=dim[i], out_channels=dim[i + 1])
            self.blocks[f'NonLocal_{i}'] = NonLocalBlock(dim[i + 1])
            if i < num_layers - 1:
                self.blocks[f'update_graph_{i}'] = GraphUpdate(num_channels=128, num_heads=1, num_layer=self.num_layers)

    def forward(self, x, W):
        global edge_score
        batch_size, num_dims, num_points = x.size()  # bs, 12, num_corr

        # 1. W[W>0] = 1
        H = W
        H[H > 0] = 1.0
        # 2. Degree of V, E.
        degree = H.sum(dim=1)
        # mask = torch.zeros_like(degree).to(x.device)
        # mask[degree > 0] = 1 

        raw_H = H
        D = torch.diag_embed(degree)  # torch.sparse_matrix
        De_n_1 = D ** -1
        De_n_1[torch.isinf(De_n_1)] = 0
        Dv_n_1 = De_n_1  # 初始超图矩阵H0是对称的

        # 3. edge weight = sum(W, dim=0)
        W_edge = W.sum(dim=1)  # [bs, num]
        W_edge = F.normalize(W_edge, p=2, dim=1)
        W_edge = W_edge.view([batch_size, num_points, 1])

        feat = self.layer0(x)
        feat0 = feat
        edge_feat = None
        for i in range(self.num_layers):
            theta = math.log(self.lamda / (i + 1) + 1)
            feat, edge_feat = self.blocks[f'GNN_layer_{i}'](feat, edge_feat, W, H, De_n_1, Dv_n_1, W_edge, self.alpha,
                                                            theta, feat0, self.k)
            feat = self.blocks[f'NonLocal_{i}'](feat, W, raw_H)
            feat = F.normalize(feat, p=2, dim=1)
            edge_feat = F.normalize(edge_feat, p=2, dim=1)
            # change hypergraph dynamically
            if i < self.num_layers - 1:
                H, edge_score, De_n_1, Dv_n_1, W_edge = self.blocks[f'update_graph_{i}'](H, feat, edge_feat,
                                                                                         i)
                #edge_score_list.append(edge_score)
                #H_list.append(H)
        return raw_H, H, edge_score, feat


class MethodName(nn.Module):
    def __init__(self, config):
        super(MethodName, self).__init__()
        self.config = config
        self.inlier_threshold = config.inlier_threshold
        self.num_iterations = 10
        self.num_channels = 128
        self.encoder = HGNN(n_emb_dims=self.num_channels, in_channel=6)
        self.nms_radius = config.inlier_threshold  # only used during testing
        self.ratio = config.seed_ratio
        self.sigma = nn.Parameter(torch.Tensor([1.0]).float(), requires_grad=True) # changes during training
        self.sigma_d = config.inlier_threshold
        self.k = 40  # neighborhood number in NSM module.

        self.classification = nn.Sequential(
            nn.Conv1d(self.num_channels, 32, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=1, bias=True),
        )

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_data):
        corr, src_pts, tgt_pts, src_normal, tgt_normal = (
            input_data['corr_pos'], input_data['src_keypts'], input_data['tgt_keypts'], input_data['src_normal'],
            input_data['tgt_normal'])
        
        bs, num_corr, num_dim = corr.size()
        FCG_K = int(num_corr * 0.1)
        
        with torch.no_grad():
            # pairwise distance compute

            src_dist = ((src_pts[:, :, None, :] - src_pts[:, None, :, :]) ** 2).sum(-1) ** 0.5
            tgt_dist = ((tgt_pts[:, :, None, :] - tgt_pts[:, None, :, :]) ** 2).sum(-1) ** 0.5

            pairwise_dist = src_dist - tgt_dist
            del src_dist, tgt_dist
            FCG = torch.clamp(1 - pairwise_dist ** 2 / self.sigma_d ** 2, min=0)
            del pairwise_dist
            FCG[:, torch.arange(FCG.shape[1]), torch.arange(FCG.shape[1])] = 0

            # Remain top matches for each row
            sorted_value, _ = torch.topk(FCG, FCG_K, dim=2, largest=True, sorted=False)
            sorted_value = sorted_value.reshape(bs, -1)
            thresh = sorted_value.mean(dim=1, keepdim=True).unsqueeze(2)
            del sorted_value  # Free memory

            # Apply threshold
            FCG = torch.where(FCG < thresh, torch.tensor(0.0, device=FCG.device), FCG)
            del thresh  # Free memory

            # Compute W
            W = torch.matmul(FCG, FCG) * FCG
            del FCG

        F0 = corr
        raw_H, H, edge_score, corr_feats = self.encoder(F0.permute(0, 2, 1), W)  # bs, dim, num_corr
        confidence = self.classification(corr_feats).squeeze(1)  # bs, 1, num_corr-> bs, num_corr loss has sigmoid


        if self.config.mode == "test":
            M = None
        else:
            # M = distance(normed_corr_feats)
            # construct the feature similarity matrix M for loss calculation
            M = torch.matmul(corr_feats.permute(0, 2, 1), corr_feats)
            M = torch.clamp(1 - (1 - M) / self.sigma ** 2, min=0, max=1)
            # # set diagnal of M to zero
            M[:, torch.arange(M.shape[1]), torch.arange(M.shape[1])] = 0

        if self.config.mode == "test":
            seeds = self.graph_filter(H=H, confidence=confidence, max_num=int(num_corr * self.ratio))
        else:
            seeds = torch.argsort(confidence, dim=1, descending=True)[:, 0:int(num_corr * self.ratio)]

        sampled_trans, pred_trans = self.hypo_sampling_and_evaluation(seeds, corr_feats.permute(0, 2, 1), H,
                                                                    src_pts, tgt_pts, src_normal, tgt_normal)
        sampled_trans = sampled_trans.view([-1, 4, 4])

        #post refinement (only used during testing and bs == 1)
        if self.config.mode == "test":
            pred_trans = self.post_refinement(H, pred_trans, src_pts, tgt_pts)
            frag1_warp = transform(src_pts, pred_trans)
            distance = torch.sum((frag1_warp - tgt_pts) ** 2, dim=-1) ** 0.5
            pred_labels = (distance < self.inlier_threshold).float()

        if self.config.mode is not "test":
            pred_labels = confidence
        res = {
            "raw_H": raw_H,
            "hypergraph": H,
            "edge_score": edge_score,
            "final_trans": pred_trans,
            "final_labels": pred_labels,
            "M": M,
            "seeds": seeds,
            "confidence": confidence,
            "sampled_trans": sampled_trans
        }
        return res

    def graph_matrix_reconstruct(self, H, thresh=1, seeds=None):
        """
        Reconstruct the graph matrix from H.
        Input:
            - H: [bs, num_corr, num_corr]
            - seeds: [bs, num_seeds] the index to the seeding correspondence
            - thresh: 0 or 1 loose or tight
        Output:
            - seed_matrix: [bs, num_seeds, num_seeds] or [bs, num_corr, num_corr]
        """
        num_seed = None
        if seeds is not None:
            if len(seeds.size()) == 2:
                _, num_seed = seeds.size()
            else:
                num_seed = len(seeds)

        if len(H.size()) == 3:
            bs, num_corr, _ = H.size()
            merge_matrix = H + H.permute(0, 2, 1)
            merge_matrix = (merge_matrix > thresh).float()

            if seeds is not None:
                seeds_matrix = merge_matrix.gather(1, seeds.unsqueeze(-1).expand(-1, -1, num_corr)).gather(
                    2, seeds.unsqueeze(-2).expand(-1, num_seed, -1))
            else:
                seeds_matrix = merge_matrix

        else:
            num_corr, _ = H.size()
            merge_matrix = H + H.t()
            merge_matrix = (merge_matrix > thresh).float()

            if seeds is not None:
                seeds_matrix = merge_matrix[seeds][:, seeds]
            else:
                seeds_matrix = merge_matrix

        return seeds_matrix.detach()

    def graph_filter(self, H, confidence, max_num):
        assert confidence.shape[0] == 1
        assert H.shape[0] == 1
        # H0 = H
        H = self.graph_matrix_reconstruct(H, 1, None)
        bs, num_corr, _ = H.size()
        D = torch.sum(H, dim=-1) # [bs, num_corr]
        L = torch.diag_embed(D) - H # [bs, num_corr, num_corr]
        xyz = torch.bmm(D[:, :, None].permute(0, 2, 1), L)
        Lscore = torch.norm(xyz, dim=1)  # [bs, num] 绝对值
        #Lscore = xyz.reshape(bs, num_corr)
        low, _ = Lscore.min(dim=1, keepdim=True)
        up, _ = Lscore.max(dim=1, keepdim=True)
        Lscore = (Lscore - low) / (up - low) * (D > 0).float()  # ignore the node that degree == 0

        #parallel Non Maximum Suppression (more efficient)
        score_relation = confidence.T >= confidence  # [num_corr, num_corr], save the relation of leading_eig
        masked_score_relation = score_relation.masked_fill(H == 0, float('inf'))  # ignore the relation that H == 0
        is_local_max = masked_score_relation.min(dim=-1)[0].float() * (
                D > 0).float()  # ignore the node that degree == 0

        score = Lscore * is_local_max
        seed1 = torch.argsort(score, dim=1, descending=True)
        seed2 = torch.argsort(Lscore, dim=1, descending=True)
        sel_len1 = min(max_num, (score > 0).sum().item())  # preserve the seed where score > 0
        set_seed1 = set(seed1[0][:sel_len1].tolist())
        unique_seed2 = [e for e in seed2[0].tolist() if e not in set_seed1][:max_num-sel_len1]

        appended_seed1 = list(set_seed1) + unique_seed2
        return torch.tensor([appended_seed1], device=H.device).detach()

    def cal_inliers_normal(self, inliers_mask, trans, src_normal, tgt_normal):

        pred_src_normal = torch.einsum('bsnm,bmk->bsnk', trans[:, :, :3, :3],
                                       src_normal.permute(0, 2, 1))  # [bs, num_seeds, num_corr, 3]
        pred_src_normal = pred_src_normal.permute(0, 1, 3, 2)
        normal_similarity = (pred_src_normal * tgt_normal[:, None, :, :]).sum(-1)
        normal_similarity = (normal_similarity > 0.7).float()
        normal_similarity = (normal_similarity * inliers_mask).mean(-1)
        return normal_similarity

    def hypo_sampling_and_evaluation(self, seeds, corr_features, H, src_keypts, tgt_keypts, src_normal, tgt_normal):
        # 1、每个seed生成初始假设
        bs, num_corr, num_channels = corr_features.size()
        _, num_seeds = seeds.size()
        assert num_seeds > 0
        k = min(10, num_corr - 1)
        mask = H  # self.graph_matrix_reconstruct(H, 0, None)
        feature_distance = 1 - torch.matmul(corr_features, corr_features.transpose(2, 1))  # normalized
        masked_feature_distance = feature_distance * mask.float() + (1 - mask.float()) * float(1e9)
        knn_idx = torch.topk(masked_feature_distance, k, largest=False)[1]
        knn_idx = knn_idx.gather(dim=1, index=seeds[:, :, None].expand(-1, -1, k))  # [bs, num_seeds, k]
        # 初始假设更精确
        #################################
        # construct the feature consistency matrix of each correspondence subset.
        #################################
        knn_features = corr_features.gather(dim=1,
                                            index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, num_channels)).view(
            [bs, -1, k, num_channels])  # [bs, num_seeds, k, num_channels]
        knn_M = torch.matmul(knn_features, knn_features.permute(0, 1, 3, 2))
        knn_M = torch.clamp(1 - (1 - knn_M) / self.sigma ** 2, min=0)
        knn_M = knn_M.view([-1, k, k])
        feature_knn_M = knn_M

        #################################
        # construct the spatial consistency matrix of each correspondence subset.
        #################################
        src_knn = src_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view(
            [bs, -1, k, 3])  # [bs, num_seeds, k, 3]
        tgt_knn = tgt_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view(
            [bs, -1, k, 3])
        knn_M = ((src_knn[:, :, :, None, :] - src_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5 - (
                (tgt_knn[:, :, :, None, :] - tgt_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        knn_M = torch.clamp(1 - knn_M ** 2 / self.sigma_d ** 2, min=0)
        knn_M = knn_M.view([-1, k, k])
        spatial_knn_M = knn_M

        #################################
        # Power iteratation to get the inlier probability
        #################################
        total_knn_M = feature_knn_M * spatial_knn_M  # 有用
        total_knn_M[:, torch.arange(total_knn_M.shape[1]), torch.arange(total_knn_M.shape[1])] = 0
        total_weight = self.cal_leading_eigenvector(total_knn_M, method='power')
        total_weight = total_weight.view([bs, -1, k])
        total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-6)
        total_weight = total_weight.view([-1, k])
        #################################
        # calculate the transformation by weighted least-squares for each subsets in parallel
        #################################
        corr_mask = (mask.sum(dim=1) > 0).float()[:, None, :]
        seed_mask = mask.gather(dim=1, index=seeds[:, :, None].expand(-1, -1, num_corr))  # [bs, num_seeds, num_corr]
        # 2、每个假设得到pred inliers
        src_knn, tgt_knn = src_knn.view([-1, k, 3]), tgt_knn.view([-1, k, 3])
        seedwise_trans = rigid_transform_3d(src_knn, tgt_knn, total_weight)  # weight 有用
        seedwise_trans = seedwise_trans.view([bs, -1, 4, 4])
        pred_position = torch.einsum('bsnm,bmk->bsnk', seedwise_trans[:, :, :3, :3],
                                     src_keypts.permute(0, 2, 1)) + seedwise_trans[:, :, :3,
                                                                    3:4]  # [bs, num_seeds, num_corr, 3]
        pred_position = pred_position.permute(0, 1, 3, 2)
        L2_dis = torch.norm(pred_position - tgt_keypts[:, None, :, :], dim=-1)  # [bs, num_seeds, num_corr]
        seed_L2_dis = L2_dis * corr_mask + (1 - corr_mask) * float(1e9)
        fitness = self.cal_inliers_normal(seed_L2_dis < self.inlier_threshold, seedwise_trans, src_normal, tgt_normal)

        h = int(num_seeds * 0.1)
        hypo_inliers_idx = torch.topk(fitness, h, -1, largest=True)[1]  # [bs, h]
        #seeds = seeds.gather(1, hypo_inliers_idx)  # [bs, h]
        seed_mask = seed_mask.gather(dim=1,
                                     index=hypo_inliers_idx[:, :, None].expand(-1, -1, num_corr))  # [bs, h, num_corr]
        L2_dis = L2_dis.gather(dim=1, index=hypo_inliers_idx[:, :, None].expand(-1, -1, num_corr))  # [bs, h, num_corr]

        # 计算最大滑动位置
        max_length = seed_mask.sum(dim=2).min()

        # prepare for sampling
        best_score, best_trans, best_labels = None, None, None
        L2_dis = L2_dis * seed_mask + (1 - seed_mask) * float(1e9)  # [bs, num_seeds, num_corr]
        # 3、滑动窗口采样
        s = 6  # 窗口长度
        m = 3  # 步长
        iters = 30
        max_iters = int((max_length - s) / m)
        if max_length > s + m:
            iters = min(max_iters, iters)

        dis, idx = torch.topk(L2_dis, s + iters * m, -1, largest=False)
        # corr_mask
        #corr_mask = (seed_mask.sum(dim=1) > 0).float()[:, None, :]
        sampled_list = []
        for i in range(iters + 1):
            knn_idx = idx[:, :, i * m: s + i * m].contiguous()
            src_knn = src_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view(
                [bs, -1, s, 3])
            tgt_knn = tgt_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view(
                [bs, -1, s, 3])
            src_knn, tgt_knn = src_knn.view([-1, s, 3]), tgt_knn.view([-1, s, 3])
            sampled_trans = rigid_transform_3d(src_knn, tgt_knn)
            sampled_trans = sampled_trans.view([bs, -1, 4, 4])

            sampled_list.append(sampled_trans[0])

            pred_position = torch.einsum('bsnm,bmk->bsnk', sampled_trans[:, :, :3, :3],
                                         src_keypts.permute(0, 2, 1)) + sampled_trans[:, :, :3,
                                                                        3:4]  # [bs, num_seeds, num_corr, 3]
            pred_position = pred_position.permute(0, 1, 3, 2)
            sampled_L2_dis = torch.norm(pred_position - tgt_keypts[:, None, :, :], dim=-1)  # [bs, num_seeds, num_corr]
            sampled_L2_dis = sampled_L2_dis * corr_mask + (1 - corr_mask) * float(1e9)
            MAE_score = (self.inlier_threshold - sampled_L2_dis) / self.inlier_threshold
            fitness = torch.sum(MAE_score * (sampled_L2_dis < self.inlier_threshold), dim=-1)
            sampled_best_guess = fitness.argmax(dim=1)  # [bs, 1]
            sampled_best_score = fitness.gather(dim=1, index=sampled_best_guess[:, None]).squeeze(1)  # [bs, 1]

            sampled_best_trans = sampled_trans.gather(dim=1,
                                                      index=sampled_best_guess[:, None, None, None].expand(-1, -1, 4,
                                                                                                           4)).squeeze(
                1)  # [bs, 4, 4]

            # sampled_best_labels = sampled_L2_dis.gather(dim=1,
            #                                             index=sampled_best_guess[:, None, None].expand(-1, -1,
            #                                                                                            sampled_L2_dis.shape[
            #                                                                                                2])).squeeze(
            #     1)  # [bs, corr_num]
            if i == 0:
                best_score = sampled_best_score
                best_trans = sampled_best_trans
                #best_labels = sampled_best_labels
            else:
                update_mask = sampled_best_score > best_score
                best_score = torch.where(update_mask, sampled_best_score, best_score)
                best_trans = torch.where(update_mask.unsqueeze(-1).unsqueeze(-1), sampled_best_trans, best_trans)
                #best_labels = torch.where(update_mask.unsqueeze(-1), sampled_best_labels, best_labels)

        final_trans = best_trans
        #final_labels = (best_labels < self.inlier_threshold).float()
        return torch.stack(sampled_list), final_trans

    def cal_leading_eigenvector(self, M, method='power'):
        """
        Calculate the leading eigenvector using power iteration algorithm or torch.symeig
        Input:
            - M:      [bs, num_corr, num_corr] the compatibility matrix
            - method: select different method for calculating the learding eigenvector.
        Output:
            - solution: [bs, num_corr] leading eigenvector
        """
        if method == 'power':
            # power iteration algorithm
            leading_eig = torch.ones_like(M[:, :, 0:1])
            leading_eig_last = leading_eig
            for i in range(self.num_iterations):
                leading_eig = torch.bmm(M, leading_eig)
                leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
                if torch.allclose(leading_eig, leading_eig_last):
                    break
                leading_eig_last = leading_eig
            leading_eig = leading_eig.squeeze(-1)
            return leading_eig
        elif method == 'eig':  # cause NaN during back-prop
            e, v = torch.symeig(M, eigenvectors=True)
            leading_eig = v[:, :, -1]
            return leading_eig
        else:
            exit(-1)

    def post_refinement(self, H, initial_trans, src_keypts, tgt_keypts, weights=None):
        """
        Perform post refinement using the initial transformation matrix, only adopted during testing.
        Input
            - initial_trans: [bs, 4, 4]
            - src_keypts:    [bs, num_corr, 3]
            - tgt_keypts:    [bs, num_corr, 3]
            - weights:       [bs, num_corr]
        Output:
            - final_trans:   [bs, 4, 4]
        """
        assert initial_trans.shape[0] == 1
        if self.inlier_threshold == 0.10:  # for 3DMatch
            inlier_threshold_list = [0.10] * 20
        else:  # for KITTI
            inlier_threshold_list = [1.2] * 20
        mask = H
        degree = mask.sum(dim=1)  # [bs, num]
        corr_mask = (degree > 0).float()

        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:
            warped_src_keypts = transform(src_keypts, initial_trans)
            L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
            L2_dis = L2_dis * corr_mask.float() + (1 - corr_mask.float()) * float(1e9)
            MAE_score = (inlier_threshold - L2_dis) / inlier_threshold
            inlier_num = torch.sum(MAE_score * (L2_dis < inlier_threshold), dim=-1)[0]
            pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
            if inlier_num <= previous_inlier_num:
                break
            else:
                previous_inlier_num = inlier_num
            initial_trans = rigid_transform_3d(
                A=src_keypts[:, pred_inlier, :],
                B=tgt_keypts[:, pred_inlier, :],
                ## https://link.springer.com/article/10.1007/s10589-014-9643-2
                # weights=None,
                weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[:, pred_inlier],
                # weights=((1-L2_dis/inlier_threshold)**2)[:, pred_inlier]
            )
        return initial_trans
