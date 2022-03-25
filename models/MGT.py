"""Meta Graph Transformer (MGT)"""
import math

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def multihead_linear_transform(W, inputs):
    B, P, N, H, d_k, d_model = W.shape
    inputs = inputs.reshape((B, P, N, 1, d_model, 1))
    out = torch.matmul(W, inputs).squeeze(-1)  # (B, P, N, H, d_k)

    return out


def multihead_temporal_attention(Q, K, V, causal=False):
    B, P1, N, H, d_k = Q.shape
    P = K.shape[1]
    Q = Q.permute((0, 2, 3, 1, 4))  # (B, N, H, P1, d_k)
    K = K.permute((0, 2, 3, 4, 1))  # (B, N, H, d_k, P)

    scaled_dot_product = torch.matmul(Q, K) / math.sqrt(d_k)  # (B, N, H, P1, P)
    if causal is True:
        assert P1 == P
        mask = scaled_dot_product.new_full((P, P), -np.inf).triu(diagonal=1)
        scaled_dot_product += mask
    alpha = F.softmax(scaled_dot_product, dim=-1)

    V = V.permute((0, 2, 3, 1, 4))  # (B, N, H, P, d_k)
    out = torch.matmul(alpha, V)  # (B, N, H, P1, d_k)
    out = out.permute((0, 3, 1, 2, 4))  # (B, P1, N, H, d_k)
    out = out.reshape((B, P1, N, H * d_k))  # (B, P1, N, H * d_k) i.e. (B, P1, N, d_model)

    return out


def multihead_spatial_attention(Q, K, V, transition_matrix):
    B, P, N, H, d_k = Q.shape

    Q = Q.permute((0, 1, 3, 2, 4))  # (B, P, H, N, d_k)
    K = K.permute((0, 1, 3, 4, 2))  # (B, P, H, d_k, N)
    V = V.permute((0, 1, 3, 2, 4))  # (B, P, H, N, d_k)

    scaled_dot_product = torch.matmul(Q, K) / math.sqrt(d_k)  # (B, P, H, N, N)
    mask = scaled_dot_product.new_full((N, N), -np.inf)  # (N, N)
    mask[transition_matrix != 0] = 0
    scaled_dot_product += mask
    alpha = F.softmax(scaled_dot_product, dim=-1)  # (B, P, H, N, N)
    out = torch.matmul(alpha * transition_matrix, V)  # (B, P, H, N, d_k)

    out = out.permute((0, 1, 3, 2, 4))  # (B, P, N, H, d_k)
    out = out.reshape((B, P, N, H * d_k))  # (B, P, N, H * d_k) i.e. (B, P, N, d_model)

    return out


class TemporalEmbedding(nn.Module):
    def __init__(self, num_embeddings, d_model, max_len):
        super(TemporalEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
        self.register_buffer('pe', pe)

        self.embedding_modules = nn.ModuleList([nn.Embedding(item, d_model) for item in num_embeddings])
        self.linear = nn.Linear((len(num_embeddings) + 1) * d_model, d_model)

    def forward(self, extras):
        assert len(extras) == 2 * len(self.num_embeddings)
        inputs_extras = extras[::2]
        targets_extras = extras[1::2]

        B, P = inputs_extras[0].shape
        _, Q = targets_extras[0].shape

        inputs_pe = self.pe[:P, :].expand(B, P, self.d_model)
        targets_pe = self.pe[:Q, :].expand(B, Q, self.d_model)

        inputs_extras_embedding = torch.cat([self.embedding_modules[i](inputs_extras[i])
                                             for i in range(len(self.num_embeddings))] + [inputs_pe], dim=-1)
        targets_extras_embedding = torch.cat([self.embedding_modules[i](targets_extras[i])
                                              for i in range(len(self.num_embeddings))] + [targets_pe], dim=-1)

        inputs_extras_embedding = self.linear(inputs_extras_embedding)
        targets_extras_embedding = self.linear(targets_extras_embedding)

        return inputs_extras_embedding, targets_extras_embedding


class SpatialEmbedding(nn.Module):
    def __init__(self, eigenmaps_k, d_model):
        super(SpatialEmbedding, self).__init__()
        self.linear = nn.Linear(eigenmaps_k, d_model)

    def forward(self, eigenmaps):
        spatial_embedding = self.linear(eigenmaps)

        return spatial_embedding


class SpatialTemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SpatialTemporalEmbedding, self).__init__()
        self.linear = nn.Linear(2 * d_model, d_model)

    def forward(self, z_inputs, z_targets, u):  # (B, P, d_model), (B, Q, d_model), (N, d_model)
        z_inputs = torch.stack((z_inputs, ) * len(u), dim=2)
        z_targets = torch.stack((z_targets, ) * len(u), dim=2)
        u_inputs = u.expand_as(z_inputs)
        u_targets = u.expand_as(z_targets)

        c_inputs = self.linear(torch.cat((z_inputs, u_inputs), dim=-1))
        c_targets = self.linear(torch.cat((z_targets, u_targets), dim=-1))

        return c_inputs, c_targets


class MetaLearner(nn.Module):
    def __init__(self, d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=3):
        super(MetaLearner, self).__init__()
        self.num_weight_matrices = num_weight_matrices
        self.num_heads = num_heads
        self.d_k = d_k

        self.linear1 = nn.Linear(d_model, d_hidden_mt)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_hidden_mt, num_weight_matrices * num_heads * d_k * d_model)

    def forward(self, c_inputs):
        B, P, N, d_model = c_inputs.shape
        out = self.relu(self.linear1(c_inputs))
        out = self.linear2(out)
        out = out.reshape((B, P, N, self.num_weight_matrices, self.num_heads, self.d_k, d_model))
        out = out.permute((3, 0, 1, 2, 4, 5, 6))  # (num_weight_matrices, B, P, N, num_heads, d_k, d_model)

        return out


class TemporalSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_hidden_mt, num_heads, noMeta, causal=False):
        super(TemporalSelfAttention, self).__init__()
        self.noMeta = noMeta
        self.causal = causal

        if self.noMeta:
            self.num_heads = num_heads
            self.d_k = d_k
            self.linear_q = nn.Linear(d_model, d_model, bias=False)
            self.linear_k = nn.Linear(d_model, d_model, bias=False)
            self.linear_v = nn.Linear(d_model, d_model, bias=False)
        else:
            self.meta_learner = MetaLearner(d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=3)

        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, inputs, c_inputs):
        if self.noMeta:
            B, P, N, _ = inputs.shape
            Q = self.linear_q(inputs).reshape((B, P, N, self.num_heads, self.d_k))
            K = self.linear_k(inputs).reshape((B, P, N, self.num_heads, self.d_k))
            V = self.linear_v(inputs).reshape((B, P, N, self.num_heads, self.d_k))
        else:
            W_q, W_k, W_v = self.meta_learner(c_inputs)  # (B, P, N, H, d_k, d_model)
            Q = multihead_linear_transform(W_q, inputs)  # (B, P, N, H, d_k)
            K = multihead_linear_transform(W_k, inputs)  # (B, P, N, H, d_k)
            V = multihead_linear_transform(W_v, inputs)  # (B, P, N, H, d_k)

        out = multihead_temporal_attention(Q, K, V, causal=self.causal)  # (B, P, N, d_model)
        out = self.linear(out)  # (B, P, N, d_model)
        out = self.layer_norm(out + inputs)  # (B, P, N, d_model)

        return out


class SpatialSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_hidden_mt, num_heads, which_transition_matrices, dropout, noMeta):
        super(SpatialSelfAttention, self).__init__()
        self.which_transition_matrices = which_transition_matrices
        self.num_transition_matrices = sum(which_transition_matrices)
        assert self.num_transition_matrices > 0
        self.noMeta = noMeta

        if self.noMeta:
            self.num_heads = num_heads
            self.d_k = d_k
            self.linear_q = nn.ModuleList([nn.Linear(d_model, d_model, bias=False)
                                           for _ in range(self.num_transition_matrices)])
            self.linear_k = nn.ModuleList([nn.Linear(d_model, d_model, bias=False)
                                           for _ in range(self.num_transition_matrices)])
            self.linear_v = nn.ModuleList([nn.Linear(d_model, d_model, bias=False)
                                           for _ in range(self.num_transition_matrices)])
        else:
            self.meta_learners = nn.ModuleList([MetaLearner(
                d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=3)
                for _ in range(self.num_transition_matrices)])

        self.linear = nn.Linear(d_model * self.num_transition_matrices, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, inputs, c_inputs, transition_matrices):
        assert transition_matrices.shape[0] == len(self.which_transition_matrices)
        transition_matrices = transition_matrices[self.which_transition_matrices]

        out = []
        for i in range(self.num_transition_matrices):
            if self.noMeta:
                B, P, N, _ = inputs.shape
                Q = self.linear_q[i](inputs).reshape((B, P, N, self.num_heads, self.d_k))
                K = self.linear_k[i](inputs).reshape((B, P, N, self.num_heads, self.d_k))
                V = self.linear_v[i](inputs).reshape((B, P, N, self.num_heads, self.d_k))
            else:
                W_q, W_k, W_v = self.meta_learners[i](c_inputs)  # (B, P, N, H, d_k, d_model)
                Q = multihead_linear_transform(W_q, inputs)  # (B, P, N, H, d_k)
                K = multihead_linear_transform(W_k, inputs)  # (B, P, N, H, d_k)
                V = multihead_linear_transform(W_v, inputs)  # (B, P, N, H, d_k)

            out.append(multihead_spatial_attention(Q, K, V, transition_matrices[i]))  # (B, P, N, d_model)
        out = torch.cat(out, dim=-1)  # (B, P, N, d_model * num_transition_matrices)
        out = self.linear(out)  # (B, P, N, d_model)
        out = self.dropout(out)
        out = self.layer_norm(out + inputs)  # (B, P, N, d_model)

        return out


class TemporalEncoderDecoderAttention(nn.Module):
    def __init__(self, d_model, d_k, d_hidden_mt, num_heads, noMeta):
        super(TemporalEncoderDecoderAttention, self).__init__()
        self.noMeta = noMeta

        if self.noMeta:
            self.num_heads = num_heads
            self.d_k = d_k
            self.linear_q = nn.Linear(d_model, d_model, bias=False)
        else:
            self.meta_learner = MetaLearner(d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=1)

        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, inputs, enc_K, enc_V, c_targets):
        if self.noMeta:
            B, P1, N, _ = inputs.shape
            Q = self.linear_q(inputs).reshape((B, P1, N, self.num_heads, self.d_k))
        else:
            W_q, = self.meta_learner(c_targets)  # (B, P1, N, H, d_k, d_model)
            Q = multihead_linear_transform(W_q, inputs)  # (B, P1, N, H, d_k)

        out = multihead_temporal_attention(Q, enc_K, enc_V, causal=False)  # (B, P1, N, d_model)
        out = self.linear(out)  # (B, P1, N, d_model)
        out = self.layer_norm(out + inputs)  # (B, P1, N, d_model)

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_hidden_ff, d_model)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, inputs):
        out = self.relu(self.linear1(inputs))
        out = self.linear2(out)
        out = self.layer_norm(out + inputs)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, cfgs):
        super(EncoderLayer, self).__init__()
        d_model = cfgs['d_model']
        d_k = cfgs['d_k']
        d_hidden_mt = cfgs['d_hidden_mt']
        d_hidden_ff = cfgs['d_hidden_ff']
        num_heads = cfgs['num_heads']
        which_transition_matrices = cfgs['which_transition_matrices']
        dropout = cfgs['dropout']
        self.noTSA = cfgs.get('noTSA', False)
        self.noSSA = cfgs.get('noSSA', False)
        self.noMeta = cfgs.get('noMeta', False)
        self.noTE = cfgs.get('noTE', False)
        self.noSE = cfgs.get('noSE', False)
        if self.noTE and self.noSE:
            self.noMeta = True

        if not self.noTSA:
            self.temporal_self_attention = TemporalSelfAttention(
                d_model, d_k, d_hidden_mt, num_heads, self.noMeta, causal=False)
        if not self.noSSA:
            self.spatial_self_attention = SpatialSelfAttention(
                d_model, d_k, d_hidden_mt, num_heads, which_transition_matrices, dropout, self.noMeta)
        self.feed_forward = FeedForward(d_model, d_hidden_ff)

    def forward(self, inputs, c_inputs, transition_matrices):
        out = inputs
        if not self.noTSA:
            out = self.temporal_self_attention(out, c_inputs)
        if not self.noSSA:
            out = self.spatial_self_attention(out, c_inputs, transition_matrices)
        out = self.feed_forward(out)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, cfgs):
        super(DecoderLayer, self).__init__()
        d_model = cfgs['d_model']
        d_k = cfgs['d_k']
        d_hidden_mt = cfgs['d_hidden_mt']
        d_hidden_ff = cfgs['d_hidden_ff']
        num_heads = cfgs['num_heads']
        which_transition_matrices = cfgs['which_transition_matrices']
        dropout = cfgs['dropout']
        self.noTSA = cfgs.get('noTSA', False)
        self.noSSA = cfgs.get('noSSA', False)
        self.noMeta = cfgs.get('noMeta', False)
        self.noTE = cfgs.get('noTE', False)
        self.noSE = cfgs.get('noSE', False)
        if self.noTE and self.noSE:
            self.noMeta = True

        if not self.noTSA:
            self.temporal_self_attention = TemporalSelfAttention(
                d_model, d_k, d_hidden_mt, num_heads, self.noMeta, causal=True)
        if not self.noSSA:
            self.spatial_self_attention = SpatialSelfAttention(
                d_model, d_k, d_hidden_mt, num_heads, which_transition_matrices, dropout, self.noMeta)
        self.temporal_encoder_decoder_attention = TemporalEncoderDecoderAttention(
            d_model, d_k, d_hidden_mt, num_heads, self.noMeta)
        self.feed_forward = FeedForward(d_model, d_hidden_ff)

    def forward(self, inputs, enc_K, enc_V, c_targets, transition_matrices):
        out = inputs
        if not self.noTSA:
            out = self.temporal_self_attention(out, c_targets)
        if not self.noSSA:
            out = self.spatial_self_attention(out, c_targets, transition_matrices)
        out = self.temporal_encoder_decoder_attention(out, enc_K, enc_V, c_targets)
        out = self.feed_forward(out)

        return out


class Project(nn.Module):
    def __init__(self, d_model, num_features):
        super(Project, self).__init__()
        self.linear = nn.Linear(d_model, num_features)

    def forward(self, inputs):
        out = self.linear(inputs)

        return out


class Encoder(nn.Module):
    def __init__(self, cfgs):
        super(Encoder, self).__init__()
        num_features = cfgs['num_features']
        d_model = cfgs['d_model']
        num_encoder_layers = cfgs['num_encoder_layers']
        self.noMeta = cfgs.get('noMeta', False)
        self.noTE = cfgs.get('noTE', False)
        self.noSE = cfgs.get('noSE', False)

        self.linear = nn.Linear(num_features, d_model)
        self.layer_stack = nn.ModuleList([EncoderLayer(cfgs) for _ in range(num_encoder_layers)])

    def forward(self, inputs, c_inputs, transition_matrices):
        if self.noMeta and ((not self.noTE) or (not self.noSE)):
            out = F.relu(self.linear(inputs) + c_inputs)
        else:
            out = F.relu(self.linear(inputs))
        skip = 0
        for encoder_layer in self.layer_stack:
            out = encoder_layer(out, c_inputs, transition_matrices)
            skip += out

        return skip


class Decoder(nn.Module):
    def __init__(self, cfgs):
        super(Decoder, self).__init__()
        d_model = cfgs['d_model']
        d_k = cfgs['d_k']
        d_hidden_mt = cfgs['d_hidden_mt']
        num_features = cfgs['num_features']
        num_heads = cfgs['num_heads']
        num_decoder_layers = cfgs['num_decoder_layers']
        self.out_len = cfgs['out_len']
        self.use_curriculum_learning = cfgs['use_curriculum_learning']
        self.cl_decay_steps = cfgs['cl_decay_steps']
        self.noMeta = cfgs.get('noMeta', False)
        self.noTE = cfgs.get('noTE', False)
        self.noSE = cfgs.get('noSE', False)

        if self.noMeta or (self.noTE and self.noSE):
            self.num_heads = num_heads
            self.d_k = d_k
            self.linear_k = nn.Linear(d_model, d_model, bias=False)
            self.linear_v = nn.Linear(d_model, d_model, bias=False)
        else:
            self.meta_learner = MetaLearner(d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=2)

        self.linear = nn.Linear(num_features, d_model)
        self.layer_stack = nn.ModuleList([DecoderLayer(cfgs) for _ in range(num_decoder_layers)])
        self.project = Project(d_model, num_features)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, inputs, targets, c_inputs, c_targets, enc_outputs, transition_matrices, batches_seen):
        if self.noMeta or (self.noTE and self.noSE):
            B, P, N, _ = enc_outputs.shape
            enc_K = self.linear_k(enc_outputs).reshape((B, P, N, self.num_heads, self.d_k))
            enc_V = self.linear_v(enc_outputs).reshape((B, P, N, self.num_heads, self.d_k))
        else:
            W_k, W_v = self.meta_learner(c_inputs)  # (B, P, N, H, d_k, d_model)
            enc_K = multihead_linear_transform(W_k, enc_outputs)  # (B, P, N, H, d_k)
            enc_V = multihead_linear_transform(W_v, enc_outputs)  # (B, P, N, H, d_k)

        use_targets = False
        if self.training and (targets is not None) and self.use_curriculum_learning:
            c = np.random.uniform(0, 1)
            if c < self._compute_sampling_threshold(batches_seen):
                use_targets = True

        if use_targets is True:
            dec_inputs = torch.cat((inputs[:, -1, :, :].unsqueeze(1), targets[:, :-1, :, :]), dim=1)  # (B, Q, N, C)
            if self.noMeta and ((not self.noTE) or (not self.noSE)):
                out = F.relu(self.linear(dec_inputs) + c_targets)
            else:
                out = F.relu(self.linear(dec_inputs))  # (B, Q, N, d_model)
            skip = 0
            for decoder_layer in self.layer_stack:
                out = decoder_layer(out, enc_K, enc_V, c_targets, transition_matrices)
                skip += out
            outputs = self.project(skip)  # (B, Q, N, C)
        else:
            dec_inputs = inputs[:, -1, :, :].unsqueeze(1)  # (B, 1, N, C)
            outputs = []
            for i in range(self.out_len):
                if self.noMeta and ((not self.noTE) or (not self.noSE)):
                    out = F.relu(self.linear(dec_inputs) + c_targets[:, :(i + 1), :, :])
                else:
                    out = F.relu(self.linear(dec_inputs))  # (B, *, N, d_model)
                skip = 0
                for decoder_layer in self.layer_stack:
                    if (not self.noTE) or (not self.noSE):
                        out = decoder_layer(out, enc_K, enc_V, c_targets[:, :(i + 1), :, :], transition_matrices)
                    else:
                        out = decoder_layer(out, enc_K, enc_V, None, transition_matrices)
                    skip += out
                out = self.project(skip)  # (B, *, N, C)
                outputs.append(out[:, -1, :, :])
                dec_inputs = torch.cat((dec_inputs, out[:, -1, :, :].unsqueeze(1)), dim=1)
            outputs = torch.stack(outputs, dim=1)  # (B, Q, N, C)

        return outputs


class MGT(nn.Module):
    def __init__(self, cfgs):
        super(MGT, self).__init__()
        d_model = cfgs['d_model']
        num_embeddings = cfgs['num_embeddings']
        eigenmaps_k = cfgs['eigenmaps_k']
        self.in_len = cfgs['in_len']
        self.out_len = cfgs['out_len']
        max_len = max(self.in_len, self.out_len)
        self.noTE = cfgs.get('noTE', False)
        self.noSE = cfgs.get('noSE', False)
        self.batches_seen = 0

        if not self.noTE:
            self.temporal_embedding = TemporalEmbedding(num_embeddings, d_model, max_len)
        if not self.noSE:
            self.spatial_embedding = SpatialEmbedding(eigenmaps_k, d_model)

        if (not self.noTE) and (not self.noSE):
            self.spatial_temporal_embedding = SpatialTemporalEmbedding(d_model)

        self.encoder = Encoder(cfgs)
        self.decoder = Decoder(cfgs)

    def forward(self, inputs, targets, *extras, **statics):
        if not self.noTE:
            z_inputs, z_targets = self.temporal_embedding(extras)  # (B, P, d_model), (B, Q, d_model)
        if not self.noSE:
            u = self.spatial_embedding(statics['eigenmaps'])  # (N, d_model)
        if (not self.noTE) and (not self.noSE):
            c_inputs, c_targets = self.spatial_temporal_embedding(
                z_inputs, z_targets, u)  # (B, P, N, d_model), (B, Q, N, d_model)
        elif self.noTE and (not self.noSE):
            B = inputs.size(0)
            P = self.in_len
            Q = self.out_len
            N = u.size(0)
            d_model = u.size(1)
            c_inputs = u.expand(B, P, N, d_model)
            c_targets = u.expand(B, Q, N, d_model)
        elif (not self.noTE) and self.noSE:
            N = inputs.size(2)
            c_inputs = torch.stack((z_inputs,) * N, dim=2)
            c_targets = torch.stack((z_targets,) * N, dim=2)
        else:
            c_inputs = None
            c_targets = None

        transition_matrices = statics['transition_matrices']

        enc_outputs = self.encoder(inputs, c_inputs, transition_matrices)
        outputs = self.decoder(inputs, targets, c_inputs, c_targets, enc_outputs, transition_matrices,
                               self.batches_seen)

        if self.training:
            self.batches_seen += 1

        return outputs


if __name__ == '__main__':
    cfgs = yaml.safe_load(open('cfgs/HZMetro_MGT.yaml'))['model']
    model = MGT(cfgs)

    # dummy data
    B, P, Q, N, C = 10, 4, 4, 80, 2
    M = 73, 2
    eigenmaps_k = 8
    n = 3

    inputs = torch.randn(B, P, N, C, dtype=torch.float32)
    targets = torch.randn(B, Q, N, C, dtype=torch.float32)

    inputs_time0 = torch.randint(M[0], (B, P), dtype=torch.int64)
    targets_time0 = torch.randint(M[0], (B, Q), dtype=torch.int64)
    inputs_time1 = torch.randint(M[1], (B, P), dtype=torch.int64)
    targets_time1 = torch.randint(M[1], (B, Q), dtype=torch.int64)

    eigenmaps = torch.randn(N, eigenmaps_k, dtype=torch.float32)

    transition_matrices = torch.rand(n, N, N, dtype=torch.float32)

    extras = [inputs_time0, targets_time0, inputs_time1, targets_time1]
    statics = {'eigenmaps': eigenmaps, 'transition_matrices': transition_matrices}

    # forward
    outputs1 = model(inputs, targets, *extras, **statics)
    outputs2 = model(inputs, None, *extras, **statics)




