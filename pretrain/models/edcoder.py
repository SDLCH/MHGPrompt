from functools import partial
import torch
import torch.nn as nn
import dgl
from dgl import DropEdge

from pretrain.utils import sce_loss
from pretrain.utils import create_norm
from pretrain.models.han import HAN
from pretrain.models.contrast import Contrast
from pretrain.models.sc_encoder import Sc_encoder
from MHGPrompt.prompt import Pretext_Token


class PreModel(nn.Module):
    def __init__(
            self,
            args,
            num_metapath: int,
            feats_dim_list: list,
    ):
        super(PreModel, self).__init__()

        self.num_metapath = num_metapath
        self.focused_feature_dim = feats_dim_list[0]
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.num_out_heads = args.num_out_heads
        self.activation = args.activation
        self.feat_drop = args.feat_drop
        self.attn_drop = args.attn_drop
        self.negative_slope = args.negative_slope
        self.residual = args.residual
        self.norm = args.norm
        self.feat_mask_rate = args.feat_mask_rate
        self.encoder_type = args.encoder
        self.decoder_type = args.decoder
        self.loss_fn = args.loss_fn
        self.enc_dec_input_dim = self.focused_feature_dim
        assert self.hidden_dim % self.num_heads == 0
        assert self.hidden_dim % self.num_out_heads == 0

        self.sample_rate = args.sample_rate
        self.nei_num = args.nei_num
        self.tau = args.tau
        self.lam = args.lam
        self.heco_loss_weight = args.heco_loss_weight
        self.p_num = args.p_num

        self.hgmaeprompt1 = Pretext_Token(self.focused_feature_dim, self.p_num).to(torch.device("cuda:0"))
        self.hgmaeprompt2 = Pretext_Token(self.focused_feature_dim, self.p_num).to(torch.device("cuda:0"))
        self.hecoprompt = Pretext_Token(self.focused_feature_dim, self.p_num).to(torch.device("cuda:0"))

        if self.encoder_type in ("gat", "dotgat", "han"):
            enc_num_hidden = self.hidden_dim // self.num_heads
            enc_nhead = self.num_heads
        else:
            enc_num_hidden = self.hidden_dim
            enc_nhead = 1

        if self.decoder_type in ("gat", "dotgat", "han"):
            dec_num_hidden = self.hidden_dim // self.num_out_heads
            dec_nhead = self.num_out_heads
        else:
            dec_num_hidden = self.hidden_dim
            dec_nhead = 1
        dec_in_dim = self.hidden_dim

        self.encoder = setup_module(
            num_metapath=self.num_metapath,
            m_type=self.encoder_type,
            enc_dec="encoding",
            in_dim=self.enc_dec_input_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=self.num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=self.activation,
            dropout=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=self.norm,
        )

        self.decoder = setup_module(
            num_metapath=self.num_metapath,
            m_type=self.decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=self.enc_dec_input_dim,
            num_layers=1,
            nhead=enc_nhead,
            nhead_out=dec_nhead,
            activation=self.activation,
            dropout=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=self.norm,
            concat_out=True,
        )

        self.alpha_l = args.alpha_l
        self.attr_restoration_loss = self.setup_loss_fn(self.loss_fn, self.alpha_l)
        self.__cache_gs = None
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.enc_dec_input_dim))
        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self._replace_rate = args.replace_rate
        self._leave_unchanged = args.leave_unchanged
        assert self._replace_rate + self._leave_unchanged < 1, "Replace rate + leave_unchanged must be smaller than 1"

        self.use_mp_edge_recon = args.use_mp_edge_recon
        self.mp_edge_recon_loss_weight = args.mp_edge_recon_loss_weight
        self.mp_edge_mask_rate = args.mp_edge_mask_rate
        self.mp_edge_alpha_l = args.mp_edge_alpha_l
        self.mp_edge_recon_loss = self.setup_loss_fn(self.loss_fn, self.mp_edge_alpha_l)
        self.encoder_to_decoder_edge_recon = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        self.mps_embedding_dim = args.mps_embedding_dim
        self.use_mp2vec_feat_pred = args.use_mp2vec_feat_pred
        self.mp2vec_feat_pred_loss_weight = args.mp2vec_feat_pred_loss_weight
        self.mp2vec_feat_alpha_l = args.mp2vec_feat_alpha_l
        self.mp2vec_feat_drop = args.mp2vec_feat_drop
        self.mp2vec_feat_pred_loss = self.setup_loss_fn(self.loss_fn, self.mp2vec_feat_alpha_l)
        self.enc_out_to_mp2vec_feat_mapping = nn.Sequential(
            nn.Linear(dec_in_dim, self.mps_embedding_dim),
            nn.PReLU(),
            nn.Dropout(self.mp2vec_feat_drop),
            nn.Linear(self.mps_embedding_dim, self.mps_embedding_dim),
            nn.PReLU(),
            nn.Dropout(self.mp2vec_feat_drop),
            nn.Linear(self.mps_embedding_dim, self.mps_embedding_dim)
        )

        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, self.hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if self.feat_drop > 0:
            self.feat_drop = nn.Dropout(self.feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.sc = Sc_encoder(self.hidden_dim, self.sample_rate, self.nei_num, self.attn_drop)
        self.contrast = Contrast(self.hidden_dim, self.tau, self.lam)
        self.act = nn.PReLU()

    def get_mask_rate(self, input_mask_rate, get_min=False, epoch=None):
        try:
            return float(input_mask_rate)
        except ValueError:
            if "~" in input_mask_rate:  # 0.6~0.8 Uniform sample
                mask_rate = [float(i) for i in input_mask_rate.split('~')]
                assert len(mask_rate) == 2
                if get_min:
                    return mask_rate[0]
                else:
                    return torch.empty(1).uniform_(mask_rate[0], mask_rate[1]).item()
            elif "," in input_mask_rate:  # 0.6,-0.1,0.4 stepwise increment/decrement
                mask_rate = [float(i) for i in input_mask_rate.split(',')]
                assert len(mask_rate) == 3
                start = mask_rate[0]
                step = mask_rate[1]
                end = mask_rate[2]
                if get_min:
                    return min(start, end)
                else:
                    cur_mask_rate = start + epoch * step
                    if cur_mask_rate < min(start, end) or cur_mask_rate > max(start, end):
                        return end
                    return cur_mask_rate
            else:
                raise NotImplementedError

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        perm_mask = torch.randperm(num_mask_nodes, device=x.device)
        num_leave_nodes = int(self._leave_unchanged * num_mask_nodes)
        num_noise_nodes = int(self._replace_rate * num_mask_nodes)
        num_real_mask_nodes = num_mask_nodes - num_leave_nodes - num_noise_nodes
        token_nodes = mask_nodes[perm_mask[: num_real_mask_nodes]]
        noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
        noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

        out_x = x.clone()
        out_x[token_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token
        if num_noise_nodes > 0:
            out_x[noise_nodes] = x[noise_to_be_chosen]

        return out_x, (mask_nodes, keep_nodes)

    def mask_attr_restoration(self, feat, gs, epoch):
        cur_feat_mask_rate = self.get_mask_rate(self.feat_mask_rate, epoch=epoch)
        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(feat, cur_feat_mask_rate)

        use_x = self.hgmaeprompt1.add(use_x)
        enc_out, _ = self.encoder(gs, use_x)

        enc_out_mapped = self.encoder_to_decoder(enc_out)

        enc_out_mapped[mask_nodes] = 0

        feat_recon, att_mp = self.decoder(gs, enc_out_mapped)

        x_init = feat[mask_nodes]
        x_rec = feat_recon[mask_nodes]
        loss = self.attr_restoration_loss(x_rec, x_init)

        return loss, feat_recon, att_mp, enc_out, mask_nodes

    def mask_mp_edge_reconstruction(self, feat, mps, epoch):
        masked_gs = self.mps_to_gs(mps)
        cur_mp_edge_mask_rate = self.get_mask_rate(self.mp_edge_mask_rate, epoch=epoch)
        drop_edge = DropEdge(p=cur_mp_edge_mask_rate)
        for i in range(len(masked_gs)):
            masked_gs[i] = drop_edge(masked_gs[i])
            masked_gs[i] = dgl.add_self_loop(masked_gs[i])
        feat1 = self.hgmaeprompt2.add(feat.clone())
        enc_rep, _ = self.encoder(masked_gs, feat1)
        rep = self.encoder_to_decoder_edge_recon(enc_rep)

        feat_recon, att_mp = self.decoder(masked_gs, rep)

        gs_recon = torch.mm(feat_recon, feat_recon.T)

        loss = None
        for i in range(len(mps)):
            if loss is None:
                loss = att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())
            else:
                loss += att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())
        return loss

    def mps_to_gs(self, mps):
        if self.__cache_gs is None:
            gs = []
            device = mps[0].device
            for mp in mps:
                indices = mp._indices()
                cur_graph = dgl.graph((indices[0], indices[1]))
                gs.append(cur_graph.to(device))
            self.__cache_gs = gs
            return gs
        else:
            device = mps[0].device
            gs_copy = []
            for g in self.__cache_gs:
                edges = g.edges()
                new_g = dgl.graph((edges[0], edges[1]), num_nodes=g.num_nodes())
                new_g = new_g.to(device)
                gs_copy.append(new_g)
            return gs_copy

    def forward(self, feats, mps, epoch, pos, nei_index):
        if self.use_mp2vec_feat_pred:
            mp2vec_feat = feats[0][:, self.focused_feature_dim:]
            origin_feat = feats[0][:, :self.focused_feature_dim]
        else:
            origin_feat = feats[0]

        gs = self.mps_to_gs(mps)
        loss, feat_recon, att_mp, enc_out, mask_nodes = self.mask_attr_restoration(origin_feat, gs, epoch)

        if self.use_mp_edge_recon:
            edge_recon_loss = self.mask_mp_edge_reconstruction(origin_feat, mps, epoch)
            loss += self.mp_edge_recon_loss_weight * edge_recon_loss

        if self.use_mp2vec_feat_pred:
            mp2vec_feat_pred = self.enc_out_to_mp2vec_feat_mapping(enc_out)

            mp2vec_feat_pred_loss = self.mp2vec_feat_pred_loss(mp2vec_feat_pred, mp2vec_feat)

            loss += self.mp2vec_feat_pred_loss_weight * mp2vec_feat_pred_loss

        origin_feat_all = [origin_feat]+feats[1:]
        h_all = []
        for i in range(len(origin_feat_all)):
            h_all.append(self.act(self.feat_drop(self.fc_list[i](origin_feat_all[i]))))
        feat2 = self.hecoprompt.add(origin_feat.clone())
        z_mp, _ = self.encoder(gs, feat2)
        z_sc = self.sc(h_all, nei_index)
        heco_loss = self.contrast(z_mp, z_sc, pos)
        loss += self.heco_loss_weight*heco_loss

        return loss, loss.item()

    def get_embeds(self, feats, mps):
        gs = self.mps_to_gs(mps)
        rep, _ = self.encoder(gs, feats)
        return rep


def setup_module(m_type, num_metapath, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual,
                 norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "han":
        mod = HAN(
            num_metapath=num_metapath,
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    else:
        raise NotImplementedError

    return mod
