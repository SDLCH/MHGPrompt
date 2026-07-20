import os

import torch
from torch_geometric.nn.models import MetaPath2Vec

from pretrain.models.edcoder import PreModel
from pretrain.utils import (load_best_configs, load_data,
                            metapath2vec_train, preprocess_features,
                            set_random_seed)
from pretrain.utils.params import build_args


def main(args):
    set_random_seed(args.seed)
    (nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test), g, processed_metapaths = \
        load_data(args.dataset, args.ratio, args.type_num)
    feats_dim_list = [i.shape[1] for i in feats]

    num_mp = int(len(mps))
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", num_mp)

    if args.use_mp2vec_feat_pred:
        assert args.mps_embedding_dim > 0
        metapath_model = MetaPath2Vec(g.edge_index_dict,
                                      args.mps_embedding_dim,
                                      processed_metapaths,
                                      args.mps_walk_length,
                                      args.mps_context_size,
                                      args.mps_walks_per_node,
                                      args.mps_num_negative_samples,
                                      sparse=True
                                      )
        metapath2vec_train(args, metapath_model, args.mps_epoch, args.device)
        mp2vec_feat = metapath_model('target').detach()
        del metapath_model
        if args.device.type == 'cuda':
            mp2vec_feat = mp2vec_feat.cpu()
            torch.cuda.empty_cache()
        mp2vec_feat = torch.FloatTensor(preprocess_features(mp2vec_feat))
        feats[0] = torch.hstack([feats[0], mp2vec_feat])

    model = PreModel(args, num_mp, feats_dim_list)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
    if args.scheduler:
        print("--- Use schedular ---")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
    else:
        scheduler = None

    model.to(args.device)
    feats = [feat.to(args.device) for feat in feats]
    mps = [mp.to(args.device) for mp in mps]
    pos = pos.to(args.device)

    cnt_wait = 0
    best = 1e9
    best_t = 0

    best_model_state_dict = None
    for epoch in range(args.mae_epochs):
        model.train()
        optimizer.zero_grad()
        loss, loss_item = model(feats, mps, epoch, pos, nei_index)
        print(f"Epoch: {epoch}, loss: {loss_item}, lr: {optimizer.param_groups[0]['lr']:.6f}")
        if loss_item < best:
            best = loss_item
            best_t = epoch
            cnt_wait = 0
            best_model_state_dict = model.state_dict()
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print(f'Early stopping in {best_t}! loss:{best:.4f}')
            save_dir = "./pre_trained_models"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{args.dataset}.pth")
            torch.save(best_model_state_dict, save_path)
            print(f"+++model saved ! {save_path}")
            break

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()


if __name__ == "__main__":
    args = build_args()

    if torch.cuda.is_available():
        args.device = torch.device("cuda:" + str(args.gpu))
        torch.cuda.set_device(args.gpu)
    else:
        args.device = torch.device("cpu")

    if args.use_cfg:
        config_file_name = "./pretrain/configs.yml"
        args = load_best_configs(args, config_file_name)

    print(args)

    main(args)
