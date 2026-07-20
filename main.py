import torch
from MHGPrompt import set_random_seed
from MHGPrompt import get_args
from pretrain.utils import load_data, load_best_configs
from MHGPrompt.task import Task


def main(args):
    set_random_seed(args.seed)

    (nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test), g, processed_metapaths = \
        load_data(args.dataset, args.ratio, args.type_num)
    feats_dim_list = [i.shape[1] for i in feats]
    focused_feature_dim = feats_dim_list[0]
    num_mp = int(len(mps))

    idx_train = idx_train[-1]
    idx_val = idx_val[-1]
    idx_test = idx_test[-1]

    feats = [feat.to(device) for feat in feats]
    mps = [mp.to(device) for mp in mps]
    label = label.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    tasker = Task(dataset=args.dataset, epochs=args.epochs, shot_num=args.shot_num, device=device,
                  num_mp=num_mp, feats_dim_list=feats_dim_list, patience=args.patience,
                  p_num=args.p_num, lr=args.lr, wd=args.decay, feats=feats, mps=mps,
                  input_dim=focused_feature_dim, output_dim=args.n_labels, task_num=args.task_num,
                  label=label, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    return tasker


if __name__ == "__main__":
    args = get_args()
    print(args)

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
        torch.cuda.set_device(args.device)
    else:
        device = torch.device("cpu")

    tasker = main(args=args)

    tasker.run()


