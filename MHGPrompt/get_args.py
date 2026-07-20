import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Heterogeneous Graph Neural Network Prompt Learning"
    )
    parser.add_argument("--dataset", type=str, default="acm", help="Choose the dataset of downstream task")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train ")
    parser.add_argument("--patience", type=int, default=20, help="Stop training after this many epochs")
    parser.add_argument("--lr", type=float, default=0.004, help="Learning rate ")
    parser.add_argument("--decay", type=float, default=0, help="Weight decay ")
    parser.add_argument("--p_num", type=int, default=45, help="The number of independent basis for Prompt")
    parser.add_argument("--shot_num", type=int, default=3, help="Number of shots")
    parser.add_argument("--task_num", type=int, default=5, help="The number of tasks for computing the mean metrices")
    parser.add_argument("--seed", type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60], help=" Dataset parameters")
    parser.add_argument("--device", type=int, default=0, help="Which gpu to use if any (default: 0)")

    args, _ = parser.parse_known_args()
    for key, value in datasets_args[args.dataset].items():
        setattr(args, key, value)
    return args


datasets_args = {
    "dblp": {
        "type_num": [4057, 14328, 7723, 20],  # the number of every node type
        "nei_num": 1,  # the number of neighbors' types
        "n_labels": 4,
    },
    "aminer": {
        "type_num": [6564, 13329, 35890],
        "nei_num": 2,
        "n_labels": 4,
    },
    "freebase": {
        "type_num": [3492, 2502, 33401, 4459],
        "nei_num": 3,
        "n_labels": 3,
    },
    "acm": {
        "type_num": [4019, 7167, 60],
        "nei_num": 2,
        "n_labels": 3,
    },
}
