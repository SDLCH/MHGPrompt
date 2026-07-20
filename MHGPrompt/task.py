import torch
import time
import os
from torch import optim, nn
import numpy as np
from torchmetrics.classification import MulticlassF1Score

from .prompt import Composite_Prompt, Prompt
from .evaluate import evaluate
from .utils import node_sample_and_save
from pretrain.models.edcoder import PreModel
from pretrain.utils.params import build_args
from pretrain.utils import load_best_configs


class Task:
    def __init__(self, dataset, epochs, shot_num, device, num_mp,
                 feats_dim_list, patience, p_num, lr, wd,
                 feats, mps, input_dim, output_dim, task_num, label, idx_train, idx_val, idx_test
                 ):
        self.device = device
        self.args = build_args()
        self.args.dataset = dataset
        self.args.p_num = p_num
        if self.args.use_cfg:
            config_file_name = "./pretrain/configs.yml"
            self.args = load_best_configs(self.args, config_file_name)
        self.dataset = dataset
        self.shot_num = shot_num
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.num_mp = num_mp
        self.feats_dim_list = feats_dim_list
        self.patience = patience
        self.p_num = p_num
        self.hid_dim = self.args.hidden_dim

        self.task_num = task_num
        self.feats = feats[0]
        self.mps = mps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.label = label
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.create_few_data_folder()
        self.criterion = nn.CrossEntropyLoss()

    def create_few_data_folder(self):
        k_shot = self.shot_num
        task_num = self.task_num
        for k in range(1, k_shot + 1):
            k_shot_folder = './data/' + self.dataset + '/' + str(k) + '_shot'
            expected_files = [os.path.join(k_shot_folder, f'train_{i}_idx.pt') for i in range(1, task_num + 1)]
            all_files_exist = all(os.path.exists(f) for f in expected_files)
            if not all_files_exist:
                if os.path.exists(k_shot_folder):
                    print(f"Folder {k_shot_folder} is incomplete. Regenerating all task data.")
                    import shutil
                    shutil.rmtree(k_shot_folder)
                else:
                    print(f"Folder {k_shot_folder} not found. Starting data generation.")
                os.makedirs(k_shot_folder, exist_ok=True)
                for i in range(1, task_num + 1):
                    node_sample_and_save(self.idx_train, self.label, k, i, k_shot_folder, self.output_dim)
                print(self.dataset + " " + str(k) + ' shot ' + ' this saved!!\n')

    def initialize_optimizer(self):
        model_param_group = [{"params": self.prompt.parameters(), "lr": self.lr * 0.001},
                             {"params": self.answering.parameters(), "lr": self.lr}]
        self.optimizer = optim.Adam(model_param_group, weight_decay=self.wd)

    def initialize_prompt(self):
        self.com_prompt = Composite_Prompt(self.model.hgmaeprompt1.prompt,
                                           self.model.hgmaeprompt2.prompt,
                                           self.model.hecoprompt.prompt).to(self.device)
        self.prompt = Prompt(self.input_dim, self.com_prompt, self.p_num).to(self.device)

    def initialize_model(self):
        self.model = PreModel(self.args, self.num_mp, self.feats_dim_list)
        model_path = f'./pre_trained_models/{self.dataset}.pth'
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print("Successfully loaded pre-trained weights!")

    def Train(self, idx_train):
        self.prompt.train()
        self.answering.train()
        self.optimizer.zero_grad()
        feat = self.prompt.add(self.feats)
        embeds = self.model.get_embeds(feat, self.mps)
        train_embs = embeds[idx_train]
        train_lbls = torch.argmax(self.label[idx_train], dim=-1)
        out = self.answering(train_embs)
        loss = self.criterion(out, train_lbls)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def GPFValidate(self):
        self.prompt.eval()
        self.answering.eval()
        with torch.no_grad():
            feat = self.prompt.add(self.feats)
            embeds = self.model.get_embeds(feat, self.mps)
            val_embs = embeds[self.idx_val]
            val_lbls = torch.argmax(self.label[self.idx_val], dim=-1)
            out = self.answering(val_embs)
            val_loss = self.criterion(out, val_lbls)
        return val_loss.item()

    def run(self):
        mi_f1s = []
        ma_f1s = []
        best_loss = []

        for i in range(1, self.task_num + 1):
            data_folder_path = f"./data/{self.dataset}/{self.shot_num}_shot/"
            if not os.path.exists(data_folder_path):
                raise FileNotFoundError(
                    f"Failed to find sample_data for shot {self.shot_num}, id {i}, path: {data_folder_path}"
                )

            layer = nn.Linear(self.hid_dim, self.output_dim)
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)
            self.answering = nn.Sequential(layer).to(self.device)

            self.initialize_model()
            self.initialize_prompt()
            self.initialize_optimizer()

            idx_train = torch.load(f"{data_folder_path}/train_{i}_idx.pt").type(torch.long).to(self.device)

            best_val_loss = 1e9
            cnt_wait = 0
            best_t = 0

            best_prompt_state = None
            best_answering_state = None

            for epoch in range(1, self.epochs + 1):
                t0 = time.time()
                train_loss = self.Train(idx_train)
                val_loss = self.GPFValidate()

                print(
                    f"Epoch {epoch:03d} | Time(s) {time.time() - t0:.4f} | Train Loss {train_loss:.4f} "
                    f"| Val Loss {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_t = epoch
                    cnt_wait = 0
                    best_prompt_state = self.prompt.state_dict()
                    best_answering_state = self.answering.state_dict()
                else:
                    cnt_wait += 1

                if cnt_wait == self.patience:
                    print('-' * 100)
                    print(f'Early stopping at {best_t} eopch! Best Val Loss:{ best_val_loss:.4f}')
                    break

            print('The best epoch is: ', best_t)
            self.prompt.load_state_dict(best_prompt_state)
            self.answering.load_state_dict(best_answering_state)
            best_loss.append(best_val_loss)

            ma_f1, mi_f1 = evaluate(self.feats, self.mps, self.idx_test, self.label, self.model,
                                         self.prompt, self.answering, self.output_dim, self.device)
            mi_f1s.append(mi_f1)
            ma_f1s.append(ma_f1)

        print("best_loss", best_loss)
        print('Mi-F1 List', mi_f1s)
        print('Ma-F1 List', ma_f1s)
        print(f" Final best | test Mi-F1 {np.mean(mi_f1s) * 100:.2f}±{np.std(mi_f1s) * 100:.2f}(std)")
        print(f" Final best | test Ma-F1 {np.mean(ma_f1s) * 100:.2f}±{np.std(ma_f1s) * 100:.2f}(std)")
