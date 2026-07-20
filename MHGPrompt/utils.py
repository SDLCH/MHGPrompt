import random
import numpy as np
import torch
import os


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def node_sample_and_save(idx_train, label, k, i, folder, num_classes):
    label = torch.argmax(label, dim=-1)
    print(f"--- Processing task {i} for {k}-shot sampling ---")
    current_train_labels = label[idx_train]
    few_shot_indices_list = []
    for class_id in range(num_classes):
        indices_of_class_in_current_split = (current_train_labels == class_id).nonzero(as_tuple=True)[0]
        if len(indices_of_class_in_current_split) < k:
            print(
                f"  Warning: Class {class_id} has only {len(indices_of_class_in_current_split)} samples , which is less than k={k}. Using all available samples.")
            selected_local_indices = indices_of_class_in_current_split
        else:
            perm = torch.randperm(len(indices_of_class_in_current_split))
            selected_local_indices = indices_of_class_in_current_split[perm[:k]]
        original_node_ids = idx_train[selected_local_indices]
        few_shot_indices_list.append(original_node_ids)
        print(f"  Class {class_id}: Sampled {len(original_node_ids)} nodes.")

    final_few_shot_indices = torch.cat(few_shot_indices_list)
    final_few_shot_indices = final_few_shot_indices[torch.randperm(final_few_shot_indices.size(0))]
    print(f"Total nodes in new few-shot training set for task {i}: {len(final_few_shot_indices)}")
    save_path = os.path.join(folder, f'train_{i}_idx.pt')
    torch.save(final_few_shot_indices, save_path)
    print(f"Saved new training indices to: {save_path}\n")

