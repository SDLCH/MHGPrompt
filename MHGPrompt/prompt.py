import torch
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F
import torch.nn as nn


class Prompt(nn.Module):
    def __init__(self, in_channels: int, com_prompt_module: nn.Module, p_num: int):
        super(Prompt, self).__init__()
        self.com_prompt = com_prompt_module
        self.lora_prompt = Low_Rank_Adaptation_Prompt(p_num, in_channels)
        self.prompt_mix_weight = nn.Parameter(torch.FloatTensor(1, 2))
        self.prompt_attention = nn.Linear(in_channels, p_num)
        self.dropout = nn.Dropout(p=0.5)
        self.reset_parameters()

    def reset_parameters(self):
        self.prompt_attention.reset_parameters()
        glorot(self.prompt_mix_weight)

    def add(self, x: torch.Tensor):
        com_prompt = self.com_prompt()
        lora_prompt = self.lora_prompt()
        mix_ratio = F.softmax(self.prompt_mix_weight, dim=1)
        prompt = lora_prompt * mix_ratio[0][0] + com_prompt * mix_ratio[0][1]
        score = self.prompt_attention(x)
        score = self.dropout(score)
        weight = F.softmax(score, dim=1)
        p = weight.mm(prompt)

        return x + p


class Composite_Prompt(nn.Module):
    def __init__(self, prompt1, prompt2, prompt3):
        super(Composite_Prompt, self).__init__()
        self.prompt1 = prompt1
        self.prompt2 = prompt2
        self.prompt3 = prompt3
        in_channels = prompt1.shape[1]
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=2, batch_first=True)

    def forward(self):
        prompt_sequence = torch.stack([self.prompt1, self.prompt2, self.prompt3], dim=1)
        attn_output, _ = self.attention(prompt_sequence, prompt_sequence, prompt_sequence)
        final_prompt_pool = attn_output.mean(dim=1)

        return final_prompt_pool


class Low_Rank_Adaptation_Prompt(nn.Module):
    def __init__(self, p_num: int, in_channels: int, rank: int = 1):
        super(Low_Rank_Adaptation_Prompt, self).__init__()
        self.lora_A = nn.Parameter(torch.Tensor(p_num, rank))
        self.lora_B = nn.Parameter(torch.Tensor(rank, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self) -> torch.Tensor:

        return self.lora_A.mm(self.lora_B)


class Pretext_Token(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(Pretext_Token, self).__init__()
        self.prompt = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.dropout = nn.Dropout(p=0.5)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.prompt)
        self.a.reset_parameters()

    def add(self, x: torch.Tensor):
        score = self.a(x)
        score = self.dropout(score)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.prompt)

        return x + p

