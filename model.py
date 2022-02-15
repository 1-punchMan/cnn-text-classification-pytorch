import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from types import SimpleNamespace
from utils import from_path_import

# from dictionary import Dictionary
from_path_import(
    name="dictionary",
    path="/home/zchen/encyclopedia-text-style-transfer/dictionary.py",
    globals=globals(),
    demands=["Dictionary"]
    )

class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

        if self.args.static:
            self.embed.weight.requires_grad = False

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
    
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit

    @classmethod
    def from_pretrained(cls, name, device="cuda"):
        if name == "ETST":
            path = "/home/zchen/encyclopedia-text-style-transfer/data/vocab"
            vocab = Dictionary.read_vocab(path)
            args = SimpleNamespace(
                dropout=0.5,
                max_norm=3,
                embed_dim=128,
                kernel_num=100,
                kernel_sizes=[3, 4, 5],
                static=False,
                embed_num = len(vocab),
                class_num = 2
                )
            model = cls(args)
            path = "/home/zchen/encyclopedia-text-style-transfer/cnn-text-classification-pytorch/experiments/2/2022-01-03_12-31-57/best.pt"
            model.load_state_dict(torch.load(path, map_location=torch.device(device)))
            return model