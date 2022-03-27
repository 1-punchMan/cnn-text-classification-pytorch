import os
import torch.nn.functional as F

def eval(dataset, model, args):
    model.eval()
    data_iter = dataset.dataloader
    probs1, probs2 = [], []
    for feature, target in data_iter:

        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        output = F.softmax(logit, dim=1).tolist()
        target = target.tolist()
        
        for ps, targ in zip(output, target):
            plist = probs2 if targ else probs1
            plist.append(ps[targ])
    return probs1, probs2

def predict_probs_for_datasets(args, data, model):
    dir1 = os.path.join(args.wiki_dir, "probs")
    dir2 = os.path.join(args.baidu_dir, "probs")
    os.makedirs(dir1)
    os.makedirs(dir2)

    for split in ["train", "valid", "test"]:
        probs1, probs2 = eval(data[split], model, args)
        path = os.path.join(dir1, split)
        save(probs1, path)
        path = os.path.join(dir2, split)
        save(probs2, path)
        
def save(probs, path):
    with open(path, 'w', encoding='utf-8') as f:
        lines = ''.join([str(p) + '\n' for p in probs])
        f.write(lines)