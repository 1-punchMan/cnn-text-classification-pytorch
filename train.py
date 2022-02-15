import os, subprocess
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from logging import getLogger

logger = getLogger()

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            model.train()
            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.item(), 
                                                                             accuracy.item(),
                                                                             corrects.item(),
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.item()+1]
    return accuracy

def ETST_train(train_set, dev_set, model, args):
    if args.cuda:
        model.cuda()

    train_iter = train_set.dataloader
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    stopping_cnt = 0
    batch_size = args.batch_size
    for epoch in range(1, args.epochs+1):
        logger.info(f"Start epoch {epoch}")

        for feature, target in train_iter:
            model.train()
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / batch_size
                logger.info(
                    'epoch {} | Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(
                        epoch,
                        steps, 
                        loss.item(), 
                        accuracy.item(),
                        corrects.item(),
                        batch_size
                        ))
            if steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)
                logger.info("")
            if steps % args.test_interval == 0:
                logger.info("")
                dev_acc = ETST_eval(dev_set, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    stopping_cnt = 0
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    stopping_cnt += 1
                    logger.info(f'Not better ({stopping_cnt} / {args.early_stop})')
                    logger.info(f'Best score: {best_acc:.4f}%')

                    if stopping_cnt >= args.early_stop:
                        logger.info('early stop by {} steps.'.format(args.early_stop))
                        exit()
                logger.info("")

        logger.info("")


def ETST_eval(dataset, model, args):
    model.eval()
    data_iter = dataset.dataloader
    corrects, sum_loss = 0, 0
    for feature, target in data_iter:
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, reduction="sum")

        sum_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum().item()

    size = len(dataset)
    avg_loss = sum_loss / size
    accuracy = 100.0 * corrects/size
    logger.info('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy

def ETST_predict(text, data, model, cuda_flag, tokenize=True):

    def preprocess(text):
        if tokenize:
            zh_char_tokenize = "/home/zchen/encyclopedia-text-style-transfer/tools/zh_char_tokenize.sh"
            command=f'{zh_char_tokenize} 2>&-'
            cwd = "/home/zchen/encyclopedia-text-style-transfer/tools/"
            CompletedProcess = subprocess.run(command, input=text, cwd=cwd, encoding="utf-8", shell=True, stdout=subprocess.PIPE)
            text = CompletedProcess.stdout.strip().split()  # tokenize.sh會多加一個'\n'在最後
        numericalized = [[params.bos_index] + [data["dico"].index(x) for x in text] + [params.eos_index]]

        return numericalized

    assert isinstance(text, str)
    numericalized = preprocess(text)
    x = torch.tensor(numericalized)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)

    model.eval()
    output = model(x)
    _, predicted = torch.max(output, 1)

    return data["id2label"][predicted.item()]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{save_prefix}.pt')
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved {save_prefix} at step {steps}")
