#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.legacy.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets
import logging
from data_utils.loader import load_data as ETST
from utils import from_path_import

# from logger import create_logger
name = "logger"
path = "/home/zchen/encyclopedia-text-style-transfer/logger.py"
demands = ["create_logger"]
from_path_import(name, path, globals(), demands)


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-dataset', type=str, default='MR', help='dataset name')
parser.add_argument('-wiki_dir', type=str, default='', help='where to load wiki data')
parser.add_argument('-baidu_dir', type=str, default='', help='where to load baidu data')
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-tokenize', type=bool, default=True, help='tokenize before predict')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


# load SST dataset
def sst(text_field, label_field,  **kargs):
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data), 
                                        batch_sizes=(args.batch_size, 
                                                     len(dev_data), 
                                                     len(test_data)),
                                        **kargs)
    return train_iter, dev_iter, test_iter


# load MR dataset
def mr(text_field, label_field, **kargs):
    train_data, dev_data = mydatasets.MR.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(args.batch_size, len(dev_data)),
                                **kargs)
    return train_iter, dev_iter

# # load ETST dataset
# def ETST(args):
#     wiki_train_path, wiki_dev_path, wiki_test_path = (
#         os.path.join(args.wiki_dir, "train"),
#         os.path.join(args.wiki_dir, "valid"),
#         os.path.join(args.wiki_dir, "test")
#     )
#     baidu_train_path, baidu_dev_path, baidu_test_path = (
#         os.path.join(args.baidu_dir, "train"),
#         os.path.join(args.baidu_dir, "valid"),
#         os.path.join(args.baidu_dir, "test")
#     )

#     train_data = mydatasets.ETST(text_field, label_field, wiki_train_path, baidu_train_path)
#     dev_data = mydatasets.ETST(text_field, label_field, wiki_dev_path, baidu_dev_path)
#     test_data = mydatasets.ETST(text_field, label_field, wiki_test_path, baidu_test_path)
#     print("Start to build vocab ...")
#     text_field.build_vocab(train_data)
#     label_field.build_vocab(train_data)

#     train_iter = data.Iterator(train_data, batch_size=args.batch_size, **kargs)
#     dev_iter = data.Iterator(dev_data, batch_size=args.batch_size, **kargs)
#     test_iter = data.Iterator(test_data, batch_size=args.batch_size, **kargs)

    # data = load_data(args)
    # return data["train"], data["valid"], data["test"]

if args.predict is None:
    """ create a logger """
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logger = create_logger(os.path.join(args.save_dir, 'train.log'), rank=0)
else:
    logger = create_logger()

# load data
logger.info("Loading data...")
if args.dataset == "MR":
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    train_iter, dev_iter = mr(text_field, label_field, device=-1)
elif args.dataset == "ETST":
    data = ETST(args)
    train_set, dev_set, test_set = data.get("train"), data.get("valid"), data.get("test")
# train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)

# update args and print
if args.dataset == "ETST":
    args.embed_num = args.n_words
    args.class_num = 2
else:
    args.embed_num = len(text_field.vocab)
    args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

logger.info("Parameters:")
for attr, value in sorted(args.__dict__.items()):
    logger.info("\t{}={}".format(attr.upper(), value))


# model
cnn = model.CNN_Text(args)
if args.snapshot is not None:
    logger.info('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()
        

# train or predict
if args.predict is not None:
    label = train.ETST_predict(args.predict, data, cnn, args.cuda, args.tokenize) if args.dataset == "ETST" else train.predict(args.predict, cnn, text_field, label_field, args.cuda)
    logger.info('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.ETST_eval(test_set, cnn, args) if args.dataset == "ETST" else train.eval(test_iter, cnn, args)
    except Exception as e:
        logger.info("\nSorry. The test dataset doesn't  exist.\n")
else:
    logger.info("")
    try:
        train.ETST_train(train_set, dev_set, cnn, args) if args.dataset == "ETST" else train.train(train_iter, dev_iter, cnn, args)
    except KeyboardInterrupt:
        logger.info('\n' + '-' * 89)
        logger.info('Exiting from training early')

