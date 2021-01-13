import os
import numpy as np
import dill
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import get_base_dataset
from models import load_backbone, BaseNet
from evals import test_acc, compute_aurocs, test_pearson

from common import CKPT_PATH, parse_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_result(model, test_loader):
    model.eval()

    def infer_model(model, loader):
        pred, maxprob, probs = list(), list(), list()
        for i, (tokens, _) in enumerate(loader):
            tokens = tokens.to(device)
            outputs = model(tokens)

            prob = torch.nn.Softmax(dim=-1)(outputs)
            res = torch.max(prob, -1)

            maxprob += [s.item() for s in res[0]]
            pred += [s.item() for s in res[1]]
            probs.append(prob.cpu().detach().numpy())
        return pred, maxprob, np.vstack(probs)

    # in-domain
    pred_id, maxprob_id, prob_id = infer_model(model, test_loader)

    outp = {
        "pred": pred_id,
        "maxprob": maxprob_id,
        "prob": prob_id,
    }

    return outp


def infer_all(model, news_loader, fake_loader, corona_loader, args):
    outp = {}

    result_news = infer_result(model, news_loader)
    result_corona = infer_result(model, corona_loader)
    result_fake = infer_result(model, fake_loader)

    if args.dataset == 'agnews':
        outp['IND'] = result_news
        outp['OOD'] = {}
        outp['OOD']['corona'] = result_corona
        outp['OOD']['fake'] = result_fake

    if args.dataset == 'fake':
        outp['IND'] = result_fake
        outp['OOD'] = {}
        outp['OOD']['corona'] = result_corona
        outp['OOD']['news'] = result_news

    if args.dataset == 'corona':
        outp['IND'] = result_corona
        outp['OOD'] = {}
        outp['OOD']['fake'] = result_fake
        outp['OOD']['news'] = result_news
    return outp


def process(model, news_loader, fake_loader, corona_loader):

    result = infer_all(model, news_test, corona_test, fake_test)

    num = args.model_path.split('/')[-1]
    save_path = os.path.join(args.save_path, 'noier', args.model_type, num)
    os.makedirs(save_path, exist_ok=True)
    save_prefix = f'{args.model_type}_{args.data_type}_abl{args.ablation}_del{args.pr_del}' \
        f'_repl{args.pr_repl}_perm{args.perm_ratio}_noise{args.do_noise}_outp.dict'
    save = os.path.join(save_path, save_prefix)
    with open(save, 'wb') as saveFile:
        dill.dump(result, saveFile)


def main():
    args = parse_args(mode='eval')

    args.batch_size = 16

    print('Loading dataset and model...')
    backbone, tokenizer = load_backbone(args.backbone)

    agnews_dataset = get_base_dataset('agnews', tokenizer, args.split_ratio, args.seed, test_only=True)
    corona_dataset = get_base_dataset('corona', tokenizer, args.split_ratio, args.seed, test_only=True)
    fake_dataset = get_base_dataset('fake', tokenizer, args.split_ratio, args.seed, test_only=True)

    agnews_loader = DataLoader(agnews_dataset.test_dataset, shuffle=False,
                               batch_size=args.batch_size, num_workers=4)
    corona_loader = DataLoader(corona_dataset.test_dataset, shuffle=False,
                               batch_size=args.batch_size, num_workers=4)
    fake_loader = DataLoader(fake_dataset.test_dataset, shuffle=False,
                               batch_size=args.batch_size, num_workers=4)
    if args.dataset == 'agenws':
        model = BaseNet(args.backbone, backbone, agnews_dataset.n_classes).to(device)
    elif args.dataset == 'fake':
        model = BaseNet(args.backbone, backbone, fake_dataset.n_classes).to(device)
    elif args.dataset == 'corona':
        model = BaseNet(args.backbone, backbone, corona_dataset.n_classes).to(device)
    else:
        raise ValueError("No corresponding dataset")

    assert args.model_path is not None
    state_dict = torch.load(os.path.join(CKPT_PATH, args.dataset, args.model_path))

    for key in list(state_dict.keys()):  # only keep base parameters
        if key.split('.')[0] not in ['backbone', 'dense', 'net_cls']:
            state_dict.pop(key)

    model.load_state_dict(state_dict)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if not args.save_path:
        args.save_path = f'result'

    result = infer_all(model, agnews_loader, fake_loader, corona_loader, args)
    os.makedirs(args.save_path, exist_ok=True)

    save = os.path.join(args.save_path, f'{args.dataset}_outp.dict')
    with open(save, 'wb') as saveFile:
        dill.dump(result, saveFile)


if __name__ == "__main__":
    main()

