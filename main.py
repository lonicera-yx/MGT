"""Train and test a deep model for traffic forecasting."""
import argparse
import os
import os.path as osp
import json
import time
from datetime import datetime

import yaml
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import *


def get_args():
    parser = argparse.ArgumentParser(description="Train and test a deep model for traffic forecasting.")
    parser.add_argument('dataset', type=str, help="traffic dataset")
    parser.add_argument('model', type=str, help="traffic forecasting model")
    parser.add_argument('name', type=str, help="experiment name")
    parser.add_argument('gpu', type=str, help="CUDA device")
    parser.add_argument('--batch_size', type=int, default=2, help="batch size")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")
    parser.add_argument('--lr', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon in optimizer")
    parser.add_argument('--weight_decay', type=float, default=0.0002, help="weight decay")
    parser.add_argument('--milestones', type=int, nargs='*', default=[50, 80], help="milestones for scheduler")
    parser.add_argument('--gamma', type=float, default=0.1, help="gamma for scheduler")
    parser.add_argument('--epochs', type=int, default=100, help="number of epochs")
    parser.add_argument('--val_freq', type=int, default=1, help="validation frequency")
    parser.add_argument('--clip_grad_norm', type=bool, default=False, help="whether to clip gradient norm")
    parser.add_argument('--max_grad_norm', type=int, default=5, help="max gradient norm")
    parser.add_argument('--test', action='store_true', help="only testing")
    parser.add_argument('--save_every', type=int, default=101, help="save the model in what frequency")

    return parser.parse_args()


def gen_train_val_data(args):
    train_set = eval(args.dataset)(args.dataset_model_args['dataset'], split='train')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_set = eval(args.dataset)(args.dataset_model_args['dataset'], split='val')
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True, drop_last=False)

    return (train_set, train_loader), (val_set, val_loader)


def gen_test_data(args):
    test_set = eval(args.dataset)(args.dataset_model_args['dataset'], split='test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True, drop_last=False)

    return test_set, test_loader


def build_model(args, mode, device, state_dict=None, **kwargs):
    cfgs = args.dataset_model_args['model']
    model = eval(args.model)(cfgs)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    exec(f'model.{mode}()')
    model.to(device)

    return model


def load_model(file):
    save_dict = torch.load(file, map_location='cpu')
    mean = save_dict['mean']
    std = save_dict['std']
    statics = save_dict['statics']
    state_dict = save_dict['model']

    return mean, std, statics, state_dict


def train_epoch(train_loader, mean, std, normtype, device, model, statics, criterion, optimizer, scheduler, args):
    ave = Average()
    statics = move2device(statics, device)
    for batch in tqdm(train_loader):
        inputs, targets, *extras = batch
        inputs, targets = normalize([inputs, targets], mean, std, normtype)
        inputs, targets, *extras = move2device([inputs, targets] + extras, device)

        outputs = model(inputs, targets, *extras, **statics)
        outputs, targets = denormalize([outputs, targets], mean.to(device), std.to(device), normtype)
        loss = criterion(outputs, targets)

        ave.add(loss.item(), 1)

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
    scheduler.step()
    ave_loss = ave.average()

    return ave_loss


def val(val_loader, mean, std, normtype, device, model, statics, args, mode):
    target, output = [], []
    statics = move2device(statics, device)
    for batch in tqdm(val_loader):
        inputs, targets, *extras = batch
        inputs, = normalize([inputs, ], mean, std, normtype)
        inputs, *extras = move2device([inputs, ] + extras, device)

        with torch.no_grad():
            outputs = model(inputs, None, *extras, **statics).cpu()

        outputs, = denormalize([outputs, ], mean, std, normtype)
        target.append(targets)
        output.append(outputs)
    target, output = torch.cat(target, dim=0), torch.cat(output, dim=0)

    rmse, mae, mape = Metrics(target, output, mode).all()

    return (rmse, mae, mape), output


def train(args, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    (train_set, train_loader), (val_set, val_loader) = gen_train_val_data(args)
    mean, std = train_set.mean, train_set.std
    normtype='zscore'
    statics = train_set.statics

    model = build_model(args, mode='train', device=device)
    logger.info('--------- Model Info ---------')
    logger.info('Model size: {:.6f}MB'.format(model_size(model, type_size=4) / 1e6))

    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           eps=args.eps, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    best_mae = np.inf
    ave_losses = []
    val_maes = []
    logger.info('---------- Training ----------')
    logger.info('num_samples: {}, num_batches: {}'.format(len(train_set), len(train_loader)))
    for epoch in range(args.epochs):

        start = time.time()
        ave_loss = train_epoch(train_loader, mean, std, normtype, device, model, statics,
                               criterion, optimizer, scheduler, args)
        time_elapsed = time.time() - start

        logger.info(f'[epoch {epoch}/{args.epochs - 1}] ave_loss: {ave_loss:.6f}, time_elapsed: {time_elapsed:.6f}(sec)')
        ave_losses.append(ave_loss)

        if (epoch + 1) % args.val_freq == 0:
            logger.info('Validating...')
            logger.info('num_samples: {}, num_batches: {}'.format(len(val_set), len(val_loader)))

            model.eval()
            start = time.time()
            (_, mae, _), _ = val(val_loader, mean, std, normtype, device, model, statics, args, mode='val')
            time_elapsed = time.time() - start

            logger.info(f'time_elapsed: {time_elapsed:.6f}(sec)')

            if mae < best_mae:
                best_mae = mae
                save_dict = {'model': model.state_dict(),
                             'statics': statics,
                             'mean': mean,
                             'std': std,
                             'epoch': epoch}
                torch.save(save_dict, osp.join(args.exp_dir, 'best.pth'))
                logger.info("The best model 'best.pth' has been updated")
            if (epoch + 1) % args.save_every == 0:
                save_dict = {'model': model.state_dict(),
                             'statics': statics,
                             'mean': mean,
                             'std': std,
                             'epoch': epoch}
                torch.save(save_dict, osp.join(args.exp_dir, 'epoch{:03d}.pth'.format(epoch)))
                logger.info("The model 'epoch{:03d}.pth' has been saved".format(epoch))
            logger.info(f'mae: {mae:.6f}, best_mae: {best_mae:.6f}')
            val_maes.append([epoch, mae])

            model.train()
    np.savetxt(osp.join(args.exp_dir, 'ave_losses.txt'), np.array(ave_losses))
    np.savetxt(osp.join(args.exp_dir, 'val_maes.txt'), np.array(val_maes), '%d %g')


def test(args, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_set, test_loader = gen_test_data(args)
    mean, std, statics, state_dict = load_model(osp.join(args.exp_dir, 'best.pth'))
    normtype = 'zscore'
    model = build_model(args, mode='eval', device=device, state_dict=state_dict)

    logger.info('---------- Testing ----------')
    logger.info('num_samples: {}, num_batches: {}'.format(len(test_set), len(test_loader)))
    start = time.time()
    (rmse, mae, mape), output = val(test_loader, mean, std, normtype, device, model, statics, args, mode='test')
    time_elapsed = time.time() - start
    logger.info(f'time_elapsed: {time_elapsed:.6f}(sec)')

    metrics = save_metrics(rmse, mae, mape, osp.join(args.exp_dir, 'metrics.csv'))
    logger.info(metrics)

    torch.save(output, osp.join(args.exp_dir, 'output.pth'))


if __name__ == "__main__":
    args = get_args()
    args.dataset_model_args = get_dataset_model_args(args.dataset, args.model)
    args.exp_dir = create_exp_dir(args.dataset, args.model, args.name)

    logger = get_logger(args.exp_dir)
    if not args.test:
        logger.info('Start time: {}'.format(datetime.now()))
        logger.info('---------- Args ----------')
        logger.info(json.dumps(args.__dict__, indent=2))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    exec('from datasets.{0} import {0}'.format(args.dataset))
    exec('from models.{0} import {0}'.format(args.model))

    if not args.test:
        train(args, logger)
    test(args, logger)

    logger.info('--------------------------')
    logger.info('End time: {}'.format(datetime.now()))
