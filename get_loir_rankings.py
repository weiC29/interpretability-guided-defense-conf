import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from multiprocessing import Pool
import random
from tqdm import tqdm

from models import resnet_val, wideresnet_trades
from models.resnet50 import ResNet50
from utils import (
    clamp,
    get_loaders,
    evaluate_standard,
    evaluate_standard_classwise,
    compare_predicted_probs,
    save_predicted_logits,
    DEVICE,
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--load-model', type=str, help='filename of checkpoint')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet'], default='cifar10')
    parser.add_argument('--arch', type=str, default='rn18_val', choices=['rn18_val', 'wrn34_10', 'rn50'])
    parser.add_argument('--preprocessing', type=str, default='Crop288',
                        choices=['Crop288', 'Res256Crop224', 'Crop288-autoaug'],
                        help='preprocessing type, only for ImageNet')
    parser.add_argument('--num-parallel-threads', default=8, type=int,
                        help='Number of parallel threads, use 8 for 24GB GPU and 4 for 12GB GPU; affects runtime')
    parser.add_argument('--layer-name', default='layer4', help='Name of layer whose output is ablated')
    parser.add_argument('--out-dir', default='saved_loir_rankings', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--start-dim', default=0, type=int, help='index of neuron to start from, only for ImageNet')
    parser.add_argument('--end-dim', default=2048, type=int, help='index of neuron to end at, only for ImageNet')
    return parser.parse_args()


def get_acc_ablated(args_):
    """
    Worker function for LO-IR: ablates a single neuron `unit` and
    measures the drop in predicted logits/probabilities.
    """
    (model_test,
     train_eval_loader,
     model_name,
     out_dir,
     layer_name,
     unit,
     num_classes,
     layer_dim) = args_

    model_save_name = os.path.splitext(os.path.basename(model_name))[0]

    # mask is created on the same DEVICE
    binary_mask = torch.ones((layer_dim), device=DEVICE)
    binary_mask[unit] = 0

    difference_accum = compare_predicted_probs(
        train_eval_loader,
        model_test,
        model_name,
        layer_name=layer_name,
        ablated_units=binary_mask,
        num_classes=num_classes,
        gt_only=True,
    )

    print('unit', unit, '| average logits change', difference_accum)
    os.makedirs(os.path.join(out_dir, model_save_name, layer_name), exist_ok=True)
    np.save(os.path.join(out_dir, model_save_name, layer_name, f'unit{unit}.npy'),
            difference_accum)


def main():
    args = get_args()

    model_save_name = os.path.splitext(os.path.basename(args.load_model))[0]
    os.makedirs(os.path.join(args.out_dir, model_save_name, args.layer_name),
                exist_ok=True)

    # resume logic for CIFAR runs
    if args.dataset != 'imagenet':
        list_of_files = os.listdir(os.path.join(args.out_dir, model_save_name, args.layer_name))
        if len(list_of_files) == 0:
            start_dim = 0
        else:
            # filenames are like "unit123.npy"
            list_of_ids = [int(fname[4:-4]) for fname in list_of_files if fname.startswith("unit")]
            start_dim = max(list_of_ids) - 1
            assert start_dim >= 0
        end_dim = None  # inferred from layer_dim
    else:
        start_dim = args.start_dim
        end_dim = args.end_dim

    # seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    print('finding important units for', args.load_model)

    # mapping layer -> dimension for different architectures
    if args.arch == 'rn18_val':
        layer2dim = {
            'layer1': 64,
            'layer2': 128,
            'layer3': 256,
            'layer4': 512,
        }
    elif args.arch == 'rn50':
        layer2dim = {
            'layer3': 1024,
            'layer4': 2048,
        }
    elif args.arch == 'wrn34_10':
        layer2dim = {'block2': 320, 'block3': 640}
    else:
        raise ValueError(f"Unknown arch {args.arch}")

    layer_dim = layer2dim[args.layer_name]

    # loaders (CIFAR or ImageNet)
    if args.num_parallel_threads > 1:
        train_eval_loader, test_loader = get_loaders(
            args.dataset, args.data_dir, args.batch_size, args.arch, args.preprocessing
        )
    else:
        train_eval_loader, test_loader = get_loaders(
            args.dataset, args.data_dir, args.batch_size, args.arch, args.preprocessing,
            num_workers=8,
        )
    print(args.dataset, 'dataset loaded')

    num_classes_map = {
        'cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000,
    }
    num_classes = num_classes_map[args.dataset]

    # build model on DEVICE
    if args.arch == 'rn18_val':
        model_test = resnet_val.ResNet18().to(DEVICE)
    elif args.arch == 'wrn34_10':
        model_test = wideresnet_trades.WideResNet34_10().to(DEVICE)
    elif args.arch == 'rn50':
        model_test = ResNet50().to(DEVICE)
    else:
        raise ValueError(f"Unknown arch {args.arch}")

    # load checkpoint; handle DataParallel "module." prefix
    ckpt = torch.load(
        os.path.join('checkpoints', args.load_model),
        map_location=DEVICE,
    )
    if isinstance(ckpt, dict) and len(ckpt) > 0 and list(ckpt.keys())[0].startswith('module.'):
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}

    model_test.load_state_dict(ckpt)
    model_test.float()
    model_test.eval()
    print('model loaded')

    # quick sanity check accuracy on test set (CIFAR only)
    if args.dataset != 'imagenet':
        _, test_acc = evaluate_standard(test_loader, model_test)
        print('Test accuracy (for checking correctness of model and data loading):', test_acc)

    # save (or reload) clean predicted logits once
    predlogits_path = os.path.join('saved_predicted_logits', f'{model_save_name}_predlogits.pth')
    if not os.path.exists(predlogits_path):
        print('saving predicted logits now')
        save_predicted_logits(train_eval_loader, model_test, args.load_model, num_classes)
    else:
        print(f'loading orig. pred. logits from {predlogits_path}')

    # LO-IR computation
    if args.num_parallel_threads > 1 and args.dataset != 'imagenet':
        # CIFAR case: parallel over groups of neurons
        total_groups = layer_dim // args.num_parallel_threads
        start_group = start_dim // args.num_parallel_threads
        print(f"Starting LO-IR from neuron {start_dim} over {layer_dim} dims "
              f"({total_groups - start_group} groups of {args.num_parallel_threads}).")

        for k in tqdm(range(start_group, total_groups),
                      desc="LO-IR over neuron groups"):
            start_time = time.time()
            p = Pool(args.num_parallel_threads)
            unitlist = []
            for idx in range(args.num_parallel_threads):
                unit_idx = args.num_parallel_threads * k + idx
                if unit_idx >= layer_dim:
                    continue
                unitlist.append((
                    model_test,
                    train_eval_loader,
                    args.load_model,
                    args.out_dir,
                    args.layer_name,
                    unit_idx,
                    num_classes,
                    layer_dim,
                ))
            p.map(get_acc_ablated, unitlist)
            p.close()
            p.join()
            print('Time taken for group', k, ':', time.time() - start_time)
    else:
        # ImageNet or manually forced single-thread mode
        if end_dim is None:
            end_dim = layer_dim
        print(f'starting from {start_dim}, ending at {end_dim}')
        for k in tqdm(range(start_dim, end_dim), desc="LO-IR over neurons"):
            start_time = time.time()
            print('Running for neuron', k)
            get_acc_ablated((
                model_test,
                train_eval_loader,
                args.load_model,
                args.out_dir,
                args.layer_name,
                k,
                num_classes,
                layer_dim,
            ))
            print('Time taken : ', time.time() - start_time)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
