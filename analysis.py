import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import time
import math
import argparse
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import models
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from models.resnet_val import ResNet18
from models.wideresnet_trades import WideResNet34_10
import random


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)

def load_neuron_importance(layer_name, checkpoint_name, important_dim, num_classes):
    layer2dim = {'layer3': 256, 'layer4': 512}
    if 'imagenet' in checkpoint_name:
        layer2dim = {'layer3': 1024, 'layer4': 2048}

    layer_dim = layer2dim[layer_name]

    root_name = 'saved_loir_rankings/'

    folder_name = root_name + checkpoint_name.split('.')[0] + '/' + layer_name

    ablated_acc = torch.zeros((layer_dim, num_classes)).cuda()

    # loading the neuron importance for every class
    for k in range(layer_dim):
        ablated_acc[k] = torch.Tensor(np.load(folder_name + '/unit' + str(k) + '.npy'))

    neuron_class_importance = torch.ones((num_classes, important_dim)) * -1

    for curr_cls in range(num_classes):
        # sorting the logit-changes of current class
        # and list of units to be ablated contains the bottom 492 i.e. (512 - 20)
        # where 512 is layer dim and 20 is number of important units
        # HIGHER CHANGE IS HIGHER IMPORTANCE --> hence we sort in descending order
        neuron_class_importance[curr_cls] = ablated_acc[:, curr_cls].sort(0, descending=True)[1][:important_dim]
            
    return neuron_class_importance


parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--trained-model', default='./',
                    help='location of the adversarially trained model')
parser.add_argument('--arch', type=str, default='rn18', choices=['rn18', 'parn18'])
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
parser.add_argument('--data-path', type=str, default='./data',
                    help='where is the dataset')
parser.add_argument('--layer-name', default='layer4', choices=['layer3', 'layer4'], help='Name of layer whose output is ablated')
parser.add_argument('--important-dim', default=50, type=int, help='Number of important neurons to be retained in forward pass')

args = parser.parse_args()


#loading data 
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

upper_limit = [(1 - mu) / std for mu, std in zip(cifar10_mean, cifar10_std)]
lower_limit = [(0 - mu) / std for mu, std in zip(cifar10_mean, cifar10_std)]

transform_test = transforms.Compose([
    transforms.ToTensor(),
])


if args.data == 'CIFAR10' or args.data == 'CIFAR100':
    # testset = getattr(datasets, args.data)(root=args.data_path, train=False, download=True, transform=transform_test)
    testset = datasets.CIFAR10(args.data_path, train=False, transform=transform_test, download=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

if args.data == 'CIFAR10':
    NUM_CLASSES = 10
    test_size = 10000
elif args.data == 'CIFAR100':
    NUM_CLASSES = 100
    test_size = 10000


if args.arch == 'rn18':
    model = ResNet18()
elif args.arch == 'wrn34_10':
    model = WideResNet34_10()

model = model.cuda()

ckpt_path = 'checkpoints/' + args.trained_model
model_dict = torch.load(ckpt_path)
print('Loading weights from', ckpt_path)

# Some checkpoints are saved as {"state_dict": ...}
state_dict = model_dict.get('state_dict', model_dict)

# If keys start with "module.", strip that prefix
from collections import OrderedDict
first_key = next(iter(state_dict.keys()))
if first_key.startswith('module.'):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[len("module."):]] = v
    state_dict = new_state_dict

model.load_state_dict(state_dict)
model.eval()


neuron_class_importance = load_neuron_importance(args.layer_name, args.trained_model, args.important_dim, NUM_CLASSES)
neuron_class_importance = neuron_class_importance.detach().cpu().numpy()
# print(neuron_class_importance.shape)
neuron_class_importance = neuron_class_importance.astype(np.uint32)
    
    
def max_margin_loss(x,y):
    B = y.size(0)
    corr = x[range(B),y]

    x_new = x - 1000*torch.eye(NUM_CLASSES, device='cuda')[y].cuda()
    tar = x[range(B),x_new.argmax(dim=1)]
    loss = tar - corr
    loss = torch.mean(loss)
    
    return loss



def GAMA_PGD(model,data,target,eps,eps_iter,bounds,steps,w_reg,lin,SCHED,drop,normalize=None):
    """
    model
    loss : loss used for training
    data : input to network
    target : ground truth label corresponding to data
    eps : perturbation srength added to image
    eps_iter
    """
    #Raise error if in training mode
    if model.training:
        assert 'Model is in  training mode'
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(np.random.uniform(-eps,eps,(B,C,H,W))).cuda()
    noise  = eps*torch.sign(noise)
    img_arr = []
    W_REG = w_reg
    orig_img = data+noise
    orig_img = Variable(orig_img,requires_grad=True)
    for step in range(steps):
        # convert data and corresponding into cuda variable
        img = data + noise
        img = Variable(img,requires_grad=True)
        
        if step in SCHED:
            eps_iter /= drop
        
        # make gradient of img to zeros
        zero_gradients(img)

        # forward pass        
        orig_out = model(orig_img)
        P_out = nn.Softmax(dim=1)(orig_out)

        out = model(img)
        Q_out = nn.Softmax(dim=1)(out)
        #compute loss using true label
        if step <= lin:
            cost =  W_REG*((P_out - Q_out)**2.0).sum(1).mean(0) + max_margin_loss(Q_out,tar)
            W_REG -= w_reg/lin
        else:
            cost = max_margin_loss(Q_out,tar)
        #backward pass
        cost.backward()
        #get gradient of loss wrt data
        per =  torch.sign(img.grad.data)
        #convert eps 0-1 range to per channel range 
        per[:,0,:,:] = (eps_iter * (bounds[0,1] - bounds[0,0])) * per[:,0,:,:]
        if(per.size(1)>1):
            per[:,1,:,:] = (eps_iter * (bounds[1,1] - bounds[1,0])) * per[:,1,:,:]
            per[:,2,:,:] = (eps_iter * (bounds[2,1] - bounds[2,0])) * per[:,2,:,:]
        #  ascent
        adv = img.data + per.cuda()
        #clip per channel data out of the range
        img.requires_grad =False
        img[:,0,:,:] = torch.clamp(adv[:,0,:,:],bounds[0,0],bounds[0,1])
        if(per.size(1)>1):
            img[:,1,:,:] = torch.clamp(adv[:,1,:,:],bounds[1,0],bounds[1,1])
            img[:,2,:,:] = torch.clamp(adv[:,2,:,:],bounds[2,0],bounds[2,1])
        img = img.data
        noise = img - data
        noise  = torch.clamp(noise,-eps,eps)

    return data + noise


loss=nn.CrossEntropyLoss()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
    
command = f"model.{args.layer_name}.register_forward_hook(get_activation(args.layer_name))"
eval(command)

eps = 8 / 255

# Make this small while debugging
steps = 100 
loss = nn.CrossEntropyLoss()
acc = 0
start_time = time.time()
important_samecls_change = 0.0
important_diffcls_change = 0.0
important_advcls_change = 0.0
unimportant_change = 0.0
sample_count = 0

def safe_rel_change(pre_vec, post_vec, idx):
    """
    pre_vec, post_vec: 1D arrays (length = layer_dim)
    idx: array-like of indices
    returns 0.0 if idx is empty or denominator is zero/NaN
    """
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return 0.0

    before = pre_vec[idx]
    after = post_vec[idx]

    denom = before.mean()
    if denom == 0 or np.isnan(denom):
        return 0.0

    num = (-before + after).mean()
    return float(num / denom)


for batch_idx, (inputs, targets) in enumerate(test_loader):
    data = inputs.cuda()
    target = targets.cuda()

   
    if batch_idx % 100 == 0:
        print('processed', batch_idx)
    if batch_idx >= 1000:
        break

    # 1. Forward on clean image
    with torch.no_grad():
        pred = model(data)
        pred = torch.argmax(pred, dim=1)
        if pred == target:
            preactivation_layer = activation[args.layer_name].cpu().numpy()
            preactivation_layer = np.mean(preactivation_layer[0], axis=(1, 2))  # shape (layer_dim,)
            gt_label = target.cpu().numpy()[0]
        else:
            # skip misclassified clean images
            continue

    # 2. Craft adversarial example
    with torch.enable_grad():
        adv_img = GAMA_PGD(
            model,
            data.cuda(),
            target.cuda(),
            eps=eps,
            eps_iter=2 * eps,
            bounds=np.array([[0, 1], [0, 1], [0, 1]]),
            steps=steps,
            w_reg=50,
            lin=25,
            SCHED=[60, 85],
            drop=10
        )

    # 3. Check if attack succeeds
    pred_adv = torch.argmax(model(adv_img), dim=1)
    if pred_adv != target:
        postactivation_layer = activation[args.layer_name].cpu().numpy()
        postactivation_layer = np.mean(postactivation_layer[0], axis=(1, 2))  # shape (layer_dim,)

        # Indices of important neurons
        important_samecls_idx = neuron_class_importance[gt_label]
        important_advcls_idx = neuron_class_importance[pred_adv.cpu().numpy()[0]]

        # Flatten once for convenience
        all_important = neuron_class_importance.reshape(-1)

        # indices that are important for *some* class but not GT or adv class
        important_diffcls_idx = np.array(
            list(
                set(all_important)
                - set(important_samecls_idx)
                - set(important_advcls_idx)
            ),
            dtype=np.int64,
        )

        # dimension of the layer (e.g., 512), inferred from activation vector length
        layer_dim = preactivation_layer.shape[0]

        # indices that are never important for any class
        unimportant_idx = np.array(
            list(set(range(layer_dim)) - set(all_important)),
            dtype=np.int64,
        )

        # Debug: only print sizes on the first successful adversarial example
        if sample_count == 0:
            print("len GT idx:", len(important_samecls_idx))
            print("len adv idx:", len(important_advcls_idx))
            print("len diff idx:", len(important_diffcls_idx))
            print("len unimportant idx:", len(unimportant_idx))

        # 4. Accumulate relative changes using the safe helper
        important_samecls_change  += safe_rel_change(
            preactivation_layer, postactivation_layer, important_samecls_idx
        )
        important_advcls_change   += safe_rel_change(
            preactivation_layer, postactivation_layer, important_advcls_idx
        )
        important_diffcls_change  += safe_rel_change(
            preactivation_layer, postactivation_layer, important_diffcls_idx
        )
        unimportant_change        += safe_rel_change(
            preactivation_layer, postactivation_layer, unimportant_idx
        )

        sample_count += 1

# 5. Final summary
if sample_count == 0:
    print("No successful adversarial misclassifications found in the sampled data.")
else:
    print(f'average change in GT class activation: {(important_samecls_change / sample_count) * 100.0:.6f}%')
    print(f'average change in post-attack class activation: {(important_advcls_change / sample_count) * 100.0:.6f}%')
    print(f'average change in remaining class activation: {(important_diffcls_change / sample_count) * 100.0:.6f}%')
    print(f'average change in unimportant neuron activations: {(unimportant_change / sample_count) * 100.0:.6f}%')
