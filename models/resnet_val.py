import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=None, binary_mask=None, pred_cls=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if layer is not None and "layer1" in layer:
            if pred_cls is None and binary_mask is not None:
                # binary_mask: (C,) or (B,C)
                if binary_mask.dim() == 1:
                    out = out * binary_mask.view(1, -1, 1, 1)
                else:
                    out = out * binary_mask.unsqueeze(-1).unsqueeze(-1)
            elif pred_cls is not None and binary_mask is not None:
                out = out * (pred_cls @ binary_mask).unsqueeze(-1).unsqueeze(-1)
            else:
                raise NotImplementedError

        out = self.layer2(out)
        if layer is not None and "layer2" in layer:
            if pred_cls is None and binary_mask is not None:
                if binary_mask.dim() == 1:
                    out = out * binary_mask.view(1, -1, 1, 1)
                else:
                    out = out * binary_mask.unsqueeze(-1).unsqueeze(-1)
            elif pred_cls is not None and binary_mask is not None:
                out = out * (pred_cls @ binary_mask).unsqueeze(-1).unsqueeze(-1)
            else:
                raise NotImplementedError

        out = self.layer3(out)
        if layer is not None and "layer3" in layer:
            if pred_cls is None and binary_mask is not None:
                if binary_mask.dim() == 1:
                    out = out * binary_mask.view(1, -1, 1, 1)
                else:
                    out = out * binary_mask.unsqueeze(-1).unsqueeze(-1)
            elif pred_cls is not None and binary_mask is not None:
                out = out * (pred_cls @ binary_mask).unsqueeze(-1).unsqueeze(-1)
            else:
                raise NotImplementedError

        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if layer is not None and "layer4" in layer:
            if pred_cls is None and binary_mask is not None:
                # Here out and binary_mask can both be (B, D) or (D,)
                if binary_mask.dim() == 1:
                    out = out * binary_mask
                else:
                    out = out * binary_mask
            elif pred_cls is not None and binary_mask is not None:
                out = out * (pred_cls @ binary_mask)
            else:
                raise NotImplementedError

        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


class InterpMaskedResNet(ResNet):
    """
    InterpMaskedResNet with:
    - mask_which = 'none'      : vanilla / RS-only
    - mask_which = 'loir'      : original LOIR masking
    - mask_which = 'cdir'      : CDIR masking (if rankings exist)
    - mask_which = 'random'    : random masking
    - mask_which = 'conf'      : **our confidence-adaptive masking**, built on LOIR rankings
    """

    def __init__(
        self,
        layer_name,
        checkpoint_name,
        mask_which,
        important_dim,
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        num_classes=10,
        rs=False,
        rs_sigma=8 / 255,
        rs_nsmooth=1,
        batch_size=1000,
        conf_thresh=0.9,
    ):
        super(InterpMaskedResNet, self).__init__(block, num_blocks, num_classes)
        self.layer_name = layer_name
        self.checkpoint_name = checkpoint_name
        self.mask_which = mask_which
        self.important_dim = important_dim
        self.num_classes = num_classes
        self.layer2dim = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
        self.layer_dim = self.layer2dim[layer_name]

        self.rs = rs
        self.conf_thresh = conf_thresh  # used only when mask_which == 'conf'

        if self.rs:
            self.rs_sigma = rs_sigma
            self.rs_nsmooth = rs_nsmooth
            print(
                "using RS with",
                self.rs_nsmooth,
                "samples and sigma",
                self.rs_sigma,
                "instead of vanilla forward pass",
            )
        else:
            print("using vanilla forward pass before masking")

        if self.mask_which not in ["none", "twoforwardpasses"]:
            self.neuron_importance, self.binary_masks = self.load_neuron_importance()
        else:
            self.neuron_importance = None
            self.binary_masks = None

    def load_neuron_importance(self):
        # Reuse LOIR rankings for both 'loir' and 'conf' (confidence-adaptive)
        if self.mask_which in ["loir", "conf"]:
            root_name = "saved_loir_rankings/"
        elif self.mask_which == "cdir":
            root_name = "saved_cdir_rankings/"
        elif self.mask_which == "random":
            root_name = None
        else:
            raise NotImplementedError("check argument mask_which")

        model_save_name = os.path.splitext(os.path.basename(self.checkpoint_name))[0]

        if root_name is not None:
            folder_name = root_name + model_save_name + "/" + self.layer_name

            ablated_acc = torch.zeros(
                (self.layer_dim, self.num_classes)
            ).cuda()

            # loading the neuron importance for every class
            for k in range(self.layer_dim):
                ablated_acc[k] = torch.Tensor(
                    np.load(folder_name + "/unit" + str(k) + ".npy")
                )
        else:
            # if root_name is None, then make sure it's not cdir or loir/conf
            assert self.mask_which not in ["cdir", "loir", "conf"]
            folder_name = "N/A (random masking)"

        # neuron_class_importance stores indices of neurons to be **ablated**
        neuron_class_importance = (
            torch.ones((self.num_classes, self.layer_dim - self.important_dim)) * -1
        )

        for curr_cls in range(neuron_class_importance.shape[0]):
            if self.mask_which == "random":
                neuron_class_importance[curr_cls] = torch.Tensor(
                    random.sample(range(self.layer_dim), self.important_dim)
                ).cuda()
            elif self.mask_which in ["cdir", "loir", "conf"]:
                # HIGHER SIMILARITY/CHANGE IS HIGHER IMPORTANCE --> sort descending
                # We mask neurons with **lowest** importance (last layer_dim - important_dim)
                neuron_class_importance[curr_cls] = (
                    ablated_acc[:, curr_cls]
                    .sort(0, descending=True)[1][-(self.layer_dim - self.important_dim) :]
                )

        binary_masks = torch.ones(
            (self.num_classes, self.layer_dim)
        ).cuda()
        for curr_cls in range(self.num_classes):
            curr_unitslist = [
                curr_unit.item() for curr_unit in neuron_class_importance[curr_cls]
            ]
            binary_masks[curr_cls, curr_unitslist] = 0

        print(
            "Neuron importance loaded with method {} from folder {}".format(
                self.mask_which, folder_name
            )
        )
        return neuron_class_importance, binary_masks

    def forward(self, x, layer=None, ablated_units=None, pred_cls=None, tau=1.0):
        # ---------------------------------------------------------------------
        # 1) "twoforwardpasses": debugging mode from original code
        # ---------------------------------------------------------------------
        if self.mask_which == "twoforwardpasses":
            # no masking, just to see the effect of gradient masking in two forward passes
            pred_logits = super(InterpMaskedResNet, self).forward(x)
            return super(InterpMaskedResNet, self).forward(x)

        # ---------------------------------------------------------------------
        # 2) Baseline: no masking, optional RS
        # ---------------------------------------------------------------------
        if self.mask_which == "none":
            if self.rs:
                sigma = self.rs_sigma
                n_smooth = self.rs_nsmooth
                tmp = torch.zeros(
                    (x.shape[0] * n_smooth, *x.shape[1:]), device=x.device
                )
                x_tmp = (
                    x.repeat((1, n_smooth, 1, 1))
                    .view(tmp.shape)
                    .to(x.device)
                )

                noise = torch.randn_like(x_tmp) * sigma
                noisy_x = x_tmp + noise
                noisy_preds = super(InterpMaskedResNet, self).forward(noisy_x)

                smoothed = torch.ones(
                    (x.shape[0], self.num_classes), device=x.device
                ) * -1
                # get smoothed prediction for each point
                for m in range(x.shape[0]):
                    smoothed[m] = torch.mean(
                        noisy_preds[(m * n_smooth) : ((m + 1) * n_smooth)], dim=0
                    )
                return smoothed
            else:
                return super(InterpMaskedResNet, self).forward(x)

        # ---------------------------------------------------------------------
        # 3) Masking modes: loir / cdir / random / conf
        #    First forward pass to get class scores (with optional RS)
        # ---------------------------------------------------------------------
        if self.rs:
            sigma = self.rs_sigma
            n_smooth = self.rs_nsmooth
            tmp = torch.zeros(
                (x.shape[0] * n_smooth, *x.shape[1:]), device=x.device
            )
            x_tmp = (
                x.repeat((1, n_smooth, 1, 1))
                .view(tmp.shape)
                .to(x.device)
            )

            noise = torch.randn_like(x_tmp) * sigma
            noisy_x = x_tmp + noise
            noisy_preds = super(InterpMaskedResNet, self).forward(noisy_x)

            logits = torch.ones(
                (x.shape[0], self.num_classes), device=x.device
            ) * -1
            for m in range(x.shape[0]):
                logits[m] = torch.mean(
                    noisy_preds[(m * n_smooth) : ((m + 1) * n_smooth)], dim=0
                )
        else:
            logits = super(InterpMaskedResNet, self).forward(x)

        # Softmax with large temperature, same as original IG-Def
        probs = F.softmax(logits * 100.0, dim=1)

        # ---------------------------------------------------------------------
        # 3a) Our confidence-adaptive masking (mask_which == 'conf')
        # ---------------------------------------------------------------------
        if self.mask_which == "conf":
            # LOIR-style class-weighted mask first
            # binary_masks: (num_classes, layer_dim)
            # probs:        (B, num_classes)
            # base_mask:    (B, layer_dim)
            base_mask = probs @ self.binary_masks

            # Confidence per sample = max class prob
            conf_vals, _ = probs.max(dim=1, keepdim=True)  # (B, 1)

            # Strength in [0,1]:
            #   conf <= conf_thresh -> 0 (no masking, baseline)
            #   conf >= 1          -> 1 (full LOIR masking)
            strength = torch.clamp(
                (conf_vals - self.conf_thresh) / (1.0 - self.conf_thresh),
                min=0.0,
                max=1.0,
            )

            # Interpolate between "all ones" (no masking) and LOIR mask
            ones = torch.ones_like(base_mask)
            eff_mask = (1.0 - strength) * ones + strength * base_mask
            # eff_mask is (B, layer_dim)

            # Second forward pass: apply *per-sample* mask at self.layer_name
            return super(InterpMaskedResNet, self).forward(
                x, self.layer_name, pred_cls=None, binary_mask=eff_mask
            )

        # ---------------------------------------------------------------------
        # 3b) Original IG-Def masking (loir / cdir / random)
        # ---------------------------------------------------------------------
        return super(InterpMaskedResNet, self).forward(
            x, self.layer_name, pred_cls=probs, binary_mask=self.binary_masks
        )


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
