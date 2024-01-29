"""Train CIFAR10 with PyTorch."""
from math import inf
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import *
from storm import *
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


parser = argparse.ArgumentParser(description="PyTorch CIFAR10 STORM1 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--l2", default=1e-4, type=float, help="l2 regularization")
parser.add_argument(
    "--frequency", "-f", default=10, type=int, help="rho frequency update"
)
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument(
    "--sgd", "--SGD", action="store_true", help="use SGD instead of STORM1"
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


seed = 420  # any number
set_deterministic(seed=seed)

# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Model
print("==> Building model..")
# net = VGG('VGG19')
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = ResNet18()
net = net.to(device)
if device == "cuda":
    net = torch.nn.DataParallel(net)

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()
decrease_ex = next(iter(trainloader))

def closure():
    outputs = net(decrease_ex[0].to(device))
    loss = criterion(outputs, decrease_ex[1].to(device))

    for param in net.parameters():
        loss += args.l2 * torch.norm(param, p=2)
    return loss

optimizer = (
    STORM1(net.parameters(), lr=args.lr, momentum=0.9,loss=closure)
    if not args.sgd
    else optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
)
scheduler = (
    optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) if args.sgd else None
)


# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(
        enumerate(trainloader),
        total=len(trainloader),
        desc="Epoch {}, lr {}".format(epoch, optimizer.param_groups[0]["lr"]),
    )
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        with torch.autocast(device_type=device):
            outputs = net(inputs)

            # We use L2 regularization instead of weight decay
            l2_lambda = args.l2
            l2_reg = torch.tensor(0.0).to(device)
            for param in net.parameters():
                l2_reg += torch.norm(param, p=2)
            loss = criterion(outputs, targets) + l2_lambda * l2_reg
        
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        scaler.scale(loss).backward()
        if not args.sgd:
            scaler.step(optimizer)
        else:
            scaler.step(optimizer)
        scaler.update()

        # Updates the scale for next iteration.
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if not args.sgd:
            pbar.set_postfix(
                {
                    "loss": train_loss / (batch_idx + 1),
                    "acc": 100.0 * correct / total,
                    "rho": optimizer.rho,
                    # "lr": optimizer.param_groups[0]["lr"],
                }
            )
        else:
            pbar.set_postfix(
                {
                    "loss": train_loss / (batch_idx + 1),
                    "acc": 100.0 * correct / total,
                    # "lr": optimizer.param_groups[0]["lr"],
                }
            )


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(testloader), total=len(testloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(
                {
                    "test_loss": test_loss / (batch_idx + 1),
                    "test_acc": 100.0 * correct / total,
                }
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, "./checkpoint/ckpt.pth")
        best_acc = acc


# loss_prev = criterion(outputs, decrease_ex[1].to(device)).item()
lr = args.lr
rho = inf
# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()



for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)

    # Update scheduler if using SGD
    if scheduler is not None:
        scheduler.step()

    # Compute rho every 10 epochs if using STORM1
    # if epoch % args.frequency == 0 and not args.sgd:
    #     with torch.no_grad():
    #         outputs = net(decrease_ex[0].to(device))
    #     loss = criterion(outputs, decrease_ex[1].to(device)).item()
    #     l2_lambda = args.l2
    #     l2_reg = torch.tensor(0.0).to(device)
    #     for param in net.parameters():
    #         l2_reg += torch.norm(param, p=2)
    #     loss += l2_lambda * l2_reg
    #     rho = (loss_prev - loss) / (lr * torch.linalg.norm(optimizer.updates, 2))
    #     optimizer.updates = None
    #     loss_prev = loss

    #     # Update lr
    #     if rho < 0.25:
    #         lr = lr / 2
    #     elif rho > 0.75:
    #         lr = lr * 2

    #     # Update optimizer
    #     optimizer = STORM1(net.parameters(), lr=lr, momentum=0.9)
