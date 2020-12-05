import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random

from tqdm import tqdm

from resnet import ResNet18


if torch.cuda.is_available() == True:
    device = torch.device('cuda:0')
    print(torch.cuda.get_device_name())
else:
    device = torch.device('cpu')
device

batch_size = 128
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def load_data():
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    dataset = {'train': trainset, 'test': testset}
    data_loader = {'train': train_loader, 'test': test_loader}
    return dataset, data_loader

def load_iter(data_loader, data_type):
    bar_format = '{bar:30} {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}] {desc}'

    if data_type == 'train':
        train_loader = data_loader['train']
        train_iter = tqdm(enumerate(train_loader), total=len(train_loader), unit_scale=batch_size, bar_format=bar_format)
        return train_iter
    elif data_type == 'test':
        test_loader = data_loader['test']
        test_iter = tqdm(enumerate(test_loader), total=len(test_loader), unit_scale=batch_size, bar_format=bar_format)
        return test_iter
    else:
        print('Data Error!!!')

def collect_advs(model, data_loader, epsilon):
    model.eval()
    test_loss = 0
    correct = 0
    success = 0
    total = 0
    adv_instances = []
    train_iter = load_iter(data_loader, 'train')

    for j, (batch, label) in train_iter:
        batch, label = batch.to(device), label.to(device)
        batch.requires_grad = True
        output = model(batch)
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(output, label)
        _, predicted = output.max(1)

        model.zero_grad() #Note to my self:
                          #IF all your model parameters are in that optimizer,
                          #model.zero_grad() and optimizer.zero_grad() are the same
        loss.backward()
        batch_grad = batch.grad.data #Derive gradient value w.r.t each data instance(in the batch)
        for i, data in enumerate(batch.clone()):
            if label[i].item() == predicted[i].item():
                data_grad = batch_grad[i]
                perturbed_image = fgsm_attack(data, epsilon, data_grad)
                #batch[i] = perturbed_image
                _, perturb_predict = model(perturbed_image.view(1,3,32,32)).max(1)
                #print("Output shape", perutb_predict)
                #check perturbed one is also adversarial
                if perturb_predict.item() != predicted[i].item():
                    adv_instances.append((perturbed_image.cpu(), label[i].item(), perturb_predict.item(), data.cpu().detach()))
        train_iter.set_description(f'[# of Collected Adv Instances : {len(adv_instances)}]', False)
    return adv_instances

def load_model(name):
    state_dict = torch.load(f'./models/{name}.pth', map_location=device)
    model = ResNet18()
    model.to(device)
    model.load_state_dict(state_dict['model'])
    optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
#     optimizer.load_state_dict(state_dict['optimizer'])
    return model, optimizer

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

class AdvDataSet(Dataset):
    def __init__(self, adv_instances, need_perturb_label):
        self.need_perturb_label = need_perturb_label
        if self.need_perturb_label:
            self.adv_instances = adv_instances
        else:
            self.adv_instances = []
            for instance in adv_instances:
                self.adv_instances.append((instance[0], instance[1]))

    def slicing(self, shuffle_seed=None, slice_by=1000):
        if not shuffle_seed:
            random.Random(shuffle_seed).shuffle(self.adv_instances)

        self.slices = []
        for i in range(0, len(self.adv_instances), slice_by):
            self.slices.append(self.adv_instances[i:i + slice_by])

        return self.slices

    def concat_dataset(self, trainset):
        dataset_list = []
        for data_slice in self.slices:
            dataset = ConcatDataset([data_slice, trainset])
            dataset_list.append(dataset)
        return dataset_list

    def __len__(self):
        return len(self.adv_instances)

    def __getitem__(self, idx):
        return self.adv_instances[idx]