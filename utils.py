import math
import copy
import collections

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def loss_func(loss_name_list):
    """
    Return list of loss function

    Args:
        loss_name_list: List contains loss name

    Returns:
        loss_func_list: List contrains torch loss
    """
    loss_func_list = []
    for loss_name in loss_name_list:
        loss_name = loss_name.lower()
        if loss_name == 'mseloss':
            loss = nn.MSELoss()
        elif loss_name == 'crossentropyloss':
            loss = nn.CrossEntropyLoss()
        elif loss_name == 'huberloss':
            loss = nn.HuberLoss()
        elif loss_name == 'kldivloss':
            loss = nn.KLDivLoss()
        elif loss_name == 'bceloss':
            loss = nn.BCELoss()
        else:
            raise NotImplementedError
        loss_func_list.append(loss)
    return loss_func_list

def optim_func(model, cfg):
    """
    Return torch optimizer

    Args:
        model: Model you want to train
        cfg: Dictionary of optimizer configuration 

    Returns:
        optimizer
    """
    optim_name = cfg['name'].lower()
    learning_rate = cfg['learning_rate']
    others = cfg['others']
    if optim_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, **others)
    elif optim_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, **others)
    elif optim_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, **others)
    else:
        raise NotImplementedError
    return optimizer

def lr_scheduler_func(optimizer, cfg):
    """
    Return torch learning rate scheduler

    Args:
        optimizer
        cfg: Dictionary of learning rate scheduler configuration 
    
    Returns:
        lr_scheduler
    """
    scheduler_name = cfg['name'].lower()
    others = cfg['others']
    if scheduler_name == 'steplr':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **others)
    elif scheduler_name == 'multisteplr':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **others)
    elif scheduler_name == 'cosineannealinglr':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **others)
    elif scheduler_name == 'cycliclr':
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, **others)
    elif scheduler_name == 'lambdalr':
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, **others)
    else:
        raise NotImplementedError
    return lr_scheduler

def plot_progress(history, epoch, file_path='./train_progress'):
    """
    Plot train progress

    Args:
        history: Dictionary contains train/validation loss history
        epoch: Current train epoch
        file_path: Path to save graph
    """
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, epoch+1, dtype=np.int16), history['train'])
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, epoch+1, dtype=np.int16), history['validation'])
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f'{file_path}/{epoch}epochs.png')
    plt.close()
    
def save_model(model, epoch, optimizer, lr_scheduler, history, model_path):
    """
    Save training ckeckpoint

    Args:
        epoch: Current epoch
        model: Trained model
        optimizer
        lr_scheduler
        file_path: Path to save checkpoint
    """
    torch.save({
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'lr_scheduler' : lr_scheduler.state_dict(),
        'history' : history,
        }, f'{model_path}/model_{epoch}.pt')