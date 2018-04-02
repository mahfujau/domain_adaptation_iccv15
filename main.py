import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
from models import Encoder, ClassClassifier, DomainClassifier
from dataset import get_dataloader

# Parameters
data_dir = '/home/lucliu/dataset/domain_adaptation/office31'
src_dir = 'amazon'
tgt_dir = 'webcam'
test_dir = 'test'

tempreture = 2

