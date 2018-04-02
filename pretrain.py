import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from models import Encoder, ClassClassifier 
from dataset import get_dataloader 
# parameters
batch_size = 15
lr = 1e-4 
momentum = 0.9
interval = 100
epochs = 5000
data_dir = '/home/lucliu/dataset/domain_adaptation/office31'
src_dir = 'amazon'

cuda = torch.cuda.is_available() 
# dataloader
src_train_loader = get_dataloader(data_dir, src_dir, batch_size, train=True)
# model
# Pretrained Model
alexnet = torchvision.models.alexnet(pretrained=True)
pretrained_dict = alexnet.state_dict()
# Train source data
# Model parameters
src_encoder = Encoder()
src_classifier = ClassClassifier(num_classes=31)
src_encoder_dict = src_encoder.state_dict()
# Load pretrained model 
# filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in src_encoder_dict}
# overwrite entries in the existing state dict
src_encoder_dict.update(pretrained_dict) 
# load the new state dict
src_encoder.load_state_dict(src_encoder_dict)
optimizer = optim.SGD(
    list(src_encoder.parameters()) + list(src_classifier.parameters()),
    lr=lr,
    momentum=momentum)

criterion = nn.CrossEntropyLoss()

if cuda: 
    src_encoder = src_encoder.cuda()
    src_classifier = src_classifier.cuda() 
    criterion = criterion.cuda() 

src_encoder.train()
src_classifier.train()
# begin train
for epoch in range(1, epochs+1):
    correct = 0
    for batch_idx, (src_data, label) in enumerate(src_train_loader):
        if cuda:
            src_data, label = src_data.cuda(), label.cuda()
        src_data, label = Variable(src_data), Variable(label)
        optimizer.zero_grad()
        src_feature = src_encoder(src_data)
        output = src_classifier(src_feature)
        loss = criterion(output, label)
        pred = output.src_data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
    acc = correct / len(src_train_loader.dataset)
    print("epoch: %d, loss: %f, acc: %f"%(epoch, loss.data[0], acc))

    # save parameters
    if (epoch % interval == 0):
        torch.save(src_encoder.state_dict(), "./src_encoder{}.pth".format(epoch))

