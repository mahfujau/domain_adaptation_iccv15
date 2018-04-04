import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models import Encoder, ClassClassifier, DomainClassifier
from dataset import get_dataloader
from utils import gen_soft_labels, ret_soft_label
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Parameters
epochs = 10000
temperature = 2
batch_size = 15
lr = 1e-4
momentum = 0.9 
interval = 50
data_dir = '/home/lucliu/dataset/domain_adaptation/office31'
src_dir = 'amazon'
tgt_train_dir = 'webcam_tgt'
tgt_dir = 'webcam'
test_dir = 'test'
cuda = torch.cuda.is_available()
test_loader = get_dataloader(data_dir, tgt_dir, batch_size=15, train=False)
# lam for confusion 
lam = 0.01
# nu for soft
nu = 0.1

# load the pretrained and fine-tuned alex model

encoder = Encoder()
cl_classifier = ClassClassifier(num_classes=31)
dm_classifier = DomainClassifier()

encoder.load_state_dict(torch.load('./checkpoints/a2w/src_encoder_final.pth'))
cl_classifier.load_state_dict(torch.load('./checkpoints/a2w/src_classifier_final.pth'))

src_train_loader = get_dataloader(data_dir, src_dir, batch_size, train=True)
tgt_train_loader = get_dataloader(data_dir, tgt_train_dir, batch_size, train=True)
criterion = nn.CrossEntropyLoss()
# criterion_kl = nn.KLDivLoss()
if cuda:
    criterion = criterion.cuda()
    cl_classifier = cl_classifier.cuda()
    dm_classifier = dm_classifier.cuda()
    encoder = encoder.cuda()
soft_labels = gen_soft_labels(31, src_train_loader, encoder, cl_classifier)
# optimizer
optimizer = optim.SGD(
    list(encoder.parameters()) + list(cl_classifier.parameters()),
    lr=lr,
    momentum=momentum)

optimizer_conf = optim.SGD(
    encoder.parameters(),
    lr=lr,
    momentum=momentum)

optimizer_dm = optim.SGD(
    dm_classifier.parameters(),
    lr=lr,
    momentum=momentum)
# begin training
encoder.train()
cl_classifier.train()
dm_classifier.train()
for epoch in range(1, epochs+1):
    correct = 0
    for batch_idx, ((src_data, src_label_cl), (tgt_data, tgt_label_cl)) in enumerate(zip(src_train_loader, tgt_train_loader)):
        src_label_dm = torch.ones(src_label_cl.size()).long()
        tgt_label_dm = torch.zeros(tgt_label_cl.size()).long()
        if cuda:
            src_data, src_label_cl, src_label_dm = src_data.cuda(), src_label_cl.cuda(), src_label_dm.cuda()
            tgt_data, tgt_label_cl, tgt_label_dm = tgt_data.cuda(), tgt_label_cl.cuda(), tgt_label_dm.cuda()
        src_data, src_label_cl, src_label_dm = Variable(src_data), Variable(src_label_cl), Variable(src_label_dm)
        tgt_data, tgt_label_cl, tgt_label_dm = Variable(tgt_data), Variable(tgt_label_cl), Variable(tgt_label_dm)
        soft_label_for_batch = ret_soft_label(tgt_label_cl, soft_labels)
        
        # update encoder & class classifier
        optimizer.zero_grad()
        src_feature = encoder(src_data)
        tgt_feature = encoder(tgt_data)
        # class output
        src_output_cl = cl_classifier(src_feature)
        tgt_output_cl = cl_classifier(tgt_feature) 

        soft_label_for_batch = ret_soft_label(tgt_label_cl, soft_labels)
        if cuda:
            soft_label_for_batch = soft_label_for_batch.cuda()
        soft_label_for_batch = Variable(soft_label_for_batch)
        output_cl_score = F.softmax(tgt_output_cl/temperature, dim=1)
        loss_cl = criterion(tgt_output_cl, tgt_label_cl)
        loss_soft = - (torch.sum(soft_label_for_batch * torch.log(output_cl_score)))/float(output_cl_score.size(0))
        # loss_soft = criterion_kl(tgt_output_cl, soft_label_for_batch)
        loss = loss_cl + nu * loss_soft
        # loss = loss_cl
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # update domain classifier only
        optimizer_dm.zero_grad()
        # domain output
        src_output_dm = dm_classifier(src_feature.detach())
        tgt_output_dm = dm_classifier(tgt_feature.detach())
        loss_dm_src = criterion(src_output_dm, src_label_dm)
        loss_dm_tgt = criterion(tgt_output_dm, tgt_label_dm)
        loss_dm = lam * (loss_dm_src + loss_dm_tgt)
        loss_dm.backward()
        optimizer_dm.step()

        # update encoder only using domain loss
        optimizer_conf.zero_grad()
        feature_concat =  torch.cat((src_feature, tgt_feature), 0)
        # src_output_dm_conf = dm_classifier(src_feature)
        # tgt_output_dm_conf = dm_classifier(tgt_feature)
        output_dm_conf = dm_classifier(feature_concat)
        output_dm_conf = F.softmax(output_dm_conf, dim=1)
        uni_distrib = torch.FloatTensor(output_dm_conf.size()).uniform_(0, 1)
        if cuda:
            uni_distrib = uni_distrib.cuda()
        uni_distrib = Variable(uni_distrib)
        # loss_conf = lam * criterion_kl(tgt_output_dm_conf, uni_distrib)
        loss_conf = - lam * (torch.sum(uni_distrib * torch.log(output_dm_conf)))/float(output_dm_conf.size(0)) 
        loss_conf.backward()
        optimizer_conf.step()
        # acc
        tgt_output_cl_score = F.softmax(tgt_output_cl, dim=1) # softmax first
        pred = tgt_output_cl_score.data.max(1, keepdim=True)[1]
        correct += pred.eq(tgt_label_cl.data.view_as(pred)).cpu().sum()

    acc = correct / len(tgt_train_loader.dataset)
    print("epoch: %d, class loss: %f, domain loss: %f, confusion loss: %f, acc: %f"%(epoch, loss.data[0], loss_dm.data[0], loss_conf.data[0], acc))

        # save parameters
    if (epoch % interval == 0):
        torch.save(encoder.state_dict(), "./checkpoints/a2w/encoder{}.pth".format(epoch))
        torch.save(cl_classifier.state_dict(), "./checkpoints/a2w/class_classifier{}.pth".format(epoch))

torch.save(encoder.state_dict(), "./checkpoints/a2w/encoder_final.pth")
torch.save(cl_classifier.state_dict(), "./checkpoints/a2w/class_classifier_final.pth")

