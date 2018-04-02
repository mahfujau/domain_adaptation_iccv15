import torch
import torch.nn.functional as F
from torch.autograd import Variable

def gen_soft_labels(num_classes, src_train_loader, src_encoder, src_classifier):
    cuda = torch.cuda.is_available() 
    temperature = 2

    soft_labels = torch.zeros(num_classes, 1, num_classes)
    sum_classes = torch.zeros(num_classes)
    pred_scores_total = []
    label_total = []
    #if cuda:
        #src_encoder = src_encoder.cuda()
        #src_classifier = src_classifier.cuda()
    
    for _, (src_data, label) in enumerate(src_train_loader):
        label_total.append(label)
        if cuda:
            src_data, label = src_data.cuda(), label.cuda()
            src_data, label = Variable(src_data), Variable(label)
        
        src_feature = src_encoder(src_data)
        output = src_classifier(src_feature)

        pred_scores = F.softmax(output/temperature, dim=1).data.cpu()
        pred_scores_total.append(pred_scores)
    
    pred_scores_total = torch.cat(pred_scores_total)
    label_total = torch.cat(label_total)

    # sum of each class
    for i in range(len(src_train_loader.dataset)):
        sum_classes[label_total[i]] += 1    
        soft_labels[label_total[i]][0] += pred_scores_total[i]
    # average
    for cl_idx in range(num_classes): 
        soft_labels[cl_idx][0] /= sum_classes[cl_idx]
    return soft_labels

# soft label for each batch
def ret_soft_label(label, soft_labels):
    num_classes = 31
    soft_label_for_batch = torch.zeros(label.size(0), num_classes)
    for i in range(label.size(0)):
        soft_label_for_batch[i] = soft_labels[label.data[i]]
        
    return soft_label_for_batch