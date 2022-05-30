import pickle
from sklearn.metrics import roc_curve, auc
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from framework.conag.logreg import LogReg
from framework.conag.network import Conag
from framework.conag.test_focal_loss import FocalLoss
from framework.execution import data_preparer

from global_object.file import *
from global_object.weight import *

from framework.execution.load_whole_data import r_load_whole_data
from framework.execution.neg_sam import get_neg_para
from framework.alg_generator.sorter import get_anomaly_uids, \
    get_nodes_sorted_device

batch_size = 1
nb_epochs = 10000
patience = 40
# lr = 0.001
lr = 0.0005
l2_coef = 0.0
drop_prob = 0.0
hid_units = 512
sparse = True
nonlinearity = 'prelu'

# ctrl
train = False
reload_whole_data = False
reload_neg_data = False

# get data
if reload_whole_data:
    adj, features, anomaly_features, labels, idx_train, idx_test = \
        r_load_whole_data()
    # dump
    with open(device_para_save, 'wb') as f:
        pickle.dump(adj, f)
        pickle.dump(features, f)
        pickle.dump(anomaly_features, f)
        pickle.dump(labels, f)
        pickle.dump(idx_train, f)
        pickle.dump(idx_test, f)
else:
    with open(device_para_save, 'rb') as f:
        adj = pickle.load(f)
        features = pickle.load(f)
        anomaly_features = pickle.load(f)
        labels = pickle.load(f)
        idx_train = pickle.load(f)
        idx_test = pickle.load(f)

features, _ = data_preparer.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

adj = data_preparer.normalize_adj(adj + sp.eye(adj.shape[0]))

if sparse:
    sp_adj = data_preparer.sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])

if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_test = torch.LongTensor(idx_test)

# build a model
model = Conag(ft_size, hid_units, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
    else:
        adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_test = idx_test.cuda()

# build loss
b_xent = nn.BCEWithLogitsLoss()
f_loss = FocalLoss(logits=True)
ce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCEWithLogitsLoss()

cnt_wait = 0
best = 1e9
best_t = 0

if reload_neg_data:
    neg_num, adj3, seq3 = get_neg_para(anomaly_features)
    with open(device_para_neg_save, 'wb') as f:
        pickle.dump(neg_num, f)
        pickle.dump(adj3, f)
        pickle.dump(seq3, f)
else:
    with open(device_para_neg_save, 'rb') as f:
        neg_num = pickle.load(f)
        adj3 = pickle.load(f)
        seq3 = pickle.load(f)

# neg_num = r_get_neg_num()
# adj3 = get_neg_adj(anomaly_seeds)
# seq3 = get_neg_feature_matrix(anomaly_features)
# lbl_3 = torch.ones(batch_size, neg_num)
# anomaly_num = len(anomaly_seeds)

print('neg num:', neg_num)
print('adj3 shape:', adj3.shape)
print('seq3 shape:', seq3.shape)

if torch.cuda.is_available():
    seq3 = seq3.cuda()
    adj3 = adj3.cuda()

# train_features = features[:, idx_train, :]
# start
print('start train')
nodes = get_nodes_sorted_device()
anomaly_uids = get_anomaly_uids()
if train:
    for epoch in range(nb_epochs):
        # each epoch
        model.train()
        optimiser.zero_grad()
        # row
        # idx = np.random.permutation(len(idx_train))
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]

        # lbl1：train set
        lbl_1_list = []
        for node in nodes:
            if node.uid in anomaly_uids:
                lbl_1_list.append(1)
            else:
                lbl_1_list.append(0)
        lbl_1 = torch.tensor(lbl_1_list)
        lbl_1 = lbl_1.unsqueeze(0)

        # lbl2：row
        lbl_2 = torch.ones(batch_size, nb_nodes)

        # lbl3：anomaly
        lbl_3 = torch.ones(batch_size, neg_num)

        lbl = torch.cat((lbl_1, lbl_2), 1)
        lbl = torch.cat((lbl, lbl_3), 1)
        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()

        logits = model(features, shuf_fts, seq3,
                       sp_adj if sparse else adj,
                       adj3,
                       sparse, None, None, None)

        # get loss: you could try different loss here
        loss = f_loss(logits, lbl)
        print('epoch ', epoch, ' ; loss:', loss.detach())
        with open('write_log.txt', 'a', encoding='utf-8') as f:
            str_w = str(epoch) + '  ' + str(loss) + '\n'
            f.write(str_w)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), embedding_result)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break

        # update
        loss.backward()
        optimiser.step()

# load embedding
print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load(embedding_result))
embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)

# m = nn.BatchNorm1d(embeds.size()[1], affine=True)
# m = nn.LayerNorm(embeds.size()[1:], elementwise_affine=True)
# embeds = m(embeds)

# test
test_train = False
test_load_model = True

train_embs = embeds[0, idx_train]
test_embs = embeds[0, idx_test]
train_lbls = torch.argmax(labels[0, idx_train], dim=1).float()
test_lbls = torch.argmax(labels[0, idx_test], dim=1)
tot = torch.zeros(1)
# tot = tot.cuda()
accs = []
print('start testing')

# print(idx_train)
# print(idx_test)
# print(train_embs)
best_loss = 1e9
best_epoch = 0
cnt_wait = 0
patience = 50

log = LogReg(hid_units, 1)
opt = torch.optim.Adam(log.parameters(), lr=0.001, weight_decay=0.01)
# log.cuda()
f_loss2 = FocalLoss(logits=True, alpha=0.25, gamma=2)
bce = torch.nn.BCELoss()

pat_steps = 1
best_acc = torch.zeros(1)
# best_acc = best_acc.cuda()

if test_load_model:
    log.load_state_dict(torch.load(test_result))

if test_train:
    for _ in range(1000):
        log.train()
        opt.zero_grad()

        logits = log.forward(train_embs)
        logits_resize = logits.view(len(train_lbls))
        logits_resize = torch.sigmoid(logits_resize)
        loss = f_loss2(logits_resize, train_lbls)

        print(_, 'epoch, loss :', loss.detach().numpy())

        t = 0
        f = 0
        for i in range(len(logits_resize)):
            pre = logits_resize[i]
            lbl = train_lbls[i]
            if pre < 0.5 and lbl == 0:
                t += 1
            elif pre > 0.5 and lbl == 1:
                t += 1
            else:
                f += 1
        acc = t/(t+f)
        print('   acc:', acc)

        with open('r_test_log.txt', 'a', encoding='utf-8') as f:
            str_w = str(_) + ' epoch, loss: ' + str(loss.detach().numpy()) + \
                    ', acc: ' + str(acc) + '\n'
            f.write(str_w)

        if loss < best_loss:
            best_loss = loss
            best_epoch = _
            cnt_wait = 0
            torch.save(log.state_dict(), test_result)
        else:
            cnt_wait += 1

        if cnt_wait >= patience:
            print('Early stopping!')
            break

        loss.backward()
        opt.step()

logits = log(test_embs)
logits_resize = torch.sigmoid(logits)
# torch.set_printoptions(profile="full")
# preds = torch.argmax(logits, dim=1)

preds = []
for out in logits_resize:
    if out < 0.5:
        preds.append(0)
    else:
        preds.append(1)
preds = torch.tensor(preds)
torch.set_printoptions(profile="full")
# print(preds)
# print(test_lbls)

tp = float(0)
fp = float(0)
fn = float(0)
tn = float(0)
pre_list = []
lbl_list = []
for i in range(len(preds)):
    # print(test_lbls[i], preds[i])
    if test_lbls[i] == 0:
        # 正常
        if preds[i] == 0:
            tn += 1
        else:
            fp += 1
    else:
        # 异常节点
        if preds[i] == 0:
            fn += 1
        else:
            tp += 1

    lbl_list.append(test_lbls[i])
    pre_list.append(preds[i])

if tp + fp + fn + tn != 0:
    print('acc = ' + str((tp + tn) / (tp + fp + fn + tn)))
if tp + fn != 0:
    print('recall = ' + str(tp / (tp + fn)))
if tp + fp != 0:
    print('precision = ' + str(tp / (tp + fp)))

print('tp:', tp)
print('fp:', fp)
print('fn:', fn)
print('tn:', tn)
# print(acc_add, recall_add, precision_add)
# print(test_lbls)
# print(logits_resize.detach().numpy())

# draw ROC
# fpr, tpr, threshold = roc_curve(test_lbls, logits_resize.detach().numpy())
# roc_auc = auc(fpr, tpr)
# print('auc:', roc_auc)
# lw = 2
# plt.figure(figsize=(8, 5))
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw,
#          label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc="lower right")
# plt.show()
