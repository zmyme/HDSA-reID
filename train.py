import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

import utils

import resources

def extract(data):
    img, people_id, camera_id, relative_path = data
    
    # info = [x for x in zip(relative_path, people_id.numpy().tolist(), camera_id.numpy().tolist())]
    # print('\n'.join(['[{0}] {1} {2}'.format(*x) for x in info]))
    
    images = img.cuda().float()
    labels = people_id.cuda().long()
    return images, labels

def exclude_items(original, exclude):
    exclude_id = [id(p) for p in exclude]
    items_left = [p for p in original if id(p) not in exclude_id]
    return items_left

def remove_freeze_params(params):
    params_trainable = [p for p in params if p.requires_grad]
    print('Total Params removed:', len(params) - len(params_trainable))
    return params_trainable

# def decompose_params(net, verify=True):
#     params = list(net.parameters())
#     params_pretrain = list(net.feature_extract.base_model.model.parameters())
#     params_conv = list(net.feature_extract.base_model.parameters())
#     params_others = exclude_items(params_conv, params_pretrain)
#     # params_conv = prams_pretrain + params_others
#     params_fc = exclude_items(params, params_conv)
#     # params = params_conv + params_fc

#     # filter params
#     params_pretrain = remove_freeze_params(params_pretrain)
#     params_fc = remove_freeze_params(params_fc)
#     params_others = remove_freeze_params(params_others)
#     return params_pretrain, params_others, params_fc

def decompose_params(net:nn.Module, conf):
    finetune_mode = conf.get('no_fc_classifier', False)
    params = list(net.parameters())
    params_pretrain = list(net.backbone.parameters())
    params_pooling = []
    for head in net.heads:
        this_pooling = list(head.pooling.parameters())
        if finetune_mode:
            this_pooling_no_pretrain = list(head.pooling.bn.parameters())
            this_pooling_pretrain = exclude_items(this_pooling, this_pooling_no_pretrain)
            params_pretrain += this_pooling_pretrain
            this_pooling = this_pooling_no_pretrain
        params_pooling += this_pooling
    params_fc = exclude_items(params, params_pooling + params_pretrain)

    # some hint.
    print('num params_all:', len(params))
    print('num params_pretrain before remvoe:', len(params_pretrain))
    print('num params_pooling before remvoe:', len(params_pooling))
    print('num params_fc before remvoe:', len(params_fc))

    # remove freeze params
    params_pretrain = remove_freeze_params(params_pretrain)
    params_pooling = remove_freeze_params(params_pooling)
    params_fc = remove_freeze_params(params_fc)
    print('num params_pretrain after remvoe:', len(params_pretrain))
    print('num params_pooling after remvoe:', len(params_pooling))
    print('num params_fc after remvoe:', len(params_fc))
    return params_pretrain, params_pooling, params_fc

# def decompose_params_false(net:nn.Module):
#     params = list(net.parameters())
#     print('num_params:', len(params))
#     fc_layers = filt_modules(net, 'Linear')
#     fc_params = []
#     for fc in fc_layers:
#         fc_params += list(fc.parameters())
#     conv_params = exclude_items(params, fc_params)
#     # params = conv_params + fc_params
#     pretrain_params = net.backbone.parameters()
#     other_params = exclude_items(conv_params, pretrain_params)
#     # conv_params = pretrain_params + other_params
#     print('conv_params:', len(conv_params))
#     print('fc_params:', len(fc_params))
#     conv_params = remove_freeze_params(conv_params)
#     fc_params = remove_freeze_params(fc_params)
#     return pretrain_params, other_params, fc_params

def freeze_module(m):
	params = m.parameters()
	for param in params:
		param.requires_grad = False
	m.eval()

# def test(net, dataloader):

def train(conf):
    dataloader = resources.get_dataloader(conf, names='train')
    net = resources.get_network(conf)
    num_outputs = net.num_outputs
    criterion = resources.get_criterion(conf)
    criterion.num = num_outputs
    # freeze modules
    for layer in net.backbone.get_frozen_layers():
        freeze_module(layer)
    if conf['model']['head']['pooling'].get('static', False):
        vlads = utils.filt_modules(net, 'vlad_core_euc')
        for vlad in vlads:
            freeze_module(vlad)
    # obtain optimizer
    pretrain_params, other_params, fc_params = decompose_params(net, conf)
    optimconf = conf['optim']
    base_lr = optimconf['lr']
    weight_decay = optimconf['wd']
    # optimizer = optim.Adam(net.parameters(), lr=0.00035, weight_decay=1e-4)
    optimizer = optim.SGD([
         {'params': pretrain_params, 'lr': base_lr * 0.1},
         {'params': other_params, 'lr': base_lr * 0.1},
         {'params': fc_params, 'lr': base_lr},
         ], weight_decay=weight_decay, momentum=0.9, nesterov=True)
    # optimizer = optim.Adam([
    #      {'params': pretrain_params, 'lr': base_lr * 0.1},
    #      {'params': other_params, 'lr': base_lr * 0.1},
    #      {'params': fc_params, 'lr': base_lr},
    #      ], weight_decay=weight_decay)
    
    scheduler = utils.lr_manager(optimizer, **conf['scheduler'])
    lossmgr = utils.LossManager()
    lossmgr.config(**criterion.get_weights())
    num_batches = len(dataloader)
    summarizer = utils.TrainTableSummarizer(num_batches)
    # generate summarizer configs.
    configs = [{"name": 'AvgLoss', "width": 8}]
    for key in lossmgr.weights:
        configs.append({"name": key})
    for i in range(num_outputs):
        configs.append({"name": "ACC{0}".format(i+1), "mode": "latest"})
    summarizer.config(configs)
    summarizer.start()

    for epoch in range(conf['num_epoch']):
        net = net.train(True)
        scheduler.step()
        num_samples = 0
        num_corrects = [0] * num_outputs
        for data in dataloader:
            optimizer.zero_grad()
            images, labels = extract(data)
            preds = net(images)
            losses = criterion(preds, labels)
            loss = lossmgr.feed(losses)
            loss.backward()
            optimizer.step()

            # record num_samples and num_corrects
            num_samples += labels.shape[0]
            if net.triplet:
                preds = [pred[0] for pred in preds]
            preds = [torch.max(pred[-1], dim=1)[1] for pred in preds]
            # print("preds: ", preds)
            # print([pred == labels for pred in preds])
            num_corrects = [num_correct + (pred == labels).sum().item() for num_correct, pred in zip(num_corrects, preds)]
            info = {}
            info['AvgLoss'] = loss.item()
            info.update({name:losses[name].item() for name in losses})
            for i in range(num_outputs):
                info['ACC{0}'.format(i+1)] = num_corrects[i]/num_samples
            # print("info:", info)
            summarizer.feed_batch(info)
        
        if (epoch+1)%1 == 0:
            filename = 'epoch_{0}.pth'.format(epoch+1)
            savepath = os.path.join(conf['paths']['checkpoint'], filename)
            torch.save(net.state_dict(), savepath)
        
        summarizer.summary_epoch()
    summarizer.finish()
    
    filename = [
        conf['flag'] if conf['flag'] is not None else conf['summary'],
        time.strftime("%Y%m%d_%H%M%S", time.localtime())
    ]
    filename = '_'.join(filename) + '.pth'
    savepath = conf['paths']['checkpoint'] + '/' + filename
    torch.save(net.state_dict(), savepath)
    print('Saving final checkpoint to', savepath)
    return net

if __name__ == '__main__':
    from config import conf
    from test import test_model
    net = train(conf)
    test_model(conf, net)