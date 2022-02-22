import cv2
import numpy as np
import torch
from config import conf
from resources import get_network, get_dataloader
from utils import filt_modules
from dataset import ReIDDataSet
import torch.nn.functional as torch_func

# generate person mask.
def get_person_mask_ellipse(h, w):
    y = torch.arange(h).float()
    x = torch.arange(w).float()
    grid_y, grid_x = torch.meshgrid(y, x)
    grid_y = grid_y * 2 / h - 1.0
    grid_x = grid_x * 2 / w - 1.0
    mask = torch.where(grid_y**2 + grid_x**2 <= 1, torch.zeros(h, w), torch.ones(h, w))
    return mask

def get_assign_score_by_mask(assigns, masks, normalize=False):
    scores = []
    b = assigns[0].shape[0]
    if len(masks.shape) == 2:
        masks = masks.repeat(b, 1, 1)
    _, h, w = masks.shape
    for assign in assigns:
        assign = torch_func.interpolate(assign, (h, w), mode='bicubic', align_corners=True).clamp(0, 1) # b * c * h * w
        b, c, _, _ = assign.shape
        # assign = torch_func.normalize(assign.view(b, c, -1), p=2, dim=-1).view(b, c, h, w)
        forgrounds = masks.repeat(c, 1, 1, 1).permute(1, 0, 2, 3) # b * c * h * w
        score = (assign * forgrounds).view(b, c, -1).sum(dim=-1)
        if normalize:
            full = assign.view(b, c, -1).sum(dim=-1) # b * c
            score = score / full
        score = score.sum(dim=0)
        scores.append(score)
    return scores, b

def random_erase(imgs):
    b, _, h, w = imgs.shape
    masks = []
    erase_h = h // 2
    erase_w = w //2
    moving_h = h - erase_h
    moving_w = w - erase_w
    masks = torch.zeros(b, h, w).type(imgs.type())
    for i in range(b):
        left = torch.randint(0, moving_w, (1,)).item()
        top = torch.randint(0, moving_h, (1,)).item()
        randpatch = torch.rand(erase_h, erase_w).type(imgs.type())
        randpatch = randpatch - randpatch.mean()
        randpatch = randpatch / randpatch.std()
        imgs[i, :, top:top+erase_h, left:left+erase_w] = randpatch
        masks[i, top:top+erase_h, left:left+erase_w] = 1.0
    return imgs, masks

def extract_assigns(vlads):
    assigns = []
    for i, vlad in enumerate(vlads):
        assign = vlad.get_assign(tonp=False, cpu=False) # b * c * h * w
        assigns.append(assign)
    return assigns

if __name__ == '__main__':
    person_mask = get_person_mask_ellipse(384, 128).cuda() # h * w
    # print('person_mask =', person_mask)
    # print('person_mask.shape =', person_mask.shape)
    # mask_view = person_mask.cpu().detach().numpy().astype(np.uint8) * 254
    # mask_view = np.dstack([mask_view, mask_view, mask_view])
    # print('mask_view.shape =', mask_view.shape)
    # imshow('mask', mask_view, waitKey=0)
    # raise KeyboardInterrupt
    dataloader = get_dataloader(conf, 'train')
    dataset:ReIDDataSet = dataloader.dataset
    dataset.random_crop = False
    dataset.random_erase = False
    dataset.random_filp = False
    net = get_network(conf)
    net = net.eval()
    vlads = filt_modules(net, 'vlad_wrap')


    accumulated_scores = None

    mineconf = conf.get('mine', {})
    topk = mineconf.get('k', 8)
    escape = mineconf.get('escape', False)
    show = mineconf.get('show', False)
    normalize = mineconf.get('normalize', False)

    def stringfy_cluster(clusters, escape=False):
        starter, ender = '[', ']'
        if escape:
            starter, ender = '\\[', '\\]'
        infos = []
        for head in clusters:
            head = starter + ','.join([str(c) for c in head]) + ender
            infos.append(head)
        info = starter + ','.join(infos) + ender
        return info

    with torch.no_grad():
        for data in dataloader:
            imgs, labels, cameras, paths = data
            imgs = imgs.cuda() # b * c * h * w
            imgs_np = imgs.cpu().numpy()
            b = imgs.shape[0]
            features = net(imgs)
            assigns = extract_assigns(vlads)
            person_mask_scores, _ = get_assign_score_by_mask(assigns, person_mask)
            if accumulated_scores is None:
                accumulated_scores = person_mask_scores
            else:
                accumulated_scores = [a + p for a, p in zip(accumulated_scores, person_mask_scores)]
            topk_bg_cluster = [score.topk(topk, dim=0, largest=True)[1].cpu().detach().numpy().tolist() for score in accumulated_scores]
            print('TOPK:', stringfy_cluster(topk_bg_cluster, escape=escape))