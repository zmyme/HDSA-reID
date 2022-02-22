import numpy as np
from scipy.io import loadmat
import torch
import time

class evaluator():
    """
    result: the given result dict.
    """
    def __init__(self, result=None):
        self.result = result

        self.scores = None
        self.indexs = None
        self.match_flags = None
        self.clean_match = None
        self.aps = None

    def load(self, path):
        self.result = loadmat(path)

    def get_failed_ids(self, rank=1):
        flags = (self.clean_match[:,:rank]==1).any(axis=1)
        failed_ids = np.argwhere(~flags).reshape(-1)
        return failed_ids

    # this function return labels_real ids, flags, distance
    # labels_real: the label of the query_image
    # ids: the id of the image in gallery (not person ids)
    # labels: person id of the matched image.
    # flags: whether this is a correct match
    # distance: the distance of this id
    def get_match_info(self, index, max_num=10, mod="seq", num_correct=1, num_wrong=1):
        labels_real = self.result['query_label'][0][index]
        match = self.match_flags[index]
        valid_matchs = np.argwhere(match!=0).reshape(-1)
        ids = self.indexs[index][valid_matchs]
        labels = self.result['gallery_label'][0][ids]
        flags = (match[valid_matchs]==1)
        distance = self.scores[index][ids]
        if mod == "seq":
            if max_num <= 0:
                max_num = len(ids)
            return labels_real, ids[:max_num], labels[:max_num], flags[:max_num], distance[:max_num]
        elif mod == "compare":
            correct_matches = np.argwhere(match[valid_matchs]==1)
            wrong_matches = np.argwhere(match[valid_matchs]==-1)

            correct_ids = ids[correct_matches].reshape(-1)[:num_correct]
            correct_labels = labels[correct_matches].reshape(-1)[:num_correct]
            correct_flags = flags[correct_matches].reshape(-1)[:num_correct]
            correct_distances = distance[correct_matches].reshape(-1)[:num_correct]

            wrong_ids = ids[wrong_matches].reshape(-1)[:num_wrong]
            wrong_labels = labels[wrong_matches].reshape(-1)[:num_wrong]
            wrong_flags = flags[wrong_matches].reshape(-1)[:num_wrong]
            wrong_distances = distance[wrong_matches].reshape(-1)[:num_wrong]

            ids = np.asarray(wrong_ids.tolist() + correct_ids.tolist())
            labels = np.asarray(wrong_labels.tolist() + correct_labels.tolist())
            flags = np.asarray(wrong_flags.tolist() + correct_flags.tolist())
            distances = np.asarray(wrong_distances.tolist() + correct_distances.tolist())
            return labels_real, ids, labels, flags, distances
            pass
        else:
            raise NotImplementedError('Mode {0} is not implenmented'.format(mod))


    def evaluate(self):
        if self.result is None:
            print('No result mat given, please load it first')
            return None

        # time_counter_start = time.time()

        query_feature = self.result['query_f']
        query_cam = self.result['query_cam'][0]
        query_label = self.result['query_label'][0]
        gallery_feature = self.result['gallery_f']
        gallery_cam = self.result['gallery_cam'][0]
        gallery_label = self.result['gallery_label'][0]
        scores = self.result.get('scores', None)

        num_query = len(query_label)
        num_gallery = len(gallery_label)
        # time_counter_prepare = time.time()

        # compute scores(aka cosine similarity) & sort it (result stored in indexs)
        # scores are sorted from big to small, here is an example.
        # scores:
        # [ 0.1,  0.3,  0.7, -0.2]
        # [ 0.2,  0.5, -0.3, -0.2]
        # [-0.8,  0.5,  0.4,  0.6]
        # the indexs will be:
        # [2 1 0 3]
        # [1 0 3 2]
        # [3 1 2 0]
        # query_feature = torch.from_numpy(query_feature)
        # gallery_feature = torch.from_numpy(gallery_feature.transpose(1, 0))
        # scores = query_feature.mm(gallery_feature)
        # scores = scores.numpy()

        # normalize query_feature and gallery_feature
        def norm_feature(f):
            l2norm = np.linalg.norm(f, ord=2, axis=1)
            l2norm = np.broadcast_to(l2norm, (f.shape[1], f.shape[0])).transpose()
            f = f / l2norm
            return f
        if scores is None:
            query_feature = norm_feature(query_feature)
            gallery_feature = norm_feature(gallery_feature)
            # calculate scores.
            scores = np.dot(query_feature, gallery_feature.transpose()) # num_query * num_gallery
        
        # time_counter_distance = time.time()
        indexs = np.argsort(-scores, axis=1)
        # indexs = np.flip(indexs, axis=1)
        self.scores = scores
        self.indexs = indexs
        # time_counter_sort = time.time()

        # find correct_labels and same_cameras.
        # the shape of the two matix will be (num_query, num_gallery)
        # True means query_label==gallery_label or query_cam==gallery_cam
        match_cameras = gallery_cam[indexs]
        match_labels = gallery_label[indexs]
        query_cam_expand = np.broadcast_to(query_cam, (num_gallery, num_query)).transpose()
        query_label_expand = np.broadcast_to(query_label, (num_gallery, num_query)).transpose()
        correct_labels = (query_label_expand==match_labels)
        same_cameras = (query_cam_expand==match_cameras)

        # generate match_flags, shape=(num_query, num_gallery), value meaning:
        # 1: correct match (same label, different camera)
        # 0: ignored match (same label, same camera or ignore junk indexs.)
        # -1: wrong match (different label)
        match_flags = np.where(correct_labels & (~same_cameras), 1, -1) # set wrong match to -1, others 1
        match_flags = np.where(correct_labels & same_cameras, 0, match_flags) # set same camera as ignored
        match_flags = np.where(match_labels==-1, 0, match_flags) # set junk as ingored.
        self.match_flags = match_flags
        # time_counter_match = time.time()

        # remove ignored match & compute ap for each query.
        # aps stored the ap for each query, clean_match is the match_flags after removing ignored labels.
        # the values in clean_match is only 1 -1 -2, therie meaning are as follows:
        # 1: correct match
        # -1: wrong match
        # -2: empty area, after removing the ignored match, the length of a single match will not be
        # num_gallery, the left space are filled with -2.0. Below is a example of how it is:
        # match_flags:
        # [ 0  0  1 -1  0 -1  1 -1 -1 -1]
        # [ 1  0 -1  0  1 -1 -1 -1 -1 -1]
        # [ 1 -1  0  0 -1 -1 -1 -1 -1 -1]
        # the calculated clean_match matrix will be:
        # [ 1 -1 -1  1 -1 -1 -1 -2 -2 -2]
        # [ 1 -1  1 -1 -1 -1 -1 -1 -2 -2]
        # [ 1 -1 -1 -1 -1 -1 -1 -1 -2 -2]
        # after this, the Rank@N will be easy to compute.
        aps = []
        cmc = np.ones((num_query, num_gallery))
        clean_match = np.ones(match_flags.shape) * (-2.0)
        for i in range(num_query):
            # clean a single match result.
            this_match = match_flags[i] # get a single match
            ignored_match = np.argwhere(this_match==0).reshape(-1) # find ignored match
            this_match = np.delete(this_match, ignored_match) # delete ignored match
            clean_match[i][:len(this_match)] = this_match # write this clean match to clean_match

            # compute this AP.
            good_match = np.argwhere(this_match==1).reshape(-1)
            cmc[i,:good_match[0]] = 0
            good_match = good_match + 1.0
            mask = np.arange(len(good_match)) + 1.0
            precision = mask / good_match
            old_precision = np.ones(precision.shape)
            old_precision[1:] = mask[:-1] / (good_match[1:] - 1.0)
            if good_match[0] == 1:
                old_precision[0] = 1.0
            else:
                old_precision[0] = 0
            ap = (precision + old_precision)/2.0
            ap = ap.mean()
            aps.append(ap)
        aps = np.asarray(aps, dtype=np.float32)
        mAP = aps.mean()
        cmc = cmc.mean(axis=0)
        # print('cmc:', list(cmc))
        self.clean_match = clean_match
        self.aps = aps
        self.mAP = mAP
        self.cmc = cmc
        # time_counter_ap_cmc = time.time()

        # compute Rank@1, Rank@5, Rank@10 and mAP
        rank_1 = cmc[0]
        rank_5 = cmc[4]
        rank_10 = cmc[9]
        # time_counter_end = time.time()

        # print('time_counter_prepare:', time_counter_prepare - time_counter_start)
        # print('time_counter_distance:', time_counter_distance - time_counter_prepare)
        # print('time_counter_sort:', time_counter_sort - time_counter_distance)
        # print('time_counter_match:', time_counter_match - time_counter_sort)
        # print('time_counter_ap_cmc:', time_counter_ap_cmc - time_counter_match)
        # print('time_counter_end:', time_counter_end - time_counter_ap_cmc)
        return rank_1, rank_5, rank_10, mAP


    def results(self, precision=6, indent='    ', ranks=[1, 5, 10], mAP=True):
        info = ''
        for rank in ranks:
            info += 'Rank@{0}: %.{1}f{2}'.format(rank, precision, indent)%(self.cmc[rank-1])
        if mAP:
            info += 'mAP: %.{0}f'.format(precision)%(self.mAP)
        return info

    def show(self, precision=6, indent='    ', ranks=[1, 5, 10], mAP=True):
        print(self.results(precision=6, indent='    ', ranks=[1, 5, 10], mAP=True))

if __name__ == '__main__':
    from config import conf
    evaluate = evaluator()
    start = time.time()
    result_name = 'result'
    result_flag = conf.get('flag', None)
    if result_flag is not None:
        result_name += '_' + result_flag
    evaluate.load(conf['paths']['params']+'/{0}.mat'.format(result_name))
    load = time.time()
    evaluate.evaluate()
    evaluate.show()
    end = time.time()
    print('Time used: %.6fs (load: %.4fs calculate: %.4fs)'%(end-start, load-start, end-load))
