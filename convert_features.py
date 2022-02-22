from collections import defaultdict
import os
from utils import load_arrays, save_arrays
import numpy as np

def convert_single_mat(mat):
    features = defaultdict(list) # num_heads * num_samples * dim
    beliefs = defaultdict(list)
    assigns = [] # num_heads * num_samples * c * h * w
    labels = mat['labels']
    cameras = mat['cameras']
    for key in mat:
        words = key.split('_')
        if words[-1].isdigit():
            if words[0] == 'assign':
                assigns.append(mat[key])
            elif words[0] == 'feature':
                features[words[1]].append(mat[key])
            elif words[0] == 'belief':
                beliefs[words[1]].append(mat[key])
    return features, beliefs, assigns, labels, cameras

def convert(query_mat, gallery_mat, conf):
    """
    convert features to evaluate mat.
    requires conf area:
    conf = {
        "covf": {
            "type": 5 # feature trans type.
            "flag": None # a string that indicates the save name of the mat.
        }
    }
    query_mat = {
        feature_{int}: num_samples * dim
        assign_{int}: num_samples * c * h * w
        labels: 1 * num_samples
        cameras: 1 * num_samples
        paths: num_samples
    }
    return: the converted mat.
    """
    def weighted_concatenation(features, weights, axis=1):
        features = [f * w for f, w in zip(features, weights)]
        features = np.concatenate(features, axis=axis)
        return features
    cvtconf = conf['covf']
    trans_type = cvtconf.get('type', 'cat')
    trans_args = cvtconf.get('args', {})

    qf, qb, qa, ql, qc = convert_single_mat(query_mat)
    gf, gb, ga, gl, gc = convert_single_mat(gallery_mat)

    trans_feature = trans_args.get('feature', 'trans')
    if trans_feature is not None:
        qf = qf[trans_feature]
        gf = gf[trans_feature]

    scores = None
    if trans_type == 'cat':
        weights = trans_args.get('weights', [1.0] * len(qa))
        print('concatenting with weights:', weights)
        qf = weighted_concatenation(qf, weights)
        gf = weighted_concatenation(gf, weights)
    elif trans_type == 'idx':
        idx = trans_args.get('id', -1)
        print('extracting index:', idx)
        qf = qf[idx]
        gf = gf[idx]
    else:
        raise ValueError('Transtype {0} not understood'.format(trans_type))

    result = {
        'query_f':qf,
        'query_label':ql,
        'query_cam':qc,
        'query_path': query_mat['paths'],
        'gallery_f':gf,
        'gallery_label':gl,
        'gallery_cam':gc,
        'gallery_path': gallery_mat['paths'],
    }
    if scores is not None:
        result['scores'] = scores

    flag = conf['flag']
    save_name = 'result'
    if flag is not None:
        save_name += '_' + flag
    save_arrays(conf['paths']['params'] + '/{0}'.format(save_name), result, backend='matlab')
    return result

def load_features(conf):
    flag = conf.get('flag', None)
    query_name, gallery_name = 'query', 'gallery'
    if flag is not None:
        query_name = flag + '_' + query_name
        gallery_name = flag + '_' + gallery_name
    query = load_arrays(os.path.join(conf['paths']['params'], query_name), backend='numpy', hint=True)
    gallery = load_arrays(os.path.join(conf['paths']['params'], gallery_name), backend='numpy', hint=True)
    return query, gallery

if __name__ == '__main__':
    from config import conf
    # query_mat = loadmat(conf['paths']['params'] + '/query.mat')
    # gallery_mat = loadmat(conf['paths']['params'] + '/gallery.mat')
    query_mat, gallery_mat = load_features(conf)
    result = convert(query_mat, gallery_mat, conf)
    for key in result:
        if key[:2] != '__':
            print('{0}.shape = {1}'.format(key, result[key].shape))
