import os
from resources import get_network
from calculate_features import calculate_all
from convert_features import convert, load_features
from evaluate import evaluator
from utils import load_arrays

def test_model(conf, net=None):
    levels = {
        "calf": 0,
        "covf": 1,
        "eval": 2
    }
    start = conf.get('test', {}).get('from', 'calf') # get start point.
    flag = conf.get('flag', None)
    start = levels[start]
    query, gallery = None, None
    if start <= levels['calf']:
        if net is None:
            net = get_network(conf)
        query, gallery = calculate_all(net, conf)
    else:
        query, gallery = load_features(conf)
    result = None
    if start <= levels['covf']:
        result = convert(query, gallery, conf)
    else:
        result_name = 'result'
        if flag is not None:
            result_name += '_' + flag
        result_path = conf['paths']['params']+'/{0}'.format(result_name)
        result = load_arrays(result_path, backend='matlab')
    evaluater = evaluator(result=result)
    evaluater.evaluate()
    evaluater.show()

if __name__ == '__main__':
    from config import conf
    test_model(conf)