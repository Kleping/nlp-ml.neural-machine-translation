
def split_encoded_data_to_train_and_valid():
    print('splitting encoded data to train and valid')
    from random import random
    _valid_coefficient = .3

    _samples = open('data/encoded_data.txt', encoding='utf-8').readlines()

    _range_data = list(range(len(_samples)))
    _range_train = list()
    _range_valid = list()

    _count_train = int(len(_range_data) * (1. - _valid_coefficient))
    _count_valid = len(_range_data) - _count_train

    for _i in _range_data:
        if _count_train != 0 and _count_valid != 0:
            if random() < .5:
                _range_train.append(_i)
                _count_train -= 1
            else:
                _range_valid.append(_i)
                _count_valid -= 1
        elif _count_train != 0:
            _range_train.append(_i)
            _count_train -= 1
        else:
            _range_valid.append(_i)
            _count_valid -= 1

    _lines_train = [_samples[_i][:-1] if _i == _range_train[-1] else _samples[_i] for _i in _range_train]
    _lines_valid = [_samples[_i][:-1] if _i == _range_valid[-1] else _samples[_i] for _i in _range_valid]
    open('data/encoded_data_train.txt', 'w+', encoding='utf-8').writelines(_lines_train)
    open('data/encoded_data_valid.txt', 'w+', encoding='utf-8').writelines(_lines_valid)


def calculate_max_values(_path):
    _samples_train = open(_path, encoding='utf-8').readlines()

    x_max = 0
    y_max = 0

    for i, s in enumerate(_samples_train):
        x, y = s.split('\t')

        _x_max = len(x.split(' '))
        if _x_max > x_max:
            x_max = _x_max

        _y_max = len(y.split(' '))
        if _y_max > y_max:
            y_max = _y_max

    return x_max, y_max


def write_config(_path, _data):
    import json
    with open(_path, 'w', encoding='utf-8') as f:
        json.dump(_data, f, ensure_ascii=False, indent=4)


def encode_sample(_i, _bpe):
    return str(_bpe.encode_ids(_i))[1:-1].replace(',', '')


def encode_data():
    print('encoding data')
    from bpemb import BPEmb

    _vs = 25000
    _bpe_ru = BPEmb(lang='ru', vs=_vs, dim=50)
    _bpe_en = BPEmb(lang='en', vs=_vs, dim=50)

    f_encoded_data = open('data/encoded_data.txt', 'w+')

    for _s in open('data/data.txt', encoding='utf-8').readlines():
        _x, _y = _s.split('\t')
        _y = _y[:-1]

        if _x == '' or _y == '':
            continue

        _encoded_x = encode_sample(_x, _bpe_ru)
        _encoded_y = encode_sample(_y, _bpe_en)
        f_encoded_data.write('{}\t{}\n'.format(_encoded_x, _encoded_y))

    f_encoded_data.close()


def create_and_write_data_parameters():
    print('creating and writing data parameters')
    max_x_train, max_y_train = calculate_max_values('data/encoded_data_train.txt')
    max_x_valid, max_y_valid = calculate_max_values('data/encoded_data_valid.txt')
    write_config('data/encoded_data_train.config', {'max_x': max_x_train, 'max_y': max_y_train})
    write_config('data/encoded_data_valid.config', {'max_x': max_x_valid, 'max_y': max_y_valid})


def remove_files(_path_coll):
    import os
    [os.remove(p) for p in _path_coll if os.path.exists('data/encoded_data_train.txt')]


def initial_clear():
    print('initial cleaning')
    remove_files([
        'data/encoded_data_train.txt',
        'data/encoded_data_valid.txt',
        'data/encoded_data_train.config',
        'data/encoded_data_valid.config',
        'data/encoded_data.txt',
    ])


def final_clear():
    print('final cleaning')
    remove_files([
        'data/encoded_data.txt',
    ])


def test():
    from bpemb import BPEmb

    _vs = 25000
    _bpe_ru = BPEmb(lang='ru', vs=_vs, dim=50)
    _bpe_en = BPEmb(lang='en', vs=_vs, dim=50)
    print(_bpe_en.decode_ids([120, 1005, 2360, 24935, 0]))
    print(_bpe_ru.encode_ids_with_bos_eos(['Сегодня гулял с тобой за ручку', 'Мы действительно одиноки?']))


# test()
# exit()

initial_clear()
encode_data()
split_encoded_data_to_train_and_valid()
create_and_write_data_parameters()
final_clear()
