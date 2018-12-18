import find_click
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def random_crop(xs, batch_num, n_total):
    num = xs.shape[0]
    rc_train_list = []

    if n_total == 0:
        n_total = int(num / batch_num)

    hanming_win = np.hamming(xs.shape[1])
    for i in range(0, n_total):
        # for j in range(batch_num * i, batch_num * (i + 1)):

        j = batch_num * i
        while j >= (batch_num * i) and j < (batch_num * (i + 1)):
            index = j % xs.shape[0]
            temp_x = xs[index] * hanming_win
            # beg_idx = np.random.randint(0, 32)
            # beg_idx = np.random.randint(64, (64 + 32))
            # crop_x = temp_x[beg_idx:(beg_idx + 192)]
            # crop_x = np.reshape(crop_x, [1, 192])
            crop_x = np.fft.fft(temp_x, 2048)
            crop_x = np.sqrt(crop_x.real ** 2 + crop_x.imag ** 2)

            crop_x = crop_x[:1024]
            crop_x = np.fft.fft(np.log(crop_x))
            crop_x = np.sqrt(crop_x.real ** 2 + crop_x.imag ** 2)
            crop_x = crop_x[1:15]
            # crop_x = np.reshape(crop_x, [1, 14])
            # crop_x = energy_normalize(crop_x)
            rc_train_list.append(crop_x)
            j += 1
    # print('sampled from %d clicks' % xs.shape[0])
    return rc_train_list


def energy_normalize(xs):
    energy = np.sqrt(np.sum(xs ** 2))
    xs /= energy
    xs = np.reshape(xs, [-1])
    return xs


def load_data(batch_num=20):

    dict = {'0': '', '1': '', '2': ''}
    # dict["0"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Melon"
    # dict["1"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Spinner"
    # dict["2"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Tt"
    dict["0"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Melon"
    dict["1"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Spinner"
    dict["2"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Tt"


    n_class = len(dict)
    test_ys = np.empty((0, n_class))
    test_xs = []

    # gmm_dict = {'0': None, '1': None, '2': None}
    train_dict = {'0': None, '1': None, '2': None}

    for key in dict:
        path = dict[key]

        print(path)

        c = int(key)

        # npy_files = find_click.list_npy_files(path)

        file_list = find_click.list_files(path)

        random_index = np.random.permutation(len(file_list))

        test_set = file_list[random_index[0]]

        train_set = [file_list[i] for i in random_index[1:]]

        label = np.zeros(n_class)
        label[c] = 1

        # training set
        xs = np.empty((0, 320))
        count = 0
        print('training set loading.......')
        for folder in train_set:
            # print('loading %s' % folder[-6:])
            npy_list = find_click.list_npy_files(folder)
            for npy in npy_list:
                # print('loading %s' % npy)
                npy_data = np.load(npy)
                if npy_data.shape[0] == 0:
                    continue
                xs = np.vstack((xs, npy_data))
                count += npy_data.shape[0]
        print('loaded clicks:', count)

        # test set
        txs = np.empty((0, 320))
        count = 0
        print('test set loading.......')
        print('loading %s' % test_set[-6:])
        npy_list = find_click.list_npy_files(test_set)
        for npy in npy_list:
            # print('loading %s' % npy)
            npy_data = np.load(npy)
            if npy_data.shape[0] == 0:
                continue
            txs = np.vstack((txs, npy_data))
            count += npy_data.shape[0]
        print('loaded clicks:', count)

        print('crop training clicks...')
        train_xs = random_crop(xs, batch_num, n_total=0)
        train_xs = np.array(train_xs)
        print('crop testing clicks...')
        temp_test_xs = random_crop(txs, batch_num, n_total=0)

        train_dict[key] = train_xs

        # gmm = GMM(n_components=16).fit(train_xs)
        #
        # gmm_dict[key] = gmm

        temp_test_ys = np.tile(label, (len(temp_test_xs), 1))
        test_xs += temp_test_xs
        test_ys = np.vstack((test_ys, temp_test_ys))
    test_xs = np.array(test_xs)
    return train_dict, test_xs, test_ys


def train_gmm(n_class, batch_num=20):
    train_dict, test_xs, test_ys = load_data(batch_num)
    gmm_dict = {'0': None, '1': None, '2': None}
    print(train_dict['0'].shape)

    for key in gmm_dict:
        gmm = GaussianMixture(n_components=16).fit(train_dict[key])
        gmm_dict[key] = gmm

    print('train acc single:')
    key = ['0', '1', '2']
    for i in key:
        train_xs = train_dict[i]
        label = int(i)
        # prob = np.empty((train_xs.shape[0], 3))
        prob0 = gmm_dict['0'].score_samples(train_xs)
        # prob0 = np.sum(prob0, 1)
        prob1 = gmm_dict['1'].score_samples(train_xs)
        # prob1 = np.sum(prob1, 1)
        prob2 = gmm_dict['2'].score_samples(train_xs)
        # prob2 = np.sum(prob2, 1)
        prob = np.vstack((prob0, prob1, prob2))
        predcit = np.argmax(prob, 0)
        correct_id = np.where(predcit==label)[0]
        correct_num = correct_id.size
        print('label %d num: %d correct: %d acc: %f' % (label, train_xs.shape[0], correct_num, correct_num/train_xs.shape[0]))

    sample_num = test_xs.shape[0]
    batch_index = 0
    test_cout = 0
    correct_cout = 0
    confusion_mat = np.zeros((n_class, n_class))
    while (True):
        if batch_num * (batch_index + 1) > sample_num:
            break

        test_cout += 1
        label = np.zeros(n_class)
        txs = test_xs[batch_num * batch_index:batch_num * (batch_index + 1), :]
        prob0 = gmm_dict['0'].score_samples(txs)
        # prob0 = np.sum(prob0, 1)
        prob1 = gmm_dict['1'].score_samples(txs)
        # prob1 = np.sum(prob1, 1)
        prob2 = gmm_dict['2'].score_samples(txs)
        # prob2 = np.sum(prob2, 1)
        prob = np.vstack((prob0, prob1, prob2))
        # prob = np.log(prob)
        prob = np.sum(prob, 1)
        # c = np.argmax(prob, 1)
        sample_index = batch_num * batch_index
        ref_y = test_ys[sample_index]
        ground = np.argmax(ref_y)
        predict = np.argmax(prob)
        confusion_mat[ground, predict] += 1
        if np.equal(np.argmax(prob), np.argmax(ref_y)):
            correct_cout += 1
        batch_index += 1
    print('batch test accuracy: ', round(correct_cout / test_cout, 3))
    print(confusion_mat)
    total_sample = np.sum(confusion_mat, 1)
    acc_list = []
    for i in range(0, n_class):
        acc = confusion_mat[i, i] / total_sample[i]
        acc_list.append(acc)
        print('label ', i, 'acc = ', acc)
    return np.array(acc_list)


batch_num = 20
n_class = 3
n_round = 5
acc_arr = np.empty((0, 3))
# n_total = 2000
#
for i in range(n_round):
    print('=================round %d=================' % i)
    acc = train_gmm(n_class=n_class , batch_num=batch_num)
    acc_arr = np.vstack((acc_arr, acc))

acc_mean = np.mean(acc_arr, 0)
acc_std = np.std(acc_arr, 0)
# acc_arr /= n_round
print(acc_mean)
print(acc_std)