import numpy as np
from wk5_GMM_CNN import spectrum_crop
from wk5_GMM_CNN import load_data

def feature_extractor_KNN(train_dict, test_dict, batch_num):
    keys = ['0', '1', '2']
    n_class = len(keys)
    test_ys = np.empty((0, n_class))
    test_xs = []
    train_out_dict = {'0': None, '1': None, '2': None}
    for key in keys:
        xs = train_dict[key]
        txs = test_dict[key]
        c = int(key)
        label = np.zeros(n_class)
        label[c] = 1
        train_xs = spectrum_crop(xs, batch_num, n_total=0)
        train_xs = np.array(train_xs)
        temp_test_xs = spectrum_crop(txs, batch_num, n_total=0)
        train_out_dict[key] = train_xs
        temp_test_ys = np.tile(label, (len(temp_test_xs), 1))
        test_xs += temp_test_xs
        test_ys = np.vstack((test_ys, temp_test_ys))
    test_xs = np.array(test_xs)
    return train_out_dict, test_xs, test_ys


def knnro(train_xs, train_ys, n_class, txs):
    k = 9
    reject_rate = 4
    reject_label = n_class
    inner_product = np.matmul(txs, np.transpose(train_xs))
    product_index = np.argsort(-inner_product) # 降序排列
    product_index = product_index[:k]
    nearest_label = [train_ys[i] for i in product_index]
    count_list = [nearest_label.count(j) for j in range(n_class)]
    count_arr = np.array(count_list)
    max_index = np.argmax(count_arr)
    predict = reject_label
    if count_arr[max_index] >= reject_rate:
        predict = max_index
    return predict


def train_jdknnro(train_dict_in, test_dict, batch_num=20):
    n_class = len(train_dict_in)
    train_dict, test_xs, test_ys = feature_extractor_KNN(train_dict_in, test_dict, batch_num)

    # 取训练数据中类别数最少的数量作为训练集的size
    size_list = [train_dict[key].shape[0] for key in train_dict]
    size = min(size_list)
    # 构造训练集
    feature_size =  test_xs.shape[1]
    xs = np.empty((0, feature_size))
    xs_y = []
    for key in train_dict:
        c = int(key)
        temp_xs = train_dict[key]
        np.random.shuffle(temp_xs)
        xs = np.vstack((xs, temp_xs[:size]))
        xs_y += ([c] * size)

    sample_num = test_xs.shape[0]
    knnro_list = []
    for i in range(sample_num):
        txs = test_xs[i]
        txs = np.reshape(txs, (1, feature_size))
        knnro_result = knnro(xs, xs_y, n_class, txs)
        knnro_list.append(knnro_result)
    reject_num = 0
    total = 0
    correct = 0
    confusion_mat = np.zeros((n_class,n_class))
    for i in range(batch_num, test_xs.shape[0]):
        current_knnro = knnro_list[i-20:i]
        count_list = [current_knnro.count(j) for j in range(n_class+1)]
        count_arr = np.array(count_list)
        sorted_index = np.argsort(-count_arr)
        ground = test_ys[i]
        # predict = n_class
        if sorted_index[0] != n_class:
            predict = sorted_index[0]
        elif sorted_index[1] == 0:
            reject_num += 1
            continue
        else:
            predict = sorted_index[1]

        if predict != n_class and predict == ground:
            correct += 1
        total += 1
        confusion_mat[ground, predict] += 1

    print('reject num:', reject_num)
    print('recog num:', total)
    print('GMM test accuracy: ', round(correct / total, 3))
    print(confusion_mat)
    total_sample = np.sum(confusion_mat, 1)
    acc_list = []
    for i in range(0, n_class):
        acc = confusion_mat[i, i] / total_sample[i]
        acc_list.append(acc)
        print('label ', i, 'acc = ', acc)
    return acc_list


if __name__ == '__main__':
    batch_num = 20
    n_round = 50
    acc_arr_knn = np.empty((0, 3))
    # n_total = 2000
    #
    dict = {'0': '', '1': '', '2': ''}
    # dict["0"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete_filtered/Melon"
    # dict["1"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete_filtered/Spinner"
    # dict["2"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete_filtered/Tt"
    dict["0"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Melon"
    dict["1"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Spinner"
    dict["2"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Tt"
    print(dict)
    for i in range(n_round):
        print('=================round %d=================' % i)
        train_dict, test_dict = load_data(dict)
        print('=========================JDKNNRO %d===========================' % i)
        acc_knn = train_jdknnro(train_dict, test_dict, batch_num=batch_num)
        acc_arr_knn = np.vstack((acc_arr_knn, acc_knn))

    print('knn:')
    acc_mean_knn = np.mean(acc_arr_knn, 0)
    acc_std_knn = np.std(acc_arr_knn, 0)
    # acc_arr /= n_round
    print(acc_mean_knn)
    print(acc_std_knn)