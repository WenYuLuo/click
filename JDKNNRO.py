import numpy as np
from wk5_GMM_CNN import spectrum_crop
from wk5_GMM_CNN import load_data
from wk5_GMM_CNN import cepstrum_crop


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
        # train_xs = cepstrum_crop(xs, batch_num)

        train_xs = np.array(train_xs)

        temp_test_xs = spectrum_crop(txs, batch_num, n_total=0)
        # temp_test_xs = cepstrum_crop(txs, batch_num)

        train_out_dict[key] = train_xs
        temp_test_ys = np.tile(label, (len(temp_test_xs), 1))
        test_xs += temp_test_xs
        test_ys = np.vstack((test_ys, temp_test_ys))
    test_xs = np.array(test_xs)
    return train_out_dict, test_xs, test_ys


def knnro(train_xs, train_ys, n_class, txs):
    k = 20
    reject_rate = 15
    reject_label = n_class
    inner_product = np.matmul(txs, np.transpose(train_xs))
    product_index = np.argsort(-inner_product) # 降序排列
    product_index = product_index[0][:k]
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
    print(size)
    size = 1000
    # 构造训练集
    feature_size =  test_xs.shape[1]
    xs = np.empty((0, feature_size))
    xs_y = []
    for key in train_dict:
        c = int(key)
        temp_xs = train_dict[key]
        np.random.shuffle(temp_xs)
        ref_txs = temp_xs[0]
        inner_product = np.matmul(ref_txs, np.transpose(temp_xs))
        product_index = np.argsort(-inner_product)  # 降序排列
        selected_xs = [temp_xs[product_index[i]] for i in range(0, temp_xs.shape[0], 10)]
        selected_xs = np.array(selected_xs)
        np.random.shuffle(selected_xs)
        xs = np.vstack((xs, selected_xs[:size]))
        # xs = np.vstack((xs, temp_xs[:size]))
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
        valid_num = sum(count_list[:n_class])
        sorted_index = np.argsort(-count_arr)
        ground = np.argmax(test_ys[i])
        # predict = n_class
        if sorted_index[0] != n_class:
            predict = sorted_index[0]
        elif count_arr[sorted_index[1]] == 0\
                or count_arr[sorted_index[1]]/valid_num < 0.5: # batch拒识别率
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
    # dict["0"] = "/home/fish/ROBB/CNN_click/click/CNNDet_wk3/beakedwhale"
    # dict["1"] = "/home/fish/ROBB/CNN_click/click/CNNDet_wk3/pilot"
    # dict["2"] = "/home/fish/ROBB/CNN_click/click/CNNDet_wk3/rissos"

    # dict["0"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Melon"
    # dict["1"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Spinner"
    # dict["2"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Tt"

    dict["0"] = "/home/fish/ROBB/CNN_click/click/Xiamen/bottlenose"
    dict["1"] = "/home/fish/ROBB/CNN_click/click/Xiamen/chinesewhite"
    dict["2"] = "/home/fish/ROBB/CNN_click/click/Xiamen/Neomeris"

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