import find_click
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import tensorflow as tf


def energy_normalize(xs):
    energy = np.sqrt(np.sum(xs ** 2))
    xs /= energy
    xs = np.reshape(xs, [-1])
    return xs


def load_data(dict):
    n_class = len(dict)

    train_dict = {'0': None, '1': None, '2': None}
    test_dict = {'0': None, '1': None, '2': None}
    for key in dict:
        path = dict[key]
        # print(path)
        c = int(key)

        ### split by date
        # file_list = find_click.list_files(path)
        # random_index = np.random.permutation(len(file_list))
        # test_set = file_list[random_index[0]]
        # train_set = [file_list[i] for i in random_index[1:]]
        # # label = np.zeros(n_class)
        # # label[c] = 1
        #
        # # training set
        # xs = np.empty((0, 320))
        # count = 0
        # print('training set loading.......')
        # for folder in train_set:
        #     # print('loading %s' % folder[-6:])
        #     npy_list = find_click.list_npy_files(folder)
        #     for npy in npy_list:
        #         # print('loading %s' % npy)
        #         npy_data = np.load(npy)
        #         if npy_data.shape[0] == 0:
        #             continue
        #         xs = np.vstack((xs, npy_data))
        #         count += npy_data.shape[0]
        # print('loaded clicks:', count)
        #
        # # test set
        # txs = np.empty((0, 320))
        # count = 0
        # print('test set loading.......')
        # print('loading %s' % test_set[-6:])
        # npy_list = find_click.list_npy_files(test_set)
        # for npy in npy_list:
        #     # print('loading %s' % npy)
        #     npy_data = np.load(npy)
        #     if npy_data.shape[0] == 0:
        #         continue
        #     txs = np.vstack((txs, npy_data))
        #     count += npy_data.shape[0]
        # print('loaded clicks:', count)

        ### split by file
        npy_files = find_click.list_npy_files(path)
        npy_num = len(npy_files)
        random_index = np.random.permutation(npy_num)
        split_point = int(npy_num/5)
        test_set = [npy_files[i] for i in random_index[:split_point]]
        train_set = [npy_files[i] for i in random_index[split_point:]]

        # training set
        xs = np.empty((0, 320))
        count = 0
        print('training set loading.......')
        for npy in train_set:
            npy_data = np.load(npy)
            if npy_data.shape[0] == 0:
                continue
            xs = np.vstack((xs, npy_data))
            count += npy_data.shape[0]
        print('loaded clicks:', count)

        # testing set
        txs = np.empty((0, 320))
        count = 0
        print('testing set loading.......')
        for npy in test_set:
            print(npy)
            npy_data = np.load(npy)
            if npy_data.shape[0] == 0:
                continue
            txs = np.vstack((txs, npy_data))
            count += npy_data.shape[0]
        print('loaded clicks:', count)

        train_dict[key] = xs
        test_dict[key] = txs

    return train_dict, test_dict


def cepstrum_crop(xs, batch_num):
        num = xs.shape[0]
        rc_train_list = []
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


def feature_extractor_GMM(train_dict, test_dict, batch_num):
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
        train_xs = cepstrum_crop(xs, batch_num)
        train_xs = np.array(train_xs)
        temp_test_xs = cepstrum_crop(txs, batch_num)
        train_out_dict[key] = train_xs
        temp_test_ys = np.tile(label, (len(temp_test_xs), 1))
        test_xs += temp_test_xs
        test_ys = np.vstack((test_ys, temp_test_ys))
    test_xs = np.array(test_xs)
    return train_out_dict, test_xs, test_ys


def train_gmm(train_dict_in, test_dict, batch_num=20):
    n_class = len(train_dict_in)
    # train_dict, test_dict = load_data()
    train_dict, test_xs, test_ys = feature_extractor_GMM(train_dict_in, test_dict, batch_num)

    gmm_dict = {'0': None, '1': None, '2': None}
    # print(train_dict['0'].shape)

    for key in gmm_dict:
        gmm = GaussianMixture(n_components=16).fit(train_dict[key])
        gmm_dict[key] = gmm

    # # print('train acc single:')
    # key = ['0', '1', '2']
    # for i in key:
    #     train_xs = train_dict[i]
    #     label = int(i)
    #     # prob = np.empty((train_xs.shape[0], 3))
    #     prob0 = gmm_dict['0'].score_samples(train_xs)
    #     # prob0 = np.sum(prob0, 1)
    #     prob1 = gmm_dict['1'].score_samples(train_xs)
    #     # prob1 = np.sum(prob1, 1)
    #     prob2 = gmm_dict['2'].score_samples(train_xs)
    #     # prob2 = np.sum(prob2, 1)
    #     prob = np.vstack((prob0, prob1, prob2))
    #     predcit = np.argmax(prob, 0)
    #     correct_id = np.where(predcit==label)[0]
    #     correct_num = correct_id.size
    #     print('label %d num: %d correct: %d acc: %f' % (label, train_xs.shape[0], correct_num, correct_num/train_xs.shape[0]))

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
    # print('batch test accuracy: ', round(correct_cout / test_cout, 3))
    print(confusion_mat)
    total_sample = np.sum(confusion_mat, 1)
    acc_list = []
    for i in range(0, n_class):
        acc = confusion_mat[i, i] / total_sample[i]
        acc_list.append(acc)
        print('label ', i, 'acc = ', acc)
    return np.array(acc_list)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_1x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def split_data(xs):
    num = xs.shape[0]
    split_idx = int(num * 4 / 5)
    xs0 = xs[0:split_idx, :]
    xs1 = xs[split_idx:num, :]
    return xs0, xs1


def spectrum_crop(xs, batch_num, n_total, key):
    num = xs.shape[0]
    rc_train_list = []

    if n_total == 0:
        n_total = int(num / batch_num)

    for i in range(0, n_total):
        # for j in range(batch_num * i, batch_num * (i + 1)):

        j = batch_num * i
        while j >= (batch_num * i) and j < (batch_num * (i + 1)):
            index = j % xs.shape[0]
            temp_x = xs[index]
            # beg_idx = np.random.randint(0, 32)
            beg_idx = np.random.randint(64, (64 + 32))
            crop_x = temp_x[beg_idx:(beg_idx + 192)]
            crop_x = np.reshape(crop_x, [1, 192])
            crop_x = np.fft.fft(crop_x)
            crop_x = np.sqrt(crop_x.real ** 2 + crop_x.imag ** 2)

            crop_x = crop_x[0, :96]
            crop_x = np.reshape(crop_x, [1, 96])
            crop_x = energy_normalize(crop_x)
            rc_train_list.append(crop_x)
            j += 1
    return rc_train_list


def shufflelists(xs, ys, num):
    shape = xs.shape
    ri = np.random.permutation(shape[0])
    ri = ri[0: num]
    batch_xs = np.empty((0, xs.shape[1]))
    batch_ys = np.empty((0, ys.shape[1]))
    for i in ri:
        batch_xs = np.vstack((batch_xs, xs[i]))
        batch_ys = np.vstack((batch_ys, ys[i]))

    return batch_xs, batch_ys


def shufflebatch(xs, ys, num):
    shape = xs.shape
    ri = np.random.permutation(shape[0])
    num_batch = int(shape[0]/num)
    for i in range(num_batch):
        batch_xs = np.empty((0, xs.shape[1]))
        batch_ys = np.empty((0, ys.shape[1]))
        for j in range(i*num, (i+1)*num):
            batch_xs = np.vstack((batch_xs, xs[ri[j]]))
            batch_ys = np.vstack((batch_ys, ys[ri[j]]))
        yield batch_xs, batch_ys


def feature_extractor_CNN(train_dict, test_dict, batch_num, n_total):
    keys = ['0', '1', '2']
    n_class = len(keys)
    train_ys = np.empty((0, n_class))
    test_ys = np.empty((0, n_class))
    train_xs = []
    test_xs = []
    for key in keys:
        xs = train_dict[key]
        txs = test_dict[key]
        c = int(key)
        label = np.zeros(n_class)
        label[c] = 1
        # print('cnn crop training clicks...')
        temp_train_xs = spectrum_crop(xs, batch_num, n_total, key)
        # print('crop testing clicks...')
        temp_test_xs = spectrum_crop(txs, batch_num, n_total=0, key=key)

        temp_train_ys = np.tile(label, (len(temp_train_xs), 1))
        temp_test_ys = np.tile(label, (len(temp_test_xs), 1))
        train_xs += temp_train_xs
        train_ys = np.vstack((train_ys, temp_train_ys))
        test_xs += temp_test_xs
        test_ys = np.vstack((test_ys, temp_test_ys))
    train_xs = np.array(train_xs)
    test_xs = np.array(test_xs)
    return train_xs, train_ys, test_xs, test_ys


def train_cnn(train_dict, test_dict, batch_num=20, n_total=500):
    n_class = len(train_dict)
    # train_dict, test_dict = load_data()
    train_xs, train_ys, test_xs, test_ys = feature_extractor_CNN(train_dict, test_dict, batch_num, n_total)
    # train_xs, train_ys, test_xs, test_ys = load_lwy_data(batch_num, n_total)

    # print(train_xs.shape)
    # print(test_xs.shape)
    input_size = 96

    # x = tf.placeholder("float", [None, input_size])
    # y_ = tf.placeholder("float", [None, n_class])
    #
    # # 输入
    # x_image = tf.reshape(x, [-1, 1, input_size, 1])
    #
    # # 第一个卷积层
    # W_conv1 = weight_variable([1, 5, 1, 32])
    # b_conv1 = bias_variable([32])
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # h_pool1 = max_pool_1x2(h_conv1)
    #
    # # 第二个卷积层
    # W_conv2 = weight_variable([1, 5, 32, 32])
    # b_conv2 = bias_variable([32])
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # h_pool2 = max_pool_1x2(h_conv2)
    #
    # # 密集链接层
    # downsample = int(input_size/4)
    # W_fc1 = weight_variable([1 * downsample * 32, 256])
    # b_fc1 = bias_variable([256])
    # h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * downsample * 32])
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #
    # # Dropout
    # keep_prob = tf.placeholder("float")
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)
    #
    # # 输出层
    # W_fc2 = weight_variable([256, n_class])
    # b_fc2 = bias_variable([n_class])
    # y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    x = tf.placeholder("float", [None, input_size])
    y_ = tf.placeholder("float", [None, n_class])

    # 对于卷积网络，tensorflow的输入为4维[batch, row, col, channels]
    x_signal = tf.reshape(x, [-1, 1, input_size, 1])

    # 1——卷积层 卷积长度：1*8
    W_conv1 = weight_variable([1, 9, 1, 32])
    b_conv1 = bias_variable([32])

    # TODO: 加入BN层
    h_conv1 = conv2d(x_signal, W_conv1) + b_conv1  # 192*1*32
    h_bn1 = tf.nn.relu(h_conv1)  # no batch normalization
    h_pool1 = max_pool_1x2(h_bn1)  # 输出96*1*32

    # 2——卷积层 卷积长度：1*5， 卷积核个数：8
    W_conv2 = weight_variable([1, 5, 32, 32])
    b_conv2 = bias_variable([32])

    # TODO:加入BN层
    h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2  # 96*1*8
    h_bn2 = tf.nn.relu(h_conv2)  # no batch normalization
    h_pool2 = max_pool_1x2(h_bn2)  # 输出48*1*8

    # 3——卷积层 与第2层卷积一致
    W_conv3 = weight_variable([1, 5, 32, 16])
    b_conv3 = bias_variable([16])
    #
    # TODO：加入BN层
    h_conv3 = conv2d(h_pool2, W_conv3) + b_conv3  # 48*1*8
    # h_bn3 = tf.nn.relu(batchnorm(h_conv3, is_testing, out_size=8, convolutional=True))  # batch normalization
    h_bn3 = tf.nn.relu(h_conv3)  # no batch normalization
    h_pool3 = max_pool_1x2(h_bn3)  # 输出为24*1*8

    # 4——全连接层 神经元个数：512
    downsample = int(input_size/8)
    W_fc4 = weight_variable([downsample * 1 * 16, 256])
    b_fc4 = bias_variable([256])
    h_pool3_flat = tf.reshape(h_pool3, [-1, downsample*1*16])
    h_fc4 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc4)+b_fc4)
    # TODO:dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_pro')
    h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob=keep_prob)

    # 5——全连接层 神经元个数：512
    W_fc5 = weight_variable([256, 256])
    b_fc5 = bias_variable([256])
    h_fc5 = tf.nn.relu(tf.matmul(h_fc4_drop, W_fc5) + b_fc5)

    # 输出层
    W_fc2 = weight_variable([256, n_class])
    b_fc2 = bias_variable([n_class])
    y = tf.nn.softmax(tf.matmul(h_fc5, W_fc2) + b_fc2)

    # x = tf.placeholder("float", [None, 96])
    # y_ = tf.placeholder("float", [None, n_class])
    #
    # # 输入
    # x_image = tf.reshape(x, [-1, 1, 96, 1])
    #
    # # 第一个卷积层
    # W_conv1 = weight_variable([1, 5, 1, 32])
    # b_conv1 = bias_variable([32])
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # h_pool1 = max_pool_1x2(h_conv1)
    #
    # # 第二个卷积层
    # W_conv2 = weight_variable([1, 5, 32, 32])
    # b_conv2 = bias_variable([32])
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # h_pool2 = max_pool_1x2(h_conv2)
    #
    # # 密集链接层
    # W_fc1 = weight_variable([1 * 24 * 32, 256])
    # b_fc1 = bias_variable([256])
    # h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 24 * 32])
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #
    # # Dropout
    # keep_prob = tf.placeholder("float")
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)
    #
    # # 输出层
    # W_fc2 = weight_variable([256, n_class])
    # b_fc2 = bias_variable([n_class])
    # y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    print('training...')
    with tf.Session() as sess:
        sess.run(init)
        max_acc = 0
        for i in range(1000):
            current_acc = 0
            step = 0
            for bxs, bys in shufflebatch(train_xs, train_ys, 160):
                m, acc = sess.run((train_step, accuracy), feed_dict={x: bxs, y_: bys, keep_prob: 0.5})
                current_acc += acc
                step += 1
            current_acc = float(current_acc/step)
            # print("epoch : %d, training accuracy : %g" % (i + 1, current_acc))
            if current_acc > max_acc:
                max_acc = current_acc
                saver.save(sess, "params/cnn_net_lwy.ckpt")
                if max_acc > 0.97:
                    break
            if max_acc - current_acc > 0.05:
                break
        print("training accuracy converged to : %g" % max_acc)
        saver.restore(sess, "params/cnn_net_lwy.ckpt")

        # train_num = train_xs.shape[0]
        # prob_mat = np.zeros((n_class, n_class))
        # correct_cout = 0
        # for j in range(0, train_num):
        #     txs = train_xs[j]
        #     txs = np.reshape(txs, [1, 96])
        #     out_y = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
        #     predict_t = np.argmax(out_y)
        #     ground_t = np.argmax(train_ys[j])
        #     prob_mat[ground_t, predict_t] += 1
        #     if np.equal(predict_t, ground_t):
        #         correct_cout += 1
        # print('train accuracy: ', round(correct_cout / train_num, 3))
        # prob_sum = np.sum(prob_mat, 1)
        # for i in range(n_class):
        #     prob_mat[i, 0] /= prob_sum[i]
        #     prob_mat[i, 1] /= prob_sum[i]
        #     prob_mat[i, 2] /= prob_sum[i]


        sample_num = test_xs.shape[0]

        # correct_cout = 0
        # for j in range(0, sample_num):
        #     txs = test_xs[j]
        #     txs = np.reshape(txs, [1, 96])
        #     out_y = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
        #     if np.equal(np.argmax(out_y), np.argmax(test_ys[j])):
        #         correct_cout += 1
        #
        # print('test accuracy: ', round(correct_cout / sample_num, 3))

        batch_index = 0
        test_cout = 0
        correct_cout_softmax = 0
        correct_cout_cnn = 0
        confusion_mat_softmax = np.zeros((n_class, n_class))
        confusion_mat1 = np.zeros((n_class, n_class))
        while (True):
            if batch_num * (batch_index + 1) > sample_num:
                break

            test_cout += 1
            label_softmax = np.zeros(n_class)
            label1 = np.zeros(n_class)
            for j in range(batch_num * batch_index, batch_num * (batch_index + 1)):
                txs = test_xs[j]
                txs = np.reshape(txs, [1, input_size])
                out_y = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
                c = np.argmax(out_y, 1)
                label1[c] += 1
                out_reshaped = np.reshape(out_y, label_softmax.shape)
                label_softmax = label_softmax + out_reshaped
                # label += np.reshape(prob_mat[:, c], [3])

            sample_index = batch_num * batch_index
            ref_y = test_ys[sample_index]
            ground = np.argmax(ref_y)
            predict1 = np.argmax(label1)
            confusion_mat1[ground, predict1] += 1
            if np.equal(np.argmax(label1), np.argmax(ref_y)):
                correct_cout_cnn += 1

            predict = np.argmax(label_softmax)
            confusion_mat_softmax[ground, predict] += 1
            if np.equal(predict, ground):
                correct_cout_softmax += 1

            batch_index += 1

        print('softmax sum result:')
        print('softmax test accuracy: ', round(correct_cout_softmax / test_cout, 3))
        print(confusion_mat_softmax)
        total_sample = np.sum(confusion_mat_softmax, 1)
        acc_list = []
        for i in range(0, n_class):
            acc = confusion_mat_softmax[i, i] / total_sample[i]
            acc_list.append(acc)
            print('label ', i, 'acc = ', acc)

        print('major voting result:')
        print('major voting test accuracy: ', round(correct_cout_cnn / test_cout, 3))
        print(confusion_mat1)
        total_sample = np.sum(confusion_mat1, 1)
        acc_list_mv = []
        for i in range(0, n_class):
            acc = confusion_mat1[i, i] / total_sample[i]
            acc_list_mv.append(acc)
            print('label ', i, 'acc = ', acc)

        return np.array(acc_list_mv), np.array(acc_list)


if __name__ == '__main__':
    batch_num = 20
    n_round = 50
    acc_arr_gmm = np.empty((0, 3))
    acc_arr_cnn = np.empty((0, 3))
    acc_arr_cnn_mv = np.empty((0, 3))
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
        print('=========================GMM %d===========================' % i)
        acc_gmm = train_gmm(train_dict, test_dict, batch_num=batch_num)
        acc_arr_gmm = np.vstack((acc_arr_gmm, acc_gmm))
        print('=========================CNN %d===========================' % i)
        acc_cnn_mv, acc_cnn = train_cnn(train_dict, test_dict, batch_num=batch_num, n_total=9000)
        acc_arr_cnn = np.vstack((acc_arr_cnn, acc_cnn))
        acc_arr_cnn_mv = np.vstack((acc_arr_cnn_mv, acc_cnn_mv))


    print('cnn softmax sum:')
    acc_mean__cnn = np.mean(acc_arr_cnn, 0)
    acc_std_cnn = np.std(acc_arr_cnn, 0)
    # acc_arr /= n_round
    print(acc_mean__cnn)
    print(acc_std_cnn)

    print('cnn major voting:')
    acc_mean__cnn_mv = np.mean(acc_arr_cnn_mv, 0)
    acc_std_cnn_mv = np.std(acc_arr_cnn_mv, 0)
    # acc_arr /= n_round
    print(acc_mean__cnn_mv)
    print(acc_std_cnn_mv)

    print('gmm:')
    acc_mean__gmm = np.mean(acc_arr_gmm, 0)
    acc_std_gmm = np.std(acc_arr_gmm, 0)
    # acc_arr /= n_round
    print(acc_mean__gmm)
    print(acc_std_gmm)