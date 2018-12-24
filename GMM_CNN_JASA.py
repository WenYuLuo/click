import find_click
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers


def energy_normalize(xs):
    energy = np.sqrt(np.sum(xs ** 2))
    xs /= energy
    xs = np.reshape(xs, [-1])
    return xs


def load_data(dict):
    n_class = len(dict)

    # train_dict = {'0': None, '1': None, '2': None}
    # test_dict = {'0': None, '1': None, '2': None}
    train_dict = {}
    test_dict = {}

    for key in dict:
        train_dict[key] = None
        test_dict[key] = None

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
        split_point = int(npy_num/4)
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

    n_class = len(train_dict)
    # test_ys = np.empty((0, n_class))
    # test_xs = []

    train_out_dict = {}
    test_out_dict = {}
    for key in train_dict:
        train_out_dict[key] = None
        test_out_dict[key] = None

    for key in range(n_class):
        xs = train_dict[str(key)]
        txs = test_dict[str(key)]
        if xs is None:
            continue
        c = int(key)
        label = np.zeros(n_class)
        label[c] = 1
        train_xs = cepstrum_crop(xs, batch_num)
        train_xs = np.array(train_xs)
        train_out_dict[str(key)] = train_xs
        temp_test_xs = cepstrum_crop(txs, batch_num)
        temp_test_xs = np.array(temp_test_xs)
        test_out_dict[str(key)] = temp_test_xs
        # temp_test_ys = np.tile(label, (len(temp_test_xs), 1))
        # test_xs += temp_test_xs
        # test_ys = np.vstack((test_ys, temp_test_ys))
    # test_xs = np.array(test_xs)
    return train_out_dict, test_out_dict


def train_gmm(train_dict_in, test_dict, batch_list=[]):

    train_dict, test_out_dict = feature_extractor_GMM(train_dict_in, test_dict, batch_num=10)

    gmm_dict = {}
    for key in train_dict:
        gmm_dict[key] = None

    size_list = [train_dict[key].shape[0] for key in train_dict]
    size = min(size_list)

    n_class = len(train_dict)

    for key in gmm_dict:
        temp_xs = train_dict[key]
        if temp_xs is None:
            continue
        np.random.shuffle(temp_xs)
        temp_xs = temp_xs[:size]
        gmm = GaussianMixture(n_components=16).fit(temp_xs)
        gmm_dict[key] = gmm

    # 混淆矩阵初始化
    confusion_mat_dict = {}
    for batch_ in batch_list:
        confusion_mat_dict[batch_] = np.zeros((n_class, n_class))

    for key in test_out_dict:
        test_xs = test_out_dict[key]
        ground = int(key)

        # 按batch num=10批计算
        sample_num = test_xs.shape[0]
        batch_index = 0
        test_cout = 0
        prob_arr = np.empty((n_class, 0))
        while (True):
            if batch_num * (batch_index + 1) > sample_num:
                break
            test_cout += 1
            prob = []
            txs = test_xs[batch_num * batch_index:batch_num * (batch_index + 1), :]
            for key in range(n_class):
                if gmm_dict[str(key)] is None:
                    continue
                temp_prob = gmm_dict[str(key)].score_samples(txs)
                prob.append(temp_prob)
            prob = np.array(prob)
            prob = np.sum(prob, 1)
            prob_arr = np.hstack((prob_arr, prob))
            batch_index += 1

        # 遍历gmm结果，按不同的batch整合预测
        for i in range(prob_arr.shape[1]):
            for j in batch_list:
                if (i+1) % j == 0:      # 判断是否为batch的整数倍。
                    prob = np.sum(prob_arr[(i+1-j):(i+1)], 1)
                    predict = np.argmax(prob)
                    confusion_mat_dict[j][ground, predict] += 1

    for batch_id in batch_list:
        confusion_mat = confusion_mat_dict[batch_id]
        print('batch id %d :' % batch_id*10)
        total_sample = np.sum(confusion_mat, 1)
        for i in range(0, n_class):
            confusion_mat[i, :] /= total_sample[i]
        print(confusion_mat)
        confusion_mat_dict[batch_id] = confusion_mat
    return confusion_mat_dict


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


def spectrum_crop(xs, batch_num, n_total):
    num = xs.shape[0]
    rc_train_list = []

    if n_total == 0:
        n_total = int(num / batch_num)

    for i in range(0, n_total):
        # txs = np.empty((0, 96))
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
            # txs = np.vstack((txs, crop_x))
            rc_train_list.append(crop_x)
            j += 1
        # rc_train_list.append(txs)
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


def feature_extractor_CNN(train_dict, test_dict, batch_num):
    # keys = ['0', '1', '2']
    n_class = len(train_dict)

    train_size_list = [train_dict[key].shape[0] for key in train_dict]
    train_size = max(train_size_list)*10

    train_total_batch = int(train_size / batch_num)

    train_out_dict = {}
    test_out_dict = {}
    for key in train_dict:
        train_out_dict[key] = None
        test_out_dict[key] = None

    for key in range(n_class):
        xs = train_dict[str(key)]
        if xs is None:
            continue
        txs = test_dict[str(key)]

        temp_train_xs = spectrum_crop(xs, batch_num, n_total=train_total_batch)
        temp_test_xs = spectrum_crop(txs, batch_num, n_total=0)

        train_out_dict[str(key)] = np.array(temp_train_xs)
        test_out_dict[str(key)] = np.array(temp_test_xs)

    return train_out_dict, test_out_dict


def shufflesample(train_dict, n_feature, n_class, batch_size):
    xs = np.empty((0, n_feature))
    ys = np.empty((0, n_class))
    for key in train_dict:
        c = int(key)
        label = np.zeros(n_class)
        label[c] = 1
        temp_train = train_dict[key]
        if temp_train is None:
            continue
        temp_train = np.reshape(temp_train, (-1, n_feature))
        xs = np.vstack((xs, temp_train))
        temp_train_ys = np.tile(label, (temp_train.shape[0], 1))
        ys = np.vstack((ys, temp_train_ys))
    shape = xs.shape
    # print(shape)
    ri = np.random.permutation(shape[0])
    num_batch = int(shape[0] / batch_size)
    for i in range(num_batch):
        batch_xs = np.empty((0, xs.shape[1]))
        batch_ys = np.empty((0, ys.shape[1]))
        for j in range(i * batch_size, (i + 1) * batch_size):
            batch_xs = np.vstack((batch_xs, xs[ri[j]]))
            batch_ys = np.vstack((batch_ys, ys[ri[j]]))
        yield batch_xs, batch_ys


def shuffle_lstm_batch(train_dict, n_feature, n_class, batch_size):
    ys = np.empty((0, n_class))
    xs = []
    for key in train_dict:
        c = int(key)
        label = np.zeros(n_class)
        label[c] = 1
        temp_train = train_dict[key]
        if temp_train is None:
            continue
        xs += temp_train.tolist()
        temp_train_ys = np.tile(label, (temp_train.shape[0], 1))
        ys = np.vstack((ys, temp_train_ys))
    xs = np.array(xs)
    shape = xs.shape
    # print(shape)
    ri = np.random.permutation(shape[0])
    num_batch = int(shape[0] / batch_size)
    for i in range(num_batch):
        batch_xs = np.empty((0, n_feature))
        batch_ys = np.empty((0, ys.shape[1]))
        for j in range(i * batch_size, (i + 1) * batch_size):
            batch_xs = np.vstack((batch_xs, np.reshape(xs[ri[j]], (-1, n_feature))))
            batch_ys = np.vstack((batch_ys, ys[ri[j]]))
        yield batch_xs, batch_ys


def train_cnn(train_dict, test_dict, batch_list=[]):

    train_out_dict, test_out_dict = feature_extractor_CNN(train_dict, test_dict, batch_num=max(batch_list))

    n_class = len(train_out_dict)
    # print(train_xs.shape)
    # print(test_xs.shape)
    input_size = 96
    tf.reset_default_graph()
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
    feature_out = 64
    W_fc5 = weight_variable([256, feature_out])
    b_fc5 = bias_variable([feature_out])
    h_fc5 = tf.nn.relu(tf.matmul(h_fc4_drop, W_fc5) + b_fc5)

    # 输出层
    W_fc2 = weight_variable([feature_out, n_class])
    b_fc2 = bias_variable([n_class])
    y = tf.nn.softmax(tf.matmul(h_fc5, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # lstm
    joint_train = tf.placeholder(dtype=bool, name='joint_train')
    lstm_place_input = tf.placeholder("float", [None, feature_out])

    def reshape_for_lstm(input):
        out_put = tf.reshape(input, [-1, batch_num, feature_out])
        return out_put

    lstm_input = tf.cond(joint_train, lambda: reshape_for_lstm(h_fc5), lambda: reshape_for_lstm(lstm_place_input))

    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=128, use_peepholes=True,
                                        initializer=initializers.xavier_initializer(),
                                        num_proj=n_class)
    BATCH_SIZE = 100
    init_state = lstm_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=lstm_input, initial_state=init_state,
                                        dtype=tf.float32)
    h = outputs[:, -1, :]
    output = tf.nn.softmax(h)
    # W = weight_variable([num_units, D_label])
    # b = bias_variable([D_label])
    # output = tf.nn.softmax(tf.matmul(rnn, W) + b)
    cross_entropy_lstm = -tf.reduce_sum(y_ * tf.log(output))
    lstm_optimizer = tf.train.AdamOptimizer(7e-5).minimize(loss=cross_entropy_lstm)
    correct_prediction_lstm = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy_lstm = tf.reduce_mean(tf.cast(correct_prediction_lstm, "float"))

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    print('training...')
    with tf.Session() as sess:
        sess.run(init)
        max_acc = 0
        for i in range(1000):
            current_acc = 0
            step = 0
            for bxs, bys in shufflesample(train_out_dict, n_feature=input_size, n_class=n_class, batch_size=160):
                m, acc = sess.run((train_step, accuracy), feed_dict={x: bxs, y_: bys, keep_prob: 0.5})
                current_acc += acc
                step += 1
            current_acc = float(current_acc/step)
            print("epoch : %d, training accuracy : %g" % (i + 1, current_acc))
            if current_acc > max_acc:
                max_acc = current_acc
                saver.save(sess, "params/cnn_net_lwy.ckpt")
                if max_acc > 0.97:
                    break
            if max_acc - current_acc > 0.05:
                break
        print("training accuracy converged to : %g" % max_acc)
        saver.restore(sess, "params/cnn_net_lwy.ckpt")

        correct_cout_softmax = 0
        correct_cout_mv = 0

        # 混淆矩阵初始化
        confusion_mat_mv_dict = {}
        confusion_mat_softmax_dict = {}
        for batch_ in batch_list:
            confusion_mat_mv_dict[batch_] = np.zeros((n_class, n_class))
            confusion_mat_softmax_dict[batch_] = np.zeros((n_class, n_class))

        min_batch = min(batch_list)*10
        for key in test_out_dict:
            test_data = test_out_dict[key]
            ground = key
            min_batch_num = test_data.shape[0]/min_batch
            soft_out_arr = np.empty((0, n_class))
            label_mv_arr = np.empty(n_class)
            for i in range(min_batch_num-1):
                txs = test_data[(i*min_batch):((i+1)*min_batch)]
                out_y = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
                mv_out = np.argmax(out_y, 1)

                # soft max 计算
                label_softmax = np.sum(out_y, 0)
                soft_out_arr = np.vstack((soft_out_arr, out_y))

                # majority voting 计算
                label_mv = np.zeros(n_class)
                for i in mv_out:
                    label_mv[i] += 1
                label_mv_arr = np.vstack((label_mv_arr, label_mv))

            for i in range(soft_out_arr.shape[0]):
                for j in batch_list:
                    if (i+1) % j == 0:
                        soft_predict = np.argmax(np.sum(soft_out_arr[i+1-j:i+1], 0))
                        mv_predict = np.argmax(np.sum(label_mv_arr[i+1-j:i+1], 0))
                        confusion_mat_softmax_dict[j][ground, soft_predict] += 1
                        confusion_mat_mv_dict[j][ground, mv_predict] += 1

        print('majority voting result:')
        for batch_id in batch_list:
            confusion_mat = confusion_mat_mv_dict[batch_id]
            print('batch id %d :' % batch_id * 10)
            total_sample = np.sum(confusion_mat, 1)
            for i in range(0, n_class):
                confusion_mat[i, :] /= total_sample[i]
            print(confusion_mat)
            confusion_mat_mv_dict[batch_id] = confusion_mat

        print('softmax sum result:')
        for batch_id in batch_list:
            confusion_mat = confusion_mat_softmax_dict[batch_id]
            print('batch id %d :' % batch_id * 10)
            total_sample = np.sum(confusion_mat, 1)
            for i in range(0, n_class):
                confusion_mat[i, :] /= total_sample[i]
            print(confusion_mat)
            confusion_mat_softmax_dict[batch_id] = confusion_mat

       # # train lstm...
        # for key in train_out_dict:
        #     temp_train_xs = train_out_dict[key]
        #     if temp_train_xs is None:
        #         continue
        #     cnn_result = []
        #     for bxs in temp_train_xs:
        #         cnn_out = sess.run(h_fc5, feed_dict={x: bxs, keep_prob: 1})
        #         # print(cnn_out.shape)
        #         cnn_result.append(cnn_out)
        #     train_out_dict[key] = np.array(cnn_result)
        #
        # print('train lstm....')
        # max_acc = 0
        # # input_size = 256
        # for i in range(1000):
        #     current_acc = 0
        #     step = 0
        #     for bxs, bys in shuffle_lstm_batch(train_out_dict, n_feature=feature_out, n_class=n_class, batch_size=BATCH_SIZE):
        #         m, acc = sess.run((lstm_optimizer, accuracy_lstm), feed_dict={x: np.zeros((10, input_size)),
        #                                                                       lstm_place_input: bxs,
        #                                                                       y_: bys,
        #                                                                       keep_prob: 1.0,
        #                                                                       joint_train: False})
        #         current_acc += acc
        #         step += 1
        #     current_acc = float(current_acc / step)
        #     print("epoch : %d, training accuracy : %g" % (i + 1, current_acc))
        #     if current_acc > max_acc:
        #         max_acc = current_acc
        #         saver.save(sess, "params/cnn_net_lwy.ckpt")
        #         if max_acc > 0.97:
        #             break
        #     if max_acc - current_acc > 0.05:
        #         break
        # print("lstm training accuracy converged to : %g" % max_acc)
        #
        # saver.restore(sess, "params/cnn_net_lwy.ckpt")
        #
        # lstm_batch_index = 0
        # test_cout = 0
        # correct_cout = 0
        # confusion_mat_lstm = np.zeros((n_class, n_class))
        # while (True):
        #     if BATCH_SIZE * (lstm_batch_index + 1) > sample_num:
        #         break
        #     test_cout += 1
        #     # label = np.zeros(n_class)
        #     start = lstm_batch_index * BATCH_SIZE
        #     end = BATCH_SIZE * (lstm_batch_index + 1)
        #     txs = test_xs[start:end]
        #     txs_y = test_ys[start:end]
        #     txs = np.reshape(txs, (-1, input_size))
        #     out_cnn = sess.run(h_fc5, feed_dict={x: txs, keep_prob: 1.0})
        #     out_cnn = np.reshape(out_cnn, (-1, feature_out))
        #     # print(out_cnn.shape)
        #     out_y = sess.run(h, feed_dict={x: np.zeros((10, input_size)),
        #                                    lstm_place_input: out_cnn,
        #                                    keep_prob: 1.0,
        #                                    joint_train: False})
        #     predict = np.argmax(out_y, 1)
        #     ground = np.argmax(txs_y, 1)
        #     for i in range(BATCH_SIZE):
        #         confusion_mat_lstm[ground[i], predict[i]] += 1
        #         if np.equal(ground[i], predict[i]):
        #             correct_cout += 1
        #     lstm_batch_index += 1
        #
        # print('lstm test accuracy: ', round(correct_cout / (lstm_batch_index * BATCH_SIZE), 3))
        # print(confusion_mat_lstm)
        # total_sample = np.sum(confusion_mat_lstm, 1)
        # acc_list_lstm = []
        # for i in range(0, n_class):
        #     acc = confusion_mat_lstm[i, i] / total_sample[i]
        #     acc_list_lstm.append(acc)
        #     print('label ', i, 'acc = ', acc)

        return confusion_mat_mv_dict, confusion_mat_softmax_dict


if __name__ == '__main__':
    # batch_num = 20
    n_round = 15
    n_class = 3
    batch_list = [1, 2, 3, 4, 5] # 表示batch分别为10 20 30 40 50
    confusion_gmm_dict = {}
    confusion_cnnmv_dict = {}
    confusion_cnnsoft_dict = {}
    # confusion_cnnlstm_dict = {}
    for batch_id in batch_list:
        confusion_gmm_dict[batch_id] = []
        confusion_cnnmv_dict[batch_id] = []
        confusion_cnnsoft_dict[batch_id] = []
        # confusion_cnnlstm_dict[batch_id] = np.zeros((n_class, n_class))

    # dict = {'0': '', '1': ''}
    # dict["0"] = "/home/fish/ROBB/CNN_click/click/Xiamen_filtered/chinesewhite"
    # dict["1"] = "/home/fish/ROBB/CNN_click/click/Xiamen_filtered/Neomeris"

    dict = {'0': '', '1': '', '2': ''}
    # dict["0"] = "/home/fish/ROBB/CNN_click/click/Xiamen/bottlenose"
    # dict["1"] = "/home/fish/ROBB/CNN_click/click/Xiamen/chinesewhite"
    # dict["2"] = "/home/fish/ROBB/CNN_click/click/Xiamen/Neomeris"
    #

    dict["0"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete_filtered/Melon"
    dict["1"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete_filtered/Spinner"
    dict["2"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete_filtered/Tt"

    # dict["0"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Melon"
    # dict["1"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Spinner"
    # dict["2"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Tt"

    # dict["0"] = "/home/fish/ROBB/CNN_click/click/CNNDet_wk3/beakedwhale"
    # dict["1"] = "/home/fish/ROBB/CNN_click/click/CNNDet_wk3/pilot"
    # dict["2"] = "/home/fish/ROBB/CNN_click/click/CNNDet_wk3/rissos"
    print(dict)
    for i in range(n_round):
        print('=================round %d=================' % i)
        train_dict, test_dict = load_data(dict)
        print('=========================GMM %d===========================' % i)
        confusion_gmm = train_gmm(train_dict, test_dict, batch_list=batch_list)
        print('=========================CNN %d===========================' % i)
        confusion_mv, confusion_softmax = train_cnn(train_dict, test_dict, batch_list=batch_list)

        for batch_id in batch_list:
            confusion_gmm_dict[batch_id].append(confusion_gmm[batch_id])
            confusion_cnnmv_dict[batch_id].append(confusion_mv[batch_id])
            confusion_cnnsoft_dict[batch_id].append(confusion_softmax[batch_id])
            # confusion_cnnlstm_dict[batch_id] += np.zeros((n_class, n_class))

    for batch_id in batch_list:
        print('batch_num:', batch_id*10)
        mat_gmm = np.array(confusion_gmm_dict[batch_id])
        mat_mv = np.array(confusion_cnnmv_dict[batch_id])
        mat_soft = np.array(confusion_cnnsoft_dict[batch_id])

        mean_gmm = np.mean(mat_gmm, 2)
        std_gmm = np.mean(mat_gmm, 2)
        print('gmm:')
        print('gmm mean:')
        print(mean_gmm)
        print('gmm std:')
        print(std_gmm)

        mean_mv = np.mean(mat_mv, 2)
        std_mv = np.mean(mat_mv, 2)
        print('majority voting:')
        print('majority voting mean:')
        print(mean_mv)
        print('majority voting std:')
        print(std_mv)

        mean_softmax = np.mean(mat_soft, 2)
        std_softmax = np.mean(mat_soft, 2)
        print('softmax:')
        print('softmax mean:')
        print(mean_softmax)
        print('softmax std:')
        print(std_softmax)


