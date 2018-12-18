import tensorflow as tf
import find_click
import numpy as np
import time
import matplotlib.pyplot as plt
from wk5_GMM_CNN import load_data


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


def random_crop(xs, batch_num, n_total, key):
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

    print('sampled from %d clicks' % xs.shape[0])
    return rc_train_list


def energy_normalize(xs):
    energy = np.sqrt(np.sum(xs ** 2))
    xs /= energy
    xs = np.reshape(xs, [-1])
    return xs


# def load_data(data_path, n_class, batch_num=20, n_total=500):
#     train_xs = np.empty((0, 192))
#     train_ys = np.empty((0, n_class))
#     test_xs = np.empty((0, 192))
#     test_ys = np.empty((0, n_class))
#
#     for c in range(0, n_class):
#         path = "%(path)s/%(class)d" % {'path': data_path, 'class': c}
#         wav_files = find_click.list_wav_files(path)
#
#         print("load data : %s, the number of files : %d" % (path, len(wav_files)))
#
#         label = np.zeros(n_class)
#         label[c] = 1
#
#         # xs = np.empty((0, 256))
#         xs = np.empty((0, 320))
#         count = 0
#         #
#         for pathname in wav_files:
#             wave_data, frame_rate = find_click.read_wav_file(pathname)
#
#             # energy = np.sqrt(np.sum(wave_data ** 2))
#             # wave_data /= energy
#             # wave_data = np.reshape(wave_data, [-1])
#             xs = np.vstack((xs, wave_data))
#             count += 1
#             if count >= batch_num * n_total:
#                 break
#
#         xs0, xs1 = split_data(xs)
#
#         temp_train_xs = random_crop(xs0, batch_num, int(n_total * 4 / 5))
#         temp_test_xs = random_crop(xs1, batch_num, int(n_total / 5))
#
#         temp_train_ys = np.tile(label, (temp_train_xs.shape[0], 1))
#         temp_test_ys = np.tile(label, (temp_test_xs.shape[0], 1))
#
#         train_xs = np.vstack((train_xs, temp_train_xs))
#         train_ys = np.vstack((train_ys, temp_train_ys))
#         test_xs = np.vstack((test_xs, temp_test_xs))
#         test_ys = np.vstack((test_ys, temp_test_ys))
#
#     return train_xs, train_ys, test_xs, test_ys


def load_npy_data(batch_num=20, n_total=500):

    dict = {'0': '', '1': '', '2': ''}
    # dict["0"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Melon"
    # dict["1"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Spinner"
    # dict["2"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Tt"
    dict["0"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Melon"
    dict["1"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Spinner"
    dict["2"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Tt"


    n_class = len(dict)
    # train_xs = np.empty((0, 96))
    train_ys = np.empty((0, n_class))
    # test_xs = np.empty((0, 96))
    test_ys = np.empty((0, n_class))

    train_xs = []
    # train_ys = []
    test_xs = []
    # test_ys = []

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
        temp_train_xs = random_crop(xs, batch_num, n_total , key)
        print('crop testing clicks...')
        temp_test_xs = random_crop(txs, batch_num, n_total=0, key=key)

        temp_train_ys = np.tile(label, (len(temp_train_xs), 1))
        temp_test_ys = np.tile(label, (len(temp_test_xs), 1))
        train_xs += temp_train_xs
        train_ys = np.vstack((train_ys, temp_train_ys))
        test_xs += temp_test_xs
        test_ys = np.vstack((test_ys, temp_test_ys))
    train_xs = np.array(train_xs)
    test_xs = np.array(test_xs)

    return train_xs, train_ys, test_xs, test_ys



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

from tensorflow.contrib.layers.python.layers import initializers

def train_cnn(n_class, batch_num=20, n_total=500):

    print("train cnn for one click ... ...")

    # train_xs, train_ys, test_xs, test_ys = load_data(data_path, n_class, batch_num, n_total)
    train_xs, train_ys, test_xs, test_ys = load_npy_data(batch_num, n_total)
    # train_xs, train_ys, test_xs, test_ys = load_lwy_data(batch_num, n_total)

    print(train_xs.shape)
    print(test_xs.shape)

    input_size = 96

    x = tf.placeholder("float", [None, input_size])
    y_ = tf.placeholder("float", [None, n_class])

    # 输入
    x_image = tf.reshape(x, [-1, 1, input_size, 1])

    # 第一个卷积层
    W_conv1 = weight_variable([1, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_1x2(h_conv1)

    # 第二个卷积层
    W_conv2 = weight_variable([1, 5, 32, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_1x2(h_conv2)

    # 密集链接层
    downsample = int(input_size/4)
    W_fc1 = weight_variable([1 * downsample * 32, 256])
    b_fc1 = bias_variable([256])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * downsample * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # 输出层
    W_fc2 = weight_variable([256, n_class])
    b_fc2 = bias_variable([n_class])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # lstm
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=128, use_peepholes=True,
                                        initializer=initializers.xavier_initializer(),
                                        num_proj=n_class)
    BATCH_SIZE = 128
    init_state = lstm_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=h_fc1, initial_state=init_state, dtype=tf.float32)
    h = outputs[:, -1, :]
    cross_entropy_lstm = -tf.reduce_sum(y_ * tf.log(h))
    lstm_optimizer=tf.train.AdamOptimizer(1e-4).minimize(loss=cross_entropy_lstm)
    correct_prediction_lstm = tf.equal(tf.argmax(h, 1), tf.argmax(y_, 1))
    accuracy_lstm = tf.reduce_mean(tf.cast(correct_prediction_lstm, "float"))

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

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

        sample_num = test_xs.shape[0]

        # correct_cout = 0
        # for j in range(0, sample_num):
        #     txs = test_xs[j]
        #     txs = np.reshape(txs, [1, input_size])
        #     out_y = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
        #     if np.equal(np.argmax(out_y), np.argmax(test_ys[j])):
        #         correct_cout += 1
        #
        # print('test accuracy: ', round(correct_cout / sample_num, 3))

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
        # print('prob matrix:')
        # print(prob_mat)
        # # print(prob_mat.shape)

        batch_index = 0
        test_cout = 0
        correct_cout = 0
        confusion_mat = np.zeros((n_class, n_class))
        while (True):
            if batch_num * (batch_index + 1) > sample_num:
                break

            test_cout += 1
            label = np.zeros(n_class)

            for j in range(batch_num * batch_index, batch_num * (batch_index + 1)):
                txs = test_xs[j]
                txs = np.reshape(txs, [1, input_size])
                out_y = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
                c = np.argmax(out_y, 1)
                label[c] += 1

            sample_index = batch_num * batch_index
            ref_y = test_ys[sample_index]
            ground = np.argmax(ref_y)
            predict = np.argmax(label)
            confusion_mat[ground, predict] += 1
            if np.equal(np.argmax(label), np.argmax(ref_y)):
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

        # train lstm...
        print('train lstm....')
        max_acc = 0
        for i in range(1000):
            current_acc = 0
            step = 0
            for bxs, bys in shufflebatch(train_xs, train_ys, 160):
                m, acc = sess.run((lstm_optimizer, accuracy_lstm), feed_dict={x: bxs, y_: bys, keep_prob: 0.5})
                current_acc += acc
                step += 1
            current_acc = float(current_acc / step)
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
        batch_index = 0
        test_cout = 0
        correct_cout = 0
        confusion_mat = np.zeros((n_class, n_class))
        while (True):
            if batch_num * (batch_index + 1) > sample_num:
                break

            test_cout += 1
            label = np.zeros(n_class)
            for j in range(batch_num * batch_index, batch_num * (batch_index + 1)):
                txs = test_xs[j]
                txs = np.reshape(txs, [1, input_size])
                out_y = sess.run(h, feed_dict={x: txs, keep_prob: 1.0})
                c = np.argmax(out_y, 1)
                label[c] += 1

            sample_index = batch_num * batch_index
            ref_y = test_ys[sample_index]
            ground = np.argmax(ref_y)
            predict = np.argmax(label)
            confusion_mat[ground, predict] += 1
            if np.equal(predict, np.argmax(ref_y)):
                correct_cout += 1

            batch_index += 1


        return np.array(acc_list)


if __name__ == '__main__':
    batch_num = 20
    n_class = 3
    n_round = 50
    acc_arr = np.empty((0, 3))
    # n_total = 2000
    #
    for i in range(n_round):
        print('=================round %d=================' % i)
        acc = train_cnn(n_class=n_class , batch_num=20 , n_total=9000)
        acc_arr = np.vstack((acc_arr, acc))

    acc_mean = np.mean(acc_arr, 0)
    acc_std = np.std(acc_arr, 0)
    # acc_arr /= n_round
    print(acc_mean)
    print(acc_std)
