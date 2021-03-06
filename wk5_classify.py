import tensorflow as tf
import find_click
import numpy as np
import time
import matplotlib.pyplot as plt


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
    # num = xs.shape[0]
    # rc_xs = np.empty((0, 96))


    rc_train_list = []
    # rc_test_list = []

    if n_total == 0:
        n_total = int(xs.shape[0]/batch_num)

    for i in range(0, n_total):
        # for j in range(batch_num * i, batch_num * (i + 1)):
        rc_xs = np.empty((0, 96))
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
            # # peak值位于20k以下，70k以上的滤去
            # if int(key) >= 0:
            #     peak_index = np.argmax(crop_x)
            #     if peak_index < 20 or peak_index > 70:
            #         xs = np.delete(xs, index, 0)
            #         deleted += 1
            #         continue
            crop_x = energy_normalize(crop_x)
            rc_xs = np.vstack((rc_xs, crop_x))
            j += 1
        # if i % 5 == 0:
        #     rc_test_list.append(rc_xs)
        # else:
        #     rc_train_list.append(rc_xs)
        rc_train_list.append(rc_xs)
    print('sampled from %d clicks' % xs.shape[0])
    # np.save('B.npy', rc_xs)
    return rc_train_list


def random_crop_filter_click(xs, batch_num, n_total, key):
    # num = xs.shape[0]
    rc_xs = np.empty((0, 96))

    for i in range(xs.shape[0]):
        temp_x = xs[i]
        # beg_idx = np.random.randint(0, 32)
        beg_idx = np.random.randint(64, (64 + 32))
        crop_x = temp_x[beg_idx:(beg_idx + 192)]
        crop_x = np.reshape(crop_x, [1, 192])

        crop_x = np.fft.fft(crop_x)
        crop_x = np.sqrt(crop_x.real ** 2 + crop_x.imag ** 2)

        crop_x = crop_x[0, :96]
        crop_x = np.reshape(crop_x, [1, 96])
        crop_x = energy_normalize(crop_x)
        non_96k = 0
        # peak值位于20k以下，70k以上的滤去
        if int(key) >= non_96k:
            peak_index = np.argmax(crop_x)
            if peak_index < 20 or peak_index > 70:
                continue
        rc_xs = np.vstack((rc_xs, crop_x))
    print('sampled from %d clicks' % rc_xs.shape[0])

    rc_train_list = []
    # rc_test_list = []
    if n_total == 0:
        n_total = int(rc_xs.shape[0]/batch_num)

    for i in range(0, n_total):
        rc_x = np.empty((0, 96))
        for j in range(batch_num * i, batch_num * (i + 1)):
            index = j % rc_xs.shape[0]
            temp = rc_xs[index]
            rc_x = np.vstack((rc_x, temp))
        # if i % 5 == 0:
        #     rc_test_list.append(rc_x)
        # else:
        #     rc_train_list.append(rc_x)
            rc_train_list.append(rc_x)
    np.save('A.npy', rc_xs)
    return rc_train_list


def energy_normalize(xs):
    energy = np.sqrt(np.sum(xs ** 2))
    xs /= energy
    xs = np.reshape(xs, [-1])
    return xs


def load_lwy_data(batch_num=20, n_total=500):

    dict = {'0': '', '1': '', '2': '', '3':'', '4':'', '5':'', '6':'', '7':''}

    dict["0"] = "/home/fish/ROBB/CNN_click/click/WavData/BBW/Blainvilles_beaked_whale_(Mesoplodon_densirostris)"
    dict["1"] = "/home/fish/ROBB/CNN_click/click/WavData/Gm/Pilot_whale_(Globicephala_macrorhynchus)"
    dict["2"] = "/home/fish/ROBB/CNN_click/click/WavData/Gg/Rissos_(Grampus_grisieus)"

    dict["3"] = "/home/fish/ROBB/CNN_click/click/WavData/Tt/palmyra2006"
    dict["4"] = "/home/fish/ROBB/CNN_click/click/WavData/Dc/Dc"
    dict["5"] = "/home/fish/ROBB/CNN_click/click/WavData/Dd/Dd"
    dict["6"] = "/home/fish/ROBB/CNN_click/click/WavData/Melon/palmyra2006"
    dict["7"] = "/home/fish/ROBB/CNN_click/click/WavData/Spinner/palmyra2006"

    n_class = len(dict)
    train_xs = np.empty((0, 192))
    train_ys = np.empty((0, n_class))
    test_xs = np.empty((0, 192))
    test_ys = np.empty((0, n_class))

    for key in dict:
        # path = "%(path)s/%(class)d" % {'path': data_path, 'class': c}
        path = dict[key]
        c = int(key)
        wav_files = find_click.list_wav_files(path)

        print("load data : %s, the number of files : %d, class: %d" % (path, len(wav_files), c))

        label = np.zeros(n_class)
        label[c] = 1

        # xs = np.empty((0, 256))
        xs = np.empty((0, 320))
        count = 0
        #
        for pathname in wav_files:
            wave_data, frame_rate = find_click.read_wav_file(pathname)

            # energy = np.sqrt(np.sum(wave_data ** 2))
            # wave_data /= energy
            wave_data = np.reshape(wave_data, [-1])
            xs = np.vstack((xs, wave_data))
            count += 1
            if count >= (batch_num + 10) * n_total:
                break

        xs0, xs1 = split_data(xs)

        temp_train_xs = random_crop(xs0, batch_num, int(n_total * 4 / 5))
        temp_test_xs = random_crop(xs1, batch_num, int(n_total / 5))

        temp_train_ys = np.tile(label, (temp_train_xs.shape[0], 1))
        temp_test_ys = np.tile(label, (temp_test_xs.shape[0], 1))

        train_xs = np.vstack((train_xs, temp_train_xs))
        train_ys = np.vstack((train_ys, temp_train_ys))
        test_xs = np.vstack((test_xs, temp_test_xs))
        test_ys = np.vstack((test_ys, temp_test_ys))

    return train_xs, train_ys, test_xs, test_ys


def load_npy_data(batch_num=20, n_total=500):

    dict = {'0': '', '1': '', '2': ''}
    # dict["0"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Melon"
    # dict["1"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Spinner"
    # dict["2"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Tt"
    dict["0"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete_filtered/Melon"
    dict["1"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete_filtered/Spinner"
    dict["2"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete_filtered/Tt"

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

        # xs0, xs1 = split_data(xs)
        # print('crop and split clicks...')
        # temp_train_xs = random_crop_filter_click(xs, batch_num, n_total, key)
        # temp_test_xs  = random_crop_filter_click(txs, batch_num, n_total=0, key=key)
        print('training set crop...')
        temp_train_xs = random_crop(xs, batch_num, n_total, key)
        print('testing set crop...')
        temp_test_xs = random_crop(txs, batch_num, n_total=0, key=key)

        temp_train_ys = np.tile(label, (len(temp_train_xs), 1))
        temp_test_ys = np.tile(label, (len(temp_test_xs), 1))
        train_xs += temp_train_xs
        train_ys = np.vstack((train_ys, temp_train_ys))
        test_xs += temp_test_xs
        test_ys = np.vstack((test_ys, temp_test_ys))
    train_xs = np.array(train_xs)
    test_xs = np.array(test_xs)

        # xs0, xs1 = split_data(xs)
        # print('crop training clicks...')
        # temp_train_xs = random_crop(xs0, batch_num, int(n_total * 4 / 5), key)
        # print('crop testing clicks...')
        # temp_test_xs = random_crop(xs1, batch_num, int(n_total / 5), key)

        # print('crop training clicks...')
        # temp_train_xs = random_crop_average_click(xs0, batch_num, int(n_total * 4 / 5), key)
        # print('crop testing clicks...')
        # temp_test_xs = random_crop_average_click(xs1, batch_num, int(n_total / 5), key)

        # temp_train_ys = np.tile(label, (temp_train_xs.shape[0], 1))
        # temp_test_ys = np.tile(label, (temp_test_xs.shape[0], 1))
        #
        # train_xs = np.vstack((train_xs, temp_train_xs))
        # train_ys = np.vstack((train_ys, temp_train_ys))
        # test_xs = np.vstack((test_xs, temp_test_xs))
        # test_ys = np.vstack((test_ys, temp_test_ys))

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
        batch_xs = np.empty((0, 96)) # batch_xs = np.empty((0, xs.shape[1]))
        batch_ys = np.empty((0, ys.shape[1]))
        for j in range(i*num, (i+1)*num):
            batch_xs = np.vstack((batch_xs, xs[ri[j]]))
            batch_ys = np.vstack((batch_ys, ys[ri[j]]))
        yield batch_xs, batch_ys


def train_cnn(n_class, batch_num=20, n_total=500):

    # print("train cnn for one click ... ...")

    # train_xs, train_ys, test_xs, test_ys = load_data(data_path, n_class, batch_num, n_total)
    train_xs, train_ys, test_xs, test_ys = load_npy_data(batch_num, n_total)
    # train_xs, train_ys, test_xs, test_ys = load_lwy_data(batch_num, n_total)

    print(train_xs.shape)
    print(test_xs.shape)
    # print(len(train_xs))
    # print(len(test_xs))
    tf.reset_default_graph()
    x = tf.placeholder("float", [None, 96])
    y_ = tf.placeholder("float", [None, n_class])

    # 输入
    x_image = tf.reshape(x, [-1, 1, 96, 1])

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
    W_fc1 = weight_variable([1 * 24 * 32, 256])
    b_fc1 = bias_variable([256])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 24 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # 输出层
    n_extract_feature = 16
    W_fc2 = weight_variable([256, n_extract_feature])
    b_fc2 = bias_variable([n_extract_feature])
    y = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    temp_y = tf.reshape(y, [-1, batch_num, n_extract_feature])
    temp_y_t = tf.transpose(temp_y, perm=[0, 2, 1])
    multi_y_t = tf.reshape(temp_y_t, [-1, batch_num])
    multi_y = tf.transpose(multi_y_t, perm=[1, 0])

    # 融合层
    # # 单模式融合
    # W_fuse = weight_variable([1, batch_num])
    # b_fuse = bias_variable([n_class])
    # fuse_out = tf.nn.softmax(tf.reshape(tf.matmul(W_fuse, multi_y), [-1, n_class]) + b_fuse)

    # 多模式融合
    fuse_mode = 16
    W_fuse = weight_variable([fuse_mode, batch_num])
    # b_fuse = bias_variable([fuse_mode, n_class])
    multi_fuse_out = tf.nn.relu(tf.matmul(W_fuse, multi_y))# + b_fuse)

    mode_weight = weight_variable([1, fuse_mode])
    mode_b = bias_variable([n_extract_feature])
    fuse_out = tf.nn.relu(tf.reshape(tf.matmul(mode_weight, multi_fuse_out), [-1, n_extract_feature]) + mode_b)

    W_fc3 = weight_variable([n_extract_feature, n_class])
    b_fc3 = bias_variable([n_class])
    out = tf.nn.softmax(tf.matmul(fuse_out, W_fc3) + b_fc3)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(out))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    # train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    # # train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        # for i in range(50000):
        #     bxs, bys = shufflelists(train_xs, train_ys, 160)
        #     if (i + 1) % 1000 == 0:
        #         step_acc = sess.run(accuracy, feed_dict={x: bxs, y_: bys, keep_prob: 1.0})
        #         print("step : %d, training accuracy : %g" % (i + 1, step_acc))
        #         if step_acc >= 0.80:
        #             break
        #
        #     sess.run(train_step, feed_dict={x: bxs, y_: bys, keep_prob: 0.5})
        #
        # saver.save(sess, "params/cnn_net_lwy.ckpt")
        print('training........')
        max_acc = 0
        for i in range(1000):
            current_acc = 0
            step = 0
            for bxs, bys in shufflebatch(train_xs, train_ys, 100):
                m, acc = sess.run((train_step, accuracy), feed_dict={x: bxs, y_: bys, keep_prob: 0.5})
                # print(f_out[0:10, :, :])
                current_acc += acc
                step += 1
            current_acc = float(current_acc/step)
            if current_acc > max_acc:
                max_acc = current_acc
                saver.save(sess, "params/cnn_net_lwy.ckpt")
                if max_acc > 0.97:
                    break
            if max_acc - current_acc > 0.05:
                break
            print("epoch : %d, training accuracy : %g" % (i + 1, current_acc))
        print("training accuracy converged to : %g" % max_acc)
            # if current_acc - pre_acc < 0.0005:
            #     break
            # else:
            #     pre_acc = current_acc
            #     continue

        saver.restore(sess, "params/cnn_net_lwy.ckpt")
        prob_mat = np.zeros((n_class, n_class))
        train_num = train_xs.shape[0]
        correct_cout = 0
        for j in range(0, train_num):
            txs = train_xs[j]
            # txs = np.reshape(txs, [1, 96])
            out_y = sess.run(out, feed_dict={x: txs, keep_prob: 1.0})
            ground = np.argmax(train_xs[j])
            predict = np.argmax(out_y)
            prob_mat[ground, predict] += 1
            if np.equal(predict, ground):
                correct_cout += 1

        # print("test accuracy : %g" % (sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys, keep_prob: 1.0})))
        sample_num = test_xs.shape[0]
        confusion_mat = np.zeros((n_class, n_class))
        correct_cout = 0
        for j in range(0, sample_num):
            txs = test_xs[j]
            # txs = np.reshape(txs, [1, 96])
            out_y = sess.run(out, feed_dict={x: txs, keep_prob: 1.0})
            ground = np.argmax(test_ys[j])
            predict = np.argmax(out_y)
            confusion_mat[ground, predict] += 1
            if np.equal(predict, ground):
                correct_cout += 1

        print('test accuracy: ', round(correct_cout / sample_num, 3))
        print(confusion_mat)
        total_sample = np.sum(confusion_mat, 1)
        acc_list = []
        for i in range(0, n_class):
            acc = confusion_mat[i, i] / total_sample[i]
            acc_list.append(acc)
            print('label ', i, 'acc = ', acc)
        return np.array(acc_list)


def test_cnn_data(data_path, label=3, n_class=8, batch_num=20):
    c = label
    npy_files = find_click.list_npy_files(data_path)
    random_index = np.random.permutation(len(npy_files))
    label = np.zeros(n_class)
    label[c] = 1

    # xs = np.empty((0, 256))


    count = 0
    #

    tf.reset_default_graph()
    x = tf.placeholder("float", [None, 96])
    # 输入
    x_image = tf.reshape(x, [-1, 1, 96, 1])

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
    W_fc1 = weight_variable([1 * 24 * 32, 256])
    b_fc1 = bias_variable([256])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 24 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # 输出层
    W_fc2 = weight_variable([256, n_class])
    b_fc2 = bias_variable([n_class])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    print('==============================================')
    total_correct = 0
    total = 0
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "params/cnn_net_lwy.ckpt")  # 加载训练好的网络参数

        for i in range(len(npy_files)):
            npy = npy_files[random_index[i]]
            print('loading %s' % npy)
            npy_data = np.load(npy)

            # x = np.arange(0, 320)
            # plt.plot(x, npy_data[0])
            # plt.show()

            if npy_data.shape[0] == 0:
                continue

            # npy_data = np.divide(npy_data, 2 ** 10)
            # energy = np.sqrt(np.sum(npy_data ** 2, 1))
            # energy = np.tile(energy, (npy_data.shape[1], 1))
            # energy = energy.transpose()
            # npy_data = np.divide(npy_data, energy)

            # plt.plot(x, npy_data[0])
            # plt.show()
            xs = np.empty((0, 320))
            xs = np.vstack((xs, npy_data))
            # xs = npy_data
            count = npy_data.shape[0]
            print('loaded clicks:', count)
            # if count >= batch_num * n_total:
            #     break

            click_batch = []
            sample_num = xs.shape[0]
            total_batch = int(sample_num / batch_num)
            # print('the number of data(%(datasrc)s): %(d)d' % {'datasrc': data_path, 'd': total_batch})
            for i in range(0, total_batch):
                tmp_xs = np.empty((0, 96))
                # for j in range(batch_num * i, batch_num * (i + 1)):
                j = batch_num * i
                if j > xs.shape[0]:
                    break
                while j >= (batch_num * i) and j < (batch_num * (i + 1)):
                    if xs.shape[0] == 0:
                        break
                    index = j % xs.shape[0]
                    temp_x = xs[index]
                    beg_idx = np.random.randint(64, (64 + 32))
                    crop_x = temp_x[beg_idx:(beg_idx + 192)]
                    crop_x = np.reshape(crop_x, [1, 192])

                    crop_x = np.fft.fft(crop_x)
                    crop_x = np.sqrt(crop_x.real ** 2 + crop_x.imag ** 2)

                    crop_x = crop_x[0, :96]
                    crop_x = np.reshape(crop_x, [1, 96])

                    if c >= 0:
                        # peak值位于20k以下，70k以上的滤去
                        peak_index = np.argmax(crop_x)
                        if peak_index < 20 or peak_index > 70:
                            xs = np.delete(xs, index, 0)
                            continue

                    crop_x = energy_normalize(crop_x)
                    tmp_xs = np.vstack((tmp_xs, crop_x))
                    j += 1

                label = [0] * n_class
                label[c] = 1

                label = np.array([[label]])
                label = list(label)

                tmp_xs = np.expand_dims(np.expand_dims(tmp_xs, axis=0), axis=0)
                tmp_xs = list(tmp_xs)
                sample = tmp_xs + label
                click_batch.append(sample)

            print('the number of batch:', len(click_batch))
            if len(click_batch) == 0:
                continue
            total += len(click_batch)
            count = 0
            majority_mat = [0] * n_class
            for i in range(len(click_batch)):
                temp_xs = click_batch[i][0]
                label = np.zeros(n_class)
                for j in range(0, temp_xs.shape[1]):
                    txs = temp_xs[0, j, :]
                    txs = np.reshape(txs, [1, 96])
                    out_y = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
                    pre_y = np.argmax(out_y, 1)
                    label[pre_y] += 1

                ref_y = click_batch[i][1]
                predict = np.argmax(label)
                majority_mat[int(predict)] += 1
                if np.equal(np.argmax(label), np.argmax(ref_y)):
                    count += 1
            total_correct += count
            print('correct:', count, 'total:', len(click_batch))
            print('cnn test accuracy (majority voting): ', round(count / len(click_batch), 3))
            print('result:', majority_mat)


            # count = 0
            # weight_vote_mat = [0] * n_class
            # weight = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            # for i in range(len(click_batch)):
            #     temp_xs = click_batch[i][0]
            #     label = np.zeros(n_class)
            #     for j in range(0, temp_xs.shape[1]):
            #         txs = temp_xs[0, j, :]
            #         txs = np.reshape(txs, [1, 192])
            #         out = sess.run(weight, feed_dict={x: txs, keep_prob: 1.0})
            #         out = np.reshape(out, label.shape)
            #         label = label + out
            #
            #     ref_y = click_batch[i][1]
            #     predict = np.argmax(label)
            #     weight_vote_mat[int(predict)] += 1
            #     if np.equal(np.argmax(label), np.argmax(ref_y)):
            #         count += 1
            #
            # print('cnn test accuracy (weight voting): ', round(count / len(click_batch), 3))
            # print('result:', weight_vote_mat)
            #
            # count = 0
            # softmax_mat = [0] * n_class
            # for i in range(len(click_batch)):
            #     temp_xs = click_batch[i][0]
            #     label = np.zeros(n_class)
            #     for j in range(0, temp_xs.shape[1]):
            #         txs = temp_xs[0, j, :]
            #         txs = np.reshape(txs, [1, 192])
            #         out = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
            #         out = np.reshape(out, label.shape)
            #         label = label + out
            #
            #     ref_y = click_batch[i][1]
            #     predict = np.argmax(label)
            #     softmax_mat[int(predict)] += 1
            #     if np.equal(np.argmax(label), np.argmax(ref_y)):
            #         count += 1
            #
            # print('cnn test accuracy (sum of softmax voting): ', round(count / len(click_batch), 3))
            # print('result:', softmax_mat)
    print('total correct:', total_correct, 'total batch:', total)
    print('%s mean acc: %f'%(data_path, total_correct/total))


def test_cnn_batch_learn(data_path, label=3, n_class=8, batch_num=20):
    c = label
    npy_files = find_click.list_npy_files(data_path)
    random_index = np.random.permutation(len(npy_files))
    label = np.zeros(n_class)
    label[c] = 1

    # xs = np.empty((0, 256))


    count = 0
    #

    tf.reset_default_graph()
    x = tf.placeholder("float", [None, 96])

    # 输入
    x_image = tf.reshape(x, [-1, 1, 96, 1])

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
    W_fc1 = weight_variable([1 * 24 * 32, 256])
    b_fc1 = bias_variable([256])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 24 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # 输出层
    W_fc2 = weight_variable([256, n_class])
    b_fc2 = bias_variable([n_class])
    y = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    temp_y = tf.reshape(y, [-1, batch_num, n_class])
    temp_y_t = tf.transpose(temp_y, perm=[0, 2, 1])
    multi_y_t = tf.reshape(temp_y_t, [-1, batch_num])
    multi_y = tf.transpose(multi_y_t, perm=[1, 0])

    # 融合层
    # # 单模式融合
    # W_fuse = weight_variable([1, batch_num])
    # b_fuse = bias_variable([n_class])
    # fuse_out = tf.nn.softmax(tf.reshape(tf.matmul(W_fuse, multi_y), [-1, n_class]) + b_fuse)

    # 多模式融合
    fuse_mode = 9
    W_fuse = weight_variable([fuse_mode, batch_num])
    # b_fuse = bias_variable([fuse_mode, n_class])
    multi_fuse_out = tf.nn.relu(tf.matmul(W_fuse, multi_y))# + b_fuse)

    mode_weight = weight_variable([1, fuse_mode])
    mode_b = bias_variable([n_class])
    fuse_out = tf.nn.softmax(tf.reshape(tf.matmul(mode_weight, multi_fuse_out), [-1, n_class]) + mode_b)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    print('==============================================')
    total_correct = 0
    total = 0
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "params/cnn_net_lwy.ckpt")  # 加载训练好的网络参数

        for i in range(len(npy_files)):
            npy = npy_files[random_index[i]]
            print('loading %s' % npy)
            npy_data = np.load(npy)

            if npy_data.shape[0] == 0:
                continue
            xs = np.empty((0, 320))
            xs = np.vstack((xs, npy_data))
            # xs = npy_data
            count = npy_data.shape[0]
            print('loaded clicks:', count)
            # if count >= batch_num * n_total:
            #     break

            click_batch = []
            sample_num = xs.shape[0]
            total_batch = int(sample_num / batch_num)
            # print('the number of data(%(datasrc)s): %(d)d' % {'datasrc': data_path, 'd': total_batch})
            for i in range(0, total_batch):
                tmp_xs = np.empty((0, 96))
                # for j in range(batch_num * i, batch_num * (i + 1)):
                j = batch_num * i
                if j > xs.shape[0]:
                    break
                while j >= (batch_num * i) and j < (batch_num * (i + 1)):
                    if xs.shape[0] == 0:
                        break
                    index = j % xs.shape[0]
                    temp_x = xs[index]
                    beg_idx = np.random.randint(64, (64 + 32))
                    crop_x = temp_x[beg_idx:(beg_idx + 192)]
                    crop_x = np.reshape(crop_x, [1, 192])

                    crop_x = np.fft.fft(crop_x)
                    crop_x = np.sqrt(crop_x.real ** 2 + crop_x.imag ** 2)

                    crop_x = crop_x[0, :96]
                    crop_x = np.reshape(crop_x, [1, 96])

                    if c >= 0:
                        # peak值位于20k以下，70k以上的滤去
                        peak_index = np.argmax(crop_x)
                        if peak_index < 20 or peak_index > 70:
                            xs = np.delete(xs, index, 0)
                            continue

                    crop_x = energy_normalize(crop_x)
                    tmp_xs = np.vstack((tmp_xs, crop_x))
                    j += 1
                # click_batch.append(tmp_xs)

                label = [0] * n_class
                label[c] = 1

                label = np.array([[label]])
                label = list(label)

                tmp_xs = np.expand_dims(np.expand_dims(tmp_xs, axis=0), axis=0)
                tmp_xs = list(tmp_xs)
                sample = tmp_xs + label
                click_batch.append(sample)

            print('the number of batch:', len(click_batch))
            if len(click_batch) == 0:
                continue
            total += len(click_batch)
            count = 0
            majority_mat = [0] * n_class
            for i in range(len(click_batch)):
                temp_xs = click_batch[i][0][0]
                label = np.zeros(n_class)
                out_y = sess.run(fuse_out, feed_dict={x: temp_xs, keep_prob: 1.0})
                pre_y = np.argmax(out_y, 1)
                ref_y = click_batch[i][1]
                if np.equal(pre_y, np.argmax(ref_y)):
                    count += 1
            total_correct += count
            print('correct:', count, 'total:', len(click_batch))
            print('cnn test accuracy (batch learn): ', round(count / len(click_batch), 3))
            # print('result:', majority_mat)
        print('total correct: %d, total: %d, batch learn acc: %f' % (total_correct, total, total_correct/total))


batch_num = 10
n_class = 3
n_round = 5
acc_arr = np.empty((0, 3))
# n_total = 2000
#
for i in range(n_round):
    print('=================round %d=================' % i)
    acc = train_cnn(n_class=n_class , batch_num=10 , n_total=6000)
    acc_arr = np.vstack((acc_arr, acc))

acc_mean = np.mean(acc_arr, 0)
acc_std = np.std(acc_arr, 0)
# acc_arr /= n_round
print(acc_mean)
print(acc_std)

# dict = {'0': '', '1': '', '2': ''}
#
# dict["0"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Melon/test2006"
# dict["1"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Spinner/test2006"
# dict["2"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Tt/test2006"
# # dict["0"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Melon/palmyra2006"
# # dict["1"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Spinner/palmyra2006"
# # dict["2"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Tt/palmyra2006"
#
# for key in dict:
#     path = dict[key]
#     print(path)
#     label = int(key)

    # test_cnn_data(path, label, n_class, batch_num)

    # test_cnn_batch_learn(path, label, n_class, batch_num)
    # test_cnn_data_average_click(path, label, n_class, batch_num)