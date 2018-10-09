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


def random_crop(xs, batch_num, n_total):
    num = xs.shape[0]
    rc_xs = np.empty((0, 192))

    for i in range(0, n_total):
        for j in range(batch_num * i, batch_num * (i + 1)):
            index = j % num
            temp_x = xs[index]
            # beg_idx = np.random.randint(0, 32)
            beg_idx = np.random.randint(64, (64 + 32))
            crop_x = temp_x[beg_idx:(beg_idx + 192)]
            crop_x = np.reshape(crop_x, [1, 192])
            rc_xs = np.vstack((rc_xs, crop_x))

    return rc_xs


def load_data(data_path, n_class, batch_num=20, n_total=500):
    train_xs = np.empty((0, 192))
    train_ys = np.empty((0, n_class))
    test_xs = np.empty((0, 192))
    test_ys = np.empty((0, n_class))

    for c in range(0, n_class):
        path = "%(path)s/%(class)d" % {'path': data_path, 'class': c}
        wav_files = find_click.list_wav_files(path)

        print("load data : %s, the number of files : %d" % (path, len(wav_files)))

        label = np.zeros(n_class)
        label[c] = 1

        # xs = np.empty((0, 256))
        xs = np.empty((0, 320))
        count = 0
        #
        for pathname in wav_files:
            wave_data, frame_rate = find_click.read_wav_file(pathname)

            energy = np.sqrt(np.sum(wave_data ** 2))
            wave_data /= energy
            wave_data = np.reshape(wave_data, [-1])
            xs = np.vstack((xs, wave_data))
            count += 1
            if count >= batch_num * n_total:
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


def load_lwy_data(batch_num=20, n_total=500):

    dict = {'0': '', '1': '', '2': '', '3':'', '4':'', '5':'', '6':'', '7':''}

    dict["0"] = "/home/fish/ROBB/CNN_click/click/TestData/BBW/Blainvilles_beaked_whale_(Mesoplodon_densirostris)"
    dict["1"] = "/home/fish/ROBB/CNN_click/click/TestData/Gm/Pilot_whale_(Globicephala_macrorhynchus)"
    dict["2"] = "/home/fish/ROBB/CNN_click/click/TestData/Gg/Rissos_(Grampus_grisieus)"

    dict["3"] = "/home/fish/ROBB/CNN_click/click/TestData/Tt/palmyra2006"
    dict["4"] = "/home/fish/ROBB/CNN_click/click/TestData/Dc/Dc"
    dict["5"] = "/home/fish/ROBB/CNN_click/click/TestData/Dd/Dd"
    dict["6"] = "/home/fish/ROBB/CNN_click/click/TestData/Melon/palmyra2006"
    dict["7"] = "/home/fish/ROBB/CNN_click/click/TestData/Spinner/palmyra2006"

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
            wave_data, frame_rate = find_click.read_wav_file7(pathname)

            energy = np.sqrt(np.sum(wave_data ** 2))
            wave_data /= energy
            wave_data = np.reshape(wave_data, [-1])
            xs = np.vstack((xs, wave_data))
            count += 1
            if count >= batch_num * n_total:
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
    # dict = {'0': '', '1': '', '2': '', '3':'', '4':'', '5':'', '6':'', '7':''}
    dict = {'0': '', '1': '', '2': '', '3': '', '4': '', '5': ''}

    # dict["0"] = "/home/fish/ROBB/CNN_click/click/Data/BBW/Blainvilles_beaked_whale_(Mesoplodon_densirostris)"
    # dict["1"] = "/home/fish/ROBB/CNN_click/click/Data/Gm/Pilot_whale_(Globicephala_macrorhynchus)"
    # dict["2"] = "/home/fish/ROBB/CNN_click/click/Data/Gg/Rissos_(Grampus_grisieus)"
    #
    # dict["3"] = "/home/fish/ROBB/CNN_click/click/Data/Tt/palmyra2006"
    # dict["4"] = "/home/fish/ROBB/CNN_click/click/Data/Dc/Dc"
    # dict["5"] = "/home/fish/ROBB/CNN_click/click/Data/Dd/Dd"
    # dict["6"] = "/home/fish/ROBB/CNN_click/click/Data/Melon/palmyra2006"
    # dict["7"] = "/home/fish/ROBB/CNN_click/click/Data/Spinner/palmyra2006"
    dict["0"] = "/home/fish/ROBB/CNN_click/click/CNNDetection/BBW/Blainvilles_beaked_whale_(Mesoplodon_densirostris)"
    dict["1"] = "/home/fish/ROBB/CNN_click/click/CNNDetection/Gm/Pilot_whale_(Globicephala_macrorhynchus)"
    dict["2"] = "/home/fish/ROBB/CNN_click/click/CNNDetection/Gg/Rissos_(Grampus_grisieus)"

    # dict["3"] = "/home/fish/ROBB/CNN_click/click/CNNDetection/Tt/palmyra2006"
    # dict["4"] = "/home/fish/ROBB/CNN_click/click/CNNDetection/Dc/Dc"
    # dict["5"] = "/home/fish/ROBB/CNN_click/click/CNNDetection/Dd/Dd"
    # dict["6"] = "/home/fish/ROBB/CNN_click/click/CNNDetection/Melon/palmyra2006"
    # dict["7"] = "/home/fish/ROBB/CNN_click/click/CNNDetection/Spinner/palmyra2006"

    # dict["3"] = "/home/fish/ROBB/CNN_click/click/CNNDetection/Tt/palmyra2006"
    dict["3"] = "/home/fish/ROBB/CNN_click/click/CNNDetection/Dc/Dc"
    dict["4"] = "/home/fish/ROBB/CNN_click/click/CNNDetection/Dd/Dd"
    dict["5"] = "/home/fish/ROBB/CNN_click/click/CNNDetection/Melon/palmyra2006"
    # dict["6"] = "/home/fish/ROBB/CNN_click/click/CNNDetection/Spinner/palmyra2006"

    n_class = len(dict)
    train_xs = np.empty((0, 192))
    train_ys = np.empty((0, n_class))
    test_xs = np.empty((0, 192))
    test_ys = np.empty((0, n_class))

    for key in dict:
        path = dict[key]
        c = int(key)
        npy_files = find_click.list_npy_files(path)

        random_index = np.random.permutation(len(npy_files))

        label = np.zeros(n_class)
        label[c] = 1

        # xs = np.empty((0, 256))
        xs = np.empty((0, 320))

        count = 0
        #
        for i in range(len(npy_files)):
            npy = npy_files[i]
            print('loading %s' % npy)
            npy_data = np.load(npy)

            # x = np.arange(0, 320)
            # plt.plot(x, npy_data[0])
            # plt.show()

            if npy_data.shape[0] == 0:
                continue
            npy_data = np.divide(npy_data, 2**10)
            energy = np.sqrt(np.sum(npy_data**2, 1))
            energy = np.tile(energy, (npy_data.shape[1], 1))
            energy = energy.transpose()
            npy_data = np.divide(npy_data, energy)

            # plt.plot(x, npy_data[0])
            # plt.show()

            xs = np.vstack((xs, npy_data))
            count += npy_data.shape[0]
            if count >= batch_num * n_total:
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



def train_cnn(data_path, n_class, batch_num=20, n_total=500):

    print("train cnn for one click ... ...")

    # train_xs, train_ys, test_xs, test_ys = load_data(data_path, n_class, batch_num, n_total)
    train_xs, train_ys, test_xs, test_ys = load_npy_data(batch_num, n_total)
    # train_xs, train_ys, test_xs, test_ys = load_lwy_data(batch_num, n_total)


    print(train_xs.shape)
    print(test_xs.shape)

    x = tf.placeholder("float", [None, 192])
    y_ = tf.placeholder("float", [None, n_class])

    # 输入
    x_image = tf.reshape(x, [-1, 1, 192, 1])

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
    W_fc1 = weight_variable([1 * 48 * 32, 256])
    b_fc1 = bias_variable([256])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 48 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # 输出层
    W_fc2 = weight_variable([256, n_class])
    b_fc2 = bias_variable([n_class])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

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

        for i in range(10000):
            mean_acc = 0
            step = 0
            for bxs, bys in shufflebatch(train_xs, train_ys, 160):
                m, acc = sess.run((train_step, accuracy), feed_dict={x: bxs, y_: bys, keep_prob: 0.5})
                mean_acc += acc
                step += 1
            mean_acc = float(mean_acc/step)
            print("epoch : %d, training accuracy : %g" % (i + 1, mean_acc))
            if mean_acc >= 0.87:
                break

        saver.save(sess, "params/cnn_net_lwy.ckpt")


        # print("test accuracy : %g" % (sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys, keep_prob: 1.0})))
        sample_num = test_xs.shape[0]

        correct_cout = 0
        for j in range(0, sample_num):
            txs = test_xs[j]
            txs = np.reshape(txs, [1, 192])
            out_y = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
            if np.equal(np.argmax(out_y), np.argmax(test_ys[j])):
                correct_cout += 1

        print('test accuracy: ', round(correct_cout / sample_num, 3))

        batch_index = 0
        test_cout = 0
        correct_cout = 0

        while (True):
            if batch_num * (batch_index + 1) > sample_num:
                break

            test_cout += 1
            label = np.zeros(n_class)
            for j in range(batch_num * batch_index, batch_num * (batch_index + 1)):
                txs = test_xs[j]
                txs = np.reshape(txs, [1, 192])
                out_y = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
                c = np.argmax(out_y, 1)
                label[c] += 1

            sample_index = batch_num * batch_index
            ref_y = test_ys[sample_index]
            if np.equal(np.argmax(label), np.argmax(ref_y)):
                correct_cout += 1

            batch_index += 1

        print('batch test accuracy: ', round(correct_cout / test_cout, 3))


#
def test_cnn_bottlenose_data(data_path, n_class=8, batch_num=20):
    click_batch = []
    list_files = find_click.list_files(data_path)
    if list_files == []:
        list_files = list_files + [data_path]
    c = 3  # the label of bottlenose is 3
    for path in list_files:
        # if path != './TestData/Dc/Dc':
        #     continue
        wav_files = find_click.list_wav_files(path)
        print("load data : %s, the number of files : %d" % (path, len(wav_files)))

        # 为避免训练网络用的Click用于测试, 类似于训练时区分训练和测试样本
        #  利用全部样本后1/5的Click生成测试样本
        xs = np.empty((0, 320))
        count = 0
        for pathname in wav_files:
            count += 1
            wave_data, frame_rate = find_click.read_wav_file(pathname)
            energy = np.sqrt(np.sum(wave_data ** 2))
            wave_data /= energy
            wave_data = np.reshape(wave_data, [-1])
            xs = np.vstack((xs, wave_data))

        sample_num = xs.shape[0]
        total_batch = int(sample_num / batch_num)
        print('the number of data(%(datasrc)s): %(d)d' % {'datasrc': path, 'd': total_batch})
        for i in range(0, total_batch):
            tmp_xs = np.empty((0, 192))
            for j in range(batch_num * i, batch_num * (i + 1)):
                index = j % sample_num
                temp_x = xs[index]
                beg_idx = np.random.randint(64, (64 + 32))
                crop_x = temp_x[beg_idx:(beg_idx + 192)]
                crop_x = np.reshape(crop_x, [1, 192])
                tmp_xs = np.vstack((tmp_xs, crop_x))

            label = [0] * n_class
            label[c] = 1

            label = np.array([[label]])
            label = list(label)

            tmp_xs = np.expand_dims(np.expand_dims(tmp_xs, axis=0), axis=0)
            tmp_xs = list(tmp_xs)
            sample = tmp_xs + label
            click_batch.append(sample)

    x = tf.placeholder("float", [None, 192])
    # 输入
    x_image = tf.reshape(x, [-1, 1, 192, 1])

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
    W_fc1 = weight_variable([1 * 48 * 32, 256])
    b_fc1 = bias_variable([256])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 48 * 32])
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

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "params/cnn_net_lwy.ckpt")  # 加载训练好的网络参数

        print('the number of batch:', len(click_batch))
        count = 0
        for i in range(len(click_batch)):
            temp_xs = click_batch[i][0]
            label = np.zeros(n_class)
            for j in range(0, temp_xs.shape[1]):
                txs = temp_xs[0, j, :]
                txs = np.reshape(txs, [1, 192])
                out_y = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
                pre_y = np.argmax(out_y, 1)
                label[pre_y] += 1

            ref_y = click_batch[i][1]
            if np.equal(np.argmax(label), np.argmax(ref_y)):
                count += 1

        print('cnn test accuracy (majority voting): ', round(count / len(click_batch), 3))

        count = 0
        weight = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        for i in range(len(click_batch)):
            temp_xs = click_batch[i][0]
            label = np.zeros(n_class)
            for j in range(0, temp_xs.shape[1]):
                txs = temp_xs[0, j, :]
                txs = np.reshape(txs, [1, 192])
                out = sess.run(weight, feed_dict={x: txs, keep_prob: 1.0})
                out = np.reshape(out, label.shape)
                label = label + out

            ref_y = click_batch[i][1]
            if np.equal(np.argmax(label), np.argmax(ref_y)):
                count += 1

        print('cnn test accuracy (weight voting): ', round(count / len(click_batch), 3))

        count = 0
        for i in range(len(click_batch)):
            temp_xs = click_batch[i][0]
            label = np.zeros(n_class)
            for j in range(0, temp_xs.shape[1]):
                txs = temp_xs[0, j, :]
                txs = np.reshape(txs, [1, 192])
                out = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
                out = np.reshape(out, label.shape)
                label = label + out

            ref_y = click_batch[i][1]
            if np.equal(np.argmax(label), np.argmax(ref_y)):
                count += 1

        print('cnn test accuracy (sum of softmax voting): ', round(count / len(click_batch), 3))


def test_cnn_data(data_path, label=3, n_class=8, batch_num=20):
    c = label
    npy_files = find_click.list_npy_files(data_path)
    random_index = np.random.permutation(len(npy_files))
    label = np.zeros(n_class)
    label[c] = 1

    # xs = np.empty((0, 256))
    xs = np.empty((0, 320))

    count = 0
    #
    for i in range(len(npy_files)):
        npy = npy_files[random_index[i]]
        print('loading %s' % npy)
        npy_data = np.load(npy)

        # x = np.arange(0, 320)
        # plt.plot(x, npy_data[0])
        # plt.show()

        if npy_data.shape[0] == 0:
            continue
        npy_data = np.divide(npy_data, 2 ** 10)
        energy = np.sqrt(np.sum(npy_data ** 2, 1))
        energy = np.tile(energy, (npy_data.shape[1], 1))
        energy = energy.transpose()
        npy_data = np.divide(npy_data, energy)

        # plt.plot(x, npy_data[0])
        # plt.show()

        xs = np.vstack((xs, npy_data))
        count += npy_data.shape[0]
        if count >= batch_num * n_total:
            break

    click_batch = []
    sample_num = xs.shape[0]
    total_batch = int(sample_num / batch_num)
    print('the number of data(%(datasrc)s): %(d)d' % {'datasrc': data_path, 'd': total_batch})
    for i in range(0, total_batch):
        tmp_xs = np.empty((0, 192))
        for j in range(batch_num * i, batch_num * (i + 1)):
            index = j % sample_num
            temp_x = xs[index]
            beg_idx = np.random.randint(64, (64 + 32))
            crop_x = temp_x[beg_idx:(beg_idx + 192)]
            crop_x = np.reshape(crop_x, [1, 192])
            tmp_xs = np.vstack((tmp_xs, crop_x))

        label = [0] * n_class
        label[c] = 1

        label = np.array([[label]])
        label = list(label)

        tmp_xs = np.expand_dims(np.expand_dims(tmp_xs, axis=0), axis=0)
        tmp_xs = list(tmp_xs)
        sample = tmp_xs + label
        click_batch.append(sample)

    x = tf.placeholder("float", [None, 192])
    # 输入
    x_image = tf.reshape(x, [-1, 1, 192, 1])

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
    W_fc1 = weight_variable([1 * 48 * 32, 256])
    b_fc1 = bias_variable([256])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 48 * 32])
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

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "params/cnn_net_lwy.ckpt")  # 加载训练好的网络参数

        print('the number of batch:', len(click_batch))
        count = 0
        for i in range(len(click_batch)):
            temp_xs = click_batch[i][0]
            label = np.zeros(n_class)
            for j in range(0, temp_xs.shape[1]):
                txs = temp_xs[0, j, :]
                txs = np.reshape(txs, [1, 192])
                out_y = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
                pre_y = np.argmax(out_y, 1)
                label[pre_y] += 1

            ref_y = click_batch[i][1]
            if np.equal(np.argmax(label), np.argmax(ref_y)):
                count += 1

        print('cnn test accuracy (majority voting): ', round(count / len(click_batch), 3))

        count = 0
        weight = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        for i in range(len(click_batch)):
            temp_xs = click_batch[i][0]
            label = np.zeros(n_class)
            for j in range(0, temp_xs.shape[1]):
                txs = temp_xs[0, j, :]
                txs = np.reshape(txs, [1, 192])
                out = sess.run(weight, feed_dict={x: txs, keep_prob: 1.0})
                out = np.reshape(out, label.shape)
                label = label + out

            ref_y = click_batch[i][1]
            if np.equal(np.argmax(label), np.argmax(ref_y)):
                count += 1

        print('cnn test accuracy (weight voting): ', round(count / len(click_batch), 3))

        count = 0
        for i in range(len(click_batch)):
            temp_xs = click_batch[i][0]
            label = np.zeros(n_class)
            for j in range(0, temp_xs.shape[1]):
                txs = temp_xs[0, j, :]
                txs = np.reshape(txs, [1, 192])
                out = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
                out = np.reshape(out, label.shape)
                label = label + out

            ref_y = click_batch[i][1]
            if np.equal(np.argmax(label), np.argmax(ref_y)):
                count += 1

        print('cnn test accuracy (sum of softmax voting): ', round(count / len(click_batch), 3))


def test_cnn_batch_data(data_path, n_class, batch_num=20, n_total=500):
    click_batch = []
    for c in range(0, n_class):
        path = "%(path)s/%(class)d" % {'path': data_path, 'class': c}
        wav_files = find_click.list_wav_files(path)
        print("load data : %s, the number of files : %d" % (path, len(wav_files)))

        # 为避免训练网络用的Click用于测试, 类似于训练时区分训练和测试样本
        #  利用全部样本后1/5的Click生成测试样本
        xs = np.empty((0, 320))
        count = 0
        split_idx = int(len(wav_files) * 4 / 5)
        for pathname in wav_files:
            count += 1
            if count < split_idx:
                continue
            wave_data, frame_rate = find_click.read_wav_file(pathname)
            energy = np.sqrt(np.sum(wave_data ** 2))
            wave_data /= energy
            wave_data = np.reshape(wave_data, [-1])
            xs = np.vstack((xs, wave_data))
            if count >= batch_num * n_total:
                break

        sample_num = xs.shape[0]
        for i in range(0, int(n_total / 5)):
            tmp_xs = np.empty((0, 192))
            for j in range(batch_num * i, batch_num * (i + 1)):
                index = j % sample_num
                temp_x = xs[index]
                beg_idx = np.random.randint(64, (64 + 32))
                crop_x = temp_x[beg_idx:(beg_idx + 192)]
                crop_x = np.reshape(crop_x, [1, 192])
                tmp_xs = np.vstack((tmp_xs, crop_x))

            label = [0] * n_class
            label[c] = 1

            label = np.array([[label]])
            label = list(label)

            tmp_xs = np.expand_dims(np.expand_dims(tmp_xs, axis=0), axis=0)
            tmp_xs = list(tmp_xs)
            sample = tmp_xs + label
            click_batch.append(sample)

    x = tf.placeholder("float", [None, 192])
    # 输入
    x_image = tf.reshape(x, [-1, 1, 192, 1])

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
    W_fc1 = weight_variable([1 * 48 * 32, 256])
    b_fc1 = bias_variable([256])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 48 * 32])
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

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "params/cnn_net.ckpt")  # 加载训练好的网络参数

        count = 0
        for i in range(len(click_batch)):
            temp_xs = click_batch[i][0]
            label = np.zeros(n_class)
            for j in range(0, temp_xs.shape[1]):
                txs = temp_xs[0, j, :]
                txs = np.reshape(txs, [1, 192])
                out_y = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
                c = np.argmax(out_y, 1)
                label[c] += 1

            ref_y = click_batch[i][1]
            if np.equal(np.argmax(label), np.argmax(ref_y)):
                count += 1

        print('cnn test accuracy (majority voting): ', round(count / len(click_batch), 3))

        count = 0
        weight = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        for i in range(len(click_batch)):
            temp_xs = click_batch[i][0]
            label = np.zeros(n_class)
            for j in range(0, temp_xs.shape[1]):
                txs = temp_xs[0, j, :]
                txs = np.reshape(txs, [1, 192])
                out = sess.run(weight, feed_dict={x: txs, keep_prob: 1.0})
                out = np.reshape(out, label.shape)
                label = label + out

            ref_y = click_batch[i][1]
            if np.equal(np.argmax(label), np.argmax(ref_y)):
                count += 1

        print('cnn test accuracy (weight voting): ', round(count / len(click_batch), 3))

        count = 0
        for i in range(len(click_batch)):
            temp_xs = click_batch[i][0]
            label = np.zeros(n_class)
            for j in range(0, temp_xs.shape[1]):
                txs = temp_xs[0, j, :]
                txs = np.reshape(txs, [1, 192])
                out = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
                out = np.reshape(out, label.shape)
                label = label + out

            ref_y = click_batch[i][1]
            if np.equal(np.argmax(label), np.argmax(ref_y)):
                count += 1

        print('cnn test accuracy (sum of softmax voting): ', round(count / len(click_batch), 3))

batch_num = 10
n_class = 6
n_total = 2000
label = 6

# train_cnn('./Data/Click', 3, 20, 200)
# train_cnn('./Data/ClickC8', n_class, 20, 500)
# exit()

# test_cnn_batch_data('./Data/ClickC8', n_class, batch_num, n_total)

train_cnn('./Data/ClickC8', n_class, 20, 500)
# test_cnn_data('./CNNDetection/Spinner/palmyra2007', label, n_class, batch_num)
# test_cnn_bottlenose_data('./TestData/Tt/cruise', n_class, batch_num)



