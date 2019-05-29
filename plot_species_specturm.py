#coding:utf-8

import find_click
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def energy_normalize(xs):
    energy = np.sqrt(np.sum(xs ** 2))
    xs /= energy
    xs = np.reshape(xs, [-1])
    return xs


def spectrum_crop(xs, batch_num, n_total):
    num = xs.shape[0]
    rc_train_list = []

    if n_total == 0:
        n_total = int(num / batch_num)

    for i in range(0, n_total):
        txs = np.zeros((1, 96))
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
            txs += crop_x
            # txs = np.vstack((txs, crop_x))
            # rc_train_list.append(crop_x)
            j += 1
        txs /= batch_num
        rc_train_list.append(txs[0])
    return rc_train_list


def load_data(dict):

    data_dict = {}

    for key in dict:
        data_dict[key] = None

    for key in dict:
        path = dict[key]
        print(path)

        npy_files = find_click.list_npy_files(path)

        xs = np.empty((0, 320))
        count = 0

        for npy in npy_files:
            npy_data = np.load(npy)
            if npy_data.shape[0] == 0:
                continue
            xs = np.vstack((xs, npy_data))
            count += npy_data.shape[0]
        print('loaded clicks:', count)

        data_dict[key] = xs

    return data_dict


def feature_extractor(data_dict):
    # keys = ['0', '1', '2']
    n_class = len(data_dict)
    batch_num_list = []

    for key in range(n_class):
        xs = data_dict[str(key)]
        if xs is None:
            continue

        click_num = xs.shape[0]
        total_batch = 1000
        batch_num = int(click_num/total_batch)
        batch_num_list.append(batch_num)
        # batch_num = 1
        temp_xs = spectrum_crop(xs, batch_num=batch_num, n_total=0)

        data_dict[str(key)] = np.array(temp_xs)

    return data_dict, batch_num_list

# zhfont1 = matplotlib.font_manager.FontProperties(fname=r"'/usr/share/fonts/simsun.ttf'")
plt.rcParams['font.sans-serif'] = ['simsun']
# matplotlib.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus'] = False

dict_ = {'0': '', '1': '', '2': ''}
dict_["0"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Melon"
dict_["1"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Spinner"
dict_["2"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Tt"

# dict_["0"] = "/home/fish/ROBB/CNN_click/click/CNN_Det12_WK3/beakedwhale"
# dict_["1"] = "/home/fish/ROBB/CNN_click/click/CNN_Det12_WK3/pilot"
# dict_["2"] = "/home/fish/ROBB/CNN_click/click/CNN_Det12_WK3/rissos"

# dict_ = {'0': '', '1': '', '2': ''}
# dict_["0"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk3_complete/beakedwhale"
# dict_["1"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk3_complete/pilot"
# dict_["2"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk3_complete/rissos"
# title_dict = {'0': 'Blainville\'s beaked whales', '1': 'Short-finned pilot whales', '2': 'Risso\'s dolphins'}

# dict_["0"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete_filtered/Melon"
# dict_["1"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete_filtered/Spinner"
# dict_["2"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete_filtered/Tt"


# dict_ = {'0': ''}
# dict_["0"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Spinner"

sample_rate = 192
y_label = np.arange(0, int(sample_rate/2), sample_rate/192)

data_dict = load_data(dict_)
data_dict, batch_num_list = feature_extractor(data_dict)

# fig = plt.figure()
index = ["0", "1", "2"]
style = ['--', ':', '-.']


fig, ax = plt.subplots(1, 1)
ax.set_ylabel('平均归一化幅度', fontsize=10)#, fontproperties=zhfont1)
ax.set_xlabel('频率/kHz', fontsize=10)#, fontproperties=zhfont1)

y_list = []

for i in range(len(index)):
    plot_index = int(index[i])
    data = data_dict[index[i]]

    # mean spectrum
    np.random.shuffle(data)
    data = data[:1000]
    print(data.shape)
    mean_data = np.mean(data, 0)
    std_data = np.std(data, 0)

    y = ax.plot(y_label, mean_data, linestyle=style[i], linewidth=1.5)

    y_list.append(y)

# fig = plt.figure()
# plot_split = '31'
# title_dict = {'0': '爪头鲸', '1': '长吻原海豚', '2': '瓶鼻海豚'}
# for i in index:
#     plot_index = int(i)
#     data = data_dict[i]
#     # click barch spectrum
#     data = np.transpose(data)
#     x_len = data.shape[1]
#     xticks = np.arange(0, x_len)
#
#     ax = fig.add_subplot(int(plot_split+str(plot_index+1)))
#
#     X, Y = np.meshgrid(xticks, y_label)
#
#     X *= batch_num_list[plot_index]
#     ax.pcolormesh(X, Y, data)
#     ax.set_title(title_dict[i], fontsize=14)
#     ax.set_xlabel('数量', fontsize=12)
#     ax.set_ylabel('频率/kHz', fontsize=12)

fig.legend((y_list[0][0], y_list[1][0], y_list[2][0]), (
    u'爪头鲸', u'长吻原海豚', u'瓶鼻海豚'),
         loc='upper right',
         bbox_to_anchor=(0.95, 0.5))#,
         # prop=zhfont1)

w5# fig.legend((y_list[0][0], y_list[1][0], y_list[2][0]), (
#     u'柏氏中喙鲸', u'泛热带引航鲸', u'灰海豚'),
#          loc='upper right',
#          bbox_to_anchor=(0.95, 0.95))#,
#          # prop=zhfont1)

fig.set_size_inches(5, 4.5)
fig.tight_layout()
# fig.subplots_adjust(top=0.975, bottom=0.017, left=0.017, right=0.969, hspace=0.3, wspace=0.2)
fig.savefig('wk5_mean_spectrum_CNN_2.png', dpi=300)
plt.show()

