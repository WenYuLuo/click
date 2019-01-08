from GMM_CNN_JASA import spectrum_crop
import find_click
import matplotlib.pyplot as plt
import numpy as np


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

    for key in range(n_class):
        xs = data_dict[str(key)]
        if xs is None:
            continue

        temp_xs = spectrum_crop(xs, batch_num=1, n_total=0)

        data_dict[str(key)] = np.array(temp_xs)

    return data_dict


dict_ = {'0': '', '1': '', '2': ''}
dict_["0"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Melon"
dict_["1"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Spinner"
dict_["2"] = "/home/fish/ROBB/CNN_click/click/CNNDet12_filtered/Tt"
sample_rate = 192
x_label = np.arange(0, int(sample_rate/2), sample_rate/192)

data_dict = load_data(dict_)
data_dict = feature_extractor(data_dict)

fig, ax = plt.subplots(1, 3, subplot_kw=dict(projection='3d'))

index = ["0", "1", "2"]

for i in index:
    plot_index = int(i)
    data = data_dict[i]
    y_len = data.shape[0]
    y_label = np.arange(0, y_len)

    surf = ax[plot_index].plot_surface(x_label, y_label, data, rstride=1, cstride=1)

plt.show()
