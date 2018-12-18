import numpy as np
import os
import find_click

if __name__ == '__main__':
    dict = {'0': '', '1': '', '2': ''}
    # dict["0"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Melon"
    # dict["1"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Spinner"
    # dict["2"] = "/home/fish/ROBB/CNN_click/click/TKEO_wk5_complete/Tt"
    dict["0"] = "/home/fish/ROBB/CNN_click/click/CNNDet18/Melon"
    dict["1"] = "/home/fish/ROBB/CNN_click/click/CNNDet18/Spinner"
    dict["2"] = "/home/fish/ROBB/CNN_click/click/CNNDet18/Tt"

    root_save_path = "/home/fish/ROBB/CNN_click/click/CNNDet18_filtered"
    if not os.path.exists(root_save_path):
       os.makedirs(root_save_path)

    for key in dict:
        count = 0
        print(dict[key])
        path = dict[key]
        specie_name = path.split('/')[-1]
        file_list = find_click.list_files(path)
        save_specie_path = os.path.join(root_save_path, specie_name)
        for date_path in file_list:
            date = date_path.split('/')[-1]
            save_path = os.path.join(save_specie_path, date)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            npy_list = find_click.list_npy_files(date_path)
            for npy in npy_list:
                npy_data = np.load(npy)
                num = npy_data.shape[0]
                xs = np.empty((0, 320))
                for index in range(num):
                    temp_x = npy_data[index]
                    beg_idx = np.random.randint(64, (64 + 32))
                    crop_x = temp_x[beg_idx:(beg_idx + 192)]
                    crop_x = np.reshape(crop_x, [1, 192])

                    crop_x = np.fft.fft(crop_x)
                    crop_x = np.sqrt(crop_x.real ** 2 + crop_x.imag ** 2)

                    crop_x = crop_x[0, :96]
                    crop_x = np.reshape(crop_x, [1, 96])
                    # peak值位于20k以下，75k以上的滤去
                    peak_index = np.argmax(crop_x)
                    if peak_index < 20 or peak_index > 75:
                        continue
                    else:
                        xs = np.vstack((xs, temp_x))

                npy_name = npy.split('/')[-1]
                npy_name = npy_name.split('.')[0]
                file_name = "%(path)s/%(pre)s_F%(num)d.npy" % {'path': save_path, 'pre': npy_name, 'num': xs.shape[0]}
                print('saving--->', file_name)
                np.save(file_name, xs)
                count += xs.shape[0]
        print(count)
