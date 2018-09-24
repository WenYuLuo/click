import os
import wave
import string
import numpy as np
import find_click
from detect_click import *


def shuffle_frames(data):
    ri = np.random.permutation(len(data))
    data = [data[i] for i in ri]
    return data


def detect_save_click(class_path, class_name):
    tar_fs = 192000
    signal_len = 320
    folder_list = find_click.list_files(class_path)
    if folder_list == []:
        folder_list = folder_list + [class_path]
    for folder in folder_list:
        print(folder)
        count = 0
        wav_files = find_click.list_wav_files(folder)

        # wav_files = shuffle_frames(wav_files)

        path_name = folder.split('/')[-1]

        dst_path = "./Data/%(class)s/%(type)s" % {'class': class_name, 'type': path_name}
        if not os.path.exists(dst_path):
            mkdir(dst_path)

        for pathname in wav_files:

            print(pathname)
            wave_data, frameRate = find_click.read_wav_file(pathname)
            [path, wavname_ext] = os.path.split(pathname)
            wavname = wavname_ext.split('/')[-1]

            fl = 5000
            fwhm = 0.0008
            fdr_threshold = 0.62
            click_index, xn = find_click.find_click_fdr_tkeo(wave_data, frameRate, fl, fwhm, fdr_threshold, signal_len,
                                                             8)

            scale = (2 ** 15 - 1) / max(xn)
            for i in np.arange(xn.size):
                xn[i] = xn[i] * scale

            click_arr = []
            for j in range(click_index.shape[0]):
                index = click_index[j]
                # click_data = wave_data[index[0]:index[1], 0]

                click_data = xn[index[0]:index[1]]

                click_data = resample(click_data, frameRate, tar_fs)  #

                click_data = cut_data(click_data, signal_len)

                click_data = click_data.astype(np.short)

                click_arr.append(click_data)
                # filename = "%(path)s/%(pre)s_click_%(n)06d.wav" % {'path': dst_path, 'pre': wavname, 'n': count}
                # f = wave.open(filename, "wb")
                # # set wav params
                # f.setnchannels(1)
                # f.setsampwidth(2)
                # f.setframerate(tar_fs)
                # # turn the data to string
                # f.writeframes(click_data.tostring())
                # f.close()
                count = count + 1
            wavname = "%(path)s/%(wavname)s_N%(num)d.npy" \
                      % {'path': dst_path, 'wavname': wavname, 'num': len(click_arr)}
            np.save(os.path.join(path, wavname), np.array(click_arr, dtype=np.short))

            # if count > 20000:
            #     break

        print("count = %(count)d" % {'count': count})


if __name__ == '__main__':
    detect_save_click(class_path='/media/fish/Elements/clickdata/ForCNNLSTM/3rdTraining_Data/Blainvilles_beaked_whale_'
                                 '(Mesoplodon_densirostris)', class_name='BBW')

    detect_save_click(class_path='/media/fish/Elements/clickdata/ForCNNLSTM/5th_DCL_data_bottlenose',
                      class_name='Tt')

    detect_save_click(class_path='/media/fish/Elements/clickdata/ForCNNLSTM/3rdTraining_Data/Pilot_whale_(Globicephala_'
                                 'macrorhynchus)', class_name='Gm')

    detect_save_click(class_path='/media/fish/Elements/clickdata/ForCNNLSTM/3rdTraining_Data/Rissos_(Grampus_grisieus)',
                      class_name='Gg')

    detect_save_click(class_path='/media/fish/Elements/clickdata/ForCNNLSTM/5th_DCL_data_common/Dc',
                      class_name='Dc')

    detect_save_click(class_path='/media/fish/Elements/clickdata/ForCNNLSTM/5th_DCL_data_common/Dd',
                      class_name='Dd')

    detect_save_click(class_path='/media/fish/Elements/clickdata/ForCNNLSTM/5th_DCL_data_melon-headed',
                      class_name='Melon')

    detect_save_click(class_path='/media/fish/Elements/clickdata/ForCNNLSTM/5th_DCL_data_spinner',
                      class_name='Spinner')
