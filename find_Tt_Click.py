import os
import wave
import string
import numpy as np
import find_click
from detect_click import *
import math


def calcu_click_energy(x):
    # x = high_pass_filter(x, )
    pow_x = x**2
    # x_norm = np.linalg.norm(x, ord=2)**2
    start_idx = int(x.shape[1]/2)
    # energy_impulse = 0
    half_size = 50
    size = 2 * half_size
    energy_impulse = np.sum(pow_x[0][start_idx-half_size:start_idx+half_size])
    # print('size %g' % size)
    return energy_impulse/size


def calcu_energy(x):
    # x_norm = np.linalg.norm(x, ord=2)
    energy = np.sum(x**2)
    energy = energy / len(x)
    return energy


def shuffle_frames(data):
    ri = np.random.permutation(len(data))
    data = [data[i] for i in ri]
    return data


def detect_save_click(class_path, class_name, snr_threshold_low=5, snr_threshold_high=100):
    tar_fs = 96000
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

        dst_path = "./TKEO_wk3_complete/%(class)s/%(type)s" % {'class': class_name, 'type': path_name}
        if not os.path.exists(dst_path):
            mkdir(dst_path)

        for pathname in wav_files:

            print(pathname)

            wave_data, frameRate = find_click.read_wav_file(pathname)

            # wave_data = resample(wave_data, frameRate, tar_fs)  #

            [path, wavname_ext] = os.path.split(pathname)
            wavname = wavname_ext.split('/')[-1]
            wavname = wavname.split('.')[0]

            fl = 5000
            fwhm = 0.0004
            fdr_threshold = 0.65
            click_index, xn = find_click.find_click_fdr_tkeo(wave_data, frameRate, fl, fwhm, fdr_threshold, signal_len,
                                                             8)

            scale = (2 ** 12 - 1) / max(xn)
            for i in np.arange(xn.size):
                xn[i] = xn[i] * scale

            click_arr = []
            for j in range(click_index.shape[0]):
                index = click_index[j]
                # click_data = wave_data[index[0]:index[1], 0]

                click_data = xn[index[0]:index[1]]

                #  信噪比过滤
                detected_clicks_energy = calcu_click_energy(click_data.reshape(1, -1))
                noise_estimate1 = xn[index[0] - 256:index[0]]
                noise_estimate2 = xn[index[1] + 1:index[1] + 257]
                noise_estimate = np.hstack((noise_estimate1, noise_estimate2))
                noise_energy = calcu_energy(noise_estimate)
                if noise_energy <= 0 or detected_clicks_energy <= 0:
                    continue
                snr = 10 * math.log10(detected_clicks_energy / noise_energy)
                if snr < snr_threshold_low or snr > snr_threshold_high:
                    continue

                click_data = resample(click_data, frameRate, tar_fs)  # 前置TKEO前

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

            dst = "%(path)s/%(pre)s_N%(num)d.npy" \
                      % {'path': dst_path, 'pre': wavname, 'num': len(click_arr)}
            print(dst)
            np.save(dst, np.array(click_arr, dtype=np.short))

            # if count > 20000:
            #     break

        print("count = %(count)d" % {'count': count})


if __name__ == '__main__':
    # detect_save_click(class_path='/ForCNNLSTM/3rdTraining_Data/Blainvilles_beaked_whale_'
    #                              '(Mesoplodon_densirostris)', class_name='BBW')
    #
    # detect_save_click(class_path='/ForCNNLSTM/workshop5_filter/bottlenose',
    #                   class_name='Tt')
    #
    # detect_save_click(class_path='/ForCNNLSTM/3rdTraining_Data/Pilot_whale_(Globicephala_'
    #                              'macrorhynchus)', class_name='Gm')
    #
    # detect_save_click(class_path='/ForCNNLSTM/3rdTraining_Data/Rissos_(Grampus_grisieus)',
    #                   class_name='Gg')
    #
    # detect_save_click(class_path='/ForCNNLSTM/workshop5_filter/Dc',
    #                   class_name='Dc')
    #
    # detect_save_click(class_path='/ForCNNLSTM/workshop5_filter/Dd',
    #                   class_name='Dd')
    #
    # detect_save_click(class_path='/ForCNNLSTM/workshop5_filter/melon',
    #                   class_name='Melon')
    #
    # detect_save_click(class_path='/ForCNNLSTM/workshop5_filter/spinner',
    #                   class_name='Spinner')

    # detect_save_click(class_path='/Northern right whale dolphin, Lissodelphis borealis',
    #                   class_name='RightWhale', snr_threshold_low=5)
    #
    # detect_save_click(class_path='/Pacific white-sided dolphin, Lagenorhynchus obliquidens',
    #                   class_name='PacWhite', snr_threshold_low=5)
    #
    # detect_save_click(class_path='/Pilot whale, Globicephala macrorhynchus',
    #                   class_name='Gm', snr_threshold_low=5)
    #
    # detect_save_click(class_path='/Rissos dolphin, Grampus griseus',
    #                   class_name='Gg', snr_threshold_low=5)
    #
    # detect_save_click(class_path='/Rough-toothed dolphin, Steno bredanensis',
    #                   class_name='RoughToothed', snr_threshold_low=5)
    #
    # detect_save_click(class_path='/Sperm whale, Physeter macrocephalus',
    #                   class_name='Sperm', snr_threshold_low=5)
    #
    # detect_save_click(class_path='/Striped dolphin, Stenella coeruleoalba',
    #                   class_name='Striped', snr_threshold_low=5)
    #
    # detect_save_click(class_path='/Blainville Beaked Whale, Mesoplodon densirostris',
    #                   class_name='Mesoplodon', snr_threshold_low=5)
    #
    # detect_save_click(class_path='/Cuvier Beaked Whale, Ziphius cavirostris',
    #                   class_name='Beaked', snr_threshold_low=5)
    #
    # detect_save_click(class_path='/Melon-headed whale, Pepenocephala electra',
    #                   class_name='Melon', snr_threshold_low=5)
    #
    # detect_save_click(class_path='/ForCNNLSTM/workshop5_filter/bottlenose',
    #                   class_name='Tt', snr_threshold_low=5, snr_threshold_high=20)
    #
    # detect_save_click(class_path='/ForCNNLSTM/workshop5_filter/Dc',
    #                   class_name='Dc', snr_threshold_low=5, snr_threshold_high=20)
    #
    # detect_save_click(class_path='/ForCNNLSTM/workshop5_filter/Dd',
    #                   class_name='Dd', snr_threshold_low=5, snr_threshold_high=20)
    #
    # detect_save_click(class_path='/ForCNNLSTM/workshop5_filter/melon',
    #                   class_name='Melon', snr_threshold_low=5, snr_threshold_high=20)
    #
    # detect_save_click(class_path='/ForCNNLSTM/workshop5_filter/spinner',
    #                   class_name='Spinner', snr_threshold_low=5, snr_threshold_high=20)

    # detect_save_click(class_path='/ForCNNLSTM/5th_DCL_data_bottlenose/palmyra2006',
    #                   class_name='Tt', snr_threshold_low=5)
    #
    # # detect_save_click(class_path='/ForCNNLSTM/workshop5_filter/Dc',
    # #                   class_name='Dc', snr_threshold_low=5)
    # #
    # # detect_save_click(class_path='/ForCNNLSTM/workshop5_filter/Dd',
    # #                   class_name='Dd', snr_threshold_low=5)
    #
    # detect_save_click(class_path='/ForCNNLSTM/5th_DCL_data_melon-headed/palmyra2006',
    #                   class_name='Melon', snr_threshold_low=5)
    #
    # detect_save_click(class_path='/ForCNNLSTM/5th_DCL_data_spinner/palmyra2006',
    #                   class_name='Spinner', snr_threshold_low=5)

    detect_save_click(class_path='/ForCNNLSTM/3rdTraining_Data/Rissos_(Grampus_grisieus)',
                      class_name='rissos', snr_threshold_low=5)

    detect_save_click(
        class_path='/ForCNNLSTM/3rdTraining_Data/Pilot_whale_(Globicephala_macrorhynchus)',
        class_name='pilot', snr_threshold_low=5)

    detect_save_click(
        class_path='/ForCNNLSTM/3rdTraining_Data/Blainvilles_beaked_whale_(Mesoplodon_densirostris)',
        class_name='beakedwhale', snr_threshold_low=5)