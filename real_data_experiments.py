from scipy import signal
# import wave
import numpy as np
import matplotlib.pyplot as plt
# import math
import os
from cnn_detect_click import *
from find_Tt_Click import *
import detection_rate
import scipy.io as sio

def list_ext_files(root_path, ext='.wav'):
    list = []
    for filename in os.listdir(root_path):
        pathname = os.path.join(root_path, filename)
        if os.path.isfile(pathname):
            (shotname, extension) = os.path.splitext(filename)
            if extension == ext:
                list = list + [pathname]

    return list


def list_files(root_path):
    list_folder = []
    for filename in os.listdir(root_path):
        pathname = os.path.join(root_path, filename)
        if os.path.isfile(pathname):
            continue
        list_folder = list_folder + [pathname]
    return list_folder


def read_annot(path):
    annot = np.loadtxt(path, skiprows=1)
    if len(annot.shape) == 1:
        if annot.size == 0:
            return annot
        annot = np.reshape(annot[:2], (1, 2))
    else:
        annot = annot[:, :2]
    return annot


def FDR_CNN_Compare():
    # audio_path = '/media/fish/Elements/clickdata/ForCNNLSTM/5th_DCL_data_melon-headed/palmyra2006/palmyra102006-061023-202000_4.wav'
    audio_path = '/media/fish/Elements/clickdata/ForCNNLSTM/5th_DCL_data_spinner/palmyra2006/palmyra102006-061102-222909_4.wav'
    # cnn param
    snr_threshold_low = 12
    cnn_predict_list, fs, audio, audio_filted, detected_visual = run_cnn_detection(audio_path, snr_threshold_low,
                                                                                   snr_threshold_high=200)
    # fdr param
    fl = 5000
    fwhm = 0.0004
    fdr_threshold = 0.65
    signal_len = 320
    click_index, xn = find_click.find_click_fdr_tkeo(audio, fs, fl, fwhm, fdr_threshold, signal_len,
                                                     8)
    time = np.arange(0, audio_filted.shape[0])# / fs
    pl.plot(time, audio_filted)

    cnn_visual = np.zeros_like(audio_filted)
    for i in cnn_predict_list:
        # if int(i[0]) > 1000000:
        #     continue
        cnn_visual[int(i[0]):int(i[1])] = 1
    cnn_visual = cnn_visual * max(audio_filted) / 6
    pl.plot(time, cnn_visual, color='r')

    fdr_visual = np.zeros_like(audio_filted)
    for i in click_index.tolist():
        # if int(i[0]) > 1000000:
        #     continue
        fdr_visual[int(i[0]):int(i[1])] = 1
    fdr_visual = fdr_visual * max(audio_filted) / 5
    pl.plot(time, fdr_visual, color='g')
    pl.show()


# def is_conclude(annot, pos):
#     annot = annot.tolist()
#     for annot_pos in annot:
#
#
# def predict_annot_compare(cnn_predict_list, annot):
#     correct_detect = 0
#     annot_num = len(annot)
#
#     for pos in cnn_predict_list:

if __name__ == '__main__':
    FDR_CNN_Compare()
    # snr_list = [0,3,5,7,9,11,13]
    # for snr_threshold_low in snr_list:
    #     print('\n\n=========================== snr threshold = %ddB ===============================' % snr_threshold_low)
    #     path = '/media/fish/Elements/clickdata/CNN_jasa/cnn_test'
    #     list_folder = list_files(path)
    #     # list_folder = ['/media/fish/Elements/clickdata/CNN_jasa/cnn_test/BBW1']
    #     # list_folder = ['/media/fish/Elements/clickdata/ForCNNLSTM/5th_DCL_data_bottlenose/palmyra2006/test']
    #     for folder in list_folder:
    #         print(' ')
    #         print('==============================processing====================================')
    #         # print(folder)
    #         annot_file_list = list_ext_files(folder, ext='.txt')
    #         wave_list = list_ext_files(folder, ext='.wav')
    #         if len(wave_list) != 1:
    #             print('当前文件夹音频数大于1')
    #             continue
    #         correct_total = 0
    #         annot_total = 0
    #         print(wave_list[0])
    #         # snr_threshold_low = 0
    #         cnn_predict_list, fs, audio, audio_filted, detected_visual = run_cnn_detection(wave_list[0], snr_threshold_low, snr_threshold_high=200)
    #         # if len(cnn_predict_list) == 0:
    #         #     print('no click was detected!')
    #         #     continue
    #         # audio, fs = find_click.read_wav_file(wave_list[0])
    #         # if audio.shape[1] > 1:
    #         #     audio = audio[:, 1]
    #         # else:
    #         #     audio = audio[:, 0]
    #         # len_audio = len(audio)
    #         # fl = fs / 40
    #         # # fs = 192000
    #         # wn = 2 * fl / fs
    #         # b, a = signal.butter(8, wn, 'high')
    #         # audio_filted = signal.filtfilt(b, a, audio)
    #
    #         # time = np.arange(0, audio_filted.shape[0])# / fs
    #         # pl.subplot(211)
    #         # spec = audio_filted.tolist()
    #         # pl.specgram(spec, Fs=fs, scale_by_freq=True, sides='default')
    #         # pl.subplot(212)
    #
    #         # audio_show = audio_filted#[3500:6500]
    #         # audio_show /= max(audio_show)
    #         # pl.plot(time, audio_show*20)
    #
    #         # pl.plot(time, audio_filted)
    #
    #         # pl.show()
    #         # pl.plot(time, audio_filted)
    #         # pl.title('high pass filter')
    #         # pl.xlabel('time')
    #         # pl.show()
    #         # detected_visual = np.zeros_like(audio_filted)
    #         #
    #         # cnn_visual = np.zeros_like(audio_filted)
    #         # for i in cnn_predict_list:
    #         #     # if int(i[0]) > 1000000:
    #         #     #     continue
    #         #     cnn_visual[int(i[0]):int(i[1])] = 2
    #         # cnn_visual = cnn_visual * max(audio_filted) / 6
    #         # pl.plot(time, cnn_visual, color='r')
    #
    #         # cnn_visual = np.zeros_like(audio_show)
    #         # for i in cnn_predict_list:
    #         #     if int(i[0]) > 4000000:
    #         #         continue
    #         #     cnn_visual[int(i[0]):int(i[1])] = 1
    #         # cnn_visual = cnn_visual * max(audio_show) / 6
    #         # pl.plot(time, cnn_visual, color='r')
    #
    #         # click_mat = np.empty((0, 512))
    #         count = 0
    #         # annot_file_list = ['/media/fish/Elements/clickdata/CNN_jasa/cnn_test/RoughToothed_Marianas(MISTC)-Annotated/MISTCS070316-113000_good.txt']
    #         for annot_file in annot_file_list:
    #             try:
    #                 annot = read_annot(annot_file)
    #             except:
    #                 print('while reading txt a error happened! file_name:', annot_file)
    #                 continue
    #             if annot.size == 0:
    #                 continue
    #             print('annot file:', annot_file)
    #             annot *= fs
    #             # correct_detect, annot_num = predict_annot_compare(cnn_predict_list, annot)
    #
    #             annot = annot.tolist()
    #             # filted_annot = []
    #             filted_annot = annot
    #
    #             # for index in annot:
    #             #     #  信噪比过滤
    #             #     if index[1]-index[0] > 2048:
    #             #         continue
    #             #     int_s = int(index[0])
    #             #     int_e = int(index[1])
    #             #     click_data = audio_filted[int_s:int_e]
    #             #     # detected_clicks_energy = calcu_click_energy(click_data.reshape(1, -1))
    #             #     detected_clicks_energy = calcu_energy(click_data)
    #             #     noise_estimate1 = audio_filted[int_s - 256:int_s]
    #             #     noise_estimate2 = audio_filted[int_e + 1:int_e + 257]
    #             #     noise_estimate = np.hstack((noise_estimate1, noise_estimate2))
    #             #     noise_energy = calcu_energy(noise_estimate)
    #             #     if noise_energy <= 0 or detected_clicks_energy <= 0:
    #             #         continue
    #             #     snr = 10 * math.log10(detected_clicks_energy / noise_energy)
    #             #     if snr < 7:
    #             #         continue
    #             #     # 保存标注click，添加至训练数据
    #             #     # else:
    #             #     #     max_index = np.argmax(click_data)
    #             #     #     center = max_index + int_s
    #             #     #     click = audio[center-256:center+256]
    #             #     #     click = np.reshape(click, (1, 512))
    #             #     #     click_mat = np.vstack((click_mat, click))
    #             #     filted_annot.append(index)
    #             if filted_annot == []:
    #                 continue
    #             detect_matcher = detection_rate.marked_pool(filted_annot)
    #             recall, precision = detect_matcher.calcu_detection_rate(cnn_predict_list)
    #             annot_num = len(filted_annot)
    #             correct_detect = int(recall * annot_num)
    #             correct_total += correct_detect
    #             annot_total += annot_num
    #             print('correct: %d, total: %d, recall: %f' %(correct_detect, annot_num, recall))
    #
    #         # print('click shape:', click_mat.shape)
    #         # path_list = folder.split('/')
    #         # filename = path_list[-1] + '.mat'
    #         # sio.savemat(filename, {'click_data': click_mat, 'fs': fs})
    #
    #         #     for i in annot:
    #         #         # if int(i[0]) > 1000000:
    #         #         #     continue
    #         #         count += 1
    #         #         detected_visual[int(i[0]):int(i[1])] = 1
    #         # print(count)
    #         # detected_visual = detected_visual * max(audio_filted) / 6
    #         # pl.plot(time, detected_visual, color='b')
    #         # pl.show()
    #
    #         if annot_total == 0:
    #             continue
    #         total_recall = correct_total / annot_total
    #         print('total correct: %d, total annot: %d, recall: %f' % (correct_total, annot_total, total_recall))


