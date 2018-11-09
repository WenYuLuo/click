# from scipy import signal
# import wave
import numpy as np
# import matplotlib.pyplot as plt
# import math
import os
from cnn_detect_click import *
import detection_rate


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
    annot_list = list_txt_files(path)
    r_annot = np.empty([0, 2])

    for annot_file in annot_list:
        annot = np.loadtxt(annot_file, skiprows=1)
        annot = annot[:, :2]
        r_annot = np.vstack((r_annot, annot))

    return r_annot


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
    path = 'E:\DailyResearch\clicksclassification\GoodClick\Rissos-SCORE-annot'
    list_folder = list_files(path)

    for folder in list_folder:
        print(folder)
        annot_file_list = list_ext_files(folder, ext='.txt')
        wave_list = list_ext_files(folder, ext='.wav')
        if len(wave_list) != 1:
            print('当前文件夹音频数大于1')
            raise ValueError
        correct_total = 0
        annot_total = 0
        print('==============================processing====================================')
        print(wave_list[0])
        snr_threshold_low = 0
        cnn_predict_list, fs = run_cnn_detection(wave_list[0], snr_threshold_low)
        for annot_file in annot_file_list:
            annot = read_annot(annot_file)
            print('annot file:', annot_file)
            annot *= fs
            # correct_detect, annot_num = predict_annot_compare(cnn_predict_list, annot)
            detect_matcher = detection_rate.marked_pool(annot.tolist())
            recall, precision = detect_matcher.calcu_detection_rate(cnn_predict_list)
            annot_num = annot.shape[0]
            correct_detect = int(recall * annot_num)
            correct_total += correct_detect
            annot_total += annot_num
            print('correct: %d, total: %d, recall: %f, precision: %f' %(correct_detect, annot_num, recall, precision))
        total_recall = correct_total / annot_total
        print('total correct: %d, total annot: %d, recall: %f' % (correct_total, annot_total, total_recall))


