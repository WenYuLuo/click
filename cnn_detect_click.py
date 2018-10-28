import os
import math
import pylab as pl
import numpy as np
from scipy import signal
import tensorflow as tf
import find_click
from detect_click import resample, mkdir




# 局部能量归一化
def local_normalize(audio):
    sum_conv_kernel = np.ones((1, 256))
    audio_pow = audio**2
    sum_energy = signal.convolve(audio_pow.reshape(1, -1), sum_conv_kernel, mode='same')
    sqrt_mean_energy = np.sqrt(sum_energy / 256)
    local_norm = np.true_divide(audio, sqrt_mean_energy)
    return local_norm


# softmax输出
def softmax(vector):
    # print(vector)
    vector = np.exp(vector)
    # print(vector)
    vector[np.isinf(vector)] = 1
    vector[vector < 10e-10] = 0
    exp_sum = np.sum(vector, axis=1).reshape((-1, 1))
    return np.true_divide(vector, exp_sum)


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


def run_cnn_detection(file_name, dst_path, tar_fs=192000):

    """
        support the audio frame rate no bigger than 192000
            :param file_name:输入音频
            :param dst_path:click存储路径
            :param tar_fs:输出信号采样率
    """

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    # sess = tf.Session(config=config)
    graph = tf.get_default_graph()
    # graph = tf.reset_default_graph()

    signal_len = 320
    count = 0
    click_arr = []
    audio, fs = find_click.read_wav_file(file_name)
    if audio.shape[1] > 1:
        audio = audio[:, 1]
    else:
        audio = audio[:, 0]

    # # 重采样至
    # audio = resample(audio, fs, tar_fs)
    # fs = tar_fs

    [path, wavname_ext] = os.path.split(file_name)
    wavname = wavname_ext.split('/')[-1]
    wavname = wavname.split('.')[0]

    if fs > 192000:
        print('down sample was not supported! current sampling rate is %d' % fs)
        return None

    len_audio = len(audio)
    fl = fs/40
    # fs = 192000
    wn = 2*fl/fs
    b, a = signal.butter(8, wn, 'high')

    audio_filted = signal.filtfilt(b, a, audio)
    scale = (2 ** 15 - 1) / max(audio_filted)
    for i in np.arange(audio_filted.size):
        audio_filted[i] = audio_filted[i] * scale

    # audio_norm = local_normalize(audio_filted)
    #
    # audio_norm = audio_norm[0]
    #
    # time = np.arange(0, audio_filted.shape[0]) / fs
    # # pl.plot(time, audio)
    # # pl.show()
    # pl.plot(time, audio_filted)
    # pl.title('high pass filter')
    # pl.xlabel('time')
    # # pl.show()

    seg_length = 8000000
    data_seg = []
    if len_audio > seg_length:
        seg_num = math.ceil(len_audio / seg_length)
        for i in range(int(seg_num)):
            start_seg = seg_length * i
            if seg_length * (i + 1) > len_audio:
                end_seg = len_audio
            else:
                end_seg = seg_length * (i + 1)
            if end_seg > len_audio - 1:
                end_seg = len_audio - 1
            data = audio_filted[start_seg:end_seg]
            data_norm = local_normalize(data)
            data_norm = data_norm[0]
            data_seg.append(data_norm)
    else:
        audio_norm = local_normalize(audio_filted)
        audio_norm = audio_norm[0]
        data_seg.append(audio_norm)

        # detected_visual用于定位click
    detected_visual = np.zeros_like(audio_filted)
    # 预加载模型参数
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph('params_cnn/allconv_cnn4click_norm_quater_manual.ckpt-300.meta')
        saver.restore(sess, 'params_cnn/allconv_cnn4click_norm_quater_manual.ckpt-300')
        # graph = tf.reset_default_graph()
        # 获取模型参数
        x = sess.graph.get_operation_by_name('x').outputs[0]
        y = sess.graph.get_operation_by_name('y').outputs[0]
        is_batch = sess.graph.get_operation_by_name('is_batch').outputs[0]
        keep_pro_l4_l5 = sess.graph.get_operation_by_name('keep_pro_l4_l5').outputs[0]
        collection = sess.graph.get_collection('saved_module')
        y_net_out6 = collection[0]
        train_step = collection[1]
        accuracy = collection[2]
        click_label = 0
        # graph.finalize()

        for i in range(len(data_seg)):
            audio_norm = data_seg[i]
            y_out = sess.run(y_net_out6, feed_dict={x: audio_norm.reshape(1, -1), keep_pro_l4_l5: 1.0, is_batch: False})
            col_num = y_out.shape[2]
            y_out = y_out.reshape(col_num, 2)
            y_out = softmax(y_out)
            # print(y_out)
            predict = np.argmax(y_out, axis=1)
            for j in range(len(predict)):
                pro = y_out[j][predict[j]]
                if predict[j] == click_label and pro > 0.9:  # and pro > 0.9:
                    start_point = seg_length * i + 8 * j
                    end_point = start_point + 256
                    detected_visual[start_point:end_point] += 1
                    # num_detected = num_detected+1
                    # elif predict == 1:
                    #     detected_visual[start_point:end_point] -= 10

    # detected click 定位
    index_detected = np.where(detected_visual >= 1)[0]
    if index_detected.size == 0:
        print("count = %(count)d" % {'count': count})
        return
    detected_list = []
    is_begin = False
    pos_start = index_detected[0]
    for i in range(len(index_detected)):
        if not is_begin:
            pos_start = index_detected[i]
            is_begin = True
        # 考虑到达list终点时的情况
        if i+1 >= len(index_detected):
            pos_end = index_detected[i]
            detected_list.append((pos_start, pos_end+1))
            break
        if index_detected[i+1] - index_detected[i] > 1:
            pos_end = index_detected[i]
            detected_list.append((pos_start, pos_end+1))
            is_begin = False
        else:
            continue

    # debug: 未过滤检测click数
    print('未过滤click数： %d' % len(detected_list))

    # 去掉低于10db的click
    index_to_remove = []
    for i in range(len(detected_list)):
        detected_pos = detected_list[i]
        detected_length = detected_pos[1] - detected_pos[0]
        if detected_length < 256 + 8 * 8:
            detected_visual[detected_pos[0]:detected_pos[1] + 1] = 0
            index_to_remove.append(i)
            continue
        # snr estimate
        click = audio_filted[detected_pos[0]:detected_pos[1] + 1]
        detected_clicks_energy = calcu_click_energy(click.reshape(1, -1))
        noise_estimate1 = audio_filted[detected_pos[0] - 256:detected_pos[0]]
        noise_estimate2 = audio_filted[detected_pos[1] + 1:detected_pos[1] + 257]
        noise_estimate = np.hstack((noise_estimate1, noise_estimate2))
        noise_energy = calcu_energy(noise_estimate)
        if noise_energy <= 0:
            detected_visual[detected_pos[0]:detected_pos[1] + 1] = 0
            index_to_remove.append(i)
            continue
        snr = 10 * math.log10(detected_clicks_energy / noise_energy)
        if snr < 5 or snr > 20:
            detected_visual[detected_pos[0]:detected_pos[1] + 1] = 0
            index_to_remove.append(i)
    has_removed = 0
    for i in index_to_remove:
        detected_list.pop(i - has_removed)
        has_removed = has_removed + 1

    # # debug
    # for i in detected_list:
    #     detected_visual[i[0]:i[1]] = 1
    # detected_visual = detected_visual * 20000
    # # # print('the number of detected click: %g' % num_detected)
    # pl.plot(time, detected_visual)
    # pl.show()

    for pos_tuple in detected_list:
        temp_click = audio_filted[pos_tuple[0]:pos_tuple[1]]

        # temp_click = resample(temp_click, fs, tar_fs)

        max_index = np.argmax(temp_click)
        max_index += pos_tuple[0]
        t_start = max_index - int(signal_len/2)
        if t_start < 0:
            t_start = 0
        t_end = max_index + int(signal_len/2)
        if t_end > len_audio:
            break
        click_data = audio_filted[t_start:t_end]

        click_data = resample(click_data, fs, tar_fs)

        click_data = cut_data(click_data, signal_len)

        click_data = click_data.astype(np.short)
        # print(click_data.shape)
        click_arr.append(click_data)
        count += 1

    dst = "%(path)s/%(pre)s_N%(num)d.npy" \
          % {'path': dst_path, 'pre': wavname, 'num': len(click_arr)}
    print(dst)
    np.save(dst, np.array(click_arr, dtype=np.short))
    print("count = %(count)d" % {'count': count})


def cut_data(input_signal, out_len):

    audio_len = len(input_signal)
    if audio_len < out_len:
        return input_signal

    beg_idx = int(audio_len/2 - out_len / 2)
    end_idx = beg_idx + out_len

    return input_signal[beg_idx:end_idx]


def detect_click(class_path, class_name):
    tar_fs = 192000
    folder_list = find_click.list_files(class_path)
    if not folder_list:
        folder_list = folder_list + [class_path]
    for folder in folder_list:
        print(folder)
        count = 0
        wav_files = find_click.list_wav_files(folder)

        # wav_files = shuffle_frames(wav_files)

        path_name = folder.split('/')[-1]

        dst_path = "./CNNClear1/%(class)s/%(type)s" % {'class': class_name, 'type': path_name}
        if not os.path.exists(dst_path):
            mkdir(dst_path)

        for file_name in wav_files:
            run_cnn_detection(file_name, dst_path, tar_fs)


if __name__ == '__main__':

    # run_cnn_detection('/media/fish/Elements/clickdata/ForCNNLSTM/3rdTraining_Data/Blainvilles_beaked_whale_'
    #                   '(Mesoplodon_densirostris)/Set4-A5-092705-H57-0600-0618-1435-1453loc_filter_1200-1400mi.wav',
    #                   dst_path='./')

    # detect_click(class_path='/media/fish/Elements/clickdata/ForCNNLSTM/3rdTraining_Data/Blainvilles_beaked_whale_'
    #                         '(Mesoplodon_densirostris)', class_name='BBW')

    # detect_click(class_path='/media/fish/Elements/clickdata/ForCNNLSTM/5th_DCL_data_bottlenose',
    #                   class_name='Tt')

    # detect_click(class_path='/media/fish/Elements/clickdata/ForCNNLSTM/3rdTraining_Data/Pilot_whale_(Globicephala_'
    #                              'macrorhynchus)', class_name='Gm')
    #
    # detect_click(class_path='/media/fish/Elements/clickdata/ForCNNLSTM/3rdTraining_Data/Rissos_(Grampus_grisieus)',
    #                   class_name='Gg')
    #
    detect_click(class_path='/media/fish/Elements/clickdata/ForCNNLSTM/5th_DCL_data_common/Dc',
                      class_name='Dc')

    detect_click(class_path='/media/fish/Elements/clickdata/ForCNNLSTM/5th_DCL_data_common/Dd',
                      class_name='Dd')

    detect_click(class_path='/media/fish/Elements/clickdata/ForCNNLSTM/5th_DCL_data_melon-headed',
                      class_name='Melon')

    detect_click(class_path='/media/fish/Elements/clickdata/ForCNNLSTM/5th_DCL_data_spinner',
                      class_name='Spinner')

    # detect_click(class_path='/media/fish/Elements/clickdata/Northern right whale dolphin, Lissodelphis borealis',
    #                   class_name='RightWhale')
    #
    # detect_click(class_path='/media/fish/Elements/clickdata/Pacific white-sided dolphin, Lagenorhynchus obliquidens',
    #                   class_name='PacWhite')
    #
    # detect_click(class_path='/media/fish/Elements/clickdata/Pilot whale, Globicephala macrorhynchus',
    #                   class_name='Gm')
    #
    # detect_click(class_path='/media/fish/Elements/clickdata/Rissos dolphin, Grampus griseus',
    #                   class_name='Gg')
    #
    # detect_click(class_path='/media/fish/Elements/clickdata/Rough-toothed dolphin, Steno bredanensis',
    #                   class_name='RoughToothed')
    #
    #
    # detect_click(class_path='/media/fish/Elements/clickdata/Striped dolphin, Stenella coeruleoalba',
    #                   class_name='Striped')
    #
    # detect_click(class_path='/media/fish/Elements/clickdata/Blainville Beaked Whale, Mesoplodon densirostris',
    #                   class_name='Mesoplodon')
    #
    # detect_click(class_path='/media/fish/Elements/clickdata/Cuvier Beaked Whale, Ziphius cavirostris',
    #                   class_name='Beaked')
    #
    # detect_click(class_path='/media/fish/Elements/clickdata/Melon-headed whale, Pepenocephala electra',
    #                   class_name='Melon')
    #
    # detect_click(class_path='/media/fish/Elements/clickdata/Sperm whale, Physeter macrocephalus',
    #                   class_name='Sperm')