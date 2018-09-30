from scipy import signal
import wave
import numpy as np
import matplotlib.pyplot as plt
import math

# wave_file = wave.open("./Data/ClickC8/0/click_000016.wav", 'rb')
# # wave_file = wave.open("./Data/Click/0/click_00016.wav", 'rb')
# params = wave_file.getparams()
# channels, sampleWidth, frameRate, frames = params[:4]
# data_bytes = wave_file.readframes(frames)  # 读取音频，字符串格式
# wave_file.close()
# wave_data = np.fromstring(data_bytes, dtype=np.int16)  # 将字符串转化为int
# wave_data = np.reshape(wave_data, [frames, channels])
#
# print(len(wave_data))
# print(frameRate)
#
# plt.figure()
# plt.plot(np.arange(0, len(wave_data)) * (1.0 / frameRate), wave_data)
# plt.xlabel("Time(s)")
# plt.ylabel("Amplitude")
# plt.grid('True')  # 标尺，on：有，off:无。
# plt.show()


if __name__ == '__main__':
    click_arr = np.load('./CNNDetection/Spinner/palmyra2006/palmyra102006-061103-221145_4_N147.npy')
    print(click_arr.shape)
    rand_index = np.random.permutation(click_arr.shape[0])
    for i in rand_index:
        click = click_arr[i, :]
        plt.plot(np.arange(0,click_arr.shape[1]) / 192000, click)
        plt.xlabel("Time(s)")
        plt.ylabel("Amplitude")
        plt.grid('True')  # 标尺，on：有，off:无。
        plt.show()
