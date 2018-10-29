from scipy import signal
import wave
import numpy as np
import matplotlib.pyplot as plt
import math

# wave_file = wave.open("/home/fish/ROBB/fiber_sensing/optical-fiber-sensing/20180920001909_out.wav", 'rb')
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
# wave_file1 = wave.open("/home/fish/ROBB/fiber_sensing/optical-fiber-sensing/20180920001909.wav", 'rb')
# # wave_file = wave.open("./Data/Click/0/click_00016.wav", 'rb')
# params1 = wave_file1.getparams()
# channels1, sampleWidth1, frameRate1, frames1 = params1[:4]
# data_bytes1 = wave_file1.readframes(frames1)  # 读取音频，字符串格式
# wave_file1.close()
# wave_data1 = np.fromstring(data_bytes1, dtype=np.int16)  # 将字符串转化为int
# wave_data1 = np.reshape(wave_data1, [frames, channels])
#
# print(len(wave_data1))
# print(frameRate1)
#
# plt.figure()
# plt.plot(np.arange(0, len(wave_data)) * (1.0 / frameRate), wave_data, color='b')
# plt.plot(np.arange(0, len(wave_data1)) * (1.0 / frameRate1), wave_data1, color='r')
# plt.xlabel("Time(s)")
# plt.ylabel("Amplitude")
# plt.grid('True')  # 标尺，on：有，off:无。
# plt.show()


if __name__ == '__main__':
    click_arr = np.load('./CNNDet/Gm/Pilot_whales_Bahamas(AUTEC)-Annotated-NUWC/Set7-A1-093005-H01-0030-0100-0846-0916loc_N9526.npy')
    print(click_arr.shape)
    rand_index = np.random.permutation(click_arr.shape[0])
    for i in rand_index:
        click = click_arr[i, :]
        plt.plot(np.arange(0,click_arr.shape[1]) / 192000, click)
        plt.xlabel("Time(s)")
        plt.ylabel("Amplitude")
        plt.grid('True')  # 标尺，on：有，off:无。
        plt.show()
