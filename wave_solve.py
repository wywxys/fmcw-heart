import numpy as np


def range_window(adc):
    return np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (adc - 1)) for n in range(adc)])


# 此时对矩阵每行做FFT得到每帧的距离和相位 距离在x轴上线性对应
def range_fft(data_for_fft):
    for chirp in range(len(data_for_fft)):  # 遍历每一行
        data_for_fft[chirp] = np.fft.fft(data_for_fft[chirp] * range_window(data_for_fft.shape[1]))


def max_range_index(fft_data):
    data_sum = np.zeros(fft_data.shape[1])
    for r in fft_data:
        data_sum += np.abs(r)

    data_sum[0:5] = 0  # 两端去直流
    data_sum[-1:-3:-1] = 0

    return np.argmax(data_sum)  # 结果是55


def reshape_radar(range_fft_radar):
    radar_temp = []

    i = max_range_index(range_fft_radar)
    radar_phase = np.angle(range_fft_radar)[:, i - 1:i + 2]

    for radar_index in range(51):
        radar_temp.append(radar_phase[radar_index * 20:radar_index * 20 + 24, :].tolist())

    return np.array(radar_temp)


def read_random(random_path, train_or_test, index):
    train_test = np.load(random_path, allow_pickle=True).tolist()
    # print(train_test)
    test_list = train_test[train_or_test]
    # print(len(test_list))
    return test_list[index]
