import os
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch

from my_resnet import device, my_resnet
from wave_solve import range_fft, reshape_radar


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def frame_output(one_frame, model_path):
    one_frame_input = torch.tensor(one_frame, dtype=torch.float32)
    one_frame_input = torch.reshape(one_frame_input, [1, 1, 24, 3])
    one_frame_input = one_frame_input.to(device)

    resnet = my_resnet(model_path)
    resnet = resnet.to(device)
    resnet = resnet.eval()

    output = resnet(one_frame_input)

    output = output.cpu()
    output = output.detach().numpy()

    return output[0]


def draw_chart(radar_one_file, model_path, bpm, chart):
    col1, col2 = st.columns(2)
    progress_bar = col1.progress(0)
    status_text = col2.text('FMCW 心率预测')

    last_rows = []
    temp_rate = 0

    for i in range(len(radar_one_file)):
        result = frame_output(radar_one_file[i, :, :], model_path)
        print(result)

        int_temp = round(result[0])
        bpm.metric(label='心跳速率', value=str(int_temp) + 'BPM', delta=str(int_temp - temp_rate))
        temp_rate = int_temp

        new_rows = last_rows + list(result)
        status_text.text("%i%% Complete" % (i * 100 / (len(radar_one_file) - 1)))
        chart.line_chart(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(1)

    progress_bar.empty()
    st.button("Re-run")

    return last_rows


def file_predict(file_num, model_path, bpm, chart):
    radar_path = './radar/'
    radar = np.load(radar_path + os.listdir(radar_path)[file_num], allow_pickle=True)

    range_fft(radar)  # 进行距离FFT 有窗
    radar_one_file = reshape_radar(radar)  # (51, 24, 3)

    out_predict = draw_chart(radar_one_file, model_path, bpm, chart)
    out_csv = pd.Series(out_predict, name='HeartRate')
    return convert_df(out_csv)


def form_submit(model_path):
    with st.form(key='my_form'):
        select_num = st.slider('选择测试集中的一个文件', 0, 46, 24)
        submitted = st.form_submit_button(label='开始监测')

    bpm = st.metric(label='心跳速率', value='--BPM', delta='--')
    chart = st.line_chart([0])

    if submitted:
        csv = file_predict(select_num, model_path, bpm, chart)  # index 0-47
        st.download_button(
            label="Download HeartRate as CSV",
            data=csv,
            file_name='HeartRate.csv',
            mime='text/csv',
        )


def run_outside():
    model_path = './resource/resnet_para.pth'
    select_path = './resource/train_test.npy'

    st.set_page_config(
        page_title="单条数据分析模式-心率监测",
        page_icon="💖",
    )

    st.title('FMCW单条数据分析')
    form_submit(model_path)


run_outside()
