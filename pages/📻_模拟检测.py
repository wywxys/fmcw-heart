import time

import numpy as np
import pandas as pd
import streamlit as st


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def draw_chart(still_time, bpm, chart):
    last_rows = []
    temp_rate = 0

    for i in range(still_time):
        result = np.random.randint(60, 80)
        print(result)

        bpm.metric(label='心跳速率', value=str(result) + 'BPM', delta=str(result - temp_rate))
        temp_rate = int(result)

        last_rows.append(result)
        chart.line_chart(last_rows)
        time.sleep(1)

    st.button("Re-run")

    return last_rows


def file_predict(still_time, bpm, chart):
    out_predict = draw_chart(still_time, bpm, chart)
    out_csv = pd.Series(out_predict, name='HeartRate')
    return convert_df(out_csv)


def form_submit():
    with st.form(key='my_form'):
        select_num = st.slider('选择模拟检测的持续时间（s）', 0, 60, 30)
        submitted = st.form_submit_button(label='开始监测')

    bpm = st.metric(label='心跳速率', value='--BPM', delta='--')
    chart = st.line_chart([0])

    if submitted:
        csv = file_predict(select_num, bpm, chart)  # index 0-47
        st.download_button(
            label="Download HeartRate as CSV",
            data=csv,
            file_name='HeartRate.csv',
            mime='text/csv',
        )


def run_outside():
    st.set_page_config(
        page_title="模拟检测模式-心率监测",
        page_icon="💖",
    )

    st.title('FMCW雷达心率监测')
    form_submit()


run_outside()
