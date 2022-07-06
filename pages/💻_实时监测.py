import time

import numpy as np
import pandas as pd
import streamlit as st


def run_outside():
    st.set_page_config(
        page_title="模拟检测模式-心率监测",
        page_icon="💖",
    )

    st.title('FMCW雷达心率监测')
    form_submit()


def form_submit():
    with st.form(key='my_form'):
        st.text('本模式用于本地部署以连接雷达进行现场实时监测')
        submitted = st.form_submit_button(label='确认')


run_outside()
