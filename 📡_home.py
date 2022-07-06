import streamlit as st


def describe():
    st.set_page_config(
        page_title="欢迎页面-心率监测",
        page_icon="📡",
    )

    st.write("# 欢迎使用非接触式心率监测系统")
    st.sidebar.success("选择监测模式")

    st.markdown(
        """
        在使用前需要注意以下几点：
        ### 雷达与人员状态
        - 雷达需要正面面对被测人员，保证人员与雷达之间没有遮挡
        - 雷达至被测人员的最佳距离在0.8-2米之间
        - 雷达监测时，人员保持静止可以获得更加准确的测量
        """
    )

    st.markdown(
        """
        ### 心率检测模式
        - <a href="/complex/" target="_self">详情模式</a>——提供心率变化监测的更多信息
        - <a href="/complex/" target="_self">简略模式</a>——呃呃呃 &emsp; 这个还没写
        """
        , unsafe_allow_html=True
    )


if __name__ == '__main__':
    describe()
