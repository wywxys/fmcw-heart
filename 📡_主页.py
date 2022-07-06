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
        - <a href="/单条数据分析" target="_self">单条数据分析</a>——对雷达数据进行心率预测，可用于科研分析
        - <a href="/实时监测" target="_self">实时监测</a>——用于本地部署连接雷达
        - <a href="/模拟检测" target="_self">模拟检测</a>——模拟雷达数据输入，可用于员工培训
        """
        , unsafe_allow_html=True
    )


if __name__ == '__main__':
    describe()
