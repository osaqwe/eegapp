import streamlit as st
import mne
from io import BytesIO
from pipeline import run_pipeline, make_waveform_plotly, make_spectrum_plotly

st.set_page_config(page_title="自适应通道数量脑电预处理系统", layout="wide")
st.title("自适应通道数量脑电全自动预处理网页系统")

# 会话状态初始化
if "result_data" not in st.session_state:
    st.session_state.result_data = None

# 文件上传区域
uploaded_file = st.file_uploader("上传EDF脑电文件", type=["edf"])
col1, col2 = st.columns(2)
with col1:
    notch_mode = st.radio("工频陷波选择", ["none", "50", "60"], horizontal=True)
with col2:
    requested_ch_option = st.radio("通道选择方式", ["全部通道", "手动选择通道"], horizontal=True)

requested_channels = None
if uploaded_file and requested_ch_option == "手动选择通道":
    uploaded_file.seek(0)
    # 读取文件流获取通道名，修复preload报错
    bio = BytesIO(uploaded_file.read())
    bio.seek(0)
    raw_temp = mne.io.read_raw_edf(bio, preload=True, verbose=False)
    
    unique_ch_names = list(dict.fromkeys(raw_temp.ch_names))
    requested_channels = st.multiselect("选择待处理通道", raw_temp.ch_names)
    uploaded_file.seek(0)  # 关键：读完后再重置，供后续处理使用

# 运行预处理按钮
if uploaded_file and st.button("开始全自动预处理"):
    with st.spinner("正在运行脑电预处理流水线..."):
        try:
            uploaded_file.seek(0)
            # 包装成BytesIO传给pipeline
            bio_file = BytesIO(uploaded_file.read())
            zip_path, log, raw_before, raw_after, bad_mask, sfreq2 = run_pipeline(
                bio_file,
                requested_channels=requested_channels,
                notch_region=notch_mode
            )
            st.session_state.result_data = {
                "zip_path": zip_path,
                "log": log,
                "raw_before": raw_before,
                "raw_after": raw_after,
                "bad_mask": bad_mask,
                "sfreq": sfreq2
            }
            st.success("预处理完成！切换下方标签页查看波形、频谱与日志结果")
        except Exception as e:
            st.error(f"处理失败：{e}")

# 结果展示标签页
if st.session_state.result_data is not None:
    res = st.session_state.result_data
    tab1, tab2, tab3 = st.tabs(["波形对比图", "频谱PSD对比", "处理日志&下载"])

    # ========== tab1 波形对比 ==========
    with tab1:
        st.markdown("**处理前后波形对比**（浅红色区域为剔除的坏段）")
        st.caption("支持鼠标拖动平移、滚轮缩放、框选放大、双击复位")
        all_chs = res["raw_before"].ch_names
        st.info(f"当前绘制全部 {len(all_chs)} 个通道波形")
        fig_wave = make_waveform_plotly(
            res["raw_before"],
            res["raw_after"],
            res["bad_mask"],
            res["sfreq"],
            sel_chs=all_chs
        )
        st.plotly_chart(fig_wave, use_container_width=True)

    # ========== tab2 频谱图 ==========
    with tab2:
        st.markdown("**预处理前后功率谱密度(PSD)对比**")
        fig_spec = make_spectrum_plotly(res["raw_before"], res["raw_after"], fmax=80)
        st.plotly_chart(fig_spec, use_container_width=True)

    # ========== tab3 日志与结果下载 ==========
    with tab3:
        st.markdown("### 📝 处理流程日志")
        
        # 将英文JSON替换为易读的中文界面
        log_data = res["log"]
        
        # 第一块：基础信息
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("采样率 (Hz)", log_data.get("sfreq", "N/A"))
        col_info2.metric("有效通道数", log_data.get("n_channels", "N/A"))
        col_info3.metric("数据时长 (秒)", round(log_data.get("duration_sec", 0), 2))

        st.markdown("---")
        
        # 第二块：坏道与剔除信息
        st.write(f"**🛠️ 坏道修复记录:** {log_data.get('repair_msg', '无')}")
        st.write(f"**✂️ 剔除坏段采样点数:** {log_data.get('bad_sample_count', 0)} (约等于 {round(log_data.get('bad_sample_count', 0) / log_data.get('sfreq', 1), 2)} 秒)")
        
        st.markdown("---")
        
        # 第三块：参数配置面板
        with st.expander("⚙️ 查看详细处理参数", expanded=False):
            params = log_data.get("params", {})
            st.write(f"- **带通滤波:** {params.get('bandpass', ['-', '-'])[0]} Hz ~ {params.get('bandpass', ['-', '-'])[1]} Hz")
            st.write(f"- **工频陷波:** {params.get('notch', 'none')}")
            st.write(f"- **滤波器类型:** {params.get('filter_method', 'N/A')} (阶数: {params.get('iir_order', 'N/A')})")
            st.write(f"- **FASTER 伪迹剔除窗口:** {params.get('faster_window_sec', 'N/A')} 秒")
            st.write(f"- **FASTER 幅值范围 (μV):** {params.get('faster_uv_range', ['N/A', 'N/A'])[0]} ~ {params.get('faster_uv_range', ['N/A', 'N/A'])[1]}")
            st.write(f"- **坏通道 Z-score 阈值:** {params.get('bad_ch_z_thresh', 'N/A')}")

        st.markdown("---")
        
        # 第四块：下载区域
        st.markdown("### 📥 结果下载")
        with open(res["zip_path"], "rb") as f:
            st.download_button(
                label="⬇️ 下载打包结果（EDF+日志+电极位置json）",
                data=f.read(),
                file_name="result.zip",
                mime="application/zip"
            )
