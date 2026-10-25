# pipeline.py
import os
import json
import zipfile
import re
import numpy as np
import mne
from io import BytesIO
from config import (
    DEFAULT_CHANNELS, MIN_SFREQ, MIN_CHANNELS,
    BAD_CHANNEL_Z_THRESH, BAD_CHANNEL_RATIO_THRESH,
    BAD_CHANNEL_TERMINATE_RATIO,
    FASTER_WINDOW_SEC, FASTER_OVERLAP, FASTER_STD_FACTOR,
    FASTER_LOWER_UV, FASTER_UPPER_UV,
    FASTER_CHANNEL_RATIO,
    BANDPASS_L_FREQ, BANDPASS_H_FREQ,
    NOTCH_REGION,
    FILTER_METHOD, IIR_ORDER,
    MONTAGE_NAME,
)
# ---------- 通道名归一化 ----------
def _norm_key(name: str) -> str:
    s = name.upper().replace("EEG", "")
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s.lower()
def load_edf(file_obj, min_sfreq=MIN_SFREQ):
    # 支持文件流 / 本地路径
    if isinstance(file_obj, BytesIO):
        file_obj.seek(0)
        raw = mne.io.read_raw_edf(file_obj, preload=True, verbose=False)
    else:
        raw = mne.io.read_raw_edf(file_obj, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    if sfreq < min_sfreq:
        raise ValueError(f"采样率{sfreq}低于阈值{min_sfreq}Hz，终止任务")
    return raw
def normalize_channel_names(raw):
    rename_map = {}
    for ch in raw.ch_names:
        clean_name = re.sub(r"\.+", "", ch).strip()
        if clean_name != ch:
            rename_map[ch] = clean_name
    if rename_map:
        raw.rename_channels(rename_map)
    return raw
def select_channels(raw, requested_channels=None):
    if requested_channels is None or len(requested_channels) == 0:
        available_chs = raw.ch_names.copy()
        missing = []
    else:
        available_chs = [ch for ch in requested_channels if ch in raw.ch_names]
        missing = [ch for ch in requested_channels if ch not in raw.ch_names]
    if len(available_chs) < MIN_CHANNELS:
        raise ValueError(
            f"有效通道数 {len(available_chs)} 少于最低通道数 {MIN_CHANNELS}。缺失通道：{missing}；原始通道：{raw.ch_names}"
        )
    raw_picked = raw.copy().pick(available_chs)
    raw_picked = normalize_channel_names(raw_picked)
    available_chs = raw_picked.ch_names
    return raw_picked, available_chs, missing
def bind_montage(raw, montage_name=MONTAGE_NAME):
    try:
        montage = mne.channels.make_standard_montage(montage_name)
    except Exception:
        return raw
    std_names = montage.ch_names
    key_map = {_norm_key(n): n for n in std_names}
    rename_map = {}
    for ch in raw.ch_names:
        k = _norm_key(ch)
        if k in key_map and key_map[k] != ch:
            rename_map[ch] = key_map[k]
    if rename_map:
        raw.rename_channels(rename_map)
    raw.set_montage(montage, on_missing="ignore")
    return raw
def detect_bad_channels(raw, z_thresh=BAD_CHANNEL_Z_THRESH, ratio_thresh=BAD_CHANNEL_RATIO_THRESH):
    data = raw.get_data()
    stds = np.std(data, axis=1)
    med_std = np.median(stds)
    mad = np.median(np.abs(stds - med_std))
    robust_z = 0.6745 * (stds - med_std) / (mad + 1e-10)
    bad_ch_list = []
    for idx, z in enumerate(robust_z):
        if abs(z) > z_thresh:
            bad_ch_list.append(raw.ch_names[idx])
    total_ch = len(raw.ch_names)
    if len(bad_ch_list) / total_ch > BAD_CHANNEL_TERMINATE_RATIO:
        raise RuntimeError(f"坏通道占比过高({len(bad_ch_list)}/{total_ch})，停止处理")
    return bad_ch_list
def repair_bad_channels(raw, bad_channels):
    if len(bad_channels) == 0:
        return raw.copy(), "无坏通道"
    raw_interp = raw.copy()
    raw_interp.info["bads"] = bad_channels
    raw_interp.interpolate_bads(reset_bads=True, verbose=False)
    msg = f"已插值修复坏通道：{bad_channels}"
    return raw_interp, msg
def faster_artifact_mask(raw, window_sec=FASTER_WINDOW_SEC, overlap=FASTER_OVERLAP,
                        std_factor=FASTER_STD_FACTOR, lower_uv=FASTER_LOWER_UV, upper_uv=FASTER_UPPER_UV,
                        ch_ratio=FASTER_CHANNEL_RATIO):
    sfreq = raw.info["sfreq"]
    data = raw.get_data()
    n_chan, n_samp = data.shape
    win_samp = int(window_sec * sfreq)
    step_samp = int(win_samp * (1 - overlap))
    mask = np.zeros(n_samp, dtype=bool)
    for start in range(0, n_samp - win_samp + 1, step_samp):
        seg = data[:, start:start+win_samp]
        ptp = np.ptp(seg, axis=1) * 1e6
        bad_count = np.sum((ptp < lower_uv) | (ptp > upper_uv))
        if bad_count / n_chan >= ch_ratio:
            mask[start:start+win_samp] = True
    return mask
def apply_bad_segment_mask(raw, bad_mask):
    if np.sum(bad_mask) == 0:
        return raw.copy(), 0
    data = raw.get_data()
    keep = ~bad_mask
    new_data = data[:, keep]
    info = raw.info.copy()
    raw_clean = mne.io.RawArray(new_data, info, verbose=False)
    return raw_clean, int(np.sum(bad_mask))
def filter_raw(raw, l_freq=BANDPASS_L_FREQ, h_freq=BANDPASS_H_FREQ, notch=NOTCH_REGION,
               method=FILTER_METHOD, iir_order=IIR_ORDER):
    raw_filt = raw.copy()
    raw_filt.filter(l_freq=l_freq, h_freq=h_freq, method=method,
                    iir_params=dict(order=iir_order, ftype="butter"), verbose=False)
    if notch not in [None, "none"]:
        raw_filt.notch_filter(freqs=float(notch), verbose=False)
    return raw_filt
# ========== 修改点1：build_log增加notch入参，params["notch"]使用传入值 ==========
def build_log(raw, ch_used, ch_missing, bad_ch, bad_samp, notch, extra=None):
    log_dict = {
        "channels_used": ch_used,
        "channels_missing": ch_missing,
        "bad_channels": bad_ch,
        "bad_sample_count": bad_samp,
        "sfreq": raw.info["sfreq"],
        "n_channels": len(ch_used),
        "duration_sec": raw.times[-1] if len(raw.times) > 0 else 0,
        "params": {
            "bandpass": [BANDPASS_L_FREQ, BANDPASS_H_FREQ],
            "notch": notch,
            "filter_method": FILTER_METHOD,
            "iir_order": IIR_ORDER,
            "faster_window_sec": FASTER_WINDOW_SEC,
            "faster_overlap": FASTER_OVERLAP,
            "faster_std_factor": FASTER_STD_FACTOR,
            "faster_uv_range": [FASTER_LOWER_UV, FASTER_UPPER_UV],
            "faster_channel_ratio": FASTER_CHANNEL_RATIO,
            "bad_ch_z_thresh": BAD_CHANNEL_Z_THRESH,
            "bad_ch_terminate_ratio": BAD_CHANNEL_TERMINATE_RATIO,
        }
    }
    if extra is not None:
        log_dict.update(extra)
    return log_dict
def export_result_zip(raw_out, log_dict, out_dir="output", prefix="result"):
    os.makedirs(out_dir, exist_ok=True)
    edf_path = os.path.join(out_dir, f"{prefix}_proc.edf")
    json_path = os.path.join(out_dir, f"{prefix}_montage.json")
    zip_path = os.path.join(out_dir, f"{prefix}.zip")
    raw_out.export(edf_path, overwrite=True, verbose=False)
    pos_dict = {}
    for ch in raw_out.info["chs"]:
        pos = ch["loc"][:3].tolist()
        pos_dict[ch["ch_name"]] = [float(x) if np.isfinite(x) else None for x in pos]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pos_dict, f, indent=2, ensure_ascii=False)
    log_path = os.path.join(out_dir, f"{prefix}_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_dict, f, indent=2, ensure_ascii=False)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(edf_path, os.path.basename(edf_path))
        zf.write(json_path, os.path.basename(json_path))
        zf.write(log_path, os.path.basename(log_path))
    return zip_path, log_dict
# ---------- 绘图函数：通道名放在波形起始位置的左侧，文字不会压在波形线上 ----------
def make_waveform_plotly(raw_before, raw_after, bad_mask, sfreq, sel_chs):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # 拆分连续坏段的辅助函数
    def get_contiguous_bad_segments(bad_mask_arr, t_arr):
        segments = []
        if not np.any(bad_mask_arr):
            return segments
        bad_indices = np.where(bad_mask_arr)[0]
        start_idx = bad_indices[0]
        current_segment = [start_idx]
        for idx in bad_indices[1:]:
            if idx == current_segment[-1] + 1:
                current_segment.append(idx)
            else:
                segments.append( (t_arr[current_segment[0]], t_arr[current_segment[-1]]) )
                current_segment = [idx]
        segments.append( (t_arr[current_segment[0]], t_arr[current_segment[-1]]) )
        return segments

    chs = [c for c in sel_chs if c in raw_before.ch_names]
    if len(chs) == 0:
        chs = raw_before.ch_names[:4]
    data_before = raw_before.get_data(picks=chs) * 1e6
    data_after = raw_after.get_data(picks=chs) * 1e6
    t_before = raw_before.times
    offset_step = 200
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("处理前（红色区域为剔除的坏段）", "处理后"))
    annotations = []
    for i, chname in enumerate(chs):
        y_pos = i * offset_step
        # name=chname 绑定真实通道名，hover不再显示trace编号
        fig.add_trace(go.Scatter(x=t_before, y=data_before[i] + y_pos,
                                 name=chname, line=dict(width=0.7), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_before, y=data_after[i] + y_pos,
                                 name=chname, line=dict(width=0.7), showlegend=False), row=2, col=1)
        # x=-0.3 放在0时刻左侧，文字在波形线前面，不会压到波形
        annotations.append(go.layout.Annotation(
            x=-0.3, y=y_pos,
            xref="x1", yref="y1",
            text=chname, showarrow=False, font=dict(size=9),
            xanchor="right", yanchor="middle"
        ))
        annotations.append(go.layout.Annotation(
            x=-0.3, y=y_pos,
            xref="x2", yref="y2",
            text=chname, showarrow=False, font=dict(size=9),
            xanchor="right", yanchor="middle"
        ))
    # 红色坏段阴影（修改：分段绘制独立坏段）
    if bad_mask is not None and np.any(bad_mask):
        bad_segs = get_contiguous_bad_segments(bad_mask, t_before)
        for (t0, t1) in bad_segs:
            fig.add_vrect(x0=t0, x1=t1, fillcolor="rgba(255,0,0,0.15)",
                          line_width=0, row=1, col=1)
    fig.update_layout(
        annotations=annotations,
        height=800,
        hovermode="x unified",
        margin=dict(l=90) # 左侧预留足够画布空间放通道名称，防止文字被截断
    )
    fig.update_xaxes(title_text="时间 (s)", row=2, col=1, range=[-0.5, t_before[-1]]) # x轴左扩，留出文字区域
    fig.update_yaxes(title_text="幅值偏移 (μV)")
    return fig
def make_spectrum_plotly(raw_before, raw_after, fmax=80):
    import plotly.graph_objects as go
    psd_obj_b = raw_before.compute_psd(fmax=fmax, verbose=False)
    psd_obj_a = raw_after.compute_psd(fmax=fmax, verbose=False)
    psd_b = psd_obj_b.get_data()
    freq_b = psd_obj_b.freqs
    psd_a = psd_obj_a.get_data()
    freq_a = psd_obj_a.freqs
    psd_b_db = 10 * np.log10(np.mean(psd_b, axis=0))
    psd_a_db = 10 * np.log10(np.mean(psd_a, axis=0))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq_b, y=psd_b_db, name="预处理前", line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=freq_a, y=psd_a_db, name="预处理后", line=dict(width=1.2)))
    fig.add_vline(x=1, line_dash="dash", line_color="gray")
    fig.add_vline(x=45, line_dash="dash", line_color="gray")
    fig.add_vline(x=50, line_dash="dash", line_color="red")
    fig.add_vline(x=60, line_dash="dash", line_color="orange")
    fig.update_layout(xaxis_title="频率(Hz)", yaxis_title="功率谱 (dB)", height=450)
    return fig
def run_pipeline(file_obj, requested_channels=None, out_dir="output", notch_region="none"):
    raw = load_edf(file_obj)
    raw, ch_used, ch_missing = select_channels(raw, requested_channels)
    raw = bind_montage(raw)
    raw = filter_raw(raw, notch=notch_region)
    bad_channels = detect_bad_channels(raw)
    raw, repair_msg = repair_bad_channels(raw, bad_channels)
    raw_before = raw.copy()
    bad_mask = faster_artifact_mask(raw)
    raw_after, bad_sample_count = apply_bad_segment_mask(raw, bad_mask)
    # ========== 修改点2：build_log调用增加notch参数传入 ==========
    log_dict = build_log(raw_after, ch_used, ch_missing, bad_channels, bad_sample_count,
                         notch=notch_region,
                         extra={"repair_msg": repair_msg})
    zip_path, log_dict = export_result_zip(raw_after, log_dict, out_dir=out_dir)
    sfreq = raw.info["sfreq"]
    return zip_path, log_dict, raw_before, raw_after, bad_mask, sfreq
