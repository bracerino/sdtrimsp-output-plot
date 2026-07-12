import os
import re
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import interp1d

PREDEFINED_DENSITIES = {
    "alpha-Ti  (5.66e22 atoms/cm3)":   5.66e22,
    "TiN       (1.0431e23 atoms/cm3)": 1.0431e23,
}

LINEAR_PRESETS = {
    "alpha-Ti → TiN": (0.0, 5.66e22, 5.226e22, 1.0431e23),
    "Custom":         None,
}

DENSITIES_DB_DIR = "densities_db"
FS = 22
COLORS = ['#2563EB', '#D97706', '#16A34A', '#DC2626',
          '#7C3AED', '#0891B2', '#B45309', '#065F46']


def _parse_two_column_file(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue
        if re.search(r'[a-df-zA-DF-Z]', line):
            continue
        parts = line.replace(',', ' ').replace('\t', ' ').split()
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if len(rows) < 2:
        raise ValueError("Need at least 2 numeric rows.")
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1]


def _list_db_files():
    if not os.path.isdir(DENSITIES_DB_DIR):
        return []
    return sorted(f for f in os.listdir(DENSITIES_DB_DIR) if f.endswith('.pkl'))


def _load_db_interpolation(filename):
    path = os.path.join(DENSITIES_DB_DIR, filename)
    with open(path, 'rb') as f:
        return pickle.load(f)


def _convert(n_conc, total_density_func):
    if callable(total_density_func):
        td = total_density_func(n_conc)
    else:
        td = np.full_like(n_conc, float(total_density_func))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(td > 0, n_conc / td, np.nan)


def _density_selector(n_conc_max, key_prefix):
    mode = st.radio(
        "Total density source:",
        ["Constant (predefined material)", "From densities_db/", "Linear interpolation"],
        horizontal=True, key=f'{key_prefix}_mode'
    )

    if mode == "Constant (predefined material)":
        chosen = st.selectbox("Material:", list(PREDEFINED_DENSITIES.keys()),
                              key=f'{key_prefix}_material')
        td_value = PREDEFINED_DENSITIES[chosen]
        st.info(f"Total density: **{td_value:.3e} atoms/cm3**")
        return td_value, chosen.split('(')[0].strip()

    elif mode == "From densities_db/":
        db_files = _list_db_files()
        if not db_files:
            st.warning(f"No .pkl files found in `{DENSITIES_DB_DIR}/`.")
            return None, ""
        chosen_file = st.selectbox("Interpolation file:", db_files,
                                   key=f'{key_prefix}_db_file')
        try:
            interp_obj = _load_db_interpolation(chosen_file)
            meta = interp_obj.get('metadata', {})
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.write(f"System: **{meta.get('matrix_initial','?')} → {meta.get('matrix_final','?')}**")
            with col_m2:
                st.write(f"X range: {meta.get('x_min',0):.3e} – {meta.get('x_max',0):.3e}")
            return interp_obj['total_density'], chosen_file.replace('.pkl', '').replace('_', ' ')
        except Exception as e:
            st.error(f"Failed to load: {e}")
            return None, ""

    else:
        preset_name = st.selectbox(
            "Preset:", list(LINEAR_PRESETS.keys()), key=f'{key_prefix}_preset'
        )
        preset = LINEAR_PRESETS[preset_name]

        if preset is not None:
            n_start_def, td_start_def, n_end_def, td_end_def = preset
        else:
            n_start_def, td_start_def = 0.0, 5.66e22
            n_end_def, td_end_def     = float(n_conc_max), 9.58e22

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Start point**")
            n_start  = st.number_input("N density at start  (atoms/cm3)",
                                       value=float(n_start_def), format="%.3e",
                                       key=f'{key_prefix}_n_start',
                                       disabled=(preset is not None))
            td_start = st.number_input("Total density at start  (atoms/cm3)",
                                       value=float(td_start_def), format="%.3e",
                                       key=f'{key_prefix}_td_start',
                                       disabled=(preset is not None))
        with col_b:
            st.markdown("**End point**")
            n_end  = st.number_input("N density at end  (atoms/cm3)",
                                     value=float(n_end_def), format="%.3e",
                                     key=f'{key_prefix}_n_end',
                                     disabled=(preset is not None))
            td_end = st.number_input("Total density at end  (atoms/cm3)",
                                     value=float(td_end_def), format="%.3e",
                                     key=f'{key_prefix}_td_end',
                                     disabled=(preset is not None))

        lin_interp = interp1d(
            [n_start, n_end], [td_start, td_end],
            kind='linear', bounds_error=False,
            fill_value=(td_start, td_end)
        )

        if st.checkbox("Show interpolation preview", key=f'{key_prefix}_show_prev'):
            x_prev = np.linspace(min(n_start, 0), n_end * 1.05, 300)
            fig_prev = go.Figure()
            fig_prev.add_trace(go.Scatter(
                x=x_prev, y=lin_interp(x_prev),
                mode='lines', line=dict(color='#16A34A', width=2.5)
            ))
            fig_prev.add_trace(go.Scatter(
                x=[n_start, n_end], y=[td_start, td_end],
                mode='markers', marker=dict(size=10, color='#16A34A', symbol='circle'),
                showlegend=False
            ))
            fig_prev.update_layout(
                xaxis=dict(title=dict(text='N density (atoms/cm3)', font=dict(size=FS-4)),
                           tickfont=dict(size=FS-6), exponentformat='e'),
                yaxis=dict(title=dict(text='Total density (atoms/cm3)', font=dict(size=FS-4)),
                           tickfont=dict(size=FS-6), exponentformat='e'),
                height=240, margin=dict(t=10, b=10), font=dict(size=FS-4),
                showlegend=False
            )
            st.plotly_chart(fig_prev)

        label = preset_name if preset is not None else f"Linear ({n_start:.2e}→{n_end:.2e})"
        return lin_interp, label


def _single_result_figure(depth, n_conc, converted, depth_unit, y_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=depth, y=n_conc,
        mode='lines', name='N conc. (atoms/cm3)',
        line=dict(color='#2563EB', width=2.5), yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=depth, y=converted,
        mode='lines', name=y_label,
        line=dict(color='#D97706', width=2.5, dash='dash'), yaxis='y2'
    ))
    fig.update_layout(
        xaxis=dict(title=dict(text=f'Depth ({depth_unit})', font=dict(size=FS)),
                   tickfont=dict(size=FS)),
        yaxis=dict(title=dict(text='N concentration (atoms/cm3)', font=dict(size=FS)),
                   tickfont=dict(size=FS), color='#2563EB', exponentformat='e'),
        yaxis2=dict(title=dict(text=y_label, font=dict(size=FS)),
                    tickfont=dict(size=FS), color='#D97706',
                    overlaying='y', side='right'),
        height=500, hovermode='x unified',
        font=dict(size=FS), legend=dict(font=dict(size=FS)),
        margin=dict(t=25)
    )
    return fig


def _comparison_figure(profiles, depth_unit, y_label,
                       font_size=22, legend_pos='top-left',
                       black_font=False, show_grid=False,
                       plot_mode='lines', line_width=2.5, marker_size=8,
                       highlight_idx=-1):

    LEGEND_MAP = {
        'top-left':     dict(x=0.01, y=0.99, xanchor='left',  yanchor='top',    orientation='v'),
        'top-right':    dict(x=0.99, y=0.99, xanchor='right', yanchor='top',    orientation='v'),
        'bottom-left':  dict(x=0.01, y=0.01, xanchor='left',  yanchor='bottom', orientation='v'),
        'bottom-right': dict(x=0.99, y=0.01, xanchor='right', yanchor='bottom', orientation='v'),
        'above':        dict(x=0.5,  y=1.02, xanchor='center', yanchor='bottom', orientation='h'),
        'below':        dict(x=0.5,  y=-0.18, xanchor='center', yanchor='top',  orientation='h'),
    }

    font_color = 'black' if black_font else None
    fig = go.Figure()

    for i, (label, depth, converted) in enumerate(profiles):
        is_highlight = (i == highlight_idx)
        color = COLORS[i % len(COLORS)]

        if highlight_idx >= 0:
            opacity = 1.0 if is_highlight else 0.5
            lw      = line_width * 2 if is_highlight else line_width
            ms      = marker_size * 1.6 if is_highlight else marker_size * 0.7
        else:
            opacity, lw, ms = 1.0, line_width, marker_size

        fig.add_trace(go.Scatter(
            x=depth, y=converted,
            mode=plot_mode, name=label,
            opacity=opacity,
            line=dict(color=color, width=lw),
            marker=dict(color=color, size=ms)
        ))

    axis_style = dict(showgrid=show_grid, gridcolor='lightgrey', gridwidth=1)
    font_dict = dict(size=font_size, color=font_color) if font_color else dict(size=font_size)
    legend_cfg = dict(font=dict(size=font_size, color=font_color),
                      **LEGEND_MAP.get(legend_pos, LEGEND_MAP['top-left']))

    bottom_margin = 120 if legend_pos == 'below' else 25

    fig.update_layout(
        xaxis=dict(title=dict(text=f'Depth ({depth_unit})',
                              font=dict(size=font_size, color=font_color)),
                   tickfont=dict(size=font_size, color=font_color), **axis_style),
        yaxis=dict(title=dict(text=y_label,
                              font=dict(size=font_size, color=font_color)),
                   tickfont=dict(size=font_size, color=font_color),
                   exponentformat='e', **axis_style),
        height=560, hovermode='x unified',
        font=font_dict, legend=legend_cfg,
        margin=dict(t=50, b=bottom_margin),
        plot_bgcolor='white'
    )
    return fig


def _db_model_figure(interp_obj):
    meta   = interp_obj.get('metadata', {})
    get_td = interp_obj['total_density']
    get_std = interp_obj.get('total_std')

    cal_x  = np.array(meta.get('calibration_x',  []))
    cal_y  = np.array(meta.get('calibration_y',  []))
    cal_xe = np.array(meta.get('calibration_x_std', np.zeros_like(cal_x)))
    cal_ye = np.array(meta.get('calibration_y_std', np.zeros_like(cal_y)))

    fig = go.Figure()

    if len(cal_x):
        fig.add_trace(go.Scatter(
            x=cal_x, y=cal_y,
            error_x=dict(type='data', array=cal_xe, visible=True),
            error_y=dict(type='data', array=cal_ye, visible=True),
            mode='markers', name='Calibration data',
            marker=dict(color='#2563EB', size=7)
        ))

        x_dense = np.linspace(cal_x.min(), cal_x.max(), 800)
        y_dense = get_td(x_dense)
        fig.add_trace(go.Scatter(
            x=x_dense, y=y_dense,
            mode='lines', name='Cubic spline',
            line=dict(color='#D97706', width=2.5)
        ))

        if get_std is not None:
            std_dense = get_std(x_dense)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_dense, x_dense[::-1]]),
                y=np.concatenate([y_dense + std_dense, (y_dense - std_dense)[::-1]]),
                fill='toself', fillcolor='rgba(217,119,6,0.15)',
                line=dict(color='rgba(0,0,0,0)'),
                name='±STD band'
            ))

    fig.update_layout(
        xaxis=dict(title=dict(text=f"N density ({meta.get('x_units','atoms/cm3')})",
                              font=dict(size=FS)),
                   tickfont=dict(size=FS), exponentformat='e'),
        yaxis=dict(title=dict(text=f"Total density ({meta.get('y_units','atoms/cm3')})",
                              font=dict(size=FS)),
                   tickfont=dict(size=FS), exponentformat='e'),
        height=500, hovermode='x unified',
        font=dict(size=FS), legend=dict(font=dict(size=FS)),
        margin=dict(t=25)
    )
    return fig


def concentration_converter_interface():
    st.markdown("### 🔄 Concentration → Atomic Fraction Converter")

    if 'comp_profiles' not in st.session_state:
        st.session_state['comp_profiles'] = []

    col_xu, col_yu = st.columns(2)
    with col_xu:
        depth_unit = st.text_input("Depth unit label", value="A", key='conv_depth_unit')
    with col_yu:
        output_type = st.radio("Output type",
                               ["Atomic fraction (0-1)", "at.% (0-100)"],
                               horizontal=True, key='conv_output_type')
    y_label = "Atomic fraction" if output_type.startswith("Atomic") else "at.%"
    scale   = 1.0             if output_type.startswith("Atomic") else 100.0

    st.markdown("#### 1️⃣  Upload depth profile")
    col_up, col_paste = st.columns(2)
    with col_up:
        up_file = st.file_uploader(
            "Profile file (.txt / .csv / .dat / .xy)",
            type=['txt', 'csv', 'dat', 'xy'], key='conv_upload'
        )
    with col_paste:
        pasted = st.text_area("…or paste data here", height=110, key='conv_paste')

    raw_text = None
    if up_file is not None:
        raw_text = up_file.read().decode('utf-8')
    elif pasted.strip():
        raw_text = pasted

    if raw_text is None:
        st.info("Upload or paste a two-column depth-profile file to continue.")
        _, _, tab3 = st.tabs(["📈 Single conversion", "📊 Model comparison", "🔍 Inspect DB model"])
        with tab3:
            _tab3_content()
        return

    try:
        depth, n_conc = _parse_two_column_file(raw_text)
    except ValueError as e:
        st.error(f"Could not parse file: {e}")
        return

    st.success(f"Loaded **{len(depth)}** data points.")
    n_conc_max = float(np.nanmax(n_conc))

    tab1, tab2, tab3 = st.tabs(["📈 Single conversion", "📊 Model comparison", "🔍 Inspect DB model"])

    with tab1:
        st.markdown("#### 2️⃣  Total density source")
        total_density_func, label_suggestion = _density_selector(n_conc_max, 't1')

        if total_density_func is not None:
            converted = _convert(n_conc, total_density_func) * scale

            nan_count = int(np.sum(np.isnan(converted)))
            if nan_count:
                st.warning(f"{nan_count} point(s) outside interpolation range → NaN.")

            st.markdown("#### 3️⃣  Preview")
            st.plotly_chart(
                _single_result_figure(depth, n_conc, converted, depth_unit, y_label),
            )

            st.markdown("#### 4️⃣  Add to comparison")
            col_lbl, col_btn = st.columns([3, 1])
            with col_lbl:
                label = st.text_input("Legend label", value=label_suggestion,
                                      key='t1_add_label')
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("➕ Add to comparison", type='primary'):
                    st.session_state['comp_profiles'].append(
                        (label, depth.copy(), converted.copy())
                    )
                    st.success(f"Added **{label}** — switch to **Model comparison** tab.")

            st.markdown("#### 5️⃣  Download this conversion")
            out_df = pd.DataFrame({
                f'depth_{depth_unit}': depth,
                'N_conc_atoms_per_cm3': n_conc,
                y_label.replace(' ', '_').replace('%', 'pct'): converted,
            })
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button("📥 Download as CSV",
                                   data=out_df.to_csv(index=False).encode(),
                                   file_name="converted_profile.csv",
                                   mime="text/csv")
            with col_dl2:
                xy_str = "\n".join(f"{d:.6e}\t{v:.6e}" for d, v in zip(depth, converted))
                st.download_button("📥 Download as .xy",
                                   data=xy_str.encode(),
                                   file_name="converted_profile.xy",
                                   mime="text/plain")

    with tab2:
        profiles = st.session_state['comp_profiles']
        if not profiles:
            st.info("No models added yet. Convert on **Tab 1** and click **➕ Add to comparison**.")
        else:
            st.markdown(f"**{len(profiles)} model(s) in comparison:**")
            for i, (lbl, _, _) in enumerate(profiles):
                col_lbl, col_rm = st.columns([6, 1])
                with col_lbl:
                    st.write(f"{i+1}. {lbl}")
                with col_rm:
                    if st.button("🗑️", key=f'rm_{i}'):
                        st.session_state['comp_profiles'].pop(i)
                        st.rerun()

            with st.expander("🎨  Plot style", expanded=False):
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    cmp_fs     = st.number_input("Font size", min_value=8, max_value=40,
                                                 value=22, step=1, key='cmp_fontsize')
                    cmp_legend = st.selectbox(
                        "Legend position",
                        ['top-left', 'top-right', 'bottom-left', 'bottom-right', 'above', 'below'],
                        key='cmp_legend_pos'
                    )
                with sc2:
                    cmp_black_font = st.checkbox("Black font", value=False, key='cmp_black_font')
                    cmp_grid       = st.checkbox("Show grid",  value=False, key='cmp_grid')
                with sc3:
                    cmp_mode_label = st.radio("Plot style", ["Lines", "Points", "Lines + Points"],
                                              horizontal=True, key='cmp_plot_mode_radio')
                    _mode_map = {"Lines": "lines", "Points": "markers",
                                 "Lines + Points": "lines+markers"}
                    cmp_mode = _mode_map[cmp_mode_label]
                    if cmp_mode_label == "Points":
                        cmp_marker_size = st.slider("Marker size", min_value=2, max_value=20,
                                                    value=8, step=1, key='cmp_marker_size')
                        cmp_line_width  = 2.5
                    else:
                        cmp_line_width  = st.slider("Line width", min_value=1, max_value=8,
                                                    value=3, step=1, key='cmp_line_width')
                        cmp_marker_size = 8

            profile_labels = [lbl for lbl, _, _ in profiles]
            cmp_highlight  = st.selectbox("Highlight curve:", ["None"] + profile_labels,
                                          key='cmp_highlight')
            highlight_idx  = profile_labels.index(cmp_highlight) if cmp_highlight != "None" else -1

            col_xt, col_yt = st.columns(2)
            with col_xt:
                cmp_xlabel = st.text_input("X axis title", value=f"Depth ({depth_unit})",
                                           key='cmp_xlabel')
            with col_yt:
                cmp_ylabel = st.text_input("Y axis title", value=y_label,
                                           key='cmp_ylabel')

            st.plotly_chart(
                _comparison_figure(
                    profiles, cmp_xlabel, cmp_ylabel,
                    font_size=cmp_fs, legend_pos=cmp_legend,
                    black_font=cmp_black_font, show_grid=cmp_grid,
                    plot_mode=cmp_mode, line_width=cmp_line_width,
                    marker_size=cmp_marker_size, highlight_idx=highlight_idx,
                ),
            )

            import zipfile, io as _io
            zip_buf = _io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for lbl, dep, conv in profiles:
                    safe = re.sub(r'[^\w\-.]', '_', lbl)
                    xy = "\n".join(f"{d:.6e}\t{v:.6e}" for d, v in zip(dep, conv))
                    zf.writestr(f"{safe}.xy", xy)
            zip_buf.seek(0)

            col_clr, col_zip = st.columns(2)
            with col_clr:
                if st.button("🗑️  Clear all"):
                    st.session_state['comp_profiles'] = []
                    st.rerun()
            with col_zip:
                st.download_button("📦 Download all as ZIP",
                                   data=zip_buf,
                                   file_name="model_comparison.zip",
                                   mime="application/zip",
                                   type='primary')

    with tab3:
        _tab3_content()


def _tab3_content():
    st.markdown("#### 🔍 Inspect densities_db model")

    db_files = _list_db_files()
    if not db_files:
        st.warning(f"No .pkl files found in `{DENSITIES_DB_DIR}/`. "
                   "Run `save_interpolation_with_metadata.py` and commit the folder.")
        return

    chosen_file = st.selectbox("Select model to inspect:", db_files, key='t3_db_file')
    try:
        interp_obj = _load_db_interpolation(chosen_file)
        meta = interp_obj.get('metadata', {})
    except Exception as e:
        st.error(f"Failed to load: {e}")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**System**")
        st.write(f"Element: **{meta.get('implanted_element','?')}**")
        st.write(f"Matrix: **{meta.get('matrix_initial','?')} → {meta.get('matrix_final','?')}**")
    with col2:
        st.markdown("**Data range**")
        st.write(f"X: {meta.get('x_min',0):.3e} – {meta.get('x_max',0):.3e} {meta.get('x_units','')}")
        st.write(f"Y: {meta.get('y_min',0):.3e} – {meta.get('y_max',0):.3e} {meta.get('y_units','')}")
    with col3:
        st.markdown("**Provenance**")
        st.write(f"Points: **{meta.get('n_calibration_points','?')}**")
        st.write(f"Method: {meta.get('interpolation_kind','?')}")
        st.write(f"Created: {meta.get('created_on','?')}")

    if meta.get('description'):
        st.caption(meta['description'])

    st.plotly_chart(_db_model_figure(interp_obj))

    cal_x  = meta.get('calibration_x')
    cal_y  = meta.get('calibration_y')
    cal_xe = meta.get('calibration_x_std')
    cal_ye = meta.get('calibration_y_std')

    if cal_x and cal_y:
        with st.expander("📋 Raw calibration data"):
            cal_df = pd.DataFrame({
                f"N density ({meta.get('x_units','atoms/cm3')})": cal_x,
                "N density STD": cal_xe if cal_xe else np.zeros(len(cal_x)),
                f"Total density ({meta.get('y_units','atoms/cm3')})": cal_y,
                "Total density STD": cal_ye if cal_ye else np.zeros(len(cal_y)),
            })
            st.dataframe(cal_df)
            st.download_button("📥 Download calibration as CSV",
                               data=cal_df.to_csv(index=False).encode(),
                               file_name=f"{chosen_file.replace('.pkl','')}_calibration.csv",
                               mime="text/csv")
