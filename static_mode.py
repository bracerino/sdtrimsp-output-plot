import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import re


def parse_experimental_data(file_content, filename):
    lines = file_content.strip().split('\n')

    first_data_line = None
    for line in lines[:10]:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('//'):
            if re.search(r'\d', line):
                first_data_line = line
                break

    if not first_data_line:
        return None, "No data lines found"

    delimiters = [',', ';', '\t', ' ']
    best_delimiter = ' '
    max_parts = 0

    for delimiter in delimiters:
        if delimiter == ' ':
            parts = first_data_line.split()
        else:
            parts = first_data_line.split(delimiter)

        numeric_parts = []
        for part in parts:
            try:
                float(part.strip())
                numeric_parts.append(part.strip())
            except ValueError:
                pass

        if len(numeric_parts) == 2:
            best_delimiter = delimiter
            break
        elif len(numeric_parts) > max_parts:
            max_parts = len(numeric_parts)
            best_delimiter = delimiter

    data_points = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('//'):
            if best_delimiter == ' ':
                parts = line.split()
            else:
                parts = line.split(best_delimiter)

            if len(parts) >= 2:
                try:
                    x_val = float(parts[0].strip())
                    y_val = float(parts[1].strip())
                    data_points.append([x_val, y_val])
                except ValueError:
                    continue

    if not data_points:
        return None, "No valid data points found"

    df = pd.DataFrame(data_points, columns=['x', 'y'])
    return df, f"Successfully parsed {len(data_points)} points using delimiter '{best_delimiter}'"


def parse_static_damage_file(file_content, filename):
    css = '''
        <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.15rem !important;
            color: #1e3a8a !important;
            font-weight: 600 !important;
            margin: 0 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 20px !important;
        }

        .stTabs [data-baseweb="tab-list"] button {
            background-color: #f0f4ff !important;
            border-radius: 12px !important;
            padding: 8px 16px !important;
            transition: all 0.3s ease !important;
            border: none !important;
            color: #1e3a8a !important;
        }

        .stTabs [data-baseweb="tab-list"] button:hover {
            background-color: #dbe5ff !important;
            cursor: pointer;
        }

        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background-color: #e0e7ff !important;
            color: #1e3a8a !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 6px rgba(30, 58, 138, 0.3) !important;

            border-bottom: 4px solid #1e3a8a !important;
            border-radius: 12px 12px 0 0 !important;
        }

        .stTabs [data-baseweb="tab-list"] button:focus {
            outline: none !important;
        }
        </style>
        '''

    st.markdown(css, unsafe_allow_html=True)
    lines = file_content.strip().split('\n')

    elements_data = {}
    current_element = None
    current_data = []
    in_data_section = False
    header_columns = []

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        if line_stripped.startswith('# CPT.') or line_stripped.startswith('#Calculated Values'):
            if current_element and current_data:
                elements_data[current_element] = {
                    'data': current_data.copy(),
                    'header': header_columns.copy()
                }
            break

        if line_stripped.startswith('# DEPTH DISTRIBUTIONS') and 'COMPONENT' in line_stripped:
            if current_element and current_data:
                elements_data[current_element] = {
                    'data': current_data.copy(),
                    'header': header_columns.copy()
                }

            parts = line.split(':')
            if len(parts) >= 2:
                element_part = parts[1].strip()
                current_element = element_part.split()[0].strip()
                current_data = []
                header_columns = []
                in_data_section = False

        elif line_stripped.startswith('# DEPTH DISTRIBUTIONS') and 'ALL' in line_stripped:
            if current_element and current_data:
                elements_data[current_element] = {
                    'data': current_data.copy(),
                    'header': header_columns.copy()
                }
            current_element = None
            in_data_section = False
            continue

        elif current_element and line_stripped.startswith('#') and (
                'DEPTH' in line_stripped or 'DEPTH/LENGTH' in line_stripped) and 'STOPS' in line_stripped:
            header_parts = line_stripped.replace('#', '').split()
            header_columns = header_parts
            in_data_section = True
            continue

        elif in_data_section and current_element and line_stripped and not line_stripped.startswith('#'):
            if 'sum' in line_stripped.lower():
                in_data_section = False
                if current_element and current_data:
                    elements_data[current_element] = {
                        'data': current_data.copy(),
                        'header': header_columns.copy()
                    }
                continue

            try:
                parts = line_stripped.split()
                if len(parts) >= 2:
                    depth = float(parts[0])
                    stops = float(parts[1])
                    vacancies = float(parts[-1]) if len(parts) >= len(header_columns) else 0.0

                    if depth > 0:
                        current_data.append({
                            'depth': depth,
                            'stops': stops,
                            'vacancies': vacancies
                        })
            except (ValueError, IndexError) as e:
                continue

    if current_element and current_data:
        elements_data[current_element] = {
            'data': current_data.copy(),
            'header': header_columns.copy()
        }

    return elements_data


def calculate_processed_data_static(elements_data, column_name, fluence_per_cm2=None):
    all_depths = set()
    for element_info in elements_data.values():
        for point in element_info['data']:
            all_depths.add(point['depth'])
    all_depths = sorted(all_depths)

    depth_data = {depth: {} for depth in all_depths}
    for element, element_info in elements_data.items():
        for point in element_info['data']:
            depth = point['depth']
            depth_data[depth][element] = point[column_name]

    if len(all_depths) > 1:
        spacings = [all_depths[i + 1] - all_depths[i] for i in range(len(all_depths) - 1)]
        bin_width_angstrom = sum(spacings) / len(spacings)
    else:
        bin_width_angstrom = 1.0

    processed_data = {}

    for element in elements_data.keys():
        element_total = sum(
            point[column_name]
            for point in elements_data[element]['data']
        )

        processed_data[element] = []

        for depth in all_depths:
            value = depth_data[depth].get(element, 0)

            total_at_depth = sum(depth_data[depth].values())

            raw_value = value

            atomic_fraction = value / total_at_depth if total_at_depth > 0 else 0

            probability = value / element_total if element_total > 0 else 0

            density_per_angstrom = value / element_total if element_total > 0 else 0

            if fluence_per_cm2 is not None and element_total > 0:
                bin_width_cm = bin_width_angstrom * 1e-8
                concentration = (value / element_total) * (fluence_per_cm2 / bin_width_cm)
            else:
                concentration = 0

            processed_data[element].append({
                'depth': depth,
                'raw_value': raw_value,
                'atomic_fraction': atomic_fraction,
                'probability': probability,
                'density_per_angstrom': density_per_angstrom,
                'concentration': concentration
            })

    return processed_data


def smooth_data(x, y, method='savgol', window=11, poly_order=3, sigma=2.0):
    if len(y) < window:
        window = len(y) if len(y) % 2 == 1 else len(y) - 1
        if window < 3:
            return y

    if method == 'moving_average':
        if window % 2 == 0:
            window += 1
        half_window = window // 2
        smoothed = np.convolve(y, np.ones(window) / window, mode='same')

        for i in range(half_window):
            smoothed[i] = np.mean(y[:i + half_window + 1])
        for i in range(len(y) - half_window, len(y)):
            smoothed[i] = np.mean(y[i - half_window:])

        return smoothed

    elif method == 'savgol':
        if window % 2 == 0:
            window += 1
        if window > len(y):
            window = len(y) if len(y) % 2 == 1 else len(y) - 1
        if poly_order >= window:
            poly_order = window - 1

        try:
            return savgol_filter(y, window, poly_order)
        except:
            return y

    elif method == 'gaussian':
        return gaussian_filter1d(y, sigma)

    else:
        return y


def create_static_mode_interface():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Static Mode (Damage Profiles)")

    use_static_mode = st.sidebar.checkbox("Enable Static Mode", value=False)

    if use_static_mode:
        st.markdown("### 📊 Static Mode: Depth Profiles")

        with st.expander("ℹ️ Understanding the Different Metrics", expanded=False):
            st.markdown("""
            **Raw Values**: Direct values from the file (STOPS or VACANCIES counts)

            **Atomic Fraction**: At each depth, what fraction of all events are from this element?
            - Formula: `value_at_depth / total_all_elements_at_depth`
            - Sum across all elements at one depth = 1.0

            **Normalized Probability**: What is the probability of finding this ion at this depth?
            - Formula: `stops_at_depth / total_stops_for_element`
            - Integral over all depths = 1.0 for each element
            - **This is the most useful for comparing depth distributions!**

            **Density (ions/Å)**: Ion density per unit depth, normalized per implanted ion
            - Same as probability but with explicit depth units
            - Useful for calculating range parameters

            **Concentration (ions/cm³)**: Actual ion concentration (requires fluence input)
            - Formula: `probability × (fluence / bin_width)`
            - Only meaningful if you know your implantation fluence
            """)

        st.sidebar.info("Upload multiple depth_damage.dat or depth_proj.dat files and select elements/columns to plot")

        uploaded_static_files = st.sidebar.file_uploader(
            "Upload depth_damage.dat or depth_proj.dat files",
            type=['dat', 'txt'],
            accept_multiple_files=True,
            key="static_damage_files"
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Experimental/Comparison Data")

        uploaded_exp_files = st.sidebar.file_uploader(
            "Upload experimental data (2-column format)",
            type=['txt', 'csv', 'dat'],
            accept_multiple_files=True,
            key="static_exp_files",
            help="Upload 2-column data files (depth, value) for comparison"
        )

        experimental_data = []
        if uploaded_exp_files:
            for exp_file in uploaded_exp_files:
                exp_content = str(exp_file.read(), "utf-8")
                exp_df, exp_info = parse_experimental_data(exp_content, exp_file.name)
                if exp_df is not None:
                    experimental_data.append((exp_file.name, exp_df, exp_info))
                    st.sidebar.success(f"✅ {exp_file.name}: {exp_info}")
                else:
                    st.sidebar.error(f"❌ {exp_file.name}: {exp_info}")

        all_file_data = []
        if uploaded_static_files:
            for uploaded_file in uploaded_static_files:
                file_content = str(uploaded_file.read(), "utf-8")
                elements_data = parse_static_damage_file(file_content, uploaded_file.name)

                if elements_data:
                    all_file_data.append({
                        'filename': uploaded_file.name,
                        'elements_data': elements_data
                    })

                    sample_element = list(elements_data.keys())[0]
                    if len(elements_data[sample_element]['data']) > 1:
                        depths = [p['depth'] for p in elements_data[sample_element]['data']]
                        depths.sort()
                        spacings = [depths[i + 1] - depths[i] for i in range(len(depths) - 1)]
                        avg_spacing = sum(spacings) / len(spacings)
                        st.sidebar.info(f"📏 Detected bin width: {avg_spacing:.2f} Å")
                else:
                    st.error(f"❌ {uploaded_file.name}: No valid data found")

        if all_file_data or experimental_data:
            col_settings1, col_settings2 = st.columns(2)

            with col_settings1:
                plot_data_type = st.sidebar.radio(
                    "Data Type to Plot:",
                    ["Raw Values",
                     "Atomic Fractions",
                     "Normalized Probability",
                     "Density (ions/Å)",
                     "Concentration (ions/cm³)"],
                    index=2,
                    key="static_data_type",
                    help="Normalized Probability is recommended for comparing distributions"
                )

            with col_settings2:
                if plot_data_type == "Concentration (ions/cm³)":
                    fluence = st.sidebar.number_input(
                        "Fluence (ions/cm²):",
                        min_value=1e10,
                        max_value=1e18,
                        value=1e15,
                        format="%.2e",
                        help="Total number of ions implanted per cm²",
                        key="static_fluence"
                    )
                    st.sidebar.caption(f"= {fluence:.2e} ions/cm²")
                else:
                    fluence = None

            selected_data_to_plot = []

            if all_file_data:
                st.subheader("Select Elements and Columns to Plot")

                tab_names = [f"📄 {file_data['filename']}" for file_data in all_file_data]
                tabs = st.tabs(tab_names)

                for file_idx, (tab, file_data) in enumerate(zip(tabs, all_file_data)):
                    with tab:
                        available_elements = list(file_data['elements_data'].keys())

                        col1, col2 = st.columns([1, 1])

                        with col1:
                            selected_element = st.selectbox(
                                f"Select element",
                                options=['None'] + available_elements,
                                index=len(available_elements),
                                key=f"element_select_{file_idx}"
                            )

                        with col2:
                            if selected_element != 'None':
                                available_columns = ['STOPS', 'VACANCIES']

                                selected_column = st.selectbox(
                                    f"Select column to plot",
                                    options=available_columns,
                                    key=f"column_select_{file_idx}"
                                )
                            else:
                                selected_column = None

                        if selected_element != 'None' and selected_column:
                            label = st.text_input(
                                f"Label for plot legend",
                                value=f"{file_data['filename']} - {selected_element} - {selected_column}",
                                key=f"label_{file_idx}"
                            )

                            column_key = selected_column.lower()

                            processed_data = calculate_processed_data_static(
                                file_data['elements_data'],
                                column_key,
                                fluence
                            )
                            data_to_use = processed_data[selected_element]

                            selected_data_to_plot.append({
                                'filename': file_data['filename'],
                                'element': selected_element,
                                'column': selected_column,
                                'data': data_to_use,
                                'label': label
                            })
            elif experimental_data:
                st.info("📊 Experimental data loaded. Configure plot settings below to visualize.")

            if selected_data_to_plot or experimental_data:
                st.markdown("---")
                st.subheader("Plot Controls")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    plot_mode = st.radio("Plot Mode:", ["Lines", "Markers", "Lines+Markers"],
                                         key="static_plot_mode")

                with col2:
                    y_scale = st.radio("Y-axis Scale:", ["Linear", "Logarithmic"], key="static_y_scale")

                with col3:
                    x_unit = st.radio("Depth Unit:", ["Angstroms (Å)", "Nanometers (nm)"], key="static_x_unit")

                with col4:
                    enable_smoothing = st.checkbox("Enable Smoothing", value=False, key="enable_smooth")

                if enable_smoothing:
                    st.markdown("#### Smoothing Settings")

                    smooth_col1, smooth_col2, smooth_col3 = st.columns(3)

                    with smooth_col1:
                        smooth_method = st.selectbox(
                            "Smoothing Method:",
                            ["Savitzky-Golay", "Moving Average", "Gaussian"],
                            index=0,
                            key="smooth_method",
                            help="Savitzky-Golay: Best for preserving peak shapes | Moving Average: Simple smoothing | Gaussian: Smooth but can broaden peaks"
                        )

                    with smooth_col2:
                        if smooth_method == "Savitzky-Golay":
                            window_size = st.slider(
                                "Window Size:",
                                min_value=5,
                                max_value=51,
                                value=11,
                                step=2,
                                key="savgol_window",
                                help="Must be odd. Larger = more smoothing"
                            )
                        elif smooth_method == "Moving Average":
                            window_size = st.slider(
                                "Window Size:",
                                min_value=3,
                                max_value=51,
                                value=9,
                                step=2,
                                key="ma_window",
                                help="Must be odd. Larger = more smoothing"
                            )
                        else:
                            window_size = None

                    with smooth_col3:
                        if smooth_method == "Savitzky-Golay":
                            poly_order = st.slider(
                                "Polynomial Order:",
                                min_value=1,
                                max_value=5,
                                value=3,
                                key="poly_order",
                                help="Higher = captures more detail but less smoothing"
                            )
                        elif smooth_method == "Gaussian":
                            sigma = st.slider(
                                "Sigma (σ):",
                                min_value=0.5,
                                max_value=10.0,
                                value=2.0,
                                step=0.5,
                                key="gauss_sigma",
                                help="Standard deviation. Larger = more smoothing"
                            )
                        else:
                            poly_order = None
                            sigma = None

                st.markdown("---")

                st.markdown("#### 🎨 Simulation Data Styling")
                col_sim1, col_sim2 = st.columns(2)
                with col_sim1:
                    sim_line_width = st.slider(
                        "Simulation Line Width:",
                        min_value=1,
                        max_value=10,
                        value=3,
                        step=1,
                        key="sim_line_width"
                    )

                with col_sim2:
                    sim_marker_size = st.slider(
                        "Simulation Marker Size:",
                        min_value=3,
                        max_value=15,
                        value=6,
                        step=1,
                        key="sim_marker_size"
                    )

                st.markdown("---")

                if experimental_data:
                    st.markdown("#### 🎨 Experimental Data Styling")

                    exp_plot_mode = st.selectbox(
                        "Experimental Plot Style:",
                        ["Markers", "Lines", "Lines+Markers"],
                        index=0,
                        key="exp_plot_mode"
                    )

                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        exp_marker_size = st.slider(
                            "Marker Size:",
                            min_value=4,
                            max_value=20,
                            value=8,
                            step=1,
                            key="exp_marker_size"
                        )

                    with col_exp2:
                        exp_line_width = st.slider(
                            "Line Width:",
                            min_value=1,
                            max_value=10,
                            value=3,
                            step=1,
                            key="exp_line_width"
                        )

                    st.markdown("---")
                else:
                    exp_plot_mode = "Markers"
                    exp_marker_size = 8
                    exp_line_width = 3

                fig = go.Figure()

                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'lime']

                mode_map = {
                    "Lines": "lines",
                    "Markers": "markers",
                    "Lines+Markers": "lines+markers"
                }
                plot_mode_value = mode_map[plot_mode]

                if selected_data_to_plot and len(selected_data_to_plot) > 0:
                    if plot_data_type == "Raw Values":
                        y_key = 'raw_value'
                        y_label = f"{selected_data_to_plot[0]['column']} [counts]"
                        title_text = f"Depth Profiles: {selected_data_to_plot[0]['column']} vs DEPTH"
                    elif plot_data_type == "Atomic Fractions":
                        y_key = 'atomic_fraction'
                        y_label = "Atomic Fraction (at depth)"
                        title_text = "Depth Profiles: Atomic Fractions vs DEPTH"
                    elif plot_data_type == "Normalized Probability":
                        y_key = 'probability'
                        y_label = "Probability (normalized)"
                        title_text = "Depth Profiles: Normalized Probability Distribution"
                    elif plot_data_type == "Density (ions/Å)":
                        y_key = 'density_per_angstrom'
                        y_label = "Density (ions/Å per implanted ion)"
                        title_text = "Depth Profiles: Ion Density vs DEPTH"
                    else:
                        y_key = 'concentration'
                        y_label = "Concentration (ions/cm³)"
                        title_text = f"Depth Profiles: Concentration vs DEPTH (Fluence = {fluence:.2e} ions/cm²)"
                else:
                    y_key = None
                    y_label = "Intensity / Probability"
                    title_text = "Depth Profiles: Experimental Data"

                method_map = {
                    "Savitzky-Golay": "savgol",
                    "Moving Average": "moving_average",
                    "Gaussian": "gaussian"
                }

                for idx, data_item in enumerate(selected_data_to_plot):
                    depths = np.array([point['depth'] for point in data_item['data']])
                    y_values = np.array([point[y_key] for point in data_item['data']])

                    if x_unit == "Nanometers (nm)":
                        depths = depths / 10.0

                    color = colors[idx % len(colors)]

                    if enable_smoothing:
                        method_key = method_map[smooth_method]

                        if method_key == "savgol":
                            y_smoothed = smooth_data(depths, y_values, method='savgol',
                                                     window=window_size, poly_order=poly_order)
                        elif method_key == "moving_average":
                            y_smoothed = smooth_data(depths, y_values, method='moving_average',
                                                     window=window_size)
                        elif method_key == "gaussian":
                            y_smoothed = smooth_data(depths, y_values, method='gaussian',
                                                     sigma=sigma)

                        fig.add_trace(go.Scatter(
                            x=depths,
                            y=y_values,
                            mode='lines',
                            name=f"{data_item['label']} (original)",
                            line=dict(color=color, width=1, dash='dot'),
                            opacity=0.3,
                            showlegend=True,
                            legendgroup=f"group{idx}",
                        ))

                        fig.add_trace(go.Scatter(
                            x=depths,
                            y=y_smoothed,
                            mode=plot_mode_value,
                            name=f"{data_item['label']} (smoothed)",
                            line=dict(color=color, width=sim_line_width),
                            marker=dict(size=sim_marker_size, color=color),
                            showlegend=True,
                            legendgroup=f"group{idx}",
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=depths,
                            y=y_values,
                            mode=plot_mode_value,
                            name=data_item['label'],
                            line=dict(color=color, width=sim_line_width),
                            marker=dict(size=sim_marker_size, color=color)
                        ))

                if experimental_data:
                    exp_colors = ['darkgreen', 'darkred', 'darkblue', 'darkorange', 'darkviolet']
                    for i, (exp_name, exp_df, exp_info) in enumerate(experimental_data):
                        exp_x = exp_df['x'].values
                        exp_y = exp_df['y'].values

                        if x_unit == "Nanometers (nm)":
                            exp_x = exp_x / 10.0

                        exp_mode_map = {
                            "Markers": "markers",
                            "Lines": "lines",
                            "Lines+Markers": "lines+markers"
                        }
                        exp_plot_mode_value = exp_mode_map.get(exp_plot_mode, "markers")

                        fig.add_trace(go.Scatter(
                            x=exp_x,
                            y=exp_y,
                            mode=exp_plot_mode_value,
                            name=f"Exp: {exp_name}",
                            marker=dict(size=exp_marker_size, color=exp_colors[i % len(exp_colors)], symbol='diamond'),
                            line=dict(width=exp_line_width, color=exp_colors[i % len(exp_colors)]),
                            showlegend=True
                        ))

                x_label = "Depth (nm)" if x_unit == "Nanometers (nm)" else "Depth (Å)"

                fig.update_layout(
                    title=dict(text=title_text, font=dict(size=28, color='black')),
                    xaxis_title=dict(text=x_label, font=dict(size=24, color='black')),
                    yaxis_title=dict(text=y_label, font=dict(size=24, color='black')),
                    yaxis_type="log" if y_scale == "Logarithmic" else "linear",
                    height=650,
                    hovermode='x unified',
                    font=dict(size=20, color='black'),
                    legend=dict(font=dict(size=16, color='black')),
                    xaxis=dict(tickfont=dict(size=20, color='black')),
                    yaxis=dict(tickfont=dict(size=20, color='black'))
                )

                st.plotly_chart(fig, width='stretch')

                if st.checkbox("Show Data Table", key="static_show_table"):
                    st.subheader("Selected Data")

                    if selected_data_to_plot:
                        for data_item in selected_data_to_plot:
                            with st.expander(f"📊 {data_item['label']}", expanded=False):
                                df_display = pd.DataFrame(data_item['data'])

                                if x_unit == "Nanometers (nm)":
                                    df_display['depth_nm'] = df_display['depth'] / 10.0
                                    depth_col = 'depth_nm'
                                    depth_label = 'Depth (nm)'
                                else:
                                    depth_col = 'depth'
                                    depth_label = 'Depth (Å)'

                                if plot_data_type == "Raw Values":
                                    df_display = df_display[[depth_col, 'raw_value']]
                                    df_display.columns = [depth_label, f'{data_item["column"]} (counts)']
                                elif plot_data_type == "Atomic Fractions":
                                    df_display = df_display[[depth_col, 'atomic_fraction']]
                                    df_display.columns = [depth_label, 'Atomic Fraction']
                                elif plot_data_type == "Normalized Probability":
                                    df_display = df_display[[depth_col, 'probability']]
                                    df_display.columns = [depth_label, 'Probability']
                                elif plot_data_type == "Density (ions/Å)":
                                    df_display = df_display[[depth_col, 'density_per_angstrom']]
                                    df_display.columns = [depth_label, 'Density (ions/Å)']
                                else:
                                    df_display = df_display[[depth_col, 'concentration']]
                                    df_display.columns = [depth_label, 'Concentration (ions/cm³)']

                                st.dataframe(df_display, width='stretch')

                                csv = df_display.to_csv(index=False)
                                st.download_button(
                                    label=f"Download {data_item['label']} as CSV",
                                    data=csv,
                                    file_name=f"{data_item['filename']}_{data_item['element']}_{data_item['column']}_{plot_data_type.replace(' ', '_')}.csv",
                                    mime="text/csv",
                                    key=f"download_{data_item['filename']}_{data_item['element']}_{idx}"
                                )

                    if experimental_data:
                        if selected_data_to_plot:
                            st.markdown("---")
                        st.subheader("Experimental Data")
                        for exp_name, exp_df, exp_info in experimental_data:
                            with st.expander(f"📊 Exp: {exp_name}", expanded=False):
                                display_df = exp_df.copy()

                                if x_unit == "Nanometers (nm)":
                                    display_df['x'] = display_df['x'] / 10.0
                                    x_label_display = 'Depth (nm)'
                                else:
                                    x_label_display = 'Depth (Å)'

                                display_df.columns = [x_label_display, 'Value']
                                st.dataframe(display_df, width='stretch')

                                csv = display_df.to_csv(index=False)
                                st.download_button(
                                    label=f"Download {exp_name} as CSV",
                                    data=csv,
                                    file_name=f"exp_{exp_name}",
                                    mime="text/csv",
                                    key=f"download_exp_{exp_name}"
                                )
            else:
                if all_file_data and not experimental_data:
                    st.info("👆 Select elements and columns from the files above to plot")
                elif not all_file_data and not experimental_data:
                    st.info("👆 Upload simulation files or experimental data to start plotting")
        else:
            st.info("👆 Upload simulation files and/or experimental data in the sidebar to begin")

        return True

    return False
