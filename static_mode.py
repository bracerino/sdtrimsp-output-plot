import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def parse_static_damage_file(file_content, filename):
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


def calculate_atomic_data_static(elements_data, target_density, column_name):
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

    processed_data = {}
    for element in elements_data.keys():
        processed_data[element] = []

    for depth in all_depths:
        total_value = sum(depth_data[depth].values())

        for element in elements_data.keys():
            value = depth_data[depth].get(element, 0)

            if total_value > 0:
                atomic_fraction = value / total_value
            else:
                atomic_fraction = 0

            concentration = atomic_fraction * target_density * 1e24

            processed_data[element].append({
                'depth': depth,
                'raw_value': value,
                'atomic_fraction': atomic_fraction,
                'concentration': concentration
            })

    return processed_data


def create_static_mode_interface():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Static Mode (Damage Profiles)")

    use_static_mode = st.sidebar.checkbox("Enable Static Mode", value=False)

    if use_static_mode:
        st.markdown("### 📊 Static Mode: Damage Depth Profiles")
        st.info("Upload multiple depth_damage.dat or depth_proj.dat files and select elements/columns to plot")

        uploaded_static_files = st.file_uploader(
            "Upload depth_damage.dat or depth_proj.dat files",
            type=['dat', 'txt'],
            accept_multiple_files=True,
            key="static_damage_files"
        )

        if uploaded_static_files:
            all_file_data = []

            for uploaded_file in uploaded_static_files:
                file_content = str(uploaded_file.read(), "utf-8")
                elements_data = parse_static_damage_file(file_content, uploaded_file.name)

                if elements_data:
                    all_file_data.append({
                        'filename': uploaded_file.name,
                        'elements_data': elements_data
                    })
                    st.success(f"✅ {uploaded_file.name}: Found elements: {', '.join(elements_data.keys())}")
                else:
                    st.error(f"❌ {uploaded_file.name}: No valid data found")

            if all_file_data:
                st.markdown("---")
                st.subheader("Plot Settings")

                col_settings1, col_settings2 = st.columns(2)

                with col_settings1:
                    plot_data_type = st.radio(
                        "Data Type to Plot:",
                        ["Raw Values", "Atomic Fractions", "Concentrations (atoms/cm³)"],
                        key="static_data_type"
                    )

                with col_settings2:
                    if plot_data_type == "Concentrations (atoms/cm³)":
                        target_density = st.number_input(
                            "Target Density (atoms/Ų):",
                            min_value=0.001,
                            max_value=1.0,
                            value=0.0565,
                            step=0.001,
                            format="%.4f",
                            help="Typical values: Ti~0.0565, Si~0.05, etc.",
                            key="static_density"
                        )
                    else:
                        target_density = 0.0565

                st.markdown("---")
                st.subheader("Select Elements and Columns to Plot")

                selected_data_to_plot = []

                for file_idx, file_data in enumerate(all_file_data):
                    with st.expander(f"📄 {file_data['filename']}", expanded=True):
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

                            if plot_data_type != "Raw Values":
                                processed_data = calculate_atomic_data_static(
                                    file_data['elements_data'],
                                    target_density,
                                    column_key
                                )
                                data_to_use = processed_data[selected_element]
                            else:
                                data_to_use = [
                                    {
                                        'depth': point['depth'],
                                        'raw_value': point.get(column_key, 0)
                                    }
                                    for point in file_data['elements_data'][selected_element]['data']
                                ]

                            selected_data_to_plot.append({
                                'filename': file_data['filename'],
                                'element': selected_element,
                                'column': selected_column,
                                'data': data_to_use,
                                'label': label
                            })

                if selected_data_to_plot:
                    st.markdown("---")
                    st.subheader("Plot Controls")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        plot_mode = st.radio("Plot Mode:", ["Lines", "Markers", "Lines+Markers"],
                                             key="static_plot_mode")

                    with col2:
                        y_scale = st.radio("Y-axis Scale:", ["Linear", "Logarithmic"], key="static_y_scale")

                    with col3:
                        x_unit = st.radio("Depth Unit:", ["Angstroms (Å)", "Nanometers (nm)"], key="static_x_unit")

                    st.markdown("---")

                    fig = go.Figure()

                    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'lime']

                    mode_map = {
                        "Lines": "lines",
                        "Markers": "markers",
                        "Lines+Markers": "lines+markers"
                    }
                    plot_mode_value = mode_map[plot_mode]

                    if plot_data_type == "Raw Values":
                        y_key = 'raw_value'
                        y_label = f"{selected_data_to_plot[0]['column']} [number]"
                        title_text = f"Depth Profiles: {selected_data_to_plot[0]['column']} vs DEPTH"
                    elif plot_data_type == "Atomic Fractions":
                        y_key = 'atomic_fraction'
                        y_label = "Atomic Fraction"
                        title_text = "Depth Profiles: Atomic Fractions vs DEPTH"
                    else:
                        y_key = 'concentration'
                        y_label = "Concentration (atoms/cm³)"
                        title_text = f"Depth Profiles: Concentrations vs DEPTH (ρ={target_density:.4f} atoms/Ų)"

                    for idx, data_item in enumerate(selected_data_to_plot):
                        depths = [point['depth'] for point in data_item['data']]
                        y_values = [point[y_key] for point in data_item['data']]

                        if x_unit == "Nanometers (nm)":
                            depths = [d / 10.0 for d in depths]

                        color = colors[idx % len(colors)]

                        fig.add_trace(go.Scatter(
                            x=depths,
                            y=y_values,
                            mode=plot_mode_value,
                            name=data_item['label'],
                            line=dict(color=color, width=3),
                            marker=dict(size=6, color=color)
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
                        legend=dict(font=dict(size=18, color='black')),
                        xaxis=dict(tickfont=dict(size=20, color='black')),
                        yaxis=dict(tickfont=dict(size=20, color='black'))
                    )

                    st.plotly_chart(fig, width='stretch')

                    if st.checkbox("Show Data Table", key="static_show_table"):
                        st.subheader("Selected Data")

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
                                    df_display.columns = [depth_label, data_item['column']]
                                elif plot_data_type == "Atomic Fractions":
                                    df_display = df_display[[depth_col, 'atomic_fraction']]
                                    df_display.columns = [depth_label, 'Atomic Fraction']
                                else:
                                    df_display = df_display[[depth_col, 'concentration']]
                                    df_display.columns = [depth_label, 'Concentration (atoms/cm³)']

                                st.dataframe(df_display, width='stretch')

                                csv = df_display.to_csv(index=False)
                                st.download_button(
                                    label=f"Download {data_item['label']} as CSV",
                                    data=csv,
                                    file_name=f"{data_item['filename']}_{data_item['element']}_{data_item['column']}.csv",
                                    mime="text/csv",
                                    key=f"download_{data_item['filename']}_{data_item['element']}_{idx}"
                                )
                else:
                    st.info("👆 Select elements and columns from the files above to plot")

        return True

    return False
