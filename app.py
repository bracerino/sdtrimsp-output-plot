import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import numpy as np


def parse_experimental_data(file_content, filename):
    """Parse experimental data from various delimited formats"""
    import pandas as pd
    import re

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


def create_single_fluence_plots(df, depth_col, depth_label, plot_type, mode, y_axis_scale, selected_fluence,
                                element_names, smooth_data, selected_elements, experimental_data=None):
    if plot_type == "Atomic Fractions":
        fig = go.Figure()
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']

        for i, element in enumerate(element_names):
            frac_col = f'{element}_frac_smooth' if smooth_data and f'{element}_frac_smooth' in df.columns else f'{element}_frac'
            if frac_col in df.columns:
                display_name = f"{element} (smoothed)" if smooth_data and f'{element}_frac_smooth' in df.columns else element
                fig.add_trace(go.Scatter(
                    x=df[depth_col], y=df[frac_col],
                    mode=mode, name=display_name,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=6, color=colors[i % len(colors)])
                ))

        if selected_elements and len(selected_elements) > 1:
            if smooth_data and all(f'{elem}_frac_smooth' in df.columns for elem in selected_elements):
                combined_frac = df[[f'{elem}_frac_smooth' for elem in selected_elements]].sum(axis=1)
                combined_name = f'Combined ({"+".join(selected_elements)}) (smoothed)'
            else:
                combined_frac = df[[f'{elem}_frac' for elem in selected_elements if f'{elem}_frac' in df.columns]].sum(
                    axis=1)
                combined_name = f'Combined ({"+".join(selected_elements)})'

            fig.add_trace(go.Scatter(
                x=df[depth_col], y=combined_frac,
                mode=mode, name=combined_name,
                line=dict(color='black', width=3, dash='dash'),
                marker=dict(size=6, color='black')
            ))

        if experimental_data:
            exp_colors = ['darkgreen', 'darkred', 'darkblue', 'darkorange', 'darkviolet']
            for i, (exp_name, exp_df, exp_info) in enumerate(experimental_data):
                fig.add_trace(go.Scatter(
                    x=exp_df['x'], y=exp_df['y'],
                    mode='markers', name=f"Exp: {exp_name}",
                    marker=dict(size=8, color=exp_colors[i % len(exp_colors)], symbol='diamond'),
                    showlegend=True
                ))

        fig.update_layout(
            title=dict(text=f"Atomic Fractions vs Depth (Fluence: {selected_fluence:.1f} atoms/A² = {selected_fluence:.1f} ×10¹⁶ atoms/cm²)",
                       font=dict(size=28, color='black')),
            xaxis_title=dict(text=depth_label, font=dict(size=24, color='black')),
            yaxis_title=dict(text="Atomic Fraction", font=dict(size=24, color='black')),
            yaxis_type="log" if y_axis_scale == "Logarithmic" else "linear",
            height=650,
            hovermode='x unified',
            font=dict(size=20, color='black'),
            legend=dict(font=dict(size=20, color='black')),
            xaxis=dict(tickfont=dict(size=20, color='black')),
            yaxis=dict(tickfont=dict(size=20, color='black'))
        )

    elif plot_type == "Concentrations (atoms/cm³)":
        fig = go.Figure()
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']

        for i, element in enumerate(element_names):
            conc_col = f'{element}_conc_smooth' if smooth_data and f'{element}_conc_smooth' in df.columns else f'{element}_conc'
            if conc_col in df.columns:
                display_name = f"{element} (smoothed)" if smooth_data and f'{element}_conc_smooth' in df.columns else element
                fig.add_trace(go.Scatter(
                    x=df[depth_col], y=df[conc_col],
                    mode=mode, name=display_name,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=6, color=colors[i % len(colors)])
                ))

        if selected_elements and len(selected_elements) > 1:
            if smooth_data and all(f'{elem}_conc_smooth' in df.columns for elem in selected_elements):
                combined_conc = df[[f'{elem}_conc_smooth' for elem in selected_elements]].sum(axis=1)
                combined_name = f'Combined ({"+".join(selected_elements)}) (smoothed)'
            else:
                combined_conc = df[[f'{elem}_conc' for elem in selected_elements if f'{elem}_conc' in df.columns]].sum(
                    axis=1)
                combined_name = f'Combined ({"+".join(selected_elements)})'

            fig.add_trace(go.Scatter(
                x=df[depth_col], y=combined_conc,
                mode=mode, name=combined_name,
                line=dict(color='black', width=3, dash='dash'),
                marker=dict(size=6, color='black')
            ))

        if experimental_data:
            exp_colors = ['darkgreen', 'darkred', 'darkblue', 'darkorange', 'darkviolet']
            for i, (exp_name, exp_df, exp_info) in enumerate(experimental_data):
                fig.add_trace(go.Scatter(
                    x=exp_df['x'], y=exp_df['y'],
                    mode='markers', name=f"Exp: {exp_name}",
                    marker=dict(size=8, color=exp_colors[i % len(exp_colors)], symbol='diamond'),
                    showlegend=True
                ))

        fig.update_layout(
            title=dict(text=f"Concentrations vs Depth (Fluence: {selected_fluence:.1f})",
                       font=dict(size=28, color='black')),
            xaxis_title=dict(text=depth_label, font=dict(size=24, color='black')),
            yaxis_title=dict(text="Concentration (atoms/cm³)", font=dict(size=24, color='black')),
            yaxis_type="log" if y_axis_scale == "Logarithmic" else "linear",
            height=650,
            hovermode='x unified',
            font=dict(size=20, color='black'),
            legend=dict(font=dict(size=20, color='black')),
            xaxis=dict(tickfont=dict(size=20, color='black')),
            yaxis=dict(tickfont=dict(size=20, color='black'))
        )

    else:
        fig = go.Figure()

        density_col = 'density_smooth' if smooth_data and 'density_smooth' in df.columns else 'density'
        display_name = "Total Density (smoothed)" if smooth_data and 'density_smooth' in df.columns else "Total Density"

        fig.add_trace(go.Scatter(
            x=df[depth_col], y=df[density_col],
            mode=mode, name=display_name,
            line=dict(color='purple', width=3),
            marker=dict(size=6, color='purple')
        ))

        if experimental_data:
            exp_colors = ['darkgreen', 'darkred', 'darkblue', 'darkorange', 'darkviolet']
            for i, (exp_name, exp_df, exp_info) in enumerate(experimental_data):
                fig.add_trace(go.Scatter(
                    x=exp_df['x'], y=exp_df['y'],
                    mode='markers', name=f"Exp: {exp_name}",
                    marker=dict(size=8, color=exp_colors[i % len(exp_colors)], symbol='diamond'),
                    showlegend=True
                ))

        fig.update_layout(
            title=dict(text=f"Density vs Depth (Fluence: {selected_fluence:.2e})", font=dict(size=28, color='black')),
            xaxis_title=dict(text=depth_label, font=dict(size=24, color='black')),
            yaxis_title=dict(text="Density (atoms/Ų)", font=dict(size=24, color='black')),
            yaxis_type="log" if y_axis_scale == "Logarithmic" else "linear",
            height=650,
            hovermode='x unified',
            font=dict(size=20, color='black'),
            legend=dict(font=dict(size=20, color='black')),
            xaxis=dict(tickfont=dict(size=20, color='black')),
            yaxis=dict(tickfont=dict(size=20, color='black'))
        )

    st.plotly_chart(fig, use_container_width=True)


def create_multi_fluence_comparison(fluence_data, selected_fluences, depth_col, depth_label, plot_type, mode,
                                    y_axis_scale, element_names, smooth_data, smooth_sigma, selected_elements):
    comparison_fig = go.Figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, fluence in enumerate(selected_fluences):
        data_comp = fluence_data[fluence]
        df_comp = pd.DataFrame(data_comp)

        if smooth_data:
            try:
                from scipy.ndimage import gaussian_filter1d

                for elem in element_names:
                    if f'{elem}_conc' in df_comp.columns:
                        df_comp[f'{elem}_conc_smooth'] = gaussian_filter1d(df_comp[f'{elem}_conc'], sigma=smooth_sigma)
                    if f'{elem}_frac' in df_comp.columns:
                        df_comp[f'{elem}_frac_smooth'] = gaussian_filter1d(df_comp[f'{elem}_frac'], sigma=smooth_sigma)

                if 'N_total_conc' in df_comp.columns:
                    df_comp['N_total_conc_smooth'] = gaussian_filter1d(df_comp['N_total_conc'], sigma=smooth_sigma)
                if 'N_total_frac' in df_comp.columns:
                    df_comp['N_total_frac_smooth'] = gaussian_filter1d(df_comp['N_total_frac'], sigma=smooth_sigma)
                if 'density' in df_comp.columns:
                    df_comp['density_smooth'] = gaussian_filter1d(df_comp['density'], sigma=smooth_sigma)
            except ImportError:
                pass

        color = colors[i % len(colors)]

        if plot_type == "Atomic Fractions":
            if selected_elements and len(selected_elements) > 1:
                if smooth_data and all(f'{elem}_frac_smooth' in df_comp.columns for elem in selected_elements):
                    combined_frac = df_comp[[f'{elem}_frac_smooth' for elem in selected_elements]].sum(axis=1)
                    display_name = f'Combined ({"+".join(selected_elements)}) (Fluence: {fluence:.1f}, smoothed)'
                else:
                    combined_frac = df_comp[
                        [f'{elem}_frac' for elem in selected_elements if f'{elem}_frac' in df_comp.columns]].sum(axis=1)
                    display_name = f'Combined ({"+".join(selected_elements)}) (Fluence: {fluence:.1f})'

                comparison_fig.add_trace(go.Scatter(
                    x=df_comp[depth_col], y=combined_frac,
                    mode=mode, name=display_name,
                    line=dict(color=color, width=3),
                    marker=dict(size=6, color=color)
                ))
            elif selected_elements and len(selected_elements) == 1:
                elem = selected_elements[0]
                if smooth_data and f'{elem}_frac_smooth' in df_comp.columns:
                    frac_data = df_comp[f'{elem}_frac_smooth']
                    display_name = f'{elem} (Fluence: {fluence:.1f}, smoothed)'
                else:
                    frac_data = df_comp[f'{elem}_frac']
                    display_name = f'{elem} (Fluence: {fluence:.1f})'

                comparison_fig.add_trace(go.Scatter(
                    x=df_comp[depth_col], y=frac_data,
                    mode=mode, name=display_name,
                    line=dict(color=color, width=3),
                    marker=dict(size=6, color=color)
                ))
            else:
                elem = element_names[0] if element_names else 'Ti'
                if smooth_data and f'{elem}_frac_smooth' in df_comp.columns:
                    frac_data = df_comp[f'{elem}_frac_smooth']
                    display_name = f'{elem} (Fluence: {fluence:.1f}, smoothed)'
                else:
                    frac_data = df_comp[f'{elem}_frac']
                    display_name = f'{elem} (Fluence: {fluence:.1f})'

                comparison_fig.add_trace(go.Scatter(
                    x=df_comp[depth_col], y=frac_data,
                    mode=mode, name=display_name,
                    line=dict(color=color, width=3),
                    marker=dict(size=6, color=color)
                ))

        elif plot_type == "Concentrations (atoms/cm³)":
            if selected_elements and len(selected_elements) > 1:
                if smooth_data and all(f'{elem}_conc_smooth' in df_comp.columns for elem in selected_elements):
                    combined_conc = df_comp[[f'{elem}_conc_smooth' for elem in selected_elements]].sum(axis=1)
                    display_name = f'Combined ({"+".join(selected_elements)}) (Fluence: {fluence:.1f}, smoothed)'
                else:
                    combined_conc = df_comp[
                        [f'{elem}_conc' for elem in selected_elements if f'{elem}_conc' in df_comp.columns]].sum(axis=1)
                    display_name = f'Combined ({"+".join(selected_elements)}) (Fluence: {fluence:.1f})'

                comparison_fig.add_trace(go.Scatter(
                    x=df_comp[depth_col], y=combined_conc,
                    mode=mode, name=display_name,
                    line=dict(color=color, width=3),
                    marker=dict(size=6, color=color)
                ))
            elif selected_elements and len(selected_elements) == 1:
                elem = selected_elements[0]
                if smooth_data and f'{elem}_conc_smooth' in df_comp.columns:
                    conc_data = df_comp[f'{elem}_conc_smooth']
                    display_name = f'{elem} (Fluence: {fluence:.1f}, smoothed)'
                else:
                    conc_data = df_comp[f'{elem}_conc']
                    display_name = f'{elem} (Fluence: {fluence:.1f})'

                comparison_fig.add_trace(go.Scatter(
                    x=df_comp[depth_col], y=conc_data,
                    mode=mode, name=display_name,
                    line=dict(color=color, width=3),
                    marker=dict(size=6, color=color)
                ))
            else:
                elem = element_names[0] if element_names else 'Ti'
                if smooth_data and f'{elem}_conc_smooth' in df_comp.columns:
                    conc_data = df_comp[f'{elem}_conc_smooth']
                    display_name = f'{elem} (Fluence: {fluence:.1f}, smoothed)'
                else:
                    conc_data = df_comp[f'{elem}_conc']
                    display_name = f'{elem} (Fluence: {fluence:.1f})'

                comparison_fig.add_trace(go.Scatter(
                    x=df_comp[depth_col], y=conc_data,
                    mode=mode, name=display_name,
                    line=dict(color=color, width=3),
                    marker=dict(size=6, color=color)
                ))

        else:
            if smooth_data and 'density_smooth' in df_comp.columns:
                density_data = df_comp['density_smooth']
                display_name = f'Density (Fluence: {fluence:.1f}, smoothed)'
            else:
                density_data = df_comp['density']
                display_name = f'Density (Fluence: {fluence:.1f})'

            comparison_fig.add_trace(go.Scatter(
                x=df_comp[depth_col], y=density_data,
                mode=mode, name=display_name,
                line=dict(color=color, width=3),
                marker=dict(size=6, color=color)
            ))

    y_title = "Atomic Fraction" if plot_type == "Atomic Fractions" else \
        "Concentration (atoms/cm³)" if plot_type == "Concentrations (atoms/cm³)" else "Density (atoms/Ų)"

    comparison_fig.update_layout(
        title=dict(text=f"Multi-Fluence Comparison: {plot_type}", font=dict(size=30, color='black')),
        xaxis_title=dict(text=depth_label, font=dict(size=26, color='black')),
        yaxis_title=dict(text=y_title, font=dict(size=26, color='black')),
        yaxis_type="log" if y_axis_scale == "Logarithmic" else "linear",
        height=650,
        hovermode='x unified',
        font=dict(size=22, color='black'),
        legend=dict(font=dict(size=20, color='black')),
        xaxis=dict(tickfont=dict(size=22, color='black')),
        yaxis=dict(tickfont=dict(size=22, color='black'))
    )

    st.plotly_chart(comparison_fig, use_container_width=True)


def perform_fluence_analysis(fluence_data, element_names, fluence_unit, selected_elements, smooth_data, smooth_sigma,
                             plot_type):
    st.subheader("📊 Fluence Analysis Results")

    fluence_values = sorted([f for f in fluence_data.keys() if f > 0])

    if fluence_unit == "atoms/cm²":
        fluence_display = [f * 1e16 for f in fluence_values]
        fluence_label = "Fluence (atoms/cm²)"
    else:
        fluence_display = fluence_values
        fluence_label = "Fluence (atoms/Ų)"

    analysis_results = {}

    elements_to_analyze = element_names.copy()

    if selected_elements and len(selected_elements) > 1:
        combined_name = f"Combined_{'_'.join(selected_elements)}"
        elements_to_analyze.append(combined_name)

    use_fractions = (plot_type == "Atomic Fractions")
    data_suffix = "_frac" if use_fractions else "_conc"
    data_label = "Atomic Fraction" if use_fractions else "Concentration (atoms/cm³)"
    metric_label = "Max Atomic Fraction" if use_fractions else "Max Concentration (atoms/cm³)"

    for element in elements_to_analyze:
        max_values = []
        max_depth_values = []
        fwhm_values = []

        for fluence in fluence_values:
            data = fluence_data[fluence]
            df = pd.DataFrame(data)

            if smooth_data:
                try:
                    from scipy.ndimage import gaussian_filter1d

                    for elem in element_names:
                        if f'{elem}_conc' in df.columns:
                            df[f'{elem}_conc_smooth'] = gaussian_filter1d(df[f'{elem}_conc'], sigma=smooth_sigma)
                        if f'{elem}_frac' in df.columns:
                            df[f'{elem}_frac_smooth'] = gaussian_filter1d(df[f'{elem}_frac'], sigma=smooth_sigma)

                    if element.startswith('Combined_') and selected_elements:
                        if use_fractions:
                            combined_data = df[
                                [f'{elem}_frac' for elem in selected_elements if f'{elem}_frac' in df.columns]].sum(
                                axis=1)
                        else:
                            combined_data = df[
                                [f'{elem}_conc' for elem in selected_elements if f'{elem}_conc' in df.columns]].sum(
                                axis=1)

                        df[f'{element}{data_suffix}'] = combined_data
                        df[f'{element}{data_suffix}_smooth'] = gaussian_filter1d(combined_data, sigma=smooth_sigma)

                except ImportError:
                    smooth_data = False

            if element.startswith('Combined_') and selected_elements:
                if smooth_data and all(f'{elem}{data_suffix}_smooth' in df.columns for elem in selected_elements):
                    combined_data = df[[f'{elem}{data_suffix}_smooth' for elem in selected_elements]].sum(axis=1)
                    data_col = f'{element}{data_suffix}_smooth'
                    df[data_col] = combined_data
                else:
                    combined_data = df[[f'{elem}{data_suffix}' for elem in selected_elements if
                                        f'{elem}{data_suffix}' in df.columns]].sum(axis=1)
                    data_col = f'{element}{data_suffix}'
                    df[data_col] = combined_data
            else:
                if smooth_data and f'{element}{data_suffix}_smooth' in df.columns:
                    data_col = f'{element}{data_suffix}_smooth'
                else:
                    data_col = f'{element}{data_suffix}'

            if data_col in df.columns:
                data_values = df[data_col].values
                depth_data = df['depth_A'].values

                max_idx = np.argmax(data_values)
                max_value = data_values[max_idx]
                max_depth = depth_data[max_idx]

                half_max = max_value / 2
                indices = np.where(data_values >= half_max)[0]
                if len(indices) > 1:
                    fwhm = depth_data[indices[-1]] - depth_data[indices[0]]
                else:
                    fwhm = 0

                max_values.append(max_value)
                max_depth_values.append(max_depth)
                fwhm_values.append(fwhm)
            else:
                max_values.append(0)
                max_depth_values.append(0)
                fwhm_values.append(0)

        analysis_results[element] = {
            'max_values': max_values,
            'max_depth': max_depth_values,
            'fwhm': fwhm_values
        }

    col1, col2, col3 = st.columns(3)

    with col1:
        smoothing_note = " (smoothed)" if smooth_data else ""
        st.write(f"**Maximum {data_label} vs Fluence{smoothing_note}**")
        fig_max = go.Figure()
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']

        for i, element in enumerate(elements_to_analyze):
            if element in analysis_results and any(val > 0 for val in analysis_results[element]['max_values']):
                display_name = element.replace('Combined_', 'Combined ').replace('_', '+')
                fig_max.add_trace(go.Scatter(
                    x=fluence_display, y=analysis_results[element]['max_values'],
                    mode='lines+markers', name=display_name,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=8, color=colors[i % len(colors)])
                ))

        y_axis_type = "linear" if use_fractions else "log"

        fig_max.update_layout(
            title=dict(text=f"Maximum {data_label} vs Fluence{smoothing_note}", font=dict(size=24, color='black')),
            xaxis_title=dict(text=fluence_label, font=dict(size=22, color='black')),
            yaxis_title=dict(text=metric_label, font=dict(size=22, color='black')),
            yaxis_type=y_axis_type,
            height=400,
            font=dict(size=18, color='black'),
            legend=dict(font=dict(size=18, color='black')),
            xaxis=dict(tickfont=dict(size=18, color='black')),
            yaxis=dict(tickfont=dict(size=18, color='black'))
        )
        st.plotly_chart(fig_max, use_container_width=True)

    with col2:
        st.write(f"**Depth of Maximum vs Fluence{smoothing_note}**")
        fig_depth = go.Figure()

        for i, element in enumerate(elements_to_analyze):
            if element in analysis_results and any(val > 0 for val in analysis_results[element]['max_depth']):
                display_name = element.replace('Combined_', 'Combined ').replace('_', '+')
                fig_depth.add_trace(go.Scatter(
                    x=fluence_display, y=analysis_results[element]['max_depth'],
                    mode='lines+markers', name=display_name,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=8, color=colors[i % len(colors)])
                ))

        fig_depth.update_layout(
            title=dict(text=f"Depth of Maximum vs Fluence{smoothing_note}", font=dict(size=24, color='black')),
            xaxis_title=dict(text=fluence_label, font=dict(size=22, color='black')),
            yaxis_title=dict(text="Depth of Maximum (Å)", font=dict(size=22, color='black')),
            height=400,
            font=dict(size=18, color='black'),
            legend=dict(font=dict(size=18, color='black')),
            xaxis=dict(tickfont=dict(size=18, color='black')),
            yaxis=dict(tickfont=dict(size=18, color='black'))
        )
        st.plotly_chart(fig_depth, use_container_width=True)

    with col3:
        st.write(f"**FWHM vs Fluence{smoothing_note}**")
        fig_fwhm = go.Figure()

        for i, element in enumerate(elements_to_analyze):
            if element in analysis_results and any(val > 0 for val in analysis_results[element]['fwhm']):
                display_name = element.replace('Combined_', 'Combined ').replace('_', '+')
                fig_fwhm.add_trace(go.Scatter(
                    x=fluence_display, y=analysis_results[element]['fwhm'],
                    mode='lines+markers', name=display_name,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=8, color=colors[i % len(colors)])
                ))

        fig_fwhm.update_layout(
            title=dict(text=f"FWHM vs Fluence{smoothing_note}", font=dict(size=24, color='black')),
            xaxis_title=dict(text=fluence_label, font=dict(size=22, color='black')),
            yaxis_title=dict(text="FWHM (Å)", font=dict(size=22, color='black')),
            height=400,
            font=dict(size=18, color='black'),
            legend=dict(font=dict(size=18, color='black')),
            xaxis=dict(tickfont=dict(size=18, color='black')),
            yaxis=dict(tickfont=dict(size=18, color='black'))
        )
        st.plotly_chart(fig_fwhm, use_container_width=True)

    st.write(f"**Summary Table{smoothing_note}**")
    summary_data = []
    for element in elements_to_analyze:
        if element in analysis_results:
            display_name = element.replace('Combined_', 'Combined ').replace('_', '+')
            for i, fluence in enumerate(fluence_values):
                if i < len(analysis_results[element]['max_values']):
                    max_val = analysis_results[element]['max_values'][i]
                    max_val_str = f"{max_val:.3f}" if use_fractions else f"{max_val:.2e}"

                    summary_data.append({
                        'Element': display_name,
                        'Fluence': f"{fluence_display[i]:.2e}",
                        metric_label: max_val_str,
                        'Depth of Max (Å)': f"{analysis_results[element]['max_depth'][i]:.1f}",
                        'FWHM (Å)': f"{analysis_results[element]['fwhm'][i]:.1f}"
                    })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

    csv = summary_df.to_csv(index=False)
    filename_suffix = "_smoothed" if smooth_data else ""
    data_type_suffix = "_fractions" if use_fractions else "_concentrations"
    st.download_button(
        label="Download Analysis Results as CSV",
        data=csv,
        file_name=f"fluence_analysis{data_type_suffix}{filename_suffix}.csv",
        mime="text/csv"
    )


def parse_sdtrimsp_file(file_content):
    lines = file_content.strip().split('\n')

    fluence_data = {}
    current_fluence = None
    current_data = []
    header_found = False
    parsing_data = False
    element_names = ['Ti', 'N1', 'N2']

    total_lines = len(lines)
    data_lines_found = 0
    fluence_lines_found = 0
    metadata_lines_skipped = 0

    print("=== PARSING SDTrimSP FILE ===")
    print(f"Total lines in file: {total_lines}")

    if len(lines) > 5:
        sixth_line = lines[5].strip()
        print(f"Sixth line: {sixth_line}")

        if sixth_line and not sixth_line.startswith('#') and not sixth_line.startswith('SDTrimSP'):
            parts = sixth_line.split()
            potential_elements = []
            for part in parts:
                if not part.replace('.', '').replace('-', '').replace('+', '').replace('E', '').isdigit() and len(
                        part) <= 3:
                    potential_elements.append(part.strip())

            if potential_elements:
                element_counts = {}
                processed_elements = []

                for element in potential_elements:
                    if element in element_counts:
                        element_counts[element] += 1
                        processed_elements.append(f"{element}{element_counts[element]}")
                    else:
                        element_counts[element] = 1
                        processed_elements.append(element)

                element_names = processed_elements
                print(f"Extracted element names from line 6: {element_names}")
            else:
                print("Could not extract element names from line 6, using defaults")
        else:
            print("Line 6 does not contain element names, using defaults")
    else:
        print("File too short to contain element names on line 6, using defaults")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if i % 100 == 0:
            print(f"Processing line {i}/{total_lines}")

        if not header_found and (('#' in line and ('center' in line or 'density' in line)) or
                                 ('xxx(*)' in line and 'dns(*)' in line)):
            header_found = True
            print(f"Found header at line {i}: {line.strip()}")

        elif '!--- fluc srrc sbe:' in line:
            fluence_lines_found += 1

            if current_fluence is not None and current_data:
                fluence_data[current_fluence] = current_data.copy()
                print(f"Saved fluence {current_fluence} with {len(current_data)} data points")

            try:
                before_marker = line.split('!---')[0].strip()
                parts = before_marker.split()
                if len(parts) >= 1:
                    current_fluence = float(parts[0])
                    print(f"Found new fluence {current_fluence} at line {i}")
                    current_data = []
                    parsing_data = True
                else:
                    current_fluence = None
                    parsing_data = False
            except Exception as e:
                print(f"Error parsing fluence from line {i}: {line}, Error: {e}")
                current_fluence = None
                parsing_data = False

        elif parsing_data and '!---' in line and re.match(r'^\s*0\.\d+E[+-]\d+', line):
            metadata_lines_skipped += 1
            if metadata_lines_skipped <= 5:
                print(f"Skipping metadata line {i}: {line[:60]}...")
            elif metadata_lines_skipped == 6:
                print("... (suppressing further metadata skip messages)")

        elif (header_found and parsing_data and
              re.match(r'^\s*0\.\d+E[+-]\d+', line) and
              '!---' not in line):
            try:
                parts = line.split()
                if len(parts) >= len(element_names) + 2:
                    depth = float(parts[0])
                    density = float(parts[1])

                    element_fractions = {}
                    element_concentrations = {}

                    for idx, element in enumerate(element_names):
                        if idx + 2 < len(parts):
                            frac = float(parts[idx + 2])
                            element_fractions[f'{element}_frac'] = frac
                            element_concentrations[f'{element}_conc'] = density * frac * 1e24

                    data_entry = {
                        'depth_A': depth,
                        'depth_nm': depth / 10.0,
                        'density': density,
                    }

                    data_entry.update(element_fractions)
                    data_entry.update(element_concentrations)

                    n_elements = [elem for elem in element_names if elem.startswith('N')]
                    if len(n_elements) > 1:
                        total_n_frac = sum(element_fractions.get(f'{elem}_frac', 0) for elem in n_elements)
                        total_n_conc = sum(element_concentrations.get(f'{elem}_conc', 0) for elem in n_elements)
                        data_entry['N_total_frac'] = total_n_frac
                        data_entry['N_total_conc'] = total_n_conc

                    current_data.append(data_entry)
                    data_lines_found += 1

                    if len(current_data) % 50 == 0:
                        print(f"  - Parsed {len(current_data)} data points for fluence {current_fluence}")

            except (ValueError, IndexError) as e:
                print(f"Error parsing data line {i}: {line}, Error: {e}")

        elif parsing_data and line.strip() == '':
            if current_data:
                print(f"End of data section for fluence {current_fluence} at line {i} ({len(current_data)} points)")
            parsing_data = False

        i += 1

    if current_fluence is not None and current_data:
        fluence_data[current_fluence] = current_data.copy()
        print(f"Saved final fluence {current_fluence} with {len(current_data)} data points")

    debug_info = {
        'total_lines': total_lines,
        'data_lines_found': data_lines_found,
        'fluence_lines_found': fluence_lines_found,
        'fluence_values': sorted(fluence_data.keys()) if fluence_data else [],
        'data_points_per_fluence': {f: len(data) for f, data in fluence_data.items()},
        'header_found': header_found,
        'metadata_lines_skipped': metadata_lines_skipped,
        'element_names': element_names
    }

    print(f"\nFinal summary:")
    print(f"- Processed {total_lines} lines")
    print(f"- Found {fluence_lines_found} fluence markers")
    print(f"- Parsed {data_lines_found} data lines")
    print(f"- Skipped {metadata_lines_skipped} metadata lines")
    print(
        f"- Successfully loaded {len(fluence_data)} fluences: {sorted(fluence_data.keys()) if fluence_data else 'None'}")

    return fluence_data, debug_info


def main():
    st.set_page_config(page_title="SDTrimSP Plotter", layout="wide")

    st.title("📊 SDTrimSP Data Plotter")
    st.markdown("Upload your SDTrimSP output file to visualize concentration profiles and density distributions")

    uploaded_file = st.file_uploader("Choose SDTrimSP output file")

    if uploaded_file is None:
        st.info("👆 Please upload your SDTrimSP output file above")
        st.markdown("### Expected File Format")
        return

    if uploaded_file is not None:
        file_content = str(uploaded_file.read(), "utf-8")

        try:
            fluence_data, debug_info = parse_sdtrimsp_file(file_content)

            with st.expander("🔍 File Parsing Debug Info", expanded=False):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Lines", debug_info['total_lines'])
                with col2:
                    st.metric("Data Lines Found", debug_info['data_lines_found'])
                with col3:
                    st.metric("Fluence Sections", debug_info['fluence_lines_found'])
                with col4:
                    st.metric("Fluence Steps", len(fluence_data))
                with col5:
                    st.metric("Metadata Skipped", debug_info.get('metadata_lines_skipped', 0))

                if debug_info['fluence_values']:
                    st.info(f"Found fluence values: {[f'{x:.1f}' for x in debug_info['fluence_values']]}")

                    if debug_info.get('element_names'):
                        st.info(f"Element names: {', '.join(debug_info['element_names'])}")

                    if debug_info.get('header_found'):
                        st.success("✅ Header section found")
                    else:
                        st.warning("⚠️ No header section detected")

                if st.checkbox("Show first 20 lines of uploaded file"):
                    file_lines = file_content.split('\n')[:20]
                    for i, line in enumerate(file_lines):
                        st.text(f"{i + 1:2d}: {line}")

            if not fluence_data:
                st.error("No valid data found in the file. Please check the file format.")
                st.info("The parser is looking for:")
                st.code("""
1. Lines with '!--- fluc srrc sbe:' to identify fluence sections
   - The first number on these lines is the fluence value
   - Example: "0.48500000E+02  0.24972441E+03 !--- fluc srrc sbe:   5.60000   4.90000   4.90000"
2. Element names on line 6: "Ti        N         N"
3. Data headers: '#  center[A]  density[a/A^3]  atomic fraction' OR 'xxx(*)       dns(*)'
4. Data lines starting with '0.XXXXXE+XX' 
5. At least depth + density + element fraction columns
                """)

                st.write("**First 50 lines of your file for debugging:**")
                file_lines = file_content.split('\n')[:50]
                for i, line in enumerate(file_lines):
                    st.text(f"{i + 1:2d}: {line}")

                st.write("**Searching for key patterns:**")
                fluence_patterns = []
                data_header_patterns = []
                data_line_patterns = []

                for i, line in enumerate(file_lines):
                    if '!--- fluc srrc sbe:' in line:
                        fluence_patterns.append(f"Line {i + 1}: {line}")
                    elif '#  center[A]' in line or 'xxx(*)' in line:
                        data_header_patterns.append(f"Line {i + 1}: {line}")
                    elif re.match(r'^0\.\d+E\+\d+', line):
                        data_line_patterns.append(f"Line {i + 1}: {line}")

                if fluence_patterns:
                    st.write("Found potential fluence markers:")
                    for pattern in fluence_patterns[:5]:
                        st.text(pattern)
                else:
                    st.write("❌ No fluence markers found")

                if data_header_patterns:
                    st.write("Found data headers:")
                    for pattern in data_header_patterns:
                        st.text(pattern)
                else:
                    st.write("❌ No data headers found")

                if data_line_patterns:
                    st.write("Found potential data lines:")
                    for pattern in data_line_patterns[:5]:
                        st.text(pattern)
                else:
                    st.write("❌ No data lines found")

                return

            st.success(f"✅ Successfully loaded data for {len(fluence_data)} fluence steps")

            st.sidebar.header("Plot Controls")

            fluence_values = sorted(fluence_data.keys())
            selected_fluence = st.sidebar.selectbox(
                "Select Fluence Step:",
                fluence_values,
                format_func=lambda x: f"Fluence: {x:.1f}"
            )

            plot_type = st.sidebar.radio(
                "Select Plot Type:",
                ["Atomic Fractions", "Concentrations (atoms/cm³)", "Density vs Depth"]
            )

            st.sidebar.subheader("Plot Controls")
            col_ctrl1, col_ctrl2, col_ctrl3 = st.sidebar.columns(3)

            with col_ctrl1:
                depth_unit = st.radio("Depth Units:", ["Angstroms (Å)", "Nanometers (nm)"], key="depth_unit")

            with col_ctrl2:
                plot_style = st.radio("Plot Style:", ["Lines", "Points", "Lines + Points"], key="plot_style")

            with col_ctrl3:
                y_axis_scale = st.radio("Y-axis Scale:", ["Linear", "Logarithmic"], key="y_axis_scale")

            smooth_data = False
            smooth_sigma = 2.0

            st.sidebar.subheader("Data Smoothing")
            smooth_data = st.sidebar.checkbox("Apply Smoothing", value=False)

            if smooth_data:
                smooth_sigma = st.sidebar.slider(
                    "Smoothing Strength (σ)",
                    min_value=0.5,
                    max_value=5.0,
                    value=2.0,
                    step=0.1,
                    help="Higher values = more smoothing"
                )

            element_names = debug_info.get('element_names', ['Ti', 'N1', 'N2'])

            selected_elements = []
            if len(element_names) >= 2:
                st.sidebar.subheader("Element Selection")
                selected_elements = st.sidebar.multiselect(
                    "Select elements to combine:",
                    element_names[1:],
                    default=element_names[1:]
                )

            fluence_unit = st.sidebar.radio("Fluence Units:", ["atoms/Ų", "atoms/cm²"])

            st.sidebar.subheader("📊 Experimental Data")
            uploaded_exp_files = st.sidebar.file_uploader(
                "Upload experimental data files (2-column format)",
                type=['txt', 'csv', 'dat'],
                accept_multiple_files=True,
                help="Upload 2-column data files (depth, value) for comparison with simulation"
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

            st.sidebar.subheader("Analysis Mode")
            col_analysis1, col_analysis2 = st.sidebar.columns(2)

            with col_analysis1:
                if st.button("📊 Fluence Analysis", type="primary", use_container_width=True):
                    st.session_state.show_analysis = True

            with col_analysis2:
                if st.button("📈 Profile Plots", type="secondary", use_container_width=True):
                    st.session_state.show_analysis = False

            if 'show_analysis' not in st.session_state:
                st.session_state.show_analysis = False

            current_mode = "Fluence Analysis" if st.session_state.show_analysis else "Profile Plots"
            st.sidebar.info(f"Current Mode: **{current_mode}**")

            st.sidebar.subheader("Multi-Fluence Comparison")
            compare_multiple = st.sidebar.checkbox("Compare Multiple Fluences")

            if compare_multiple:
                selected_fluences = st.sidebar.multiselect(
                    "Select fluences to compare:",
                    fluence_values,
                    default=[fluence_values[0], fluence_values[-1]] if len(fluence_values) > 1 else [fluence_values[0]]
                )

            data = fluence_data[selected_fluence]
            df = pd.DataFrame(data)

            depth_col = 'depth_A' if depth_unit == "Angstroms (Å)" else 'depth_nm'
            depth_label = "Depth (Å)" if depth_unit == "Angstroms (Å)" else "Depth (nm)"

            if smooth_data:
                try:
                    from scipy.ndimage import gaussian_filter1d

                    for element in element_names:
                        if f'{element}_conc' in df.columns:
                            df[f'{element}_conc_smooth'] = gaussian_filter1d(df[f'{element}_conc'], sigma=smooth_sigma)
                        if f'{element}_frac' in df.columns:
                            df[f'{element}_frac_smooth'] = gaussian_filter1d(df[f'{element}_frac'], sigma=smooth_sigma)

                    if 'N_total_conc' in df.columns:
                        df['N_total_conc_smooth'] = gaussian_filter1d(df['N_total_conc'], sigma=smooth_sigma)
                    if 'N_total_frac' in df.columns:
                        df['N_total_frac_smooth'] = gaussian_filter1d(df['N_total_frac'], sigma=smooth_sigma)
                    if 'density' in df.columns:
                        df['density_smooth'] = gaussian_filter1d(df['density'], sigma=smooth_sigma)

                except ImportError:
                    st.warning("⚠️ Smoothing requires scipy. Install with: pip install scipy")
                    st.warning("Using original data without smoothing.")
                    smooth_data = False

            mode = 'lines' if plot_style == "Lines" else 'markers' if plot_style == "Points" else 'lines+markers'

            if compare_multiple and len(selected_fluences) > 1:
                create_multi_fluence_comparison(fluence_data, selected_fluences, depth_col, depth_label, plot_type,
                                                mode, y_axis_scale, element_names, smooth_data, smooth_sigma,
                                                selected_elements)
            elif st.session_state.show_analysis:
                perform_fluence_analysis(fluence_data, element_names, fluence_unit, selected_elements, smooth_data,
                                         smooth_sigma, plot_type)
            else:
                create_single_fluence_plots(df, depth_col, depth_label, plot_type, mode, y_axis_scale,
                                            selected_fluence, element_names, smooth_data, selected_elements,
                                            experimental_data)

            if not compare_multiple:
                st.subheader("📈 Data Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Data Points", len(df))
                    st.metric("Max Depth (Å)", f"{df['depth_A'].max():.1f}")

                with col2:
                    st.metric("Current Fluence", f"{selected_fluence:.2e}")
                    st.metric("Max N Concentration", f"{df['N_total_conc'].max():.2e} atoms/cm³")

                with col3:
                    st.metric("Available Fluence Steps", len(fluence_values))
                    st.metric("Avg Density", f"{df['density'].mean():.4f} atoms/Ų")

                if st.checkbox("Show Raw Data Table"):
                    st.subheader("Raw Data")
                    display_cols = [depth_col, 'density']
                    display_names = [depth_label, 'Density (atoms/Ų)']

                    if plot_type == "Atomic Fractions":
                        data_suffix = "_frac"
                        data_unit = "Fraction"
                    elif plot_type == "Concentrations (atoms/cm³)":
                        data_suffix = "_conc"
                        data_unit = "(atoms/cm³)"
                    else:
                        data_suffix = "_conc"
                        data_unit = "(atoms/cm³)"

                    for element in element_names:
                        col_name = f'{element}{data_suffix}'
                        if col_name in df.columns:
                            display_cols.append(col_name)
                            display_names.append(f'{element} {data_unit}')

                    if plot_type == "Atomic Fractions" and 'N_total_frac' in df.columns:
                        display_cols.append('N_total_frac')
                        display_names.append(f'N Total {data_unit}')
                    elif plot_type == "Concentrations (atoms/cm³)" and 'N_total_conc' in df.columns:
                        display_cols.append('N_total_conc')
                        display_names.append(f'N Total {data_unit}')

                    available_cols = [col for col in display_cols if col in df.columns]
                    available_names = [display_names[i] for i, col in enumerate(display_cols) if col in df.columns]

                    if available_cols:
                        display_df = df[available_cols].copy()
                        display_df.columns = available_names
                        st.dataframe(display_df, use_container_width=True)

                        csv = display_df.to_csv(index=False)
                        data_type_name = "fractions" if plot_type == "Atomic Fractions" else "concentrations"
                        st.download_button(
                            label="Download data as CSV",
                            data=csv,
                            file_name=f"sdtrimsp_data_{data_type_name}_fluence_{selected_fluence:.2e}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No data columns available for the selected plot type.")

        except Exception as e:
            st.error(f"Error parsing file: {str(e)}")
            st.info("Please ensure the file is a valid SDTrimSP output file.")
            st.write("First few lines of the file:")
            file_lines = file_content.split('\n')[:10]
            for i, line in enumerate(file_lines):
                st.text(f"{i + 1:2d}: {line}")



main()
