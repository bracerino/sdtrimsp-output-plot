import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import numpy as np

from helpers.density import density_calculator_interface

import zipfile
import io


def create_xy_zip(fluence_data, element_names, depth_col_key, plot_type, smooth_data, smooth_sigma, selected_elements):
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fluence in sorted(fluence_data.keys()):
            df = pd.DataFrame(fluence_data[fluence])

            if smooth_data:
                try:
                    from scipy.ndimage import gaussian_filter1d
                    for elem in element_names:
                        for suffix in ('_conc', '_frac', '_dens'):
                            col = f'{elem}{suffix}'
                            if col in df.columns:
                                df[f'{col}_smooth'] = gaussian_filter1d(df[col], sigma=smooth_sigma)
                    if 'density' in df.columns:
                        df['density_smooth'] = gaussian_filter1d(df['density'], sigma=smooth_sigma)
                except ImportError:
                    pass

            depth = df[depth_col_key].values

            # --- Per-element files ---
            for elem in element_names:
                if plot_type == "Atomic Fractions":
                    col = f'{elem}_frac_smooth' if smooth_data and f'{elem}_frac_smooth' in df.columns else f'{elem}_frac'
                elif plot_type == "Concentrations (atoms/cm³)":
                    col = f'{elem}_conc_smooth' if smooth_data and f'{elem}_conc_smooth' in df.columns else f'{elem}_conc'
                elif plot_type == "Density (ions/Å)":
                    col = f'{elem}_dens_smooth' if smooth_data and f'{elem}_dens_smooth' in df.columns else f'{elem}_dens'
                else:
                    col = None

                if col and col in df.columns:
                    xy_lines = [f"{d:.6e}\t{v:.6e}" for d, v in zip(depth, df[col].values)]
                    filename = f"fluence_{fluence:.4f}_{elem}.xy"
                    zf.writestr(filename, "\n".join(xy_lines))

            # --- Combined selected elements file ---
            if selected_elements and len(selected_elements) > 1:
                if plot_type == "Atomic Fractions":
                    suffix = '_frac_smooth' if smooth_data else '_frac'
                elif plot_type == "Concentrations (atoms/cm³)":
                    suffix = '_conc_smooth' if smooth_data else '_conc'
                elif plot_type == "Density (ions/Å)":
                    suffix = '_dens_smooth' if smooth_data else '_dens'
                else:
                    suffix = None

                if suffix:
                    cols = [f'{e}{suffix}' for e in selected_elements if f'{e}{suffix}' in df.columns]
                    if cols:
                        combined = df[cols].sum(axis=1).values
                        xy_lines = [f"{d:.6e}\t{v:.6e}" for d, v in zip(depth, combined)]
                        combo_label = "+".join(selected_elements)
                        filename = f"fluence_{fluence:.4f}_combined_{combo_label}.xy"
                        zf.writestr(filename, "\n".join(xy_lines))

            # --- Total density file ---
            if plot_type == "Density vs Depth":
                col = 'density_smooth' if smooth_data and 'density_smooth' in df.columns else 'density'
                if col in df.columns:
                    xy_lines = [f"{d:.6e}\t{v:.6e}" for d, v in zip(depth, df[col].values)]
                    filename = f"fluence_{fluence:.4f}_total_density.xy"
                    zf.writestr(filename, "\n".join(xy_lines))

    zip_buffer.seek(0)
    return zip_buffer


def parse_experimental_data(file_content, filename):
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
                                element_names, smooth_data, selected_elements, experimental_data=None,
                                display_elements=None):
    # Restrict the per-element curves to the user-selected subset (defaults to all).
    elements_to_plot = display_elements if display_elements is not None else element_names

    if plot_type == "Atomic Fractions":
        fig = go.Figure()
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']

        for i, element in enumerate(elements_to_plot):
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
            title=dict(
                text=f"Atomic Fractions vs Depth (Fluence: {selected_fluence:.1f} atoms/A² = {selected_fluence:.1f} ×10¹⁶ atoms/cm²)",
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

        for i, element in enumerate(elements_to_plot):
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
            title=dict(
                text=f"Concentrations vs Depth (Fluence: {selected_fluence:.1f} atoms/A² = {selected_fluence:.1f} ×10¹⁶ atoms/cm²)",
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

    elif plot_type == "Density (ions/Å)":
        fig = go.Figure()
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']

        for i, element in enumerate(elements_to_plot):
            dens_col = f'{element}_dens_smooth' if smooth_data and f'{element}_dens_smooth' in df.columns else f'{element}_dens'
            if dens_col in df.columns:
                display_name = f"{element} (smoothed)" if smooth_data and f'{element}_dens_smooth' in df.columns else element
                fig.add_trace(go.Scatter(
                    x=df[depth_col], y=df[dens_col],
                    mode=mode, name=display_name,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=6, color=colors[i % len(colors)])
                ))

        if selected_elements and len(selected_elements) > 1:
            if smooth_data and all(f'{elem}_dens_smooth' in df.columns for elem in selected_elements):
                combined_dens = df[[f'{elem}_dens_smooth' for elem in selected_elements]].sum(axis=1)
                combined_name = f'Combined ({"+".join(selected_elements)}) (smoothed)'
            else:
                combined_dens = df[[f'{elem}_dens' for elem in selected_elements if f'{elem}_dens' in df.columns]].sum(
                    axis=1)
                combined_name = f'Combined ({"+".join(selected_elements)})'

            fig.add_trace(go.Scatter(
                x=df[depth_col], y=combined_dens,
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
            title=dict(
                text=f"Element Density vs Depth (Fluence: {selected_fluence:.1f} atoms/A² = {selected_fluence:.1f} ×10¹⁶ atoms/cm²)",
                font=dict(size=28, color='black')),
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

    st.plotly_chart(fig, width='stretch')


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
                    if f'{elem}_dens' in df_comp.columns:
                        df_comp[f'{elem}_dens_smooth'] = gaussian_filter1d(df_comp[f'{elem}_dens'], sigma=smooth_sigma)

                if 'N_total_conc' in df_comp.columns:
                    df_comp['N_total_conc_smooth'] = gaussian_filter1d(df_comp['N_total_conc'], sigma=smooth_sigma)
                if 'N_total_frac' in df_comp.columns:
                    df_comp['N_total_frac_smooth'] = gaussian_filter1d(df_comp['N_total_frac'], sigma=smooth_sigma)
                if 'N_total_dens' in df_comp.columns:
                    df_comp['N_total_dens_smooth'] = gaussian_filter1d(df_comp['N_total_dens'], sigma=smooth_sigma)
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

        elif plot_type == "Density (ions/Å)":
            if selected_elements and len(selected_elements) > 1:
                if smooth_data and all(f'{elem}_dens_smooth' in df_comp.columns for elem in selected_elements):
                    combined_dens = df_comp[[f'{elem}_dens_smooth' for elem in selected_elements]].sum(axis=1)
                    display_name = f'Combined ({"+".join(selected_elements)}) (Fluence: {fluence:.1f}, smoothed)'
                else:
                    combined_dens = df_comp[
                        [f'{elem}_dens' for elem in selected_elements if f'{elem}_dens' in df_comp.columns]].sum(axis=1)
                    display_name = f'Combined ({"+".join(selected_elements)}) (Fluence: {fluence:.1f})'

                comparison_fig.add_trace(go.Scatter(
                    x=df_comp[depth_col], y=combined_dens,
                    mode=mode, name=display_name,
                    line=dict(color=color, width=3),
                    marker=dict(size=6, color=color)
                ))
            elif selected_elements and len(selected_elements) == 1:
                elem = selected_elements[0]
                if smooth_data and f'{elem}_dens_smooth' in df_comp.columns:
                    dens_data = df_comp[f'{elem}_dens_smooth']
                    display_name = f'{elem} (Fluence: {fluence:.1f}, smoothed)'
                else:
                    dens_data = df_comp[f'{elem}_dens']
                    display_name = f'{elem} (Fluence: {fluence:.1f})'

                comparison_fig.add_trace(go.Scatter(
                    x=df_comp[depth_col], y=dens_data,
                    mode=mode, name=display_name,
                    line=dict(color=color, width=3),
                    marker=dict(size=6, color=color)
                ))
            else:
                elem = element_names[0] if element_names else 'Ti'
                if smooth_data and f'{elem}_dens_smooth' in df_comp.columns:
                    dens_data = df_comp[f'{elem}_dens_smooth']
                    display_name = f'{elem} (Fluence: {fluence:.1f}, smoothed)'
                else:
                    dens_data = df_comp[f'{elem}_dens']
                    display_name = f'{elem} (Fluence: {fluence:.1f})'

                comparison_fig.add_trace(go.Scatter(
                    x=df_comp[depth_col], y=dens_data,
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
        "Concentration (atoms/cm³)" if plot_type == "Concentrations (atoms/cm³)" else \
            "Element Density (atoms/Ų)" if plot_type == "Density (ions/Å)" else \
                "Total Density (atoms/Ų)"

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

    st.plotly_chart(comparison_fig, width='stretch')


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
    use_density = (plot_type == "Density (ions/Å)")

    if use_fractions:
        data_suffix = "_frac"
        data_label = "Atomic Fraction"
        metric_label = "Max Atomic Fraction"
    elif use_density:
        data_suffix = "_dens"
        data_label = "Density (atoms/Ų)"
        metric_label = "Max Density (atoms/Ų)"
    else:
        data_suffix = "_conc"
        data_label = "Concentration (atoms/cm³)"
        metric_label = "Max Concentration (atoms/cm³)"

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
                        if f'{elem}_dens' in df.columns:
                            df[f'{elem}_dens_smooth'] = gaussian_filter1d(df[f'{elem}_dens'], sigma=smooth_sigma)

                    if element.startswith('Combined_') and selected_elements:
                        if use_fractions:
                            combined_data = df[
                                [f'{elem}_frac' for elem in selected_elements if f'{elem}_frac' in df.columns]].sum(
                                axis=1)
                        elif use_density:
                            combined_data = df[
                                [f'{elem}_dens' for elem in selected_elements if f'{elem}_dens' in df.columns]].sum(
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
                    if use_fractions:
                        combined_data = df[[f'{elem}{data_suffix}_smooth' for elem in selected_elements]].sum(axis=1)
                    elif use_density:
                        combined_data = df[[f'{elem}{data_suffix}_smooth' for elem in selected_elements]].sum(axis=1)
                    else:
                        combined_data = df[[f'{elem}{data_suffix}_smooth' for elem in selected_elements]].sum(axis=1)
                    data_col = f'{element}{data_suffix}_smooth'
                    df[data_col] = combined_data
                else:
                    if use_fractions:
                        combined_data = df[[f'{elem}{data_suffix}' for elem in selected_elements if
                                            f'{elem}{data_suffix}' in df.columns]].sum(axis=1)
                    elif use_density:
                        combined_data = df[[f'{elem}{data_suffix}' for elem in selected_elements if
                                            f'{elem}{data_suffix}' in df.columns]].sum(axis=1)
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
        st.plotly_chart(fig_max, width='stretch')

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
        st.plotly_chart(fig_depth, width='stretch')

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
        st.plotly_chart(fig_fwhm, width='stretch')

    st.write(f"**Summary Table{smoothing_note}**")
    summary_data = []
    for element in elements_to_analyze:
        if element in analysis_results:
            display_name = element.replace('Combined_', 'Combined ').replace('_', '+')
            for i, fluence in enumerate(fluence_values):
                if i < len(analysis_results[element]['max_values']):
                    max_val = analysis_results[element]['max_values'][i]
                    if use_fractions:
                        max_val_str = f"{max_val:.3f}"
                    elif use_density:
                        max_val_str = f"{max_val:.4f}"
                    else:
                        max_val_str = f"{max_val:.2e}"

                    summary_data.append({
                        'Element': display_name,
                        'Fluence': f"{fluence_display[i]:.2e}",
                        metric_label: max_val_str,
                        'Depth of Max (Å)': f"{analysis_results[element]['max_depth'][i]:.1f}",
                        'FWHM (Å)': f"{analysis_results[element]['fwhm'][i]:.1f}"
                    })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, width='stretch')

    csv = summary_df.to_csv(index=False)
    filename_suffix = "_smoothed" if smooth_data else ""
    if use_fractions:
        data_type_suffix = "_fractions"
    elif use_density:
        data_type_suffix = "_density"
    else:
        data_type_suffix = "_concentrations"

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
                token = part.strip()
                # Keep any non-numeric token as a component name. Do NOT cap the
                # length: SDTrimSP component names such as 'Nb_1' or 'Ti_1' are
                # 4+ characters and were previously discarded, which silently
                # dropped whole atomic-fraction columns. Those dropped components
                # are exactly the bulk/matrix species that dominate at depth, so
                # the plotted fractions no longer summed to 1 and the matrix
                # profile appeared to "decay" toward 0 instead of its bulk value.
                cleaned = (token.replace('.', '').replace('-', '').replace('+', '')
                           .replace('E', '').replace('e', ''))
                if token and not cleaned.isdigit():
                    potential_elements.append(token)

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

                    if depth == 0.0 and density == 0.0:
                        i += 1
                        continue

                    element_fractions = {}
                    element_concentrations = {}
                    element_densities = {}

                    for idx, element in enumerate(element_names):
                        if idx + 2 < len(parts):
                            frac = float(parts[idx + 2])
                            element_fractions[f'{element}_frac'] = frac
                            element_concentrations[f'{element}_conc'] = density * frac * 1e24
                            element_densities[f'{element}_dens'] = density * frac

                    data_entry = {
                        'depth_A': depth,
                        'depth_nm': depth / 10.0,
                        'density': density,
                    }

                    data_entry.update(element_fractions)
                    data_entry.update(element_concentrations)
                    data_entry.update(element_densities)

                    n_elements = [elem for elem in element_names
                                  if re.sub(r'_\d+$', '', elem) == 'N']
                    if len(n_elements) > 1:
                        total_n_frac = sum(element_fractions.get(f'{elem}_frac', 0) for elem in n_elements)
                        total_n_conc = sum(element_concentrations.get(f'{elem}_conc', 0) for elem in n_elements)
                        total_n_dens = sum(element_densities.get(f'{elem}_dens', 0) for elem in n_elements)
                        data_entry['N_total_frac'] = total_n_frac
                        data_entry['N_total_conc'] = total_n_conc
                        data_entry['N_total_dens'] = total_n_dens

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


def parse_dynamic_sputter_yields(file_content):
    """Parse sputtering yields vs fluence from an SDTrimSP dynamic run log.

    The main SDTrimSP output log prints one line per history step that looks like:
        50 flc:  10.0 E: 0. a: 0.0 dz: -37.56 dz/dt 0.0 Ysum:  0.15 0.04 0.07 0.04 0.16 qumax  ...
    where the numbers after 'Ysum:' are the per-component sputtering yields and 'flc:'
    is the accumulated fluence (in ions/A**2).

    Components that belong to the same chemical element (e.g. 'Ti' and 'Ti_1') are
    combined using the 'correspond. element' mapping from the crystal table header,
    falling back to stripping a '_<n>' suffix when that mapping is unavailable.
    """
    lines = file_content.split('\n')

    symbols = []
    correspond = []
    for line in lines:
        s = line.strip()
        if not symbols and s.startswith('symbol') and ':' in s:
            symbols = s.split(':', 1)[1].split()
        elif s.startswith('correspond. element') and ':' in s:
            try:
                correspond = [int(x) for x in s.split(':', 1)[1].split()]
            except ValueError:
                correspond = []
        if symbols and correspond:
            break

    fluences = []
    yields_rows = []
    for line in lines:
        if 'flc:' in line and 'Ysum:' in line:
            mflc = re.search(r'flc:\s*([-+0-9.eE]+)', line)
            mys = re.search(r'Ysum:\s*(.*?)(?:qumax|$)', line)
            if not mflc or not mys:
                continue
            try:
                flc = float(mflc.group(1))
            except ValueError:
                continue
            yvals = []
            for tok in mys.group(1).split():
                try:
                    yvals.append(float(tok))
                except ValueError:
                    pass
            if not yvals:
                continue
            fluences.append(flc)
            yields_rows.append(yvals)

    if not fluences:
        return None

    n_comp = max(len(r) for r in yields_rows)

    if not symbols or len(symbols) != n_comp:
        symbols = [f'comp{i + 1}' for i in range(n_comp)]
        correspond = []

    if correspond and len(correspond) == len(symbols):
        group_for = []
        for ci in correspond:
            if 1 <= ci <= len(symbols):
                group_for.append(symbols[ci - 1])
            else:
                group_for.append(re.sub(r'_\d+$', '', symbols[len(group_for)]))
    else:
        group_for = [re.sub(r'_\d+$', '', s) for s in symbols]

    n_steps = len(fluences)

    # Per-component yields, kept index-based so that two components sharing the
    # same chemical symbol (e.g. the projectile N and an N already present in the
    # target) stay separable instead of overwriting each other in a dict.
    comp_yields = [[0.0] * n_steps for _ in range(n_comp)]
    for step, row in enumerate(yields_rows):
        for i in range(min(len(row), n_comp)):
            comp_yields[i][step] = row[i]

    # Disambiguate duplicate symbols for display: N, N -> "N #1", "N #2".
    sym_counts = {s: symbols.count(s) for s in set(symbols)}
    sym_seen = {}
    comp_labels = []
    for s in symbols:
        if sym_counts.get(s, 0) > 1:
            sym_seen[s] = sym_seen.get(s, 0) + 1
            comp_labels.append(f"{s} #{sym_seen[s]}")
        else:
            comp_labels.append(s)

    group_order = []
    group_members = {}        # group -> list of component indices
    group_member_labels = {}  # group -> list of disambiguated component labels
    group_yields = {}
    for i, s in enumerate(symbols):
        g = group_for[i]
        if g not in group_yields:
            group_yields[g] = [0.0] * n_steps
            group_members[g] = []
            group_member_labels[g] = []
            group_order.append(g)
        group_members[g].append(i)
        group_member_labels[g].append(comp_labels[i])
        for step in range(n_steps):
            group_yields[g][step] += comp_yields[i][step]

    total = [sum(row[i] for i in range(min(len(row), n_comp))) for row in yields_rows]

    return {
        'symbols': symbols,
        'comp_labels': comp_labels,
        'comp_yields': comp_yields,
        'group_order': group_order,
        'group_members': group_members,
        'group_member_labels': group_member_labels,
        'group_yields': group_yields,
        'fluences': fluences,
        'total': total,
    }


def display_dynamic_sputter_yields_section():
    """Uploader + chart for sputtering yields vs fluence from a dynamic run log."""
    st.markdown("### 💥 Sputtering Yields vs Fluence")
    st.caption(
        "Upload (or paste) the main SDTrimSP run log (the file with the per-step `flc: ... Ysum: ...` lines). "
        "This file is originally called **`time_run.dat`** in the SDTrimSP output. "
        "Components of the same element are summed into a per-element yield, but each component is also "
        "available separately — so when the projectile and the target share an element (e.g. implanting N "
        "into a target that already contains N), you can plot the implanted-N and target-N yields on their own, "
        "not only their sum. The total yield sums all components."
    )

    ups = st.file_uploader(
        "Choose SDTrimSP dynamic output log(s) — originally named time_run.dat (with 'Ysum:' lines)",
        type=['dat', 'txt', 'out', 'log'],
        accept_multiple_files=True,
        key="dynamic_sputter_log",
        help="Upload one or more run logs (time_run.dat) to compare their sputtering yields in a single graph. "
             "Per-element sums, individual components, and the total are all selectable."
    )

    # Allow pasting log content directly, in addition to uploading files.
    if 'dyn_sputter_pasted' not in st.session_state:
        st.session_state.dyn_sputter_pasted = {}

    with st.expander("➕ Or paste log content directly", expanded=False):
        paste_name = st.text_input(
            "Name for this dataset:",
            value=f"pasted_{len(st.session_state.dyn_sputter_pasted) + 1}",
            key="dyn_sputter_paste_name"
        )
        paste_content = st.text_area(
            "Paste the run-log text (must contain the per-step `flc: ... Ysum: ...` lines):",
            height=200,
            key="dyn_sputter_paste_content"
        )
        add_col, clear_col = st.columns([1, 1])
        with add_col:
            if st.button("✅ Accept and add", key="dyn_sputter_paste_add", type="primary"):
                if not paste_content.strip():
                    st.warning("Nothing to add — the text box is empty.")
                elif parse_dynamic_sputter_yields(paste_content) is None:
                    st.error("Could not find any per-step `Ysum:` lines in the pasted text.")
                else:
                    name = paste_name.strip() or f"pasted_{len(st.session_state.dyn_sputter_pasted) + 1}"
                    st.session_state.dyn_sputter_pasted[name] = paste_content
                    st.success(f"Added '{name}'.")
        with clear_col:
            if st.session_state.dyn_sputter_pasted and st.button(
                "🗑️ Clear pasted datasets", key="dyn_sputter_paste_clear"
            ):
                st.session_state.dyn_sputter_pasted = {}
                st.rerun()

        if st.session_state.dyn_sputter_pasted:
            st.caption("Pasted datasets: " + ", ".join(st.session_state.dyn_sputter_pasted.keys()))

    if not ups and not st.session_state.dyn_sputter_pasted:
        return

    parsed_files = {}
    failed = []
    for up in ups:
        content = str(up.read(), "utf-8")
        p = parse_dynamic_sputter_yields(content)
        if p is None:
            failed.append(up.name)
        else:
            parsed_files[up.name] = p

    for name, content in st.session_state.dyn_sputter_pasted.items():
        p = parse_dynamic_sputter_yields(content)
        if p is None:
            failed.append(name)
        else:
            parsed_files[name] = p

    if failed:
        st.warning("No per-step `Ysum:` lines found in: " + ", ".join(failed))

    if not parsed_files:
        st.error("Could not find any per-step `Ysum:` sputtering-yield lines in the uploaded/pasted data.")
        return

    multi = len(parsed_files) > 1

    # Build the list of selectable series across all files: grouped elements,
    # individual components, and the total — each tagged with its source file.
    options = []
    option_map = {}
    default_options = []
    for fname, p in parsed_files.items():
        for g in p['group_order']:
            member_labels = p['group_member_labels'][g]
            ylabel = f"Y({g})" if member_labels == [g] else f"Y({g}={'+'.join(member_labels)})"
            opt = f"{fname} | {ylabel}"
            options.append(opt)
            option_map[opt] = (fname, 'group', g)
            if not multi:
                default_options.append(opt)
        for i, lbl in enumerate(p['comp_labels']):
            opt = f"{fname} | Y[{lbl}] (component)"
            options.append(opt)
            option_map[opt] = (fname, 'component', i)
        opt_total = f"{fname} | Total"
        options.append(opt_total)
        option_map[opt_total] = (fname, 'total', None)
        default_options.append(opt_total)

    selected = st.multiselect(
        "Select which yields to plot (compare across files):",
        options,
        default=default_options,
        key="dyn_sputter_series"
    )

    # Fluence-unit conversions, relative to the parsed value in atoms/Å².
    # 1 atom/Å² = 1e16 atoms/cm² = 0.1 × 10¹⁷ atoms/cm².
    flu_unit_factors = {
        "atoms/Ų": 1.0,
        "atoms/cm²": 1e16,
        "10¹⁷ atoms/cm²": 0.1,
    }
    flu_unit_labels = {
        "atoms/Ų": "Fluence (atoms/Ų)",
        "atoms/cm²": "Fluence (atoms/cm²)",
        "10¹⁷ atoms/cm²": "Fluence (10¹⁷ atoms/cm²)",
    }

    col1, col2, col3 = st.columns(3)
    with col1:
        flu_unit = st.radio(
            "Fluence Units:", list(flu_unit_factors.keys()), index=1, key="dyn_sputter_flu_unit"
        )
    with col2:
        y_scale = st.radio("Y-axis Scale:", ["Linear", "Logarithmic"], key="dyn_sputter_yscale")
    with col3:
        plot_style = st.radio("Plot Style:", ["Lines", "Markers", "Lines+Markers"], key="dyn_sputter_style")

    x_label = flu_unit_labels[flu_unit]
    x_factor = flu_unit_factors[flu_unit]

    mode_map = {"Lines": "lines", "Markers": "markers", "Lines+Markers": "lines+markers"}
    plot_mode = mode_map[plot_style]

    # Line / marker styling controls (shown depending on the selected plot style).
    st.sidebar.markdown("---")
    st.sidebar.subheader("💥 Sputter Plot Styling")
    uses_lines = plot_style in ("Lines", "Lines+Markers")
    uses_markers = plot_style in ("Markers", "Lines+Markers")

    if uses_lines:
        line_width = st.sidebar.slider("Line width:", 1, 10, 3, key="dyn_sputter_line_width")
        line_style = st.sidebar.selectbox(
            "Line style:", ["Auto (by role)", "solid", "dash", "dot", "dashdot"],
            key="dyn_sputter_line_style",
            help="'Auto (by role)' draws totals dashed, components dotted, elements solid."
        )
    else:
        line_width = 3
        line_style = "Auto (by role)"

    if uses_markers:
        marker_size = st.sidebar.slider("Marker size:", 3, 20, 7, key="dyn_sputter_marker_size")
        marker_symbol = st.sidebar.selectbox(
            "Marker symbol:",
            ["circle", "square", "diamond", "triangle-up", "triangle-down", "x", "cross", "star"],
            key="dyn_sputter_marker_symbol"
        )
    else:
        marker_size = 7
        marker_symbol = "circle"

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'lime',
              'teal', 'olive', 'navy', 'maroon', 'darkgreen']

    if not selected:
        st.info("👆 Select one or more yields above to plot.")
        return

    def series_values(fname, kind, key):
        p = parsed_files[fname]
        if kind == 'group':
            return p['group_yields'][key]
        if kind == 'component':
            return p['comp_yields'][key]
        return p['total']

    def series_label(opt, fname, kind, key):
        if not multi:
            # drop the redundant filename prefix when only one file is loaded
            return opt.split(' | ', 1)[1]
        return opt

    fig = go.Figure()
    for idx, opt in enumerate(selected):
        fname, kind, key = option_map[opt]
        p = parsed_files[fname]
        x_vals = np.array(p['fluences']) * x_factor
        y_vals = series_values(fname, kind, key)
        is_total = (kind == 'total')
        if line_style == "Auto (by role)":
            dash = 'dash' if is_total else ('dot' if kind == 'component' else 'solid')
        else:
            dash = line_style
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode=plot_mode,
            name=series_label(opt, fname, kind, key),
            line=dict(
                color=colors[idx % len(colors)],
                width=line_width + 1 if is_total else line_width,
                dash=dash
            ),
            marker=dict(
                size=marker_size + 1 if is_total else marker_size,
                color=colors[idx % len(colors)],
                symbol=marker_symbol
            )
        ))

    fig.update_layout(
        title=dict(text="Sputtering Yield vs Fluence", font=dict(size=28, color='black')),
        xaxis_title=dict(text=x_label, font=dict(size=24, color='black')),
        yaxis_title=dict(text="Sputtering Yield (atoms/ion)", font=dict(size=24, color='black')),
        yaxis_type="log" if y_scale == "Logarithmic" else "linear",
        height=650,
        hovermode='x unified',
        font=dict(size=20, color='black'),
        legend=dict(font=dict(size=16, color='black')),
        xaxis=dict(tickfont=dict(size=20, color='black')),
        yaxis=dict(tickfont=dict(size=20, color='black'))
    )

    st.plotly_chart(fig, width='stretch')

    # Final-value summary for the selected series.
    summary_rows = []
    for opt in selected:
        fname, kind, key = option_map[opt]
        summary_rows.append({
            'File': fname,
            'Series': opt.split(' | ', 1)[1],
            'Final fluence (atoms/Ų)': parsed_files[fname]['fluences'][-1],
            'Final fluence (10¹⁷ atoms/cm²)': parsed_files[fname]['fluences'][-1] * 0.1,
            'Final yield (atoms/ion)': series_values(fname, kind, key)[-1]
        })
    st.dataframe(pd.DataFrame(summary_rows), width='stretch', hide_index=True)

    # Long-format CSV (robust to files having different fluence grids).
    long_rows = []
    for opt in selected:
        fname, kind, key = option_map[opt]
        p = parsed_files[fname]
        label = opt.split(' | ', 1)[1]
        yv = series_values(fname, kind, key)
        for f, y in zip(p['fluences'], yv):
            long_rows.append({
                'File': fname,
                'Series': label,
                'Fluence (atoms/Ų)': f,
                'Fluence (atoms/cm²)': f * 1e16,
                'Fluence (10¹⁷ atoms/cm²)': f * 0.1,
                'Yield (atoms/ion)': y
            })
    sputter_df = pd.DataFrame(long_rows)

    with st.expander("📋 Sputtering yield data table", expanded=False):
        st.dataframe(sputter_df, width='stretch', hide_index=True)

    st.download_button(
        label="📥 Download selected sputtering yields as CSV",
        data=sputter_df.to_csv(index=False),
        file_name="dynamic_sputtering_yields_vs_fluence.csv",
        mime="text/csv",
        key="download_dyn_sputter",
        type="primary"
    )

    st.markdown("---")


def main():
    st.set_page_config(page_title="SDTrimSP Plotter", layout="wide")

    st.markdown(
        "<h2 style='font-size: 32px;'>📊 SDTrimSP Data Plotter and Helpful Tools</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style="
            display: inline-block;
            background-color: #ffffff;
            border-left: 5px solid #2563eb;
            border-radius: 10px;
            padding: 10px 16px;
            margin-top: -4px;
            margin-bottom: 24px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.10);
            color: #111827;
            font-size: 0.95rem;
            font-weight: 600;
        ">
            <span style="color:#2563eb; font-weight:800;">Release:</span>
            v0.6 &nbsp; | &nbsp;
            <span style="color:#2563eb; font-weight:800;">Updated:</span>
            May 28, 2026
        </div>
        """,
        unsafe_allow_html=True
    )
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

            /* Added underline (thicker) */
            border-bottom: 4px solid #1e3a8a !important;
            border-radius: 12px 12px 0 0 !important; /* keep rounded only on top */
        }

        .stTabs [data-baseweb="tab-list"] button:focus {
            outline: none !important;
        }
        </style>
        '''

    st.markdown(css, unsafe_allow_html=True)
    from helpers.static_mode import create_static_mode_interface
    if create_static_mode_interface():
        return

    # ── Sidebar tool checkboxes (always visible, all off by default) ──────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Tools")
    show_dynamic = st.sidebar.checkbox("📊 Dynamic SDTrimSP Mode", value=False, key="show_dynamic")
    show_density = st.sidebar.checkbox("🧮 Atomic Density Calculator", value=False, key="show_density")
    show_conv = st.sidebar.checkbox("🔄 Concentration Converter", value=False, key="show_conv")
    show_crystal = st.sidebar.checkbox("🔄 POSCAR/CIF → crystal structure", value=False, key="show_crystal")
    st.sidebar.markdown("---")

    if not any([show_dynamic, show_density, show_conv, show_crystal]):
        st.sidebar.link_button("⭐ View on GitHub", "https://github.com/bracerino/sdtrimsp-output-plot/")
        st.sidebar.info(f"❤️🫶 **[Donations always appreciated!](https://buymeacoffee.com/bracerino)**")
        st.sidebar.info(
            "🌀 Developed by **[IMPLANT team](https://implant.fs.cvut.cz/)**. "
            "📘 See our corresponding **[article](https://www.sciencedirect.com/science/article/abs/pii/S0927025626000388?via%3Dihub)**. Spot a bug or have a feature requests? Let us know at **lebedmi2@cvut.cz**."
        )
    
    # ── Landing page: shown when no tool is selected ──────────────────────────
    if not any([show_dynamic, show_density, show_conv, show_crystal]):
        st.info(
            "👈 **Select a tool from the sidebar to get started.** "
            "Choose one or more of the available tools to begin your analysis."
        )
        with st.expander("📘 How to **Cite**", expanded=False):
            st.markdown("""
            If you like this interactive app, please cite the following:
            - [Lebeda, M., et al. Interactive Analysis of Static, Dynamic, and Crystalline SDTrimSP Simulations: Application to Nitrogen Ion Implantation into Vanadium. Computational Materials Science, 2026.](https://doi.org/10.1016/j.commatsci.2026.114519)
            ---
            When using SDTrimSP,  please cite:
            - [Mutzke, A., et al. SDTrimSP Version 7.00. 2024.](https://pure.mpg.de/rest/items/item_3577532/component/file_3579585/content)

            When using the main local GUI for SDTrimSP, please cite:
            - [Szabo, P. S., et al. Graphical user interface for SDTrimSP to simulate sputtering, ion implantation and the dynamic effects of ion irradiation.](https://www.sciencedirect.com/science/article/pii/S0168583X22001069)

            """,
                        unsafe_allow_html=True)

        st.markdown("""
        #### 📁 How to Use This App

        If you find a bug or have questions, feel free to contact me at **[lebedmi2@cvut.cz](mailto:lebedmi2@cvut.cz)**  
        - 📺 1️⃣ *Video tutorial for the dynamic, fluence-dependent mode:* [Watch on YouTube](https://youtu.be/JBXGyuHMtGk?si=Twj-2FA28ogJ1jUr) 
        - 📺 2️⃣ *Video tutorial for simulations of ion channeling and the static mode:* [Watch on YouTube](https://www.youtube.com/watch?v=41fctoKS4nU)
        - 📺 3️⃣ *Video tutorial for compiling SDTrimSP and how to run calculations with its local GUI:* [Watch on YouTube](https://youtu.be/DwTXVmtTUzw?si=d5laSAouvtN7UuGl)
        ---
        #### 🔄 Workflow
        """)
        st.image("images/Workflow2.png", width='content')
        st.markdown("""
        ---
        #### 📤 Input Requirements:
        Upload the **output file** from a **dynamic SDTrimSP ion implantation simulation** containing data about **element depth distributions as a function of fluence**.

        ✅ The app will **automatically load all data**.

        ---

        #### 📊 What You Can Do:

        **📊 Dynamic SDTrimSP mode** (fluence-dependent depth profiles)
        - 📈 **Select and plot** depth concentration distributions:
          - in **atomic fraction**
          - in **atoms/cm³**
          - or in **density (atoms/Å²)**
        - 🧪 Plot **target density** vs. depth and vs. fluence
        - 📉 **Compare multiple fluences** in a single plot
        - 🔄 Upload **two-column data** (e.g., experimental profiles) for **direct comparison**
        - 🪶 Optional **Gaussian smoothing** of profiles
        - 📌 Automatically calculate, as a function of fluence:
          - **Maximum concentration values**
          - **Depth positions of maxima**
          - **FWHM (Full Width at Half Maximum)**
        - 💾 **Download profiles** as .xy files (single fluence or batch ZIP of all fluences)

        **📊 Static SDTrimSP mode** (`depth_damage.dat`, `depth_proj.dat`, `output.dat`)
        - 📈 Plot **STOPS** and **VACANCIES** depth distributions per element from multiple files in one figure, as:
          - raw counts, atomic fractions, **normalized probability**, density (ions/Å), or concentration (ions/cm³, given a fluence)
        - 🔬 Choose smoothing (**Savitzky–Golay / Moving average / Gaussian**) with adjustable window and order
        - 🔄 Overlay **experimental 2-column data** for comparison
        - 💥 Upload one or more **SDTrimSP `output.dat`** files to extract **backward and transmission sputtering yields**:
          - Cross-file **summary table** (projectile, energy, total Y, per-element Y)
          - Per-file detailed tables with **mean energy, escape depth, and spread**
          - CSV download for both summary and per-file tables

        **🧮 Atomic Density Calculator**
        - Compute atomic densities (at/Å³) for elements and compounds, with a built-in materials database

        **🔄 Concentration Converter**
        - Convert between **at.%**, **wt.%**, and **atoms/cm³** for arbitrary multi-component compositions

        **🔄 POSCAR / CIF ↔ SDTrimSP crystal structure converter**
        - **POSCAR / CIF → SDTrimSP**: generate `crystal.inp` (≤ 7.01) or `table.crystal` entries (≥ 7.02), including the `Nr-crystal` structure block and the geometry-line columns (`dx`, `dy`, `dz`, density, `matrix_id`, …)
        - **SDTrimSP → POSCAR / CIF**: paste a `table.crystal` entry or upload a `crystal.inp` and download the reconstructed **POSCAR** and **CIF**
        - Automatic handling of **non-orthogonal lattices** (minimal bounding box + de-duplication) with explicit warnings from the SDTrimSP documentation
        - Optional **lattice reorientation** (axis re-mapping with sign flips)

        ---

        Enjoy using the app!
        """)
        return

    # ── Tool panels ───────────────────────────────────────────────────────────
    if show_density:
        density_calculator_interface()

    if show_conv:
        from helpers.concentration_converter import concentration_converter_interface
        concentration_converter_interface()

    if show_crystal:
        from helpers.crystal_converter_module import crystal_converter_interface
        crystal_converter_interface()

    # ── Dynamic mode ──────────────────────────────────────────────────────────
    if not show_dynamic:
        return

    profile_tab, sputter_tab = st.tabs(
        ["📈 Dynamic Depth Profile", "💥 Sputtering Yields vs Fluence"]
    )

    # Render the sputtering-yield tab first so that the early returns in the
    # depth-profile branch below still leave this tab populated.
    with sputter_tab:
        display_dynamic_sputter_yields_section()

    with profile_tab:
        uploaded_file = st.file_uploader("Choose SDTrimSP output file")

        if uploaded_file is None:
            st.info("👆 Please upload your Dynamic SDTrimSP output file above "
                    "(or open the **💥 Sputtering Yields vs Fluence** tab).")
            return
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
            st.sidebar.header("Plot Controls")

            fluence_values = sorted(fluence_data.keys())
            selected_fluence = st.sidebar.selectbox(
                "Select Fluence Step:",
                fluence_values,
                format_func=lambda x: f"Fluence: {x:.1f}"
            )

            plot_type = st.sidebar.radio(
                "Select Plot Type:",
                ["Atomic Fractions", "Concentrations (atoms/cm³)", "Density (ions/Å)", "Density vs Depth"]
            )

            st.sidebar.subheader("Plot Controls")
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

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

            fluence_unit = st.sidebar.radio("Fluence Units:", ["atoms/Ų", "atoms/cm²"], index=1)

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

            st.sidebar.subheader("💾 Bulk Download")
            depth_key = 'depth_A' if depth_unit == "Angstroms (Å)" else 'depth_nm'
            zip_buf = create_xy_zip(
                fluence_data, element_names, depth_key,
                plot_type, smooth_data, smooth_sigma, selected_elements
            )
            smoothed_tag = "_smoothed" if smooth_data else ""
            st.sidebar.download_button(
                label="📦 Download all fluences (.xy ZIP)",
                data=zip_buf,
                file_name=f"sdtrimsp_all_fluences{smoothed_tag}.zip",
                mime="application/zip",
                type='primary'
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
                        if f'{element}_dens' in df.columns:
                            df[f'{element}_dens_smooth'] = gaussian_filter1d(df[f'{element}_dens'], sigma=smooth_sigma)

                    if 'N_total_conc' in df.columns:
                        df['N_total_conc_smooth'] = gaussian_filter1d(df['N_total_conc'], sigma=smooth_sigma)
                    if 'N_total_frac' in df.columns:
                        df['N_total_frac_smooth'] = gaussian_filter1d(df['N_total_frac'], sigma=smooth_sigma)
                    if 'N_total_dens' in df.columns:
                        df['N_total_dens_smooth'] = gaussian_filter1d(df['N_total_dens'], sigma=smooth_sigma)
                    if 'density' in df.columns:
                        df['density_smooth'] = gaussian_filter1d(df['density'], sigma=smooth_sigma)

                except ImportError:
                    st.warning("⚠️ Smoothing requires scipy. Install with: pip install scipy")
                    st.warning("Using original data without smoothing.")
                    smooth_data = False

            mode = 'lines' if plot_style == "Lines" else 'markers' if plot_style == "Points" else 'lines+markers'

            # Profile views are split into sub-tabs. The single-fluence block is
            # defined last so the "Data Summary" section below stays attached to it.
            single_tab, multi_tab, analysis_tab = st.tabs(
                ["📈 Single Fluence", "📊 Multi-Fluence Comparison", "📌 Fluence Analysis"]
            )

            with multi_tab:
                selected_fluences = st.multiselect(
                    "Select fluences to compare:",
                    fluence_values,
                    default=[fluence_values[0], fluence_values[-1]] if len(fluence_values) > 1 else [fluence_values[0]]
                )
                if len(selected_fluences) > 1:
                    create_multi_fluence_comparison(fluence_data, selected_fluences, depth_col, depth_label, plot_type,
                                                    mode, y_axis_scale, element_names, smooth_data, smooth_sigma,
                                                    selected_elements)
                else:
                    st.info("👆 Select at least two fluences above to show a comparison.")

            with analysis_tab:
                perform_fluence_analysis(fluence_data, element_names, fluence_unit, selected_elements, smooth_data,
                                         smooth_sigma, plot_type)

            with single_tab:
                plot_elements = st.multiselect(
                    "Elements to include in the plot:",
                    element_names,
                    default=element_names,
                    key="single_plot_elements"
                )
                create_single_fluence_plots(df, depth_col, depth_label, plot_type, mode, y_axis_scale,
                                            selected_fluence, element_names, smooth_data, selected_elements,
                                            experimental_data, display_elements=plot_elements)

                st.subheader("📈 Data Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Data Points", len(df))
                    st.metric("Max Depth (Å)", f"{df['depth_A'].max():.1f}")

                with col2:
                    st.metric("Current Fluence", f"{selected_fluence:.2e}")

                    if 'N_total_conc' in df.columns:
                        n_conc_max = df['N_total_conc'].max()
                    elif 'N_conc' in df.columns:
                        n_conc_max = df['N_conc'].max()
                    elif 'N1_conc' in df.columns:
                        n_conc_max = df['N1_conc'].max()
                    else:
                        n_conc_max = None

                    if n_conc_max is not None:
                        st.metric("Max N Concentration", f"{n_conc_max:.2e} atoms/cm³")
                    else:
                        st.metric("Max N Concentration", "N/A")

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
                    elif plot_type == "Density (ions/Å)":
                        data_suffix = "_dens"
                        data_unit = "(atoms/Ų)"
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
                    elif plot_type == "Density (ions/Å)" and 'N_total_dens' in df.columns:
                        display_cols.append('N_total_dens')
                        display_names.append(f'N Total {data_unit}')

                    available_cols = [col for col in display_cols if col in df.columns]
                    available_names = [display_names[i] for i, col in enumerate(display_cols) if col in df.columns]

                    if available_cols:
                        display_df = df[available_cols].copy()
                        display_df.columns = available_names
                        st.dataframe(display_df, width='stretch')

                        csv = display_df.to_csv(index=False)
                        data_type_name = "fractions" if plot_type == "Atomic Fractions" else \
                            "density" if plot_type == "Density (ions/Å)" else \
                                "concentrations"
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
