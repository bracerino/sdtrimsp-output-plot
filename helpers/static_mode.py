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


def parse_sdtrimsp_output_file(file_content, filename):
    lines = file_content.split('\n')

    components = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('CPT') and 'SYMBOL' in stripped and 'A-MASS' in stripped:
            j = i + 1
            while j < len(lines):
                row = lines[j].strip()
                if not row:
                    break
                parts = row.split()
                if len(parts) < 2:
                    break
                try:
                    cpt = int(parts[0])
                    symbol = parts[1]
                    components[cpt] = symbol
                except ValueError:
                    break
                j += 1
            break

    # A run can have several beam components (e.g. the same element implanted at
    # two energies), so collect every projectile with E0 > 0, not just one.
    projectiles = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('CPT') and 'E0' in stripped and ('AlPHA0' in stripped or 'ALPHA0' in stripped):
            j = i + 1
            while j < len(lines):
                row = lines[j].strip()
                if not row:
                    break
                parts = row.split()
                if len(parts) < 2:
                    break
                try:
                    cpt = int(parts[0])
                    e0 = float(parts[1])
                    if e0 > 0:
                        projectiles.append({
                            'cpt': cpt,
                            'symbol': components.get(cpt, f'cpt{cpt}'),
                            'energy_eV': e0
                        })
                except ValueError:
                    break
                j += 1
            break
    projectile = projectiles[0] if projectiles else None

    fluence = None
    for i, line in enumerate(lines):
        if 'fluence:' in line.lower() and 'chamax' in line.lower():
            m = re.search(r'fluence:\s*([\d.eE+\-]+)', line)
            if m:
                try:
                    fluence = float(m.group(1))
                except ValueError:
                    pass
            break

    def parse_sputter_block(start_idx, header_check):
        rows = []
        total = None
        j = start_idx
        while j < len(lines):
            if 'cpt.' in lines[j] and header_check in lines[j]:
                j += 1
                break
            j += 1

        while j < len(lines):
            row = lines[j].strip()
            if not row:
                break
            parts = row.split()
            if not parts:
                break

            if parts[0].lower() == 'all':
                try:
                    total = {
                        'sputt_coef': float(parts[1]),
                        'ener_sputt_coef': float(parts[2]),
                        'mean_energy': float(parts[3]) if len(parts) > 3 else 0.0
                    }
                except (ValueError, IndexError):
                    pass
                break

            try:
                cpt = int(parts[0])
            except ValueError:
                break

            if 'no' in row.lower() and 'sputtering' in row.lower():
                rows.append({
                    'cpt': cpt,
                    'symbol': components.get(cpt, f'cpt{cpt}'),
                    'sputt_coef': 0.0,
                    'ener_sputt_coef': 0.0,
                    'mean_energy': 0.0,
                    'escape_depth': 0.0,
                    'spread': 0.0,
                    'no_sputtering': True
                })
            else:
                try:
                    rows.append({
                        'cpt': cpt,
                        'symbol': components.get(cpt, f'cpt{cpt}'),
                        'sputt_coef': float(parts[1]),
                        'ener_sputt_coef': float(parts[2]),
                        'mean_energy': float(parts[3]) if len(parts) > 3 else 0.0,
                        'escape_depth': float(parts[4]) if len(parts) > 4 else 0.0,
                        'spread': float(parts[5]) if len(parts) > 5 else 0.0,
                        'no_sputtering': False
                    })
                except (ValueError, IndexError):
                    pass
            j += 1

        return rows, total

    # Per-cause breakdown: how many atoms of species X were back-sputtered by
    # each projectile/recoil component Y. SDTrimSP prints one line per (X, Y)
    # pair, e.g. "BACK.SPUTTERED PARTIC.( 6 BY  1)= 875". Summing over Y for a
    # given X recovers that species' total backward-sputtered count, so each
    # term's share of that sum scales the species' yield into a per-cause
    # contribution. This lets a multi-component beam (e.g. the same element at
    # two energies) have its sputtering shown separately per beam component.
    sputtered_by = {}  # (target_cpt, cause_cpt) -> count
    back_sput_re = re.compile(r'BACK\.SPUTTERED PARTIC\.\(\s*(\d+)\s+BY\s+(\d+)\)\s*=\s*([\d.eE+\-]+)')
    for line in lines:
        m = back_sput_re.search(line)
        if m:
            try:
                tgt = int(m.group(1))
                cause = int(m.group(2))
                count = float(m.group(3))
                sputtered_by[(tgt, cause)] = sputtered_by.get((tgt, cause), 0.0) + count
            except ValueError:
                pass

    sputtering = []
    sputtering_total = None
    transmission_sputtering = []
    transmission_total = None

    for k, line in enumerate(lines):
        if 'SPUTTERING DATA (BACKWARD SPUTTERING)' in line:
            sputtering, sputtering_total = parse_sputter_block(k + 1, 'sputt.coef.')
            break

    for k, line in enumerate(lines):
        if 'TRANSMISSION SPUTTERING DATA' in line:
            transmission_sputtering, transmission_total = parse_sputter_block(k + 1, 'sputt.tran.coef.')
            break

    if not sputtering and sputtering_total is None:
        return None

    return {
        'filename': filename,
        'components': components,
        'projectile': projectile,
        'projectiles': projectiles,
        'fluence': fluence,
        'sputtering': sputtering,
        'sputtering_total': sputtering_total,
        'transmission_sputtering': transmission_sputtering,
        'transmission_total': transmission_total,
        'sputtered_by': sputtered_by,
    }


def _render_sputter_cause_breakdown(pf, projs):
    """Show each species' backward-sputtering yield split by the projectile or
    recoil component that caused it.

    SDTrimSP normalises the per-species yield ``Y(X)`` to the total dose, and the
    ``BACK.SPUTTERED PARTIC.(X BY Y)`` counts partition that species' sputtered
    atoms by their cause Y. So Y's share of species X's yield is simply
    ``Y(X) × count(X BY Y) / Σ_Y count(X BY Y)``. Summing those contributions over
    all sputtered species gives the yield attributable to each cause — in
    particular each individual beam component when the beam has several.
    """
    sputtered_by = pf.get('sputtered_by') or {}
    if not sputtered_by:
        return

    components = pf.get('components') or {}
    beam_cpts = {p['cpt'] for p in projs}

    # Per-species totals over all causes, plus that species' reported yield.
    species_total_count = {}
    for (tgt, _cause), count in sputtered_by.items():
        species_total_count[tgt] = species_total_count.get(tgt, 0.0) + count
    species_yield = {
        r['cpt']: (0.0 if r.get('no_sputtering') else r['sputt_coef'])
        for r in pf['sputtering']
    }

    target_cpts = sorted(species_total_count)
    cause_cpts = sorted({cause for (_tgt, cause) in sputtered_by})

    rows = []
    for cause in cause_cpts:
        sym = components.get(cause, f'cpt{cause}')
        kind = 'beam' if cause in beam_cpts else 'recoil/cascade'
        row = {'Caused by': f"{sym} (cpt {cause})", 'Type': kind}
        cause_total_yield = 0.0
        for tgt in target_cpts:
            count = sputtered_by.get((tgt, cause), 0.0)
            tot = species_total_count.get(tgt, 0.0)
            contrib = species_yield.get(tgt, 0.0) * count / tot if tot > 0 else 0.0
            tgt_sym = components.get(tgt, f'cpt{tgt}')
            row[f'Y({tgt_sym} cpt {tgt})'] = contrib
            cause_total_yield += contrib
        row['Total Y (atoms/ion)'] = cause_total_yield
        rows.append(row)

    if not rows:
        return

    with st.expander("🔬 Yield broken down by causing component (beam vs cascade)", expanded=False):
        st.caption(
            "Each sputtered species' yield is split by the projectile/recoil that "
            "caused it, using the `BACK.SPUTTERED PARTIC.(X BY Y)` counts. When the "
            "beam has several components (e.g. the same element at two energies), "
            "this shows **each beam component's own contribution** to the yield, "
            "separately from the collision-cascade (recoil) contribution."
        )
        breakdown_df = pd.DataFrame(rows)
        st.dataframe(breakdown_df, width='stretch', hide_index=True)

        beam_yield = sum(r['Total Y (atoms/ion)'] for r in rows if r['Type'] == 'beam')
        cascade_yield = sum(r['Total Y (atoms/ion)'] for r in rows if r['Type'] != 'beam')
        b1, b2 = st.columns(2)
        with b1:
            st.metric("Yield from beam components (direct)", f"{beam_yield:.5f}")
        with b2:
            st.metric("Yield from recoil cascade", f"{cascade_yield:.5f}")

        st.download_button(
            label="📥 Download yield-by-cause breakdown as CSV",
            data=breakdown_df.to_csv(index=False),
            file_name=f"{pf['filename']}_yield_by_cause.csv",
            mime="text/csv",
            key=f"download_cause_{pf['filename']}",
            type="primary"
        )


def display_sputtering_yields_section(parsed_files):
    st.markdown("### 💥 Sputtering Yields")

    summary_rows = []

    # A chemical symbol can appear in several components (e.g. a target N and an
    # implanted N). Collect the component numbers per symbol so each one gets its
    # own column; symbols with more than one component also get a summed column.
    sym_cpts = {}
    for pf in parsed_files:
        for r in pf['sputtering']:
            sym_cpts.setdefault(r['symbol'], [])
            if r['cpt'] not in sym_cpts[r['symbol']]:
                sym_cpts[r['symbol']].append(r['cpt'])

    yield_columns = []  # (column_name, kind, symbol, cpt)
    for sym in sorted(sym_cpts):
        cpts = sym_cpts[sym]
        if len(cpts) > 1:
            for cpt in cpts:
                yield_columns.append((f'Y({sym} cpt {cpt})', 'comp', sym, cpt))
            yield_columns.append((f'Y({sym}, sum)', 'sum', sym, None))
        else:
            yield_columns.append((f'Y({sym})', 'comp', sym, cpts[0]))

    for pf in parsed_files:
        row = {'File': pf['filename']}
        projs = pf.get('projectiles') or ([pf['projectile']] if pf.get('projectile') else [])
        if projs:
            row['Projectile'] = ", ".join(p['symbol'] for p in projs)
            row['Energy (eV)'] = ", ".join(f"{p['energy_eV']:.0f}" for p in projs)
        else:
            row['Projectile'] = '-'
            row['Energy (eV)'] = '-'

        total = pf.get('sputtering_total')
        row['Total Yield (atoms/ion)'] = total['sputt_coef'] if total else float('nan')

        cpt_yields = {
            (r['symbol'], r['cpt']): (0.0 if r.get('no_sputtering') else r['sputt_coef'])
            for r in pf['sputtering']
        }
        for col_name, kind, sym, cpt in yield_columns:
            if kind == 'comp':
                row[col_name] = cpt_yields.get((sym, cpt), 0.0)
            else:  # summed over all components of this symbol
                row[col_name] = sum(v for (s, _c), v in cpt_yields.items() if s == sym)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    st.markdown("#### Summary across files")
    if any(len(cpts) > 1 for cpts in sym_cpts.values()):
        st.caption(
            "Some elements have several components (same symbol) — they are shown "
            "as separate `Y(sym cpt N)` columns plus a combined `Y(sym, sum)` column."
        )
    st.dataframe(summary_df, width='stretch', hide_index=True)

    csv_summary = summary_df.to_csv(index=False)
    st.download_button(
        label="📥 Download summary as CSV",
        data=csv_summary,
        file_name="sputtering_yields_summary.csv",
        mime="text/csv",
        key="download_sputter_summary",
        type="primary"
    )

    st.markdown("#### Per-file detail")
    tab_names = [f"📄 {pf['filename']}" for pf in parsed_files]
    file_tabs = st.tabs(tab_names)

    for tab, pf in zip(file_tabs, parsed_files):
        with tab:
            info_cols = st.columns(3)
            projs = pf.get('projectiles') or ([pf['projectile']] if pf.get('projectile') else [])
            with info_cols[0]:
                st.metric("Projectile", ", ".join(p['symbol'] for p in projs) if projs else '-')
            with info_cols[1]:
                st.metric("Energy (eV)",
                          ", ".join(f"{p['energy_eV']:.0f}" for p in projs) if projs else '-')
            with info_cols[2]:
                total = pf.get('sputtering_total')
                st.metric("Total backward yield (atoms/ion)",
                          f"{total['sputt_coef']:.5f}" if total else '-')

            st.markdown("##### Backward sputtering")
            back_rows = []
            for r in pf['sputtering']:
                if r.get('no_sputtering'):
                    back_rows.append({
                        'CPT': r['cpt'],
                        'Symbol': r['symbol'],
                        'Yield (atoms/ion)': 0.0,
                        'Energy yield': 0.0,
                        'Mean energy (eV)': 0.0,
                        'Escape depth (Å)': 0.0,
                        'Spread (Å)': 0.0,
                        'Note': 'no backward sputtering'
                    })
                else:
                    back_rows.append({
                        'CPT': r['cpt'],
                        'Symbol': r['symbol'],
                        'Yield (atoms/ion)': r['sputt_coef'],
                        'Energy yield': r['ener_sputt_coef'],
                        'Mean energy (eV)': r['mean_energy'],
                        'Escape depth (Å)': r['escape_depth'],
                        'Spread (Å)': r['spread'],
                        'Note': ''
                    })
            if total:
                back_rows.append({
                    'CPT': 'all',
                    'Symbol': '—',
                    'Yield (atoms/ion)': total['sputt_coef'],
                    'Energy yield': total['ener_sputt_coef'],
                    'Mean energy (eV)': total['mean_energy'],
                    'Escape depth (Å)': float('nan'),
                    'Spread (Å)': float('nan'),
                    'Note': 'total'
                })
            back_df = pd.DataFrame(back_rows)
            st.dataframe(back_df, width='stretch', hide_index=True)

            csv_back = back_df.to_csv(index=False)
            st.download_button(
                label="📥 Download backward yields as CSV",
                data=csv_back,
                file_name=f"{pf['filename']}_backward_sputtering.csv",
                mime="text/csv",
                key=f"download_back_{pf['filename']}",
                type="primary"
            )

            # Optional breakdown of each species' yield by the projectile/recoil
            # component that caused it. Useful when the beam is several components
            # of the same element at different energies — each beam component's
            # contribution to the sputtering yield can then be read separately.
            _render_sputter_cause_breakdown(pf, projs)

            if pf['transmission_sputtering'] or pf.get('transmission_total'):
                st.markdown("##### Transmission sputtering")
                trans_rows = []
                for r in pf['transmission_sputtering']:
                    if r.get('no_sputtering'):
                        trans_rows.append({
                            'CPT': r['cpt'],
                            'Symbol': r['symbol'],
                            'Yield (atoms/ion)': 0.0,
                            'Energy yield': 0.0,
                            'Mean energy (eV)': 0.0,
                            'Note': 'no transmission sputtering'
                        })
                    else:
                        trans_rows.append({
                            'CPT': r['cpt'],
                            'Symbol': r['symbol'],
                            'Yield (atoms/ion)': r['sputt_coef'],
                            'Energy yield': r['ener_sputt_coef'],
                            'Mean energy (eV)': r['mean_energy'],
                            'Note': ''
                        })
                trans_total = pf.get('transmission_total')
                if trans_total:
                    trans_rows.append({
                        'CPT': 'all',
                        'Symbol': '—',
                        'Yield (atoms/ion)': trans_total['sputt_coef'],
                        'Energy yield': trans_total['ener_sputt_coef'],
                        'Mean energy (eV)': trans_total['mean_energy'],
                        'Note': 'total'
                    })
                trans_df = pd.DataFrame(trans_rows)
                st.dataframe(trans_df, width='stretch', hide_index=True)

            if pf.get('components'):
                with st.expander("ℹ️ Component legend", expanded=False):
                    comp_df = pd.DataFrame(
                        [{'CPT': c, 'Symbol': s} for c, s in sorted(pf['components'].items())]
                    )
                    st.dataframe(comp_df, width='stretch', hide_index=True)


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

    # Detect the kind of depth-distribution file. Projectile files
    # (depth_proj.dat) carry only projectile range / energy-loss data and do
    # NOT contain the per-depth target composition, whereas recoil/damage
    # files (depth_recoil.dat / depth_damage.dat) list every component.
    is_projectile = 'DEPTH DISTRIBUTIONS (PROJECTILES)' in file_content
    file_type = 'projectile' if is_projectile else 'recoil'

    # Collect each COMPONENT block as a separate entry. Several components can
    # share the same chemical symbol (e.g. a target N and an implanted N, or two
    # beam N's at different energies all printed as "N"), so we must NOT key them
    # by symbol alone — that would let later components overwrite earlier ones.
    parsed_components = []
    current_symbol = None
    current_cpt = None
    current_data = []
    in_data_section = False
    header_columns = []

    def flush_component():
        if current_symbol and current_data:
            parsed_components.append({
                'symbol': current_symbol,
                'cpt': current_cpt,
                'data': current_data.copy(),
                'header': header_columns.copy(),
            })

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        if line_stripped.startswith('# CPT.') or line_stripped.startswith('#Calculated Values'):
            flush_component()
            current_data = []
            break

        if line_stripped.startswith('# DEPTH DISTRIBUTIONS') and 'COMPONENT' in line_stripped:
            flush_component()
            current_data = []

            parts = line.split(':')
            if len(parts) >= 2:
                element_part = parts[1].strip()
                current_symbol = element_part.split()[0].strip()
                m = re.search(r'COMPONENT\(\s*(\d+)\s*\)', line)
                current_cpt = int(m.group(1)) if m else None
                current_data = []
                header_columns = []
                in_data_section = False

        elif line_stripped.startswith('# DEPTH DISTRIBUTIONS') and 'ALL' in line_stripped:
            flush_component()
            current_symbol = None
            current_cpt = None
            current_data = []
            in_data_section = False
            continue

        elif current_symbol and line_stripped.startswith('#') and (
                'DEPTH' in line_stripped or 'DEPTH/LENGTH' in line_stripped) and 'STOPS' in line_stripped:
            header_parts = line_stripped.replace('#', '').split()
            header_columns = header_parts
            in_data_section = True
            continue

        elif in_data_section and current_symbol and line_stripped and not line_stripped.startswith('#'):
            if 'sum' in line_stripped.lower():
                in_data_section = False
                flush_component()
                current_data = []
                current_symbol = None
                current_cpt = None
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

    flush_component()

    # Build the final dict with display keys that stay unique when a symbol is
    # repeated: a lone symbol keeps its plain name ("Ti"), while repeats are
    # disambiguated by component number ("N (cpt 1)", "N (cpt 5)", ...).
    symbol_counts = {}
    for c in parsed_components:
        symbol_counts[c['symbol']] = symbol_counts.get(c['symbol'], 0) + 1

    elements_data = {}
    for c in parsed_components:
        sym = c['symbol']
        if symbol_counts[sym] > 1 and c['cpt'] is not None:
            key = f"{sym} (cpt {c['cpt']})"
        elif symbol_counts[sym] > 1:
            # No component number available; fall back to an occurrence index.
            n = sum(1 for k in elements_data if k.split(' (')[0] == sym) + 1
            key = f"{sym} (#{n})"
        else:
            key = sym
        elements_data[key] = {
            'data': c['data'],
            'header': c['header'],
            'symbol': sym,
            'cpt': c['cpt'],
        }

    return elements_data, file_type


def calculate_processed_data_static(elements_data, column_name, fluence_per_cm2=None):
    """Process per-depth values into the various plot metrics.

    Combined / Σ profiles are *not* computed here. They are built afterwards by
    :func:`sum_processed_series`, which sums the finished per-element metrics — so
    each combined curve is a true superposition of its parts for every data type.
    """
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

    for element, element_info in elements_data.items():
        value_by_depth = {}
        for point in element_info['data']:
            value_by_depth[point['depth']] = value_by_depth.get(point['depth'], 0) + point[column_name]
        element_total = sum(value_by_depth.values())

        processed_data[element] = []

        for depth in all_depths:
            value = value_by_depth.get(depth, 0)

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


DATA_TYPE_OPTIONS = [
    "Raw Values",
    "Atomic Fractions",
    "Normalized Probability",
    "Density (ions/Å)",
    "Concentration (ions/cm³)",
]


def build_symbol_groups(elements_data):
    """Map each chemical symbol to the list of component keys that share it.

    Returns an ordered dict-like {symbol: [keys...]} preserving first-seen order.
    Symbols mapping to more than one key are the ones that can be summed.
    """
    groups = {}
    for key, info in elements_data.items():
        sym = info.get('symbol', key)
        groups.setdefault(sym, []).append(key)
    return groups


# Every metric of a combined/Σ profile is summed here.
_SUMMABLE_METRICS = (
    'raw_value', 'atomic_fraction', 'probability', 'density_per_angstrom', 'concentration'
)


def sum_processed_series(processed_data, member_keys):
    """Point-wise sum of several already-processed element series.

    A combined (or Σ) profile must be the *superposition* of its members for
    **every** data type — raw counts, atomic fraction, normalized probability,
    density and concentration alike — exactly like the dynamic mode, where the
    combined curve is the sum of the selected element columns.

    Summing the members' processed values (rather than re-normalizing a synthetic
    summed-raw series) is what makes this correct: normalized probability,
    density and concentration are each normalized *per element* by that element's
    own total, so re-processing a summed-raw series would divide by the combined
    total and push the combined curve *below* its individual parts. Summing the
    finished metrics keeps the combined ≥ each part, as expected.

    All series produced by :func:`calculate_processed_data_static` share the same
    sorted depth grid, so a straight index-by-index sum is safe.
    """
    members = [processed_data[k] for k in member_keys if k in processed_data]
    if not members:
        return []
    n = len(members[0])
    out = []
    for i in range(n):
        point = {'depth': members[0][i]['depth']}
        for metric in _SUMMABLE_METRICS:
            point[metric] = sum(m[i].get(metric, 0.0) for m in members)
        out.append(point)
    return out


def get_supported_data_types(file_type):
    """Return the 'Data Type to Plot' options that are physically meaningful
    for a given parsed file type."""
    if file_type == 'projectile':
        # depth_proj.dat stores only the projectile depth/energy-loss data,
        # not the per-depth target composition, so atomic fractions (which need
        # the number of atoms of every species at each depth) are undefined.
        return [t for t in DATA_TYPE_OPTIONS if t != "Atomic Fractions"]
    return list(DATA_TYPE_OPTIONS)


def explain_unsupported_data_type(plot_data_type, file_type):
    if file_type == 'projectile' and plot_data_type == "Atomic Fractions":
        return (
            "**Atomic Fractions** require the number of target atoms of each "
            "species at every depth. Projectile files (`depth_proj.dat`) only "
            "store the projectile range and energy-loss data (STOPS, PATHLENGTH, "
            "NUCL./ELECT. LOSS, NUM.COLL., NRT-DPA) — they do **not** contain the "
            "per-depth target composition, so an atomic fraction cannot be "
            "computed.\n\nUse **Raw Values**, **Normalized Probability**, "
            "**Density (ions/Å)** or **Concentration (ions/cm³)** for this file."
        )
    return f"“{plot_data_type}” is not available for this file type ({file_type})."


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

        st.info(
            "Upload your SDTrimSP files below. Each file is automatically recognised as either a "
            "**depth profile** (`depth_damage.dat` / `depth_proj.dat`) or a "
            "**sputtering yield** output (`output.dat`)."
        )

        uploaded_static_files = st.file_uploader(
            "Upload depth profile and/or output.dat files",
            type=['dat', 'txt', 'out'],
            accept_multiple_files=True,
            key="static_combined_files",
            help="Depth profiles (depth_damage.dat / depth_proj.dat) and SDTrimSP output.dat "
                 "files can be mixed — the type of each file is detected automatically"
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
        exp_success_lines = []
        exp_error_lines = []
        if uploaded_exp_files:
            for exp_file in uploaded_exp_files:
                exp_content = str(exp_file.read(), "utf-8")
                exp_df, exp_info = parse_experimental_data(exp_content, exp_file.name)
                if exp_df is not None:
                    experimental_data.append((exp_file.name, exp_df, exp_info))
                    exp_success_lines.append(f"✅ **{exp_file.name}** — {exp_info}")
                else:
                    exp_error_lines.append(f"❌ **{exp_file.name}** — {exp_info}")
            if exp_success_lines:
                st.sidebar.success("\n\n".join(exp_success_lines))
            if exp_error_lines:
                st.sidebar.error("\n\n".join(exp_error_lines))

        all_file_data = []
        sputter_parsed = []
        depth_info_lines = []
        sputter_info_lines = []
        unrecognised_lines = []
        if uploaded_static_files:
            for uploaded_file in uploaded_static_files:
                file_content = str(uploaded_file.read(), "utf-8")

                # Auto-detect the file type. output.dat sputtering files carry a
                # distinctive "SPUTTERING DATA" header, so try that parser first;
                # it returns None for anything else. Otherwise fall back to the
                # depth-profile parser, which returns an empty dict on a mismatch.
                parsed_sputter = parse_sdtrimsp_output_file(file_content, uploaded_file.name)
                if parsed_sputter is not None:
                    sputter_parsed.append(parsed_sputter)
                    total = parsed_sputter.get('sputtering_total')
                    if total:
                        sputter_info_lines.append(
                            f"✅ **{uploaded_file.name}** — 💥 sputtering yields — total Y = {total['sputt_coef']:.4g}"
                        )
                    else:
                        sputter_info_lines.append(
                            f"✅ **{uploaded_file.name}** — 💥 sputtering yields"
                        )
                    continue

                elements_data, file_type = parse_static_damage_file(file_content, uploaded_file.name)
                if elements_data:
                    all_file_data.append({
                        'filename': uploaded_file.name,
                        'elements_data': elements_data,
                        'file_type': file_type
                    })

                    sample_element = list(elements_data.keys())[0]
                    bin_str = "n/a"
                    if len(elements_data[sample_element]['data']) > 1:
                        depths = [p['depth'] for p in elements_data[sample_element]['data']]
                        depths.sort()
                        spacings = [depths[i + 1] - depths[i] for i in range(len(depths) - 1)]
                        avg_spacing = sum(spacings) / len(spacings)
                        bin_str = f"{avg_spacing:.2f} Å"
                    type_label = "projectile (depth_proj)" if file_type == 'projectile' else "recoil/damage"
                    depth_info_lines.append(
                        f"✅ **{uploaded_file.name}** — 🗂️ depth profile, {type_label} — 📏 bin width: {bin_str}"
                    )
                else:
                    unrecognised_lines.append(
                        f"❌ **{uploaded_file.name}** — not recognised as a depth profile or output.dat"
                    )

            recognised_lines = depth_info_lines + sputter_info_lines
            if recognised_lines:
                st.sidebar.success("\n\n".join(recognised_lines))
            if unrecognised_lines:
                st.sidebar.error("\n\n".join(unrecognised_lines))

        if sputter_parsed:
            display_sputtering_yields_section(sputter_parsed)
            st.markdown("---")

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
                        file_type = file_data.get('file_type', 'recoil')

                        # When a chemical symbol appears in several components
                        # (e.g. target N + implanted N, or two beam N's), each
                        # component is listed separately and an extra Σ option
                        # plots their sum.
                        groups = build_symbol_groups(file_data['elements_data'])
                        sum_option_map = {}
                        plot_options = list(available_elements)
                        for sym, keys in groups.items():
                            if len(keys) > 1:
                                cpts = [
                                    str(file_data['elements_data'][k].get('cpt'))
                                    for k in keys
                                ]
                                sum_label = f"Σ {sym} (sum of components {', '.join(cpts)})"
                                plot_options.append(sum_label)
                                sum_option_map[sum_label] = keys

                        if sum_option_map:
                            st.caption(
                                "This file has multiple components of the same element — "
                                "each one is listed separately, plus a **Σ** option that sums them."
                            )

                        col1, col2 = st.columns([1, 1])

                        with col1:
                            selected_elements = st.multiselect(
                                "Select element(s) / components",
                                options=plot_options,
                                default=available_elements,
                                key=f"element_select_{file_idx}",
                                help="Pick any number of components. Σ entries plot the "
                                     "sum of all components of that element."
                            )

                        with col2:
                            # Projectile files (depth_proj.dat) have no VACANCIES
                            # column (their last column is NRT-DPA), so only STOPS
                            # is offered for them.
                            if file_type == 'projectile':
                                available_columns = ['STOPS']
                            else:
                                available_columns = ['STOPS', 'VACANCIES']

                            selected_column = st.selectbox(
                                "Select column to plot",
                                options=available_columns,
                                key=f"column_select_{file_idx}"
                            )

                        # Combine any arbitrary set of components into one summed
                        # profile, mirroring the "Select elements to combine"
                        # feature of dynamic mode. Unlike the Σ option (which only
                        # sums components of the *same* chemical symbol), this lets
                        # the user add together different elements.
                        combine_selection = st.multiselect(
                            "Combine elements into a single profile (sum):",
                            options=available_elements,
                            default=[],
                            key=f"combine_select_{file_idx}",
                            help="Pick two or more components to additionally plot "
                                 "their summed profile — just like the element "
                                 "combination in dynamic mode."
                        )

                        if (selected_elements or len(combine_selection) >= 2) and selected_column:
                            column_key = selected_column.lower()

                            processed_data = calculate_processed_data_static(
                                file_data['elements_data'],
                                column_key,
                                fluence,
                            )

                            # Build Σ (same-symbol) sums and the user-defined
                            # combination as the point-wise SUM of their members'
                            # processed metrics, so every data type (raw, atomic
                            # fraction, probability, density, concentration) is a
                            # true superposition of the parts — matching dynamic
                            # mode and keeping the combined ≥ each individual part.
                            for sel in selected_elements:
                                if sel in sum_option_map:
                                    processed_data[sel] = sum_processed_series(
                                        processed_data, sum_option_map[sel]
                                    )

                            combined_key = None
                            if len(combine_selection) >= 2:
                                combined_key = f"Combined ({'+'.join(combine_selection)})"
                                processed_data[combined_key] = sum_processed_series(
                                    processed_data, combine_selection
                                )

                            for sel in selected_elements:
                                data_to_use = processed_data.get(sel)
                                if data_to_use is None:
                                    continue
                                label = f"{file_data['filename']} - {sel} - {selected_column}"
                                selected_data_to_plot.append({
                                    'filename': file_data['filename'],
                                    'element': sel,
                                    'column': selected_column,
                                    'data': data_to_use,
                                    'label': label,
                                    'file_type': file_type
                                })

                            if combined_key and processed_data.get(combined_key):
                                label = f"{file_data['filename']} - {combined_key} - {selected_column}"
                                selected_data_to_plot.append({
                                    'filename': file_data['filename'],
                                    'element': combined_key,
                                    'column': selected_column,
                                    'data': processed_data[combined_key],
                                    'label': label,
                                    'file_type': file_type
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

                # Keep only the simulation traces for which the chosen data type
                # is physically meaningful; warn (instead of plotting) for the rest.
                compatible_items = []
                incompatible = {}
                for d_item in selected_data_to_plot:
                    ftype = d_item.get('file_type', 'recoil')
                    if plot_data_type in get_supported_data_types(ftype):
                        compatible_items.append(d_item)
                    else:
                        msg = explain_unsupported_data_type(plot_data_type, ftype)
                        incompatible.setdefault(msg, []).append(d_item['label'])

                for msg, labels in incompatible.items():
                    skipped = ", ".join(f"**{l}**" for l in labels)
                    st.warning(
                        f"⚠️ Data type “{plot_data_type}” is not available for: "
                        f"{skipped}.\n\n{msg}"
                    )

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

                for idx, data_item in enumerate(compatible_items):
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

                if compatible_items or experimental_data:
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info(
                        f"ℹ️ Nothing to plot for data type “{plot_data_type}”. "
                        "See the message(s) above for why the selected file(s) do not "
                        "support this metric, then pick a different data type."
                    )

                if st.checkbox("Show Data Table", key="static_show_table"):
                    st.subheader("Selected Data")

                    if compatible_items:
                        for idx, data_item in enumerate(compatible_items):
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
                        if compatible_items:
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
                    st.info("👆 Upload simulation files above (or experimental data in the sidebar) to start plotting")
        else:
            if not sputter_parsed:
                st.info("👆 Upload simulation files above, and/or experimental data in the sidebar, to begin")

        return True

    return False
