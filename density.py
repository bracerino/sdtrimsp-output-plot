import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


COMMON_ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm'
]


def calculate_implant_density(target_total_density, implant_concentration, substrate_elements):
    if implant_concentration <= 0 or implant_concentration >= 1:
        raise ValueError(f"Implant concentration must be between 0 and 1, got {implant_concentration}")

    if target_total_density <= 0:
        raise ValueError(f"Target total density must be positive, got {target_total_density}")

    if not substrate_elements:
        raise ValueError("Substrate elements dictionary cannot be empty")

    substrate_concentration_sum = sum(c for c, d in substrate_elements.values())
    total_concentration = substrate_concentration_sum + implant_concentration

    if not np.isclose(total_concentration, 1.0, rtol=1e-5):
        raise ValueError(
            f"Total concentration must equal 1.0, got {total_concentration:.6f} "
            f"(substrate: {substrate_concentration_sum:.6f}, implant: {implant_concentration:.6f})"
        )

    substrate_contribution = sum(c / d for c, d in substrate_elements.values())

    inverse_total_density = 1.0 / target_total_density
    denominator = inverse_total_density - substrate_contribution

    if denominator <= 0:
        raise ValueError(
            f"Invalid configuration: cannot achieve target density. "
            f"1/d_total ({inverse_total_density:.6f}) must be greater than "
            f"substrate contribution ({substrate_contribution:.6f})"
        )

    implant_density = implant_concentration / denominator

    implant_contribution = implant_concentration / implant_density
    calculated_inverse_density = substrate_contribution + implant_contribution
    calculated_total_density = 1.0 / calculated_inverse_density

    validation_info = {
        'total_concentration': total_concentration,
        'substrate_contribution': substrate_contribution,
        'implant_contribution': implant_contribution,
        'calculated_total_density': calculated_total_density,
        'density_error': abs(calculated_total_density - target_total_density)
    }

    return implant_density, validation_info


def calculate_substrate_concentrations(total_elements, implant_concentration):
    if total_elements <= 0:
        raise ValueError("Total elements must be positive")

    if implant_concentration <= 0 or implant_concentration >= 1:
        raise ValueError("Implant concentration must be between 0 and 1")

    remaining_concentration = 1.0 - implant_concentration
    substrate_concentration = remaining_concentration / total_elements

    return substrate_concentration


def initialize_session_state():
    if 'substrate_elements' not in st.session_state:
        st.session_state.substrate_elements = [
            {'name': 'Hf', 'concentration': 0.15, 'density': 0.0468},
            {'name': 'Nb', 'concentration': 0.15, 'density': 0.0468},
            {'name': 'Ta', 'concentration': 0.15, 'density': 0.0468},
            {'name': 'Ti', 'concentration': 0.15, 'density': 0.0468},
            {'name': 'Zr', 'concentration': 0.15, 'density': 0.0468},
        ]

    if 'calculation_done' not in st.session_state:
        st.session_state.calculation_done = False


def add_substrate_element():
    new_element = {
        'name': f'Element{len(st.session_state.substrate_elements)+1}',
        'concentration': 0.1,
        'density': 0.05
    }
    st.session_state.substrate_elements.append(new_element)


def remove_substrate_element(index):
    if len(st.session_state.substrate_elements) > 1:
        st.session_state.substrate_elements.pop(index)

        keys_to_remove = []
        for key in st.session_state.keys():
            if key.startswith(('name_input_', 'conc_input_', 'dens_input_', 'elem_select_')):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del st.session_state[key]


def normalize_concentrations(implant_conc):
    total_substrate = sum(elem['concentration'] for elem in st.session_state.substrate_elements)
    if total_substrate > 0:
        target_substrate = 1.0 - implant_conc
        for i, elem in enumerate(st.session_state.substrate_elements):
            new_conc = (elem['concentration'] / total_substrate) * target_substrate
            elem['concentration'] = new_conc
            st.session_state[f"conc_input_{i}"] = new_conc


def create_concentration_pie_chart(substrate_elements, implant_name, implant_conc):
    labels = [elem['name'] for elem in substrate_elements] + [implant_name]
    values = [elem['concentration'] for elem in substrate_elements] + [implant_conc]

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
              '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont_size=14
    )])

    fig.update_layout(
        title="Atomic Concentration Distribution",
        height=400,
        showlegend=True,
        font=dict(size=12)
    )

    return fig


def init_density_calc():
    initialize_session_state()


def density_calculator_interface():
    initialize_session_state()

    st.title("🧮 Atomic Density Calculator")
    st.markdown("""
    Calculate the required atomic density for an **implanted element** to achieve 
    a **target total material density** based on the mixture rule.
    """)

    with st.expander("📐 Formula", expanded=False):
        st.latex(r"\frac{1}{d_{total}} = \sum_{i=1}^{n} \frac{c_i}{d_i}")
        st.markdown("**Where:**")
        st.markdown("- $d_{total}$ = total atomic density of material (atoms/Ų)")
        st.markdown("- $c_i$ = atomic concentration of component $i$ (fraction, 0-1)")
        st.markdown("- $d_i$ = atomic density of component $i$ (atoms/Ų)")
        st.markdown("---")
        st.latex(r"d_{implant} = \frac{c_{implant}}{\frac{1}{d_{total}} - \sum_{substrate} \frac{c_i}{d_i}}")

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("⚙️ Configuration")

        st.markdown("#### 🎯 Target Parameters")
        col1, col2 = st.columns(2)

        with col1:
            target_density = st.number_input(
                "Target Total Density (atoms/Ų):",
                min_value=0.001,
                max_value=1.0,
                value=0.0562,
                step=0.0001,
                format="%.4f",
                help="The desired total atomic density of the final material"
            )

        with col2:
            implant_element = st.text_input(
                "Implanted Element:",
                value="N",
                help="Symbol or name of the element being implanted"
            )

        implant_conc = st.slider(
            "Implant Concentration (fraction):",
            min_value=0.01,
            max_value=0.99,
            value=0.25,
            step=0.01,
            format="%.2f",
            help="Target concentration of the implanted element"
        )

        st.markdown("---")

        # Substrate elements section
        st.markdown("#### 🔬 Substrate Elements")

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("➕ Add Element", width='stretch'):
                add_substrate_element()
                st.rerun()

        with col_btn2:
            if st.button("🔄 Auto-Normalize", width='stretch'):
                normalize_concentrations(implant_conc)
                st.rerun()

        with col_btn3:
            total_substrate_conc = sum(elem['concentration'] for elem in st.session_state.substrate_elements)
            total_conc = total_substrate_conc + implant_conc
            if abs(total_conc - 1.0) < 0.001:
                st.success(f"✓ {total_conc:.3f}")
            else:
                st.warning(f"⚠️ {total_conc:.3f}")


        st.markdown("**Substrate Composition:**")

        if st.session_state.substrate_elements:
            col_header1, col_header2, col_header3, col_header4 = st.columns([2, 2, 2, 1])
            with col_header1:
                st.markdown("**Element**")
            with col_header2:
                st.markdown("**Concentration**")
            with col_header3:
                st.markdown("**Density (atoms/Ų)**")
            with col_header4:
                st.markdown("**Remove**")

        for i, elem in enumerate(st.session_state.substrate_elements):
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

                with col1:
                    element_options = ['Custom...'] + COMMON_ELEMENTS

                    if elem['name'] in COMMON_ELEMENTS:
                        default_index = COMMON_ELEMENTS.index(elem['name']) + 1
                    else:
                        default_index = 0

                    selected_element = st.selectbox(
                        "Element",
                        options=element_options,
                        index=default_index,
                        key=f"elem_select_{i}",
                        label_visibility="collapsed"
                    )

                    if selected_element == 'Custom...':
                        new_name = st.text_input(
                            "Custom element",
                            value=elem['name'] if elem['name'] not in COMMON_ELEMENTS else '',
                            key=f"name_input_{i}",
                            label_visibility="collapsed",
                            placeholder="Enter element name"
                        )
                        elem['name'] = new_name
                    else:
                        elem['name'] = selected_element

                with col2:
                    new_conc = st.number_input(
                        "Concentration",
                        min_value=0.0,
                        max_value=1.0,
                        value=elem['concentration'],
                        step=0.01,
                        format="%.4f",
                        key=f"conc_input_{i}",
                        label_visibility="collapsed"
                    )
                    elem['concentration'] = new_conc

                with col3:
                    new_dens = st.number_input(
                        "Density (atoms/Ų)",
                        min_value=0.001,
                        max_value=1.0,
                        value=elem['density'],
                        step=0.0001,
                        format="%.4f",
                        key=f"dens_input_{i}",
                        label_visibility="collapsed"
                    )
                    elem['density'] = new_dens

                with col4:
                    if st.button("🗑️", key=f"del_{i}", width='stretch'):
                        remove_substrate_element(i)
                        st.rerun()

    with col_right:
        st.subheader("📊 Visualization")

        # Pie chart
        fig = create_concentration_pie_chart(
            st.session_state.substrate_elements,
            implant_element,
            implant_conc
        )
        st.plotly_chart(fig, width='stretch')

        st.markdown("#### 📈 Quick Stats")

        total_substrate_conc = sum(elem['concentration'] for elem in st.session_state.substrate_elements)
        total_conc = total_substrate_conc + implant_conc

        st.metric("Total Concentration", f"{total_conc:.4f}")
        st.metric("Substrate Elements", len(st.session_state.substrate_elements))
        st.metric("Implant Fraction", f"{implant_conc:.2%}")

        st.markdown("---")
        if st.button("🔬 Calculate Required Density", type="primary", width='stretch'):
            total_substrate_conc = sum(elem['concentration'] for elem in st.session_state.substrate_elements)
            total_conc = total_substrate_conc + implant_conc

            if abs(total_conc - 1.0) > 0.001:
                st.error(f"❌ Error: Total concentration must equal 1.0, but got {total_conc:.4f}")
                st.info("💡 Tip: Click the 'Auto-Normalize' button to adjust concentrations automatically")
            else:
                try:
                    substrate_dict = {
                        elem['name']: (elem['concentration'], elem['density'])
                        for elem in st.session_state.substrate_elements
                    }

                    implant_density, validation = calculate_implant_density(
                        target_total_density=target_density,
                        implant_concentration=implant_conc,
                        substrate_elements=substrate_dict
                    )

                    st.session_state.calculation_done = True
                    st.session_state.result = {
                        'implant_density': implant_density,
                        'validation': validation,
                        'implant_element': implant_element,
                        'implant_conc': implant_conc,
                        'target_density': target_density,
                        'substrate_dict': substrate_dict
                    }
                    st.rerun()

                except ValueError as e:
                    st.error(f"❌ Calculation Error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {str(e)}")

        if st.session_state.calculation_done and 'result' in st.session_state:
            result = st.session_state.result

            st.markdown("---")
            st.success("✅ Calculation Complete!")

            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;'>
                <h3 style='color: white; margin: 0; font-size: 1.2em;'>Required {result['implant_element']} Density</h3>
                <h1 style='color: white; font-size: 2.5em; margin: 10px 0;'>{result['implant_density']:.4f}</h1>
                <p style='color: white; font-size: 1em; margin: 0;'>atoms/Ų</p>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.calculation_done and 'result' in st.session_state:
        result = st.session_state.result

        st.markdown("---")
        st.subheader("📊 Detailed Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Target Density",
                f"{result['target_density']:.4f}",
                delta="atoms/Ų",
                delta_color="off"
            )

        with col2:
            st.metric(
                f"{result['implant_element']} Concentration",
                f"{result['implant_conc']:.2%}",
                delta="fraction",
                delta_color="off"
            )

        with col3:
            st.metric(
                "Total Concentration",
                f"{result['validation']['total_concentration']:.6f}",
                delta="✓ Valid" if abs(result['validation']['total_concentration'] - 1.0) < 1e-6 else "✗ Invalid",
                delta_color="normal" if abs(result['validation']['total_concentration'] - 1.0) < 1e-6 else "inverse"
            )

        with col4:
            st.metric(
                "Density Error",
                f"{result['validation']['density_error']:.2e}",
                delta="✓ Verified" if result['validation']['density_error'] < 1e-10 else "⚠ Check",
                delta_color="normal" if result['validation']['density_error'] < 1e-10 else "inverse"
            )

        with st.expander("📋 Detailed Breakdown", expanded=True):
            tab1, tab2, tab3 = st.tabs(["📊 Summary Table", "🔬 Validation", "📐 Calculation Steps"])

            with tab1:
                summary_data = []

                for elem, (conc, dens) in result['substrate_dict'].items():
                    summary_data.append({
                        'Element': elem,
                        'Type': 'Substrate',
                        'Concentration': f"{conc:.4f}",
                        'Density (atoms/Ų)': f"{dens:.4f}",
                        'Contribution (c/d)': f"{conc/dens:.4f}"
                    })

                summary_data.append({
                    'Element': result['implant_element'],
                    'Type': 'Implanted',
                    'Concentration': f"{result['implant_conc']:.4f}",
                    'Density (atoms/Ų)': f"{result['implant_density']:.4f}",
                    'Contribution (c/d)': f"{result['implant_conc']/result['implant_density']:.4f}"
                })

                df = pd.DataFrame(summary_data)
                st.dataframe(df, width='stretch', hide_index=True)

            with tab2:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Validation Checks:**")
                    checks = {
                        "Total Concentration": result['validation']['total_concentration'],
                        "Substrate Contribution": result['validation']['substrate_contribution'],
                        "Implant Contribution": result['validation']['implant_contribution'],
                        "Calculated Total Density": result['validation']['calculated_total_density'],
                    }

                    for check, value in checks.items():
                        if "Concentration" in check:
                            st.info(f"**{check}:** {value:.6f}")
                        elif "Contribution" in check:
                            st.info(f"**{check}:** {value:.6f}")
                        else:
                            st.info(f"**{check}:** {value:.4f} atoms/Ų")

                with col2:
                    st.markdown("**Status:**")
                    if abs(result['validation']['total_concentration'] - 1.0) < 1e-6:
                        st.success("✅ Total concentration = 1.0")
                    else:
                        st.error("❌ Total concentration ≠ 1.0")

                    if result['validation']['density_error'] < 1e-10:
                        st.success("✅ Density calculation verified")
                    else:
                        st.warning("⚠️ Small numerical error detected")

                    st.success("✅ All validations passed")

            with tab3:
                st.markdown("**Step-by-step calculation:**")

                substrate_sum = sum(conc/dens for conc, dens in result['substrate_dict'].values())

                st.code(f"""
Step 1: Calculate substrate contribution
    Σ(c_substrate/d_substrate) = {substrate_sum:.6f}

Step 2: Calculate denominator
    1/d_total - Σ(c_substrate/d_substrate)
    = 1/{result['target_density']:.4f} - {substrate_sum:.6f}
    = {1/result['target_density']:.6f} - {substrate_sum:.6f}
    = {1/result['target_density'] - substrate_sum:.6f}

Step 3: Calculate implant density
    d_implant = c_implant / (1/d_total - Σ(c_substrate/d_substrate))
    d_implant = {result['implant_conc']:.4f} / {1/result['target_density'] - substrate_sum:.6f}
    d_implant = {result['implant_density']:.4f} atoms/Ų
                """)

        st.markdown("---")
        st.markdown("### 💾 Export Results")

        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            export_text = f"""Atomic Density Calculation Results
{'='*60}

TARGET CONFIGURATION:
  Total Density: {result['target_density']:.4f} atoms/Ų
  Implanted Element: {result['implant_element']}
  Implant Concentration: {result['implant_conc']:.4f} ({result['implant_conc']:.2%})

SUBSTRATE ELEMENTS:
"""
            for elem, (conc, dens) in result['substrate_dict'].items():
                export_text += f"  {elem:>5s}: c = {conc:.4f}, d = {dens:.4f} atoms/Ų\n"

            export_text += f"""
CALCULATED RESULT:
  Required {result['implant_element']} Density: {result['implant_density']:.4f} atoms/Ų

VALIDATION:
  Total Concentration: {result['validation']['total_concentration']:.6f}
  Calculated Total Density: {result['validation']['calculated_total_density']:.6f} atoms/Ų
  Density Error: {result['validation']['density_error']:.2e} atoms/Ų
  Status: {'VERIFIED ✓' if result['validation']['density_error'] < 1e-10 else 'WARNING ⚠'}

FORMULA USED:
  d_implant = c_implant / (1/d_total - Σ(c_substrate/d_substrate))
"""

            st.download_button(
                label="📥 Download as TXT",
                data=export_text,
                file_name=f"density_calculation_{result['implant_element']}.txt",
                mime="text/plain",
                width='stretch'
            )

        with col_exp2:
            csv_data = pd.DataFrame(summary_data)
            csv_string = csv_data.to_csv(index=False)

            st.download_button(
                label="📊 Download as CSV",
                data=csv_string,
                file_name=f"density_calculation_{result['implant_element']}.csv",
                mime="text/csv",
                width='stretch'
            )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Atomic Density Calculator",
        page_icon="🧮",
        layout="wide"
    )
    density_calculator_interface()
