import streamlit as st
import numpy as np
import io
import re


def _format_poscar(crystal_name, v1, v2, v3, elements, frac_positions, elem_indices):
    grouped = {i + 1: [] for i in range(len(elements))}
    for frac, eidx in zip(frac_positions, elem_indices):
        grouped[eidx].append(frac)

    counts = [len(grouped[i + 1]) for i in range(len(elements))]

    lines = [crystal_name, "1.0"]
    for v in (v1, v2, v3):
        lines.append(f"  {v[0]:18.12f}  {v[1]:18.12f}  {v[2]:18.12f}")
    lines.append("  " + "  ".join(elements))
    lines.append("  " + "  ".join(str(c) for c in counts))
    lines.append("Direct")
    for i in range(len(elements)):
        for frac in grouped[i + 1]:
            lines.append(f"  {frac[0]:18.12f}  {frac[1]:18.12f}  {frac[2]:18.12f}")
    return "\n".join(lines) + "\n"
    
def _is_orthogonal(v1, v2, v3, tol=1e-4):
    a, b, c = np.array(v1), np.array(v2), np.array(v3)
    return (
        abs(np.dot(a, b)) < tol * np.linalg.norm(a) * np.linalg.norm(b) and
        abs(np.dot(a, c)) < tol * np.linalg.norm(a) * np.linalg.norm(c) and
        abs(np.dot(b, c)) < tol * np.linalg.norm(b) * np.linalg.norm(c)
    )


def _lattice_angles(v1, v2, v3):
    a, b, c = np.array(v1), np.array(v2), np.array(v3)
    def angle(u, w):
        cos = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w))
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return angle(b, c), angle(a, c), angle(a, b)


def _frac_to_cart(positions, v1, v2, v3):
    M = np.column_stack([v1, v2, v3])
    return np.array([M @ np.array(p[:3]) for p in positions])


def _orthogonal_box_from_non_ortho(v1, v2, v3, positions):
    M = np.column_stack([v1, v2, v3])
    corners = np.array([
        M @ np.array([fx, fy, fz])
        for fx in (0, 1) for fy in (0, 1) for fz in (0, 1)
    ])
    lo = corners.min(axis=0)
    hi = corners.max(axis=0)
    dx, dy, dz = hi - lo

    cart = _frac_to_cart(positions, v1, v2, v3) - lo
    frac_box = (cart / np.array([dx, dy, dz])) % 1.0
    return dx, dy, dz, frac_box


def _deduplicate_atoms(frac_positions, element_indices, tol=1e-3):
    kept_frac, kept_elem = [], []
    for frac, eidx in zip(frac_positions, element_indices):
        wrapped = np.array(frac) % 1.0
        duplicate = False
        for kf in kept_frac:
            diff = np.abs(wrapped - np.array(kf))
            diff = np.minimum(diff, 1.0 - diff)
            if np.all(diff < tol):
                duplicate = True
                break
        if not duplicate:
            kept_frac.append(wrapped.tolist())
            kept_elem.append(eidx)
    return kept_frac, kept_elem


def _atomic_density(dx, dy, dz, n_atoms):
    vol = dx * dy * dz
    return n_atoms / vol if vol > 0 else 0.0


def _parse_poscar(lines):
    crystal_name = lines[0].strip()
    scale = float(lines[1].strip())
    v1 = [float(x) * scale for x in lines[2].split()]
    v2 = [float(x) * scale for x in lines[3].split()]
    v3 = [float(x) * scale for x in lines[4].split()]
    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    coord_type = lines[7].strip().lower()
    positions = []
    atom_idx = 0
    for elem_idx, (elem, count) in enumerate(zip(elements, counts)):
        for _ in range(count):
            row = lines[8 + atom_idx].split()
            x, y, z = float(row[0]), float(row[1]), float(row[2])
            if coord_type.startswith('c') or coord_type.startswith('k'):
                lat_matrix = np.array([v1, v2, v3]).T
                frac = np.linalg.inv(lat_matrix) @ np.array([x, y, z])
                x, y, z = frac
            positions.append([x % 1.0, y % 1.0, z % 1.0, elem_idx + 1])
            atom_idx += 1
    return crystal_name, v1, v2, v3, elements, counts, positions


def _parse_crystal_inp(lines):
    crystal_name = lines[0].strip()
    num_species = int(lines[2].strip())
    elements = []
    line_idx = 3
    for _ in range(num_species):
        elements.append(lines[line_idx].strip().strip('"').strip("'"))
        line_idx += 1
    v1 = [float(x) for x in lines[line_idx].split()]
    v2 = [float(x) for x in lines[line_idx + 1].split()]
    v3 = [float(x) for x in lines[line_idx + 2].split()]
    line_idx += 3
    num_atoms = int(lines[line_idx].strip())
    line_idx += 1
    positions = []
    counts_map = {}
    for _ in range(num_atoms):
        row = lines[line_idx].split()
        x, y, z, eidx = float(row[0]), float(row[1]), float(row[2]), int(row[3])
        positions.append([x, y, z, eidx])
        counts_map[eidx] = counts_map.get(eidx, 0) + 1
        line_idx += 1
    counts = [counts_map.get(i + 1, 0) for i in range(num_species)]
    return crystal_name, v1, v2, v3, elements, counts, positions


def _parse_cif(file_content, filename):
    try:
        from pymatgen.io.cif import CifParser
        from pymatgen.io.vasp import Poscar
        parser = CifParser(io.StringIO(file_content))
        structure = parser.parse_structures(primitive=False)[0]
        poscar_lines = str(Poscar(structure)).strip().split('\n')
        poscar_lines[0] = filename.replace('.cif', '').replace('.txt', '')
        return _parse_poscar(poscar_lines)
    except ImportError:
        st.error("pymatgen is required for CIF parsing: pip install pymatgen")
        st.stop()
    except Exception as e:
        st.error(f"Error converting CIF: {e}")
        import traceback; st.code(traceback.format_exc())
        st.stop()


def _detect_format(lines, filename):
    fn = filename.lower()
    if fn.endswith('.inp') or 'crystal.inp' in fn:
        if len(lines) > 3 and lines[1].strip().isdigit():
            return 'crystal.inp'
    if fn.endswith('.cif') or any('_cell_length' in l for l in lines[:20]):
        return 'cif'
    if len(lines) > 7:
        try:
            float(lines[1].strip())
            return 'POSCAR'
        except ValueError:
            pass
    return None



def _apply_reorientation(v1, v2, v3, positions, new_x, new_y, new_z):
    sign = lambda s: -1 if s.startswith('-') else 1
    base = lambda s: s.lstrip('-')
    vm = {'a': v1, 'b': v2, 'c': v3}
    nv1 = [sign(new_x) * x for x in vm[base(new_x)]]
    nv2 = [sign(new_y) * x for x in vm[base(new_y)]]
    nv3 = [sign(new_z) * x for x in vm[base(new_z)]]
    old_mat = np.array([v1, v2, v3]).T
    new_mat = np.array([nv1, nv2, nv3]).T
    T_inv = np.linalg.inv(np.linalg.inv(old_mat) @ new_mat)
    new_positions = []
    for pos in positions:
        new_frac = (T_inv @ np.array(pos[:3])) % 1.0
        new_positions.append([new_frac[0], new_frac[1], new_frac[2], pos[3]])
    return nv1, nv2, nv3, new_positions



def _format_crystal_inp(crystal_name, v1, v2, v3, elements, positions, cell_ext=3):
    out = f"{crystal_name}\n1\n{len(elements)}\n"
    for e in elements:
        out += f'"{e}"\n'
    out += f"{v1[0]:.4f} {v1[1]:.4f} {v1[2]:.4f}\n"
    out += f"{v2[0]:.4f} {v2[1]:.4f} {v2[2]:.4f}\n"
    out += f"{v3[0]:.4f} {v3[1]:.4f} {v3[2]:.4f}\n"
    out += f"{len(positions)}\n"
    for p in positions:
        out += f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {p[3]}\n"
    out += f"0.0\n0.0 0.0\n{cell_ext}\n"
    return out


def _format_table_structure(crystal_index, n_atoms, label,
                             frac_positions, elem_indices, elements):
    header = (
        f" {crystal_index:2d} {n_atoms:2d} {label:<24s}"
        " !Nr-crystal, number-of-atoms, name\n"
    )
    out = header
    for i, (frac, eidx) in enumerate(zip(frac_positions, elem_indices)):
        prefix = "    !relative coordinates x y z of atoms" if i == 0 else ""
        elem_label = elements[eidx - 1] if (eidx - 1) < len(elements) else "?"
        on_edge = any(abs(f) < 1e-4 or abs(f - 1.0) < 1e-4 for f in frac)
        edge_note = "  !<- on boundary: implicit mirror at opposite face" if on_edge else ""
        out += (
            f"    {frac[0]:.4f}    {frac[1]:.4f}    {frac[2]:.4f}"
            f"{prefix:<42s} ! {elem_label}{edge_note}\n"
        )
    return out


def _format_table_geometry(crystal_name, dx, dy, dz, n_atoms,
                            matrix_id, crystal_type,
                            n_elem_types, elements, counts,
                            delta_hf=0.0, p_max=3.0):
    dens = _atomic_density(dx, dy, dz, n_atoms)
    elem_list = []
    for elem, count in zip(elements, counts):
        elem_list.extend([elem] * count)
    elem_str = "   ".join(f"{e:<4s}" for e in elem_list)
    dens_comment = (
        f"{dens:.5f}  = {n_atoms}./"
        f"{dx:.6f} /{dy:.6f} /{dz:.6f}"
    )
    return (
        f"  {crystal_name:<14s}"
        f" {dx:10.6f} {dy:10.6f} {dz:10.6f}"
        f"   {dens:.5f}"
        f"    {delta_hf:.4f}"
        f"   {p_max:.4f}"
        f"      {matrix_id}"
        f"       {crystal_type}"
        f"       {n_elem_types}"
        f"       {n_atoms}"
        f"     {elem_str}"
        f"    {dens_comment}\n"
    )


def _reorder_elements(elements, counts, frac_positions, elem_indices, new_order):
    """Reorder the element list (and everything tied to it) to ``new_order``.

    ``new_order`` is a permutation of ``elements``. To keep Part 1 (the structure
    block, which lists atoms in position order) consistent with Part 2 (the
    geometry line, whose per-atom element list is built grouped by element order),
    the atoms are regrouped to follow ``new_order`` while preserving each element's
    internal atom order. Returns ``(elements, counts, frac_positions, elem_indices)``.
    """
    new_elements = list(new_order)
    new_counts = [counts[elements.index(e)] for e in new_elements]

    atoms = list(zip(frac_positions, elem_indices))
    new_frac = []
    new_eidx = []
    for new_i, e in enumerate(new_elements):
        old_idx = elements.index(e) + 1
        for frac, eidx in atoms:
            if eidx == old_idx:
                new_frac.append(frac)
                new_eidx.append(new_i + 1)
    return new_elements, new_counts, new_frac, new_eidx


def _clean_element_symbol(label):
    if not label:
        return label
    m = re.match(r'^([A-Z][a-z]?)', label)
    return m.group(1) if m else label


def _parse_table_crystal_structure_block(text):
    raw_lines = [l for l in text.split('\n') if l.strip()]
    if not raw_lines:
        return None

    first = raw_lines[0]
    if '!' in first:
        first = first.split('!', 1)[0]
    parts = first.split()
    if len(parts) < 3:
        return None
    try:
        nr_crystal = int(parts[0])
        n_atoms = int(parts[1])
    except ValueError:
        return None
    crystal_label = ' '.join(parts[2:]).strip()

    frac_positions = []
    elem_labels = []
    for line in raw_lines[1:]:
        if '!' in line:
            segs = line.split('!')
            data_part = segs[0]
            element_label = None
            for piece in reversed(segs[1:]):
                p = piece.strip()
                if not p:
                    continue
                lower = p.lower()
                if p.startswith('<-') or 'boundary' in lower or 'mirror' in lower:
                    continue
                if 'coordinate' in lower or 'relative' in lower or 'atom' in lower:
                    continue
                first_word = p.split()[0]
                if first_word and first_word[0].isalpha():
                    element_label = first_word
                    break
        else:
            data_part = line
            element_label = None

        toks = data_part.split()
        if len(toks) < 3:
            continue
        try:
            x = float(toks[0])
            y = float(toks[1])
            z = float(toks[2])
        except ValueError:
            continue
        frac_positions.append([x, y, z])
        elem_labels.append(element_label)
        if len(frac_positions) >= n_atoms:
            break

    return {
        'nr_crystal': nr_crystal,
        'n_atoms_declared': n_atoms,
        'crystal_label': crystal_label,
        'frac_positions': frac_positions,
        'elem_labels': elem_labels,
    }


def _parse_table_crystal_geometry_line(text):
    line = text.strip()
    if not line:
        return None
    if '!' in line:
        line = line.split('!', 1)[0]
    toks = line.split()
    if len(toks) < 8:
        return None
    try:
        name = toks[0]
        dx = float(toks[1])
        dy = float(toks[2])
        dz = float(toks[3])
        density = float(toks[4])
        delta_hf = float(toks[5])
        p_max = float(toks[6])
        matrix_id = int(toks[7])
    except (ValueError, IndexError):
        return None

    crystal_type = None
    n_elem_types = None
    n_atoms = None
    if len(toks) > 8:
        try:
            crystal_type = int(toks[8])
        except ValueError:
            pass
    if len(toks) > 9:
        try:
            n_elem_types = int(toks[9])
        except ValueError:
            pass
    if len(toks) > 10:
        try:
            n_atoms = int(toks[10])
        except ValueError:
            pass

    elem_list = []
    if n_atoms and len(toks) > 11:
        elem_list = toks[11:11 + n_atoms]
    elif len(toks) > 11:
        elem_list = toks[11:]

    return {
        'name': name,
        'dx': dx, 'dy': dy, 'dz': dz,
        'density': density,
        'delta_hf': delta_hf,
        'p_max': p_max,
        'matrix_id': matrix_id,
        'crystal_type': crystal_type,
        'n_elem_types': n_elem_types,
        'n_atoms': n_atoms,
        'elem_list': elem_list,
    }


def _build_cif_from_orthogonal(crystal_name, dx, dy, dz, species_per_atom, frac_positions):
    try:
        from pymatgen.core import Structure, Lattice
        from pymatgen.io.cif import CifWriter
    except ImportError:
        return None, "pymatgen is required to write CIF (pip install pymatgen)"
    try:
        lattice = Lattice.from_parameters(dx, dy, dz, 90, 90, 90)
        structure = Structure(lattice, species_per_atom, frac_positions)
        return str(CifWriter(structure)), None
    except Exception as e:
        return None, str(e)


def _build_cif_from_lattice(crystal_name, v1, v2, v3, species_per_atom, frac_positions):
    try:
        from pymatgen.core import Structure, Lattice
        from pymatgen.io.cif import CifWriter
    except ImportError:
        return None, "pymatgen is required to write CIF (pip install pymatgen)"
    try:
        lattice = Lattice([v1, v2, v3])
        structure = Structure(lattice, species_per_atom, frac_positions)
        return str(CifWriter(structure)), None
    except Exception as e:
        return None, str(e)


def _present_poscar_cif_outputs(crystal_name, v1, v2, v3, elements, counts,
                                 frac_positions, elem_indices, cif_str, key_prefix):
    poscar_str = _format_poscar(
        crystal_name, v1, v2, v3, elements, frac_positions, elem_indices
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**POSCAR**")
        st.text_area("POSCAR preview:", poscar_str, height=320, key=f"{key_prefix}_poscar_prev")
        st.download_button(
            "⬇️ Download POSCAR",
            data=poscar_str,
            file_name=f"{crystal_name}_POSCAR",
            mime="text/plain",
            type="primary",
            key=f"{key_prefix}_dl_poscar",
        )
    with col_b:
        st.markdown("**CIF**")
        if cif_str:
            st.text_area("CIF preview:", cif_str, height=320, key=f"{key_prefix}_cif_prev")
            st.download_button(
                "⬇️ Download CIF",
                data=cif_str,
                file_name=f"{crystal_name}.cif",
                mime="chemical/x-cif",
                type="primary",
                key=f"{key_prefix}_dl_cif",
            )
        else:
            st.warning("CIF could not be generated (pymatgen unavailable or invalid elements).")


def _sdtrimsp_to_poscar_cif_interface():
    st.markdown("#### Convert SDTrimSP crystal data back to POSCAR / CIF")
    st.caption(
        "Either paste the two `table.crystal` entries (structure block + geometry line) "
        "or upload a `crystal.inp` file."
    )

    source = st.radio(
        "Input format:",
        ["table.crystal entries (SDTrimSP ≥ 7.02)", "crystal.inp file (SDTrimSP ≤ 7.01)"],
        horizontal=True,
        key="reverse_input_source",
    )

    if source.startswith("table.crystal"):
        st.markdown(
            "Paste **Part 1** (structure block) and **Part 2** (geometry line) "
            "exactly as they appear in `table.crystal`."
        )
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            st.markdown("**Part 1 — structure block (fractional coords)**")
            part1_text = st.text_area(
                "Structure block:",
                height=280,
                key="reverse_part1_text",
                placeholder=(
                    "  5  8 TiNb39                  !Nr-crystal, number-of-atoms, name\n"
                    "    0.0000    0.0000    0.0000  !relative coordinates x y z of atoms        ! Ti\n"
                    "    0.5000    0.3750    0.2500                                              ! Ti\n"
                    "    ...\n"
                ),
            )
        with col_in2:
            st.markdown("**Part 2 — geometry line (dx, dy, dz, …)**")
            part2_text = st.text_area(
                "Geometry line:",
                height=280,
                key="reverse_part2_text",
                placeholder=(
                    "  TiNb39          3.306500   6.613000   6.613000   0.05533    0.0000   "
                    "3.0000      3       5       2       8     Ti   Nb   ...\n"
                ),
            )

        if not (part1_text.strip() and part2_text.strip()):
            st.info("Paste both Part 1 and Part 2 to begin.")
            return

        part1 = _parse_table_crystal_structure_block(part1_text)
        if part1 is None:
            st.error("Could not parse the structure block. Check the first line format.")
            return
        part2 = _parse_table_crystal_geometry_line(part2_text)
        if part2 is None:
            st.error("Could not parse the geometry line. Need at least 8 whitespace-separated tokens.")
            return

        n_parsed = len(part1['frac_positions'])
        if part2.get('n_atoms') and n_parsed != part2['n_atoms']:
            st.warning(
                f"Atom count mismatch: structure block has {n_parsed} positions, "
                f"geometry line declares {part2['n_atoms']}. Using parsed positions."
            )

        elem_per_atom_raw = []
        for i, label in enumerate(part1['elem_labels']):
            if label:
                elem_per_atom_raw.append(label)
            elif part2.get('elem_list') and i < len(part2['elem_list']):
                elem_per_atom_raw.append(part2['elem_list'][i])
            else:
                elem_per_atom_raw.append('X')

        st.markdown("##### Element labels")
        st.caption(
            "SDTrimSP labels (e.g. `Ti_1`) are mapped to chemical symbols (e.g. `Ti`). "
            "Edit below if needed."
        )
        clean_default = ", ".join(_clean_element_symbol(e) for e in elem_per_atom_raw)
        clean_input = st.text_input(
            "Per-atom element symbols (comma-separated, one per atom):",
            value=clean_default,
            key="reverse_clean_elems",
        )
        chem_per_atom = [e.strip() for e in clean_input.split(',') if e.strip()]
        if len(chem_per_atom) != n_parsed:
            st.error(
                f"You provided {len(chem_per_atom)} symbols but parsed {n_parsed} atoms. "
                "Fix the list above before continuing."
            )
            return

        seen = {}
        for e in chem_per_atom:
            if e not in seen:
                seen[e] = len(seen)
        elements = list(seen.keys())
        elem_indices = [seen[e] + 1 for e in chem_per_atom]
        counts = [chem_per_atom.count(e) for e in elements]

        dx, dy, dz = part2['dx'], part2['dy'], part2['dz']
        crystal_name = part2.get('name') or part1.get('crystal_label') or 'crystal'
        v1 = [dx, 0.0, 0.0]
        v2 = [0.0, dy, 0.0]
        v3 = [0.0, 0.0, dz]

        with st.expander("Reconstructed structure summary", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("dx (Å)", f"{dx:.6f}")
            c2.metric("dy (Å)", f"{dy:.6f}")
            c3.metric("dz (Å)", f"{dz:.6f}")
            c1.metric("# atoms", n_parsed)
            c2.metric("# elements", len(elements))
            c3.metric("Density (at/Å³)", f"{_atomic_density(dx, dy, dz, n_parsed):.5f}")
            st.code(
                f"Crystal label    : {crystal_name}\n"
                f"Unique elements  : {', '.join(elements)}\n"
                f"Counts           : {', '.join(str(c) for c in counts)}\n"
                f"Per-atom symbols : {' '.join(chem_per_atom)}"
            )

        cif_str, cif_err = _build_cif_from_orthogonal(
            crystal_name, dx, dy, dz, chem_per_atom, part1['frac_positions']
        )
        if cif_err and cif_str is None:
            st.warning(f"CIF generation failed: {cif_err}")

        _present_poscar_cif_outputs(
            crystal_name, v1, v2, v3, elements, counts,
            part1['frac_positions'], elem_indices, cif_str, "reverse_table",
        )
        return

    crystal_inp_file = st.file_uploader(
        "Upload crystal.inp",
        key="reverse_crystal_inp_file",
    )
    if crystal_inp_file is None:
        st.info("Upload a crystal.inp file to begin.")
        return

    try:
        text = crystal_inp_file.read().decode("utf-8")
        crystal_name, v1, v2, v3, elements, counts, positions = _parse_crystal_inp(
            text.strip().split('\n')
        )
    except Exception as e:
        st.error(f"Could not parse crystal.inp: {e}")
        return

    frac_positions = [p[:3] for p in positions]
    elem_indices = [p[3] for p in positions]
    dx = float(np.linalg.norm(v1))
    dy = float(np.linalg.norm(v2))
    dz = float(np.linalg.norm(v3))
    ortho = _is_orthogonal(v1, v2, v3)

    clean_elements = [_clean_element_symbol(e) for e in elements]
    species_per_atom = [clean_elements[ei - 1] for ei in elem_indices]

    with st.expander("Reconstructed structure summary", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("|a| (Å)", f"{dx:.6f}")
        c2.metric("|b| (Å)", f"{dy:.6f}")
        c3.metric("|c| (Å)", f"{dz:.6f}")
        c1.metric("# atoms", len(positions))
        c2.metric("# elements", len(elements))
        c3.metric("Orthogonal", "yes" if ortho else "no")
        st.code(
            f"Crystal name    : {crystal_name}\n"
            f"Raw labels      : {', '.join(elements)}\n"
            f"Clean symbols   : {', '.join(clean_elements)}\n"
            f"Counts          : {', '.join(str(c) for c in counts)}"
        )

    cif_str, cif_err = _build_cif_from_lattice(
        crystal_name, v1, v2, v3, species_per_atom, frac_positions
    )
    if cif_err and cif_str is None:
        st.warning(f"CIF generation failed: {cif_err}")

    _present_poscar_cif_outputs(
        crystal_name, v1, v2, v3, clean_elements, counts,
        frac_positions, elem_indices, cif_str, "reverse_inp",
    )


def crystal_converter_interface():
    st.markdown("### Crystal Structure File Converter")

    direction = st.radio(
        "**Conversion direction:**",
        ["POSCAR / CIF → SDTrimSP", "SDTrimSP → POSCAR / CIF"],
        horizontal=True,
        key="crystal_conv_direction",
    )
    if direction.startswith("SDTrimSP →"):
        _sdtrimsp_to_poscar_cif_interface()
        return

    version = st.radio(
        "**SDTrimSP version:**",
        ["<= 7.01  (crystal.inp format)", ">= 7.02 and above (table.crystal format)"],
        horizontal=True,
    )
    use_table = version.startswith(">=")

    crystal_file = st.file_uploader(
        "Upload POSCAR, .cif, or crystal.inp",
        key="crystal_converter_v2",
    )
    if crystal_file is None:
        st.info("Upload a structure file to begin.")
        return

    file_content = crystal_file.read().decode("utf-8")
    lines = file_content.strip().split('\n')
    file_type = _detect_format(lines, crystal_file.name)
    if file_type is None:
        st.error("Could not detect file format (POSCAR, .cif, or crystal.inp expected).")
        return
    st.success(f"Detected format: **{file_type}**")

    # Parse
    try:
        if file_type == "POSCAR":
            crystal_name, v1, v2, v3, elements, counts, positions = _parse_poscar(lines)
        elif file_type == "crystal.inp":
            crystal_name, v1, v2, v3, elements, counts, positions = _parse_crystal_inp(lines)
        else:
            crystal_name, v1, v2, v3, elements, counts, positions = _parse_cif(
                file_content, crystal_file.name)
    except Exception as e:
        st.error(f"Parsing error: {e}")
        import traceback; st.code(traceback.format_exc())
        return

    ortho = _is_orthogonal(v1, v2, v3)
    alpha, beta, gamma = _lattice_angles(v1, v2, v3)

    # Orthogonality status
    if ortho:
        st.success("Orthogonal crystal (a=b=g=90) — direct conversion is reliable.")
    else:
        st.warning(
            f"**Non-orthogonal crystal** (a={alpha:.2f}, b={beta:.2f}, g={gamma:.2f} deg).\n\n"
            "SDTrimSP table.crystal requires an **axis-aligned cuboid** (dx, dy, dz).  \n"
            "The converter will build the minimal Cartesian bounding box and re-express "
            "atom positions inside it.  From the **SDTrimSP documentation**:\n\n"
            "> *'This has not yet been tested.'*\n"
            "> *'When creating new crystal structures, the edges must be taken into "
            "account. (High error rate.)'*\n\n"
            "**You must verify** the resulting fractional coordinates against your "
            "known crystal structure before running any simulation."
        )
        st.info(
            "Tip: try reorienting so that the vector closest to [100] is aligned to X first "
            "— this often minimises off-diagonal components and improves the bounding box."
        )

    # Reorientation
    with st.expander("Reorient lattice vectors (optional)"):
        do_reorient = st.checkbox("Apply reorientation")
        if do_reorient:
            opts = ["a", "b", "c", "-a", "-b", "-c"]
            c1, c2, c3 = st.columns(3)
            new_x = c1.selectbox("New X", opts, index=0)
            new_y = c2.selectbox("New Y", opts, index=1)
            new_z = c3.selectbox("New Z", opts, index=2)
            v1, v2, v3, positions = _apply_reorientation(v1, v2, v3, positions, new_x, new_y, new_z)
            st.info(f"Reoriented: X={new_x}, Y={new_y}, Z={new_z}")
            ortho = _is_orthogonal(v1, v2, v3)

    # Compute working box
    if ortho:
        dx = np.linalg.norm(v1)
        dy = np.linalg.norm(v2)
        dz = np.linalg.norm(v3)
        frac_positions = [p[:3] for p in positions]
        elem_indices   = [p[3]  for p in positions]
    else:
        dx, dy, dz, frac_box = _orthogonal_box_from_non_ortho(v1, v2, v3, positions)
        frac_positions = frac_box.tolist()
        elem_indices   = [p[3] for p in positions]
        frac_positions, elem_indices = _deduplicate_atoms(frac_positions, elem_indices)
        counts_new = {i: 0 for i in range(1, len(elements) + 1)}
        for ei in elem_indices:
            counts_new[ei] = counts_new.get(ei, 0) + 1
        counts = [counts_new.get(i + 1, 0) for i in range(len(elements))]
        st.info(
            f"Bounding box: dx={dx:.4f} A, dy={dy:.4f} A, dz={dz:.4f} A  "
            f"| {len(frac_positions)} atoms after deduplication."
        )

    n_atoms = len(frac_positions)
    dens = _atomic_density(dx, dy, dz, n_atoms)

    # Summary
    with st.expander("Structure summary", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("dx (A)", f"{dx:.6f}")
        c2.metric("dy (A)", f"{dy:.6f}")
        c3.metric("dz (A)", f"{dz:.6f}")
        c1.metric("# atoms", n_atoms)
        c2.metric("# element types", len(elements))
        c3.metric("Density (at/A3)", f"{dens:.5f}")
        st.code(
            f"Elements : {', '.join(elements)}\n"
            f"Counts   : {', '.join(str(c) for c in counts)}\n"
            f"Angles   : a={alpha:.3f}  b={beta:.3f}  g={gamma:.3f} deg  "
            f"({'orthogonal' if ortho else 'NON-ORTHOGONAL — bounding box used'})"
        )
    poscar_str = _format_poscar(
        crystal_name, v1, v2, v3, elements, frac_positions, elem_indices
    )
    st.download_button(
        "⬇️ Download as POSCAR",
        data=poscar_str,
        file_name=f"{crystal_name}_POSCAR",
        mime="text/plain",
    )
    if n_atoms > 16:
        st.warning(f"Structure has {n_atoms} atoms. SDTrimSP 7.01 supports max 16.")

    st.markdown("---")

    # ─── Branch A: crystal.inp (<=7.01) ───────────────────────────────────────
    if not use_table:
        st.markdown("#### Output: crystal.inp  (SDTrimSP <= 7.01)")
        if not ortho:
            st.error(
                "crystal.inp does not support non-orthogonal lattices.  \n"
                "The file stores raw lattice vectors; SDTrimSP 7.01 requires them to be "
                "mutually perpendicular.  \n"
                "Options: (1) reorient to an orthogonal supercell manually, "
                "(2) upgrade to SDTrimSP >= 7.02 and use the table.crystal route."
            )
        cell_ext = st.radio("Automatic cell extension:", [3, 5], horizontal=True,
                            help="Only 3 and 5 are supported in version 7.01.")
        output = _format_crystal_inp(
            crystal_name, v1, v2, v3, elements,
            [[*f, e] for f, e in zip(frac_positions, elem_indices)],
            cell_ext,
        )
        col_prev, col_dl = st.columns(2)
        with col_prev:
            st.text_area("Preview:", output, height=380)
        with col_dl:
            st.download_button("Download crystal.inp", data=output,
                               file_name=f"{crystal_name}_crystal.inp",
                               mime="text/plain", type="primary")
        return

    # ─── Branch B: table.crystal (>=7.02) ────────────────────────────────────
    st.markdown("#### Output: table.crystal entries  (SDTrimSP >= 7.02)")
    st.info(
        "ℹ️ **When adding a new crystal structure to `table.crystal`**, also update "
        "the count on **line 2** of the file (the integer in front of "
        "`number of crystal-structure or type`)"
    )
    if not ortho:
        st.warning(
            "Non-orthogonal crystal: the bounding-box coordinates below are a "
            "best-effort approximation.  The SDTrimSP documentation states this "
            "case 'has not yet been tested' and warns of a high error rate near "
            "cell edges.  Cross-check all fractional coordinates manually."
        )

    # Let the user choose the order in which elements are written into the
    # table.crystal entry (affects both Part 1 atom labels and the Part 2
    # element list). E.g. Ti, Nb, N instead of the parsed order.
    if len(elements) > 1:
        st.markdown("**Element order in the output**")
        st.caption(
            "Choose the order the elements are written into `table.crystal` "
            "(Part 1 atom labels and the Part 2 element list). "
            "Select all elements in the desired order — e.g. Ti, then Nb, then N."
        )
        order_sel = st.multiselect(
            "Element order (selection order = output order):",
            options=list(elements),
            default=list(elements),
            key="crystal_elem_order",
        )
        if len(order_sel) == len(elements) and set(order_sel) == set(elements):
            if order_sel != list(elements):
                elements, counts, frac_positions, elem_indices = _reorder_elements(
                    elements, counts, frac_positions, elem_indices, order_sel
                )
                st.success(f"Element order set to: {' → '.join(elements)}")
        else:
            st.warning(
                "Select **all** elements exactly once to define the order. "
                f"Using the current order: {', '.join(elements)}."
            )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Part 1 — Crystal-structure block**")
        crystal_index = st.number_input(
            "Crystal index (Nr-crystal):", min_value=1, max_value=9999, value=5, step=1,
            help="Integer index in the crystal-structure list. Default entries use 1–4.",
        )
        crystal_label = st.text_input("Crystal name / label:", value=crystal_name,
                                      help="Free label shown in the coordinate block header.")

    with col_b:
        st.markdown("**Part 2 — Geometry table line**")
        st.caption(
            "`typ` is set automatically to the Nr-crystal index above. "
            "`matrix_id` controls the **internal supercell multiplication** — "
            "use **3** for a 3×3×3 expansion or **5** for a 5×5×5 expansion of the unit cell."
        )
        table_name = st.text_input("Entry name (table column):", value=crystal_name,
                                   help="Free name, e.g. TiN_NaCl, VN_NaCl.")
        matrix_id = st.radio(
            "matrix_id (internal supercell multiplication):",
            options=[3, 5], horizontal=True,
            help="Internal multiplication factor used to build the supercell. "
                 "Only 3 (3×3×3 expansion) or 5 (5×5×5 expansion) are valid.",
        )
        delta_hf = st.number_input("Heat of formation dH_f (eV):",
                                   value=0.0, step=0.001, format="%.4f",
                                   help="0.0 for elements; see table.compounds for compounds.")
        p_max = st.number_input("p_max (A):", value=3.0, step=0.1, format="%.4f")

    # typ = Nr-crystal from Part 1 (same index used there)
    crystal_type = int(crystal_index)

    part1 = _format_table_structure(
        int(crystal_index), n_atoms, crystal_label,
        frac_positions, elem_indices, elements,
    )
    part2 = _format_table_geometry(
        table_name, dx, dy, dz, n_atoms,
        int(matrix_id), crystal_type, len(elements),
        elements, counts, delta_hf=float(delta_hf), p_max=float(p_max),
    )

    st.markdown("---")
    st.markdown("##### Part 1 — paste into the crystal-structure list of table.crystal")
    st.markdown(
        "_Upper section starting with `! crystal-structure`._"
    )
    st.code(part1, language="text")
    st.download_button("Download Part 1 (structure block)", data=part1,
                       file_name=f"{table_name}_structure_block.txt", mime="text/plain")

    st.markdown("---")
    st.markdown("##### Part 2 — paste into the geometry table of table.crystal")
    st.markdown("_Add this line below the existing entries in the lower table section._")
    st.code(part2, language="text")
    st.download_button("Download Part 2 (geometry line)", data=part2,
                       file_name=f"{table_name}_geometry_line.txt", mime="text/plain")

    combined = (
        "! -- Part 1: paste into the crystal-structure list --\n" + part1 +
        "\n! -- Part 2: paste into the geometry table ----------\n" + part2
    )
    st.markdown("---")
    st.download_button("Download both parts (.txt)", data=combined,
                       file_name=f"{table_name}_table_crystal_entries.txt",
                       mime="text/plain", type="primary")

    with st.expander("table.crystal column reference and non-orthogonal notes"):
        st.markdown("""
| Column | Unit | Description |
|---|---|---|
| name | – | Free label (e.g. `TiN_NaCl`) |
| dx, dy, dz | Å | Orthogonal cuboid dimensions |
| density | at./Å³ | N / (dx·dy·dz) |
| ΔH_f | eV | Heat of formation (0 for elements) |
| p_max | Å | Max interaction cutoff (3.0–3.5 typical) |
| matrix_id | – | Internal supercell multiplication — **3** (3×3×3) or **5** (5×5×5) |
| typ | – | Nr-crystal index from the crystal-structure list (Part 1): 1=NaCl-type, 2=fcc, 3=bcc, 4=diamond, or your custom index |
| n_elements | – | Number of distinct element types |
| n_atoms | – | Total atoms per unit cell |
| elements | – | Element symbol listed once per atom |

---
**Edge-atom rule (SDTrimSP documentation)**

An atom at `0.0000  0.0000  0.0000` is implicitly also present at  
`0.0000  0.0000  1.0000`, `0.0000  1.0000  0.0000`, ... `1.0000  1.0000  1.0000`.  
Similarly `0.0000  0.5000  0.5000` implies `1.0000  0.5000  0.5000`.  
Do **not** add these duplicates manually.

---
**Non-orthogonal crystals (SDTrimSP documentation)**

> *"Therefore, it should be possible to model non-orthogonal crystals as well.  
> The condition is that a period exists in the x, y, z directions.  
> This has **not yet been tested**."*  
> *"When creating new crystal structures, the edges must be taken into  
> account. (High error rate.)"*

This converter uses the minimal Cartesian bounding box as the cuboid.  
For truly non-orthogonal structures (hexagonal, monoclinic, triclinic)  
you may need to construct an explicit orthogonal supercell manually.
""")
