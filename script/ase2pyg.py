'''
v1.2
Convert ASE molecular data to PyTorch Geometric graphs. 
Support flexiable node and edge features, default: atomic number and distance.
Energy is taken from the attached calculator (eV). Forces are eV/Å.
Returned PyG graph
=========
x            : (N, F)  float32   — node feature matrix. Columns appear in this order if requested and available:
                                   ["Z", "mulliken_charge", "lowdin_charge", "nbo_charge",
                                    "mulliken_spin", "lowdin_spin", "nbo_spin"].
pos          : (N, 3)  float32   Å — Cartesian positions (atoms.get_positions()).
z            : (N,)    int64     — atomic numbers (atoms.get_atomic_numbers()).
edge_index   : (2, E)  int64     — directed edges; built with bothways=True (i→j and j→i), no self loops.
edge_attr    : (E, D)  float32   — edge features in the exact order requested by edge_features:
                                   "distance" → 1 col (||Δr|| in Å)
                                   "delta"    → 3 cols (Δx, Δy, Δz in Å; minimum-image if PBC)
                                   "unit"     → 3 cols (Δr / ||Δr||, dimensionless)
pbc          : (3,)    bool      — per-axis periodic flags (copied from atoms.pbc).
cell         : (3, 3)  float32   Å — lattice vectors (atoms.cell.array). Present for molecules too.
y            : (1,)    float32   eV — total electronic energy from the attached Calculator(atoms.get_potential_energy()).                             
force        : (N, 3)  float32   eV/Å — atomic forces (atoms.get_forces()) when available and finite.
charge       : (1,)    int64     electrons — total system charge. Sourced from DB row.data["charge"], default 0.
spin         : (1,)    int64     dataset-defined — copied from DB row.data["spin"] (often multiplicity 2S+1).
natoms       : (1,)    int64     — number of atoms N.
'''

import os
import torch
from torch_geometric.data import Data
from ase import neighborlist
from ase.geometry import get_distances
from ase.db import connect
import numpy as np
from typing import List, Literal, Optional, Tuple

def _get_per_atom_array(atoms, keys: Tuple[str, ...]) -> Optional[np.ndarray]:
    """Try atoms.arrays first, then atoms.info for any of the given keys."""
    for k in keys:
        if k in getattr(atoms, "arrays", {}):
            return np.asarray(atoms.arrays[k])
        if k in atoms.info:
            return np.asarray(atoms.info[k])
    return None


def ase_to_pyg_graph(
    row, # assumes that the .aselmdb row 
    node_features: List[str] = ["Z"],
    edge_features: List[str] = ["distance", "delta", "unit"],
    cutoff_factor: float = 1.2,
    include_forces: bool = True
) -> Data:
    """
    Convert an ASE Atoms object into a PyTorch Geometric Data graph with total charge\&spin for whole graph.
    Args:
        atoms: ASE Atoms object
        node_features: list of node features to include ("Z", "mulliken_charge", "spin")
        edge_features: list of edge features to include ("distance","delta", "unit")
        cutoff_factor: scaling factor for atural atomic cutoffs, 6 or 12, but if you wants go realistic, use 1.2!
    Returns:
        torch_geometric.data.Data object
    """
    atoms = row.toatoms()
    Z = atoms.get_atomic_numbers()
    pos_np = atoms.get_positions()
    pos = torch.as_tensor(atoms.get_positions(), dtype=torch.float32)
    n = len(Z)

    # --------- node features ---------
    feats = []
    if "Z" in node_features:
        feats.append(torch.as_tensor(Z, dtype=torch.float32).view(-1, 1))

    # charges spins per atom
    charge_maps = {
        "mulliken_charge": ("mulliken_charges",),
        "lowdin_charge": ("lowdin_charges",),
        "nbo_charge": ("nbo_charges",),
    }
    spin_maps = {
        "mulliken_spin": ("mulliken_spins",),
        "lowdin_spin": ("lowdin_spins",),
        "nbo_spin": ("nbo_spins",),
    }
    for nf, keys in charge_maps.items():
        if nf in node_features:
            arr = _get_per_atom_array(atoms, keys)
            if arr is not None and len(arr) == n:
                feats.append(torch.as_tensor(arr, dtype=torch.float32).view(-1, 1))
    for nf, keys in spin_maps.items():
        if nf in node_features:
            arr = _get_per_atom_array(atoms, keys)
            if arr is not None and len(arr) == n:
                feats.append(torch.as_tensor(arr, dtype=torch.float32).view(-1, 1))

    x = torch.cat(feats, dim=1) if feats else None

    # --------- neighbor & edge features ---------
    natural_cutoffs = neighborlist.natural_cutoffs(atoms, cutoff_factor)
    # print("NATURAL CUTOFFS", natural_cutoffs)
    cutoffs = np.full(n, float(cutoff_factor))
    nl = neighborlist.NeighborList(natural_cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)

    rows, cols, edge_attr = [], [], []
    # If PBC, we'll compute MIC vectors in one go for speed
    use_pbc = bool(getattr(atoms, "pbc", np.array([False, False, False])).any())

    # Precompute edge feature dimension for empty graphs
    edge_dim = 0
    if "distance" in edge_features:
        edge_dim += 1
    if "delta" in edge_features:
        edge_dim += 3
    if "unit" in edge_features:
        edge_dim += 3

    for i in range(n):
        nbr_ids, _ = nl.get_neighbors(i)
        for j in nbr_ids:
            rows.append(i); cols.append(j)

            if edge_features:
                # vector & distance (MIC if periodic)
                if use_pbc:
                     # Å; returns 3-vector with minimum image
                    vec_np = atoms.get_distance(i, j, vector=True, mic=True)
                else:
                    vec_np = pos_np[j] - pos_np[i]

                dist = float(np.linalg.norm(vec_np))
                feat = []
                if "distance" in edge_features:
                    feat.append(dist)
                if "delta" in edge_features:
                    feat.extend([float(vec_np[0]), float(vec_np[1]), float(vec_np[2])])
                if "unit" in edge_features:
                    if dist > 0.0:
                        u = vec_np / dist
                        feat.extend([float(u[0]), float(u[1]), float(u[2])])
                    else:
                        feat.extend([0.0, 0.0, 0.0])
                edge_attr.append(feat)

    if rows:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else None
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr_t = torch.empty((0, edge_dim), dtype=torch.float32) if edge_dim > 0 else None
    
    # --------- build data ---------
    data = Data(x=x, pos=pos, z=torch.as_tensor(Z, dtype=torch.long),
                edge_index=edge_index, edge_attr=edge_attr_t)
    
    # Keep PBC context
    try:
        data.pbc = torch.as_tensor(atoms.pbc, dtype=torch.bool)
        data.cell = torch.as_tensor(np.asarray(atoms.cell.array), dtype=torch.float32)
    except Exception:
        pass

    # attach energy / forces if present
    if include_forces:
        try:
            F = atoms.get_forces(apply_constraint=False)
            if F is not None and np.shape(F) == (n, 3) and np.isfinite(F).all():
                data.force = torch.as_tensor(F, dtype=torch.float32)
        except Exception:
            pass

    # total energy
    E = None
    try:
        E = float(atoms.get_potential_energy())
    except Exception:
        if "energy" in atoms.info:
            try:
                E = float(atoms.info["energy"])
            except Exception:
                E = None
    if E is not None and np.isfinite(E):
        data.y = torch.tensor([E], dtype=torch.float32)


    # row keys : 'source', 'reference_source', 'data_id', 'charge', 'spin', 'num_atoms', 'num_electrons', 'num_ecp_electrons', 'n_scf_steps', 'n_basis', 'unrestricted', 'nl_energy', 'integrated_densities', 'homo_energy', 'homo_lumo_gap', 's_squared', 's_squared_dev', 'warnings', 'mulliken_charges', 'lowdin_charges', 'nbo_charges', 'mulliken_spins', 'lowdin_spins', 'nbo_spins', 'composition'])
    # charge = None
    # spin = None
    # natoms = None
    # num_electrons = None
    # num_ecp_electrons = None
    # nl_energy = None
    # integrated_densities = None
    # homo_energy = None
    # homo_lumo_gap = None
    # s_squared = None
    # s_squared_dev = None
    # mulliken_charges = None 
    # lowdin_charges = None 
    # nbo_charges = None 
    # mulliken_spins = None 
    # lowdin_spins = None
    # nbo_spins = None

    # --------- global total charge, spin, natoms, electrons ---------
    try:
        data.charge = torch.tensor([row.data.charge], dtype=torch.long)
        data.spin = torch.tensor([row.data.spin], dtype=torch.long)
        data.natoms = torch.tensor([row.data.num_atoms], dtype=torch.long)
        data.num_electrons = torch.tensor([row.data.num_electrons], dtype=torch.long)
        data.num_ecp_electrons = torch.tensor([row.data.num_ecp_electrons], dtype=torch.long)

        data.nl_energy = torch.tensor([row.data.nl_energy], dtype=torch.float32)
        data.integrated_densities = torch.tensor(row.data.integrated_densities, dtype=torch.float32)
        data.homo_energy = torch.tensor(row.data.homo_energy, dtype=torch.float32)
        data.homo_lumo_gap = torch.tensor(row.data.homo_lumo_gap, dtype=torch.float32)

        data.s_squared = torch.tensor([row.data.s_squared], dtype=torch.float32)
        data.s_squared_dev = torch.tensor([row.data.s_squared_dev], dtype=torch.float32)

        data.mulliken_charges = torch.tensor(row.data.mulliken_charges, dtype=torch.float32)
        data.lowdin_charges = torch.tensor(row.data.lowdin_charges, dtype=torch.float32)
        data.nbo_charges = torch.tensor(row.data.nbo_charges, dtype=torch.float32)

        data.mulliken_spins = torch.tensor(row.data.mulliken_spins, dtype=torch.float32)
        data.lowdin_spins = torch.tensor(row.data.lowdin_spins, dtype=torch.float32)
        data.nbo_spins = torch.tensor(row.data.nbo_spins, dtype=torch.float32)
    except Exception as e:
        print("Exception", e)


    return data


'''
Sample call of new ase2pyg.py function:
'''

# db_path = "/mnt/data/projects/TOODLES/OMol25/metals_train/"
# metals_file = "metal_complexes_data0000.aselmdb"

# ase_data = []

# try:
#     db = connect(db_path+metals_file, use_lock_file=False, readonly=True)
#     print("Processing database entries:")
#     # Get a few entries to inspect their keys and values
#     for i, row in enumerate(db.select(limit=1)):
#         print(f"\n--- Entry {i+1} (ID: {row.id}) ---")
#         ase_data.append( ase_to_pyg_graph(row) ) 

#     db.close()
    
# except Exception as e:
#     print(f"Error connecting to database: {e}")
#     print("Please check the path and if the file exists.")
#     exit()


















































def convert_ase_db_to_pyg(
    db_path: str,
    node_features: List[str] = ["Z"],
    edge_features: List[str] = ["distance"],
    limit: int = 100,
    **graph_kwargs
) -> List[Data]:
    """
    Convert all molecules in an ASE DB file to PyG graph objects.
    Args:
        db_path: Path to ASE database (.db)
        node_features: Features for nodes (atoms)
        edge_features: Features for edges (bonds)
        limit: Number of molecules to convert
    Returns:
        List of PyG Data objects

    """
    from ase.db import connect
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")

    graphs = []
    with connect(db_path) as db:
        for i, row in enumerate(db.select()):
            if i >= limit:
                break
            atoms = row.toatoms()  # attaches SinglePointCalculator if present

            # Promote global fields to atoms.info so ase_to_pyg_graph can see them
            for k in ("charge", "spin", "data_id"):
                if k in row.data:
                    atoms.info[k] = row.data[k]

            # Promote common per-atom arrays to atoms.arrays (if present)
            per_atom_keys = (
                "mulliken_charges", "lowdin_charges", "nbo_charges",
                "mulliken_spins", "lowdin_spins", "nbo_spins",
            )
            for k in per_atom_keys:
                if k in row.data:
                    arr = np.asarray(row.data[k])
                    if arr.shape[0] == len(atoms):
                        atoms.set_array(k, arr)

            data = ase_to_pyg_graph(
                atoms,
                node_features=node_features,
                edge_features=edge_features,
                **graph_kwargs
            )
            graphs.append(data)
    return graphs



