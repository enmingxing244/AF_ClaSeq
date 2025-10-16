import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import PDB
from Bio.PDB import *
from tqdm import tqdm

# Set publication-quality font
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24


def calculate_rmsd(pred_pdb, ref_pdb, superposition_start, superposition_end, rmsd_start, rmsd_end):
    # Load structures
    parser = PDB.PDBParser()
    pred_struct = parser.get_structure('pred', pred_pdb)
    ref_struct = parser.get_structure('ref', ref_pdb)
    
    # Get CA atoms for superposition
    pred_super_atoms = []
    ref_super_atoms = []
    
    # Get CA atoms for RMSD calculation
    pred_rmsd_atoms = []
    ref_rmsd_atoms = []
    
    # Only use first model and chain A for pred structure
    model = pred_struct[0]
    chain = model['A']
    for res in chain:
        if superposition_start <= res.id[1] <= superposition_end:
            if 'CA' in res:
                pred_super_atoms.append(res['CA'])
        if rmsd_start <= res.id[1] <= rmsd_end:
            if 'CA' in res:
                pred_rmsd_atoms.append(res['CA'])
                
    # Only use first model and chain A for ref structure  
    model = ref_struct[0]
    chain = model['A']
    for res in chain:
        if superposition_start <= res.id[1] <= superposition_end:
            if 'CA' in res:
                ref_super_atoms.append(res['CA'])
        if rmsd_start <= res.id[1] <= rmsd_end:
            if 'CA' in res:
                ref_rmsd_atoms.append(res['CA'])
                
    # Superimpose structures using superposition atoms
    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(ref_super_atoms, pred_super_atoms)
    
    # Apply transformation to all atoms
    super_imposer.apply(pred_struct.get_atoms())
    
    # Calculate RMSD using RMSD atoms
    rmsd_calc = PDB.Superimposer()
    rmsd_calc.set_atoms(ref_rmsd_atoms, pred_rmsd_atoms)
    return rmsd_calc.rms

# Only analyze 2LHD
case_config = {
    'super_start': 1,
    'super_end': 56,
    'rmsd_start': 1, 
    'rmsd_end': 56
}

print("Calculating RMSDs...")
all_rmsds = []

default_dir = '/fs/ess/PAA0203/xing244/AF_Vote/results/supple_cases/GB98/default'
ref_pdb = '/fs/ess/PAA0203/xing244/AF_Vote/results/supple_cases/GB98_all_conf/2LHD_a4b/ref/2LHD.pdb'

if not os.path.exists(ref_pdb):
    print(f"Warning: Reference PDB not found")
    exit(1)
    
pred_pdbs = glob.glob(f'{default_dir}/*.pdb')

if not pred_pdbs:
    print(f"Warning: No prediction PDBs found")
    exit(1)
    
for pred_pdb in tqdm(pred_pdbs, desc="Processing predictions"):
    rmsd = calculate_rmsd(pred_pdb, ref_pdb,
                         case_config['super_start'], case_config['super_end'],
                         case_config['rmsd_start'], case_config['rmsd_end'])
    all_rmsds.append(rmsd)
    print(f"RMSD for {os.path.basename(pred_pdb)}: {rmsd:.2f} Å")

print(f"\nAverage RMSD: {np.mean(all_rmsds):.2f} Å")
print(f"Standard deviation: {np.std(all_rmsds):.2f} Å")
print(f"Min RMSD: {np.min(all_rmsds):.2f} Å")
print(f"Max RMSD: {np.max(all_rmsds):.2f} Å")