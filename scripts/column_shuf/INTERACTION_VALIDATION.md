# Interaction Validation for Unique Pairs

## Overview

The column shuffle pipeline now includes **interaction validation** to ensure that unique pairs identified from contact map differences have actual non-bonded interactions in the PDB structures. This adds a critical layer of biological validation to the computational analysis.

## Implementation Details

### What Was Added

1. **Import of ProteinResidueInteractionAnalyzer**: The pipeline now uses the comprehensive interaction analyzer from `protein_residue_interactions.py` to check for:
   - Hydrogen bonds (H-bonds)
   - Salt bridges
   - Hydrophobic contacts
   - Pi-pi stacking
   - Pi-cation interactions

2. **New Method: `_validate_unique_pairs_with_interactions()`**
   - Located in `ColumnShufflePipeline` class (lines 762-913)
   - Analyzes each unique pair from contact map analysis
   - Checks for actual geometric interactions using validated criteria
   - Filters pairs to only those with detectable interactions
   - Generates detailed interaction reports

3. **Integration into Contact Map Stage**
   - After identifying unique pairs from difference maps (Stage 1)
   - Validates both state1 and state2 pairs independently
   - Saves both raw and validated pairs
   - Uses validated pairs for downstream MSA shuffling

## How It Works

### Step-by-Step Process

1. **Contact Map Analysis** (existing):
   - Build contact maps from PDB structures using CB distances and sigmoid normalization
   - Compute difference map (state1 - state2)
   - Identify unique pairs based on threshold and sequence separation

2. **Interaction Validation** (NEW):
   - For each unique pair (i, j):
     - Load residues i and j from PDB structure
     - Check for H-bonds (distance ≤ 3.5 Å)
     - Check for salt bridges (opposite charges, distance ≤ 4.0 Å)
     - Check for hydrophobic contacts (both hydrophobic, C-C distance ≤ 4.5 Å)
     - Check for pi-pi stacking (both aromatic, centroid distance ≤ 6.0 Å, angle criteria)
     - Check for pi-cation (one aromatic + one cationic, distance ≤ 6.0 Å)
   - Keep only pairs with at least one detectable interaction

3. **Output Generation**:
   - `unique_pairs_raw.json`: All pairs from contact map analysis
   - `unique_pairs.json`: **Validated pairs only** (used for shuffling)
   - `{state}_interaction_validation.txt`: Detailed report for each state

## Output Files

### Interaction Validation Report Format

Each state gets a detailed report: `{state_name}_interaction_validation.txt`

**Section 1: Summary**
- Total pairs from contact map analysis
- Number of validated pairs with interactions
- Number of pairs without detectable interactions
- Validation rate (percentage)
- Interaction type distribution

**Section 2: Detailed Pair-by-Pair Analysis**
For each validated pair:
- Residue identities (e.g., A:VAL52 <-> A:PHE145)
- Minimum distance between any atoms
- CB distance (or CA for GLY)
- Total number of interactions
- Detailed list of each interaction type found with geometric parameters

**Section 3: Pairs Without Detectable Interactions**
- Lists pairs that showed contact map differences but no specific interactions
- Explains these may be distant or weak contacts

### Example Output

```
================================================================================
Interaction Validation Report: active_state
================================================================================

PDB Structure: /path/to/structure.pdb
Chain ID: A

SUMMARY
--------------------------------------------------------------------------------
Total pairs from contact map analysis: 78
Pairs with validated interactions: 52
Pairs without detectable interactions: 26
Validation rate: 66.7%

Interaction Type Distribution:
  hydrogen_bonds: 15 pairs
  salt_bridges: 3 pairs
  hydrophobic: 28 pairs
  pi_stacking: 2 pairs
  pi_cation: 4 pairs


DETAILED PAIR-BY-PAIR ANALYSIS
================================================================================

Pair 1: A:VAL52 <-> A:PHE145
--------------------------------------------------------------------------------
Minimum distance: 4.45 Å
CB distance: 5.05 Å
Total interactions: 1

HYDROPHOBIC:
  {'distance': 4.45, 'atom1': 'CG1', 'atom2': 'CZ'}
```

## Geometric Criteria Used

The interaction analyzer uses validated geometric criteria from structural biology literature:

| Interaction Type | Distance Cutoff | Additional Criteria |
|-----------------|----------------|-------------------|
| Hydrogen Bond | D-A ≤ 3.5 Å | Angle D-H-A ≥ 120° (if H present) |
| Salt Bridge | ≤ 4.0 Å | Opposite charges (LYS/ARG/HIS vs ASP/GLU) |
| Hydrophobic | C-C ≤ 4.5 Å | Both residues hydrophobic |
| Pi-Pi Stacking | Centroid ≤ 6.0 Å | Ring plane angle (parallel or T-shaped) |
| Pi-Cation | ≤ 6.0 Å | One aromatic + one cationic, angle from normal ≤ 60° |

## Impact on Pipeline

### No Changes Required to Usage
- **YAML config files**: No changes needed
- **Command line usage**: No changes needed
- **Stage execution**: Same as before

### What Changed Internally
1. **Stage 1 (Contact Maps)** now includes validation
2. **`unique_pairs.json`** now contains only validated pairs
3. **New output files**:
   - `unique_pairs_raw.json`: Original contact map pairs
   - `{state}_interaction_validation.txt`: Detailed reports

### Downstream Effects
- **Stage 2 (Shuffling)**: Uses validated pairs → shuffles fewer but more biologically relevant positions
- **Stage 3 (Prediction)**: Predictions focus on positions with actual interactions
- **Stage 4-6 (Analysis/Plotting)**: Results reflect validated interactions only

## Example Validation Results

From test with ABL1 active state:

```
Pair (16, 40): A:TYR16 <-> A:THR40
  Min distance: 4.48 Å
  CB distance: 6.63 Å
  Has interaction: False ✗
  → Filtered out (no specific interactions detected)

Pair (52, 145): A:VAL52 <-> A:PHE145
  Min distance: 4.45 Å
  CB distance: 5.05 Å
  Has interaction: True ✓
  hydrophobic: 1 contact(s)
  → Kept for shuffling

Pair (99, 266): A:ASN99 <-> A:SER266
  Min distance: 5.51 Å
  CB distance: 6.96 Å
  Has interaction: False ✗
  → Filtered out (no specific interactions detected)
```

This shows that the validation successfully distinguishes between:
- **True interactions**: Close contacts with specific geometric arrangements (hydrophobic, H-bonds, etc.)
- **Distant/weak contacts**: Pairs with sigmoid-normalized contact probability difference but no specific interaction type

## Biological Interpretation

### Why This Matters

1. **Contact maps use sigmoid normalization** which can identify pairs with relative distance changes but not necessarily specific interactions
2. **Validation ensures biological relevance** by requiring actual non-bonded interactions
3. **Reduces false positives** from contact map analysis
4. **Focuses shuffling** on positions with confirmed molecular interactions

### Typical Validation Rates

Expected validation rates vary by system:
- **Well-defined active sites**: 70-90% validation (most contact map pairs are real interactions)
- **Conformational interfaces**: 50-70% validation (some pairs are distant contacts)
- **Flexible regions**: 30-50% validation (many pairs are weak or transient)

## Troubleshooting

### If validation rate is very low (<30%)

Possible causes:
1. Contact threshold too low (picking up distant pairs)
2. PDB structure quality issues
3. Sigmoid parameters not optimal for the system

Solutions:
- Increase `contact_threshold` in config (e.g., from 0.8 to 0.9)
- Check PDB structure completeness
- Adjust `sigmoid_center` or `sigmoid_steepness`

### If validation fails

The pipeline will log errors but continue with original pairs to avoid breaking existing workflows.

## Technical Notes

### Indexing
- Contact map residue indices: 1-indexed (matching PDB)
- PDB residue numbering: 1-indexed
- No conversion needed between systems

### Chain ID
- Currently assumes chain 'A' (default for AF structures)
- Can be modified in `_validate_unique_pairs_with_interactions()` if needed

### Performance
- Validation adds ~1-2 minutes for typical systems (50-100 pairs)
- Dominated by PDB parsing and geometric calculations
- Negligible compared to structure prediction time

## References

The interaction detection criteria are based on:
- H-bonds: Baker & Hubbard (1984), Kabsch & Sander (1983)
- Salt bridges: Kumar & Nussinov (2002)
- Hydrophobic: Tsai et al. (1999)
- Pi interactions: McGaughey et al. (1998), Gallivan & Dougherty (1999)

See `protein_residue_interactions.py` for detailed implementation.
