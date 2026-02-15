# Side Chain-Only Interaction Validation Update

## Issue Identified

The original interaction validation was counting **backbone hydrogen bonds** (N-H···O=C), which are:
- Present in secondary structure elements (helices, sheets)
- **Not residue-specific** - they form regardless of side chain identity
- Misleading for mutation/shuffling analysis

### Example Problem

Consider two residues at positions 3 and 91:
- Query structure: **THR3 and GLU91**
- They form 2 backbone H-bonds (THR:N → GLU:O and GLU:N → THR:O)
- Validation marks this pair as having "interactions" ✓

**Problem**: If we shuffle and replace:
- THR3 → **LYS3** (different side chain!)
- GLU91 → **ARG91** (different side chain!)
- The backbone H-bonds would **still form** (same backbone, different side chains)
- But functionally, LYS-ARG is completely different from THR-GLU!

## Solution: Side Chain-Only Validation

Modified the interaction analyzer to focus **only on side chain interactions**:

### What Changed

**File**: `protein_residue_interactions.py`

1. **Added `sidechain_only` parameter** to methods:
   - `analyze_pair(sidechain_only=True)` - default is now True
   - `find_hydrogen_bonds(sidechain_only=True)`
   - `_get_hbond_atoms(sidechain_only=True)`

2. **Backbone atoms excluded by default**:
   - Backbone N (donor) - excluded
   - Backbone O (acceptor) - excluded
   - Only side chain atoms are considered

3. **H-bond detection now focuses on**:
   - SER/THR: OG, OG1 (hydroxyl)
   - ASP/GLU: OD1, OD2, OE1, OE2 (carboxyl)
   - LYS: NZ (amine)
   - ARG: NE, NH1, NH2 (guanidinium)
   - HIS: ND1, NE2 (imidazole)
   - ASN/GLN: ND2, NE2, OD1, OE1 (amide)
   - TYR: OH (phenolic)
   - CYS: SG (thiol)
   - TRP: NE1 (indole)

### Other Interactions Unchanged

These were already side chain-specific:
- ✓ **Salt bridges**: Always side chain (charged groups)
- ✓ **Hydrophobic**: Always side chain (hydrophobic carbons)
- ✓ **Pi-stacking**: Always side chain (aromatic rings)
- ✓ **Pi-cation**: Always side chain (aromatic + charged)

## Impact on Results

### Before (with backbone H-bonds)

```
Pair: A:THR3 <-> A:GLU91
Total interactions: 2
  - Hydrogen bonds: 2
    - THR:N -> GLU:O (BACKBONE)
    - GLU:N -> THR:O (BACKBONE)
Has interaction: True ✓
```

**Problem**: Would validate this pair even though side chains don't interact!

### After (side chain only)

```
Pair: A:THR3 <-> A:GLU91
Total interactions: 0
  - Hydrogen bonds: 0 (no sidechain H-bonds)
Has interaction: False ✗
```

**Correct**: Pair filtered out because side chains don't form specific interactions.

## Example: True Side Chain H-Bond

For a pair with **actual side chain H-bonds**:

```
Pair: A:SER65 <-> A:ASP92
Sidechain only: True
  - Hydrogen bonds: 1
    - SER:OG (sidechain) -> ASP:OD1 (sidechain), dist=2.8 Å
Has interaction: True ✓
```

This is a **residue-specific** interaction - if we mutate SER→ALA or ASP→GLY, the interaction is lost.

## Validation Rate Changes

Expect **lower validation rates** after this fix (which is correct):

### Example: KaiB Ground State

**Before** (with backbone):
```
Total pairs from contact map: 77
Pairs with interactions: 30 (38.9%)
```

**After** (sidechain only):
```
Total pairs from contact map: 77
Pairs with interactions: ~15-20 (estimated 19.5-26.0%)
```

This is **expected and correct** because:
- Backbone H-bonds were inflating the counts
- We now only count truly residue-specific interactions
- Filtered pairs weren't suitable for mutation/shuffling analysis anyway

## Compatibility Analysis Impact

The `analyze_interaction_compatibility.py` tool will now:
- Check compatibility against **true side chain requirements**
- Exclude pairs where only backbone interactions exist
- More accurately predict shuffling outcomes

Example update needed in compatibility rules:
- **H-bond donors**: Must have side chain with donor atoms (not just backbone N)
- **H-bond acceptors**: Must have side chain with acceptor atoms (not just backbone O)

## Migration Guide

### For Existing Results

If you have existing results from the old pipeline:
1. **Re-run Stage 1** (contact maps) to regenerate validation reports
2. The new `unique_pairs.json` will have fewer pairs (correct!)
3. Downstream stages (shuffle, predict, analyze) will automatically use new pairs

### Command

```bash
# Re-run just the contact map stage
python column_shuffle_pipeline.py config.yaml --stages contact_maps

# Then continue with other stages
python column_shuffle_pipeline.py config.yaml --stages shuffle predict analyze plot show_positions
```

### Expected Changes

You should see in the logs:

```
VALIDATION SUMMARY
================================================================================
State1 (active_state):
  Contact map pairs: 78
  Validated pairs:   18  ← Lower than before
  Validation rate:   23.1%  ← Lower than before, but more accurate
```

This is **correct** - we're now validating against actual residue-specific interactions!

## Technical Implementation

### Code Changes

**`protein_residue_interactions.py`**:

```python
# Old (included backbone)
def _get_hbond_atoms(self, residue) -> Tuple[List, List]:
    donors = []
    acceptors = []

    # Backbone atoms - ALWAYS included
    if 'N' in residue:
        donors.append(residue['N'])
    if 'O' in residue:
        acceptors.append(residue['O'])

    # ... sidechain atoms ...

# New (sidechain only by default)
def _get_hbond_atoms(self, residue, sidechain_only: bool = True) -> Tuple[List, List]:
    donors = []
    acceptors = []

    # Backbone atoms - only if explicitly requested
    if not sidechain_only:
        if 'N' in residue:
            donors.append(residue['N'])
        if 'O' in residue:
            acceptors.append(residue['O'])

    # ... sidechain atoms ...
```

**`column_shuffle_pipeline.py`**:

```python
# Always use sidechain-only mode
result = analyzer.analyze_pair(chain_id, res_i, chain_id, res_j, sidechain_only=True)
```

## Validation

To verify the fix is working:

```python
from protein_residue_interactions import ProteinResidueInteractionAnalyzer

analyzer = ProteinResidueInteractionAnalyzer('structure.pdb')

# Test both modes
result_with_bb = analyzer.analyze_pair('A', 3, 'A', 91, sidechain_only=False)
result_sc_only = analyzer.analyze_pair('A', 3, 'A', 91, sidechain_only=True)

print(f"With backbone: {result_with_bb['summary']['total_interactions']} interactions")
print(f"Sidechain only: {result_sc_only['summary']['total_interactions']} interactions")
```

Expected output for backbone H-bond pair:
```
With backbone: 2 interactions
Sidechain only: 0 interactions
```

## Summary

✅ **Fixed**: Backbone H-bonds no longer counted
✅ **Focus**: Only residue-specific side chain interactions
✅ **Impact**: More accurate validation, lower rates (but correct!)
✅ **Benefit**: Shuffling analysis now truly tests residue importance

This change ensures that the column shuffling pipeline correctly identifies pairs where **specific residue identities matter**, not just spatial proximity with backbone interactions.
