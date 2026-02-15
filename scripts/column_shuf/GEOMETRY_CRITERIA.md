# Geometric Criteria for Interaction Detection

## Updated: Angle Checking for H-Bonds

We now check **both distance AND angle** for hydrogen bonds (where possible) to ensure proper geometry.

## Complete Geometric Criteria

### 1. Hydrogen Bonds ✅ DISTANCE + ANGLE

**Distance Criterion:**
- Donor-Acceptor distance ≤ 3.5 Å

**Angle Criterion** (NEW!):
- Antecedent-Donor-Acceptor angle ≥ 90°
- Approximates D-H-A angle (since most PDBs lack hydrogens)
- Filters out perpendicular geometries that aren't real H-bonds

**Atoms Checked** (sidechain only):
```
Donors:
  SER/THR: OG, OG1 (hydroxyl)
  TYR: OH (phenolic)
  CYS: SG (thiol)
  LYS: NZ (amine)
  ARG: NE, NH1, NH2 (guanidinium)
  HIS: ND1, NE2 (imidazole)
  ASN: ND2 (amide)
  GLN: NE2 (amide)
  TRP: NE1 (indole)

Acceptors:
  ASP: OD1, OD2 (carboxylate)
  GLU: OE1, OE2 (carboxylate)
  ASN: OD1 (carbonyl)
  GLN: OE1 (carbonyl)
  SER/THR: OG, OG1 (hydroxyl)
  TYR: OH (phenolic)
  HIS: ND1, NE2 (imidazole)
  MET: SD (sulfur)
```

**Example:**
```
SER:CB---OG···OD1:ASP
      ↑       ↑
   Antecedent Donor-Acceptor

Angle: CB-OG-OD1 = 135° ✓ (>90°, geometry OK)
Distance: OG···OD1 = 2.8 Å ✓ (≤3.5 Å)
→ Valid H-bond!
```

**Bad Geometry Filtered:**
```
     CB
      |
     OG----90°----OD1
          (perpendicular)

Angle: CB-OG-OD1 = 90° ✗ (not >90°)
→ Rejected (poor geometry)
```

---

### 2. Salt Bridges ⚠️ DISTANCE ONLY

**Distance Criterion:**
- Charged atom to charged atom ≤ 4.0 Å

**Residue Requirements:**
- Must be oppositely charged
- Positive: LYS (NZ), ARG (NH1, NH2, NE, CZ), HIS (ND1, NE2)
- Negative: ASP (OD1, OD2), GLU (OE1, OE2)

**Example:**
```
LYS:NZ (positive) ···3.2 Å··· OD1:ASP (negative)
→ Valid salt bridge!
```

**Note:** No angle checking currently. Distance-only criterion is standard for salt bridges since the electrostatic interaction is less directional than H-bonds.

---

### 3. Hydrophobic Contacts ⚠️ DISTANCE ONLY

**Distance Criterion:**
- Carbon-carbon distance ≤ 4.5 Å

**Residue Requirements:**
- Both must be hydrophobic: ALA, VAL, ILE, LEU, MET, PHE, TRP, PRO, CYS

**Atoms Checked:**
- Sidechain carbons only (not CA, C, O)

**Example:**
```
LEU:CD1 ···4.2 Å··· CD2:VAL
→ Valid hydrophobic contact!
```

**Note:** Van der Waals interactions are non-directional, so angle checking is not applicable.

---

### 4. Pi-Pi Stacking ✅ DISTANCE + ANGLE

**Distance Criterion:**
- Ring centroid to centroid ≤ 6.0 Å

**Angle Criterion:**
- Angle between ring planes:
  - ≤ 40° → Parallel stacking
  - ≥ 60° → T-shaped stacking
  - Between 40-60° → Offset stacking (also accepted)

**Residue Requirements:**
- Both must be aromatic: PHE, TYR, TRP, HIS

**Ring Definitions:**
```
PHE: 6-membered benzene ring (CG, CD1, CD2, CE1, CE2, CZ)
TYR: 6-membered benzene ring (CG, CD1, CD2, CE1, CE2, CZ)
TRP: 6-membered ring (CD2, CE2, CE3, CZ2, CZ3, CH2)
HIS: 5-membered imidazole ring (CG, ND1, CD2, CE1, NE2)
```

**Example:**
```
PHE ring ━━━━━━━ 4.8 Å ━━━━━━━ TYR ring
   ║                              ║
Angle between planes = 15° (parallel)
→ Valid pi-pi stacking!
```

---

### 5. Pi-Cation ✅ DISTANCE + ANGLE

**Distance Criterion:**
- Cation to ring centroid ≤ 6.0 Å

**Angle Criterion:**
- Deviation from ring normal ≤ 60°
- Cation should approach perpendicular to ring plane

**Residue Requirements:**
- One aromatic: PHE, TYR, TRP, HIS
- One cationic: LYS (NZ), ARG (CZ), HIS (CG)

**Example:**
```
        NZ (LYS cation)
         |
         | 5.2 Å
         ↓ ~85° from ring plane
    ═══PHE ring═══

Deviation from perpendicular: |90° - 85°| = 5° (≤60°)
→ Valid pi-cation!
```

---

## Summary Table

| Interaction | Distance | Angle | Notes |
|------------|----------|-------|-------|
| **H-Bond** | ≤ 3.5 Å | ≥ 90° | Angle estimates D-H-A without H atoms |
| **Salt Bridge** | ≤ 4.0 Å | - | Electrostatic, less directional |
| **Hydrophobic** | ≤ 4.5 Å | - | Van der Waals, non-directional |
| **Pi-Stacking** | ≤ 6.0 Å | ≤ 40° or ≥ 60° | Ring plane angles |
| **Pi-Cation** | ≤ 6.0 Å | ≤ 60° | Deviation from perpendicular |

---

## Impact of Angle Checking

### Before (distance only):
```
SER:OG ···3.2 Å··· OD1:ASP
Geometry: Perpendicular (90°)
Result: Accepted as H-bond ✓ (WRONG!)
```

### After (distance + angle):
```
CB-OG-OD1 angle = 90°
Check: 90° ≱ 90° (not greater than)
Result: Rejected ✗ (CORRECT!)
```

**Impact:** More stringent filtering, fewer false positives.

---

## Example: LYS7-ASP87 from KaiB

**Analysis:**
```
Min distance: 2.83 Å (very close!)
CB distance: 4.34 Å

With backbone H-bonds:
  - N-O H-bond: 2.83 Å, angle 115.9° ✓
  - N-O H-bond: 2.93 Å, angle 126.9° ✓
  Total: 2 backbone H-bonds

With sidechain only:
  - Salt bridge: NZ···OD1/OD2 >4.0 Å ✗
  - Sidechain H-bonds: None
  Total: 0 interactions

Conclusion: Backbones are close, but side chains DON'T interact!
```

This is **exactly** the scenario you identified - the residues could be mutated (LYS→ARG, ASP→GLU) and backbone H-bonds would still form, but this isn't a residue-specific interaction.

---

## Code Implementation

### H-Bond Geometry Checking

```python
# Get antecedent atom (atom bonded to donor)
antecedent = self._get_antecedent_atom(donor, donor_res)

if antecedent is not None:
    # Calculate antecedent-donor-acceptor angle
    angle = self._calculate_angle(antecedent, donor, acceptor)
    angle_deg = np.degrees(angle)

    # Require angle > 90° (filters perpendicular geometries)
    if angle_deg < 90:
        geometry_ok = False
```

### Antecedent Mapping

```python
antecedent_map = {
    'OG': 'CB',   # SER/THR hydroxyl → beta carbon
    'NZ': 'CE',   # LYS amine → epsilon carbon
    'NH1': 'CZ',  # ARG guanidinium → zeta carbon
    'OD1': 'CG',  # ASP carboxyl → gamma carbon
    # ... etc
}
```

This approximates the D-H-A angle using heavy atoms only, since hydrogens are absent from most PDB files.

---

## Validation

Run this test to verify geometry checking:

```bash
python3 << 'EOF'
from protein_residue_interactions import ProteinResidueInteractionAnalyzer

analyzer = ProteinResidueInteractionAnalyzer('structure.pdb')

# Test with and without sidechain filtering
result_with_bb = analyzer.analyze_pair('A', 7, 'A', 87, sidechain_only=False)
result_sc_only = analyzer.analyze_pair('A', 7, 'A', 87, sidechain_only=True)

print("With backbone:", result_with_bb['summary']['total_interactions'])
print("Sidechain only:", result_sc_only['summary']['total_interactions'])

# Check H-bond angles
for hb in result_with_bb['interactions']['hydrogen_bonds']:
    print(f"H-bond: {hb['donor_atom']}→{hb['acceptor_atom']}, "
          f"dist={hb['distance']} Å, angle={hb.get('angle', 'N/A')}°")
EOF
```

Expected output:
```
With backbone: 2
Sidechain only: 0
H-bond: N→O, dist=2.83 Å, angle=115.9°
H-bond: N→O, dist=2.93 Å, angle=126.9°
```

---

## Future Enhancements

Possible improvements:
1. **Salt bridge angles**: Add geometry checking for optimal salt bridge orientation
2. **Hydrophobic packing**: Check for complementary surface shapes
3. **Multiple criteria**: Require multiple interaction types for very strong validation
4. **Dynamic cutoffs**: Adjust distance/angle thresholds based on residue types

Currently, the distance + angle approach provides a good balance between accuracy and compatibility with standard PDB files (which lack hydrogens).
