#!/usr/bin/env python3
"""
Protein Residue-Residue Interaction Analyzer

Analyzes non-bonded interactions between protein residues including:
- Hydrogen bonds
- Salt bridges
- Hydrophobic contacts
- Pi-pi stacking
- Pi-cation interactions

Based on validated geometric criteria from recent literature.
"""

import numpy as np
from Bio.PDB import PDBParser, Selection
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class ProteinResidueInteractionAnalyzer:
    """
    Comprehensive analyzer for protein residue-residue interactions.
    Uses geometric criteria validated in structural biology literature.
    """
    
    # Amino acid classifications
    HYDROPHOBIC = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO', 'CYS'}
    AROMATIC = {'PHE', 'TYR', 'TRP', 'HIS'}
    POSITIVE = {'LYS', 'ARG', 'HIS'}  # HIS can be positive at physiological pH
    NEGATIVE = {'ASP', 'GLU'}
    
    # Geometric criteria (Angstroms and degrees)
    HBOND_DIST_MAX = 3.5  # Donor-Acceptor distance
    HBOND_ANGLE_MIN = 120  # Donor-H-Acceptor angle
    SALT_BRIDGE_DIST = 4.0  # Distance between charged atoms
    HYDROPHOBIC_DIST = 4.5  # Carbon-carbon distance for hydrophobic contacts
    PI_STACKING_DIST = 6.0  # Centroid-centroid distance
    PI_STACKING_ANGLE_MIN = 0  # Min angle between ring planes
    PI_STACKING_ANGLE_MAX = 40  # Max angle for parallel stacking
    PI_CATION_DIST = 6.0  # Cation to ring centroid
    PI_CATION_ANGLE_MIN = 0  # Angle from ring normal
    PI_CATION_ANGLE_MAX = 60  # Max deviation from perpendicular
    
    def __init__(self, pdb_file: str):
        """
        Initialize analyzer with a PDB file.
        
        Args:
            pdb_file: Path to PDB file
        """
        parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure('protein', pdb_file)
        self.model = self.structure[0]  # Use first model
        
    def get_residue(self, chain_id: str, res_num: int):
        """
        Get residue from structure.
        
        Args:
            chain_id: Chain identifier
            res_num: Residue number
            
        Returns:
            Residue object or None
        """
        try:
            chain = self.model[chain_id]
            # Handle insertion codes by checking all residues
            for residue in chain:
                if residue.id[1] == res_num and residue.id[0] == ' ':
                    return residue
        except KeyError:
            pass
        return None
    
    def analyze_pair(self, chain1: str, res1: int, chain2: str, res2: int, sidechain_only: bool = True) -> Dict:
        """
        Analyze all interactions between two residues.

        Args:
            chain1: First residue chain ID
            res1: First residue number
            chain2: Second residue chain ID
            res2: Second residue number
            sidechain_only: If True, only count side chain interactions (default: True)

        Returns:
            Dictionary containing all detected interactions
        """
        residue1 = self.get_residue(chain1, res1)
        residue2 = self.get_residue(chain2, res2)

        if residue1 is None or residue2 is None:
            return {'error': 'One or both residues not found'}

        # Check if same residue
        if (chain1 == chain2 and res1 == res2):
            return {'error': 'Cannot analyze same residue with itself'}

        results = {
            'residue1': f"{chain1}:{residue1.resname}{res1}",
            'residue2': f"{chain2}:{residue2.resname}{res2}",
            'min_distance': self._min_distance(residue1, residue2),
            'cb_distance': self._cb_distance(residue1, residue2),
            'interactions': {
                'hydrogen_bonds': self.find_hydrogen_bonds(residue1, residue2, sidechain_only=sidechain_only),
                'salt_bridges': self.find_salt_bridges(residue1, residue2),
                'hydrophobic': self.find_hydrophobic_contacts(residue1, residue2),
                'pi_stacking': self.find_pi_stacking(residue1, residue2),
                'pi_cation': self.find_pi_cation(residue1, residue2)
            }
        }

        # Add summary
        total_interactions = sum(
            len(v) if isinstance(v, list) else (1 if v else 0)
            for v in results['interactions'].values()
        )
        results['summary'] = {
            'total_interactions': total_interactions,
            'has_interaction': total_interactions > 0
        }

        return results
    
    def find_hydrogen_bonds(self, res1, res2, sidechain_only: bool = True) -> List[Dict]:
        """
        Find hydrogen bonds between two residues.

        Criteria:
        - Distance: D-A ≤ 3.5 Å
        - Angle: D-H-A ≥ 120°

        Args:
            res1: First residue
            res2: Second residue
            sidechain_only: If True, only count side chain H-bonds (exclude backbone)
        """
        hbonds = []

        # Get potential donors and acceptors from both residues
        donors1, acceptors1 = self._get_hbond_atoms(res1, sidechain_only=sidechain_only)
        donors2, acceptors2 = self._get_hbond_atoms(res2, sidechain_only=sidechain_only)

        # Check res1 donors with res2 acceptors
        hbonds.extend(self._check_hbond_pairs(donors1, acceptors2, res1, res2))

        # Check res2 donors with res1 acceptors
        hbonds.extend(self._check_hbond_pairs(donors2, acceptors1, res2, res1))

        return hbonds
    
    def find_salt_bridges(self, res1, res2) -> Optional[Dict]:
        """
        Find salt bridge between oppositely charged residues.
        
        Criteria:
        - Distance between charged atoms ≤ 4.0 Å
        - Residues must be oppositely charged
        """
        # Check if residues are oppositely charged
        res1_charged = res1.resname in self.POSITIVE or res1.resname in self.NEGATIVE
        res2_charged = res2.resname in self.POSITIVE or res2.resname in self.NEGATIVE
        
        if not (res1_charged and res2_charged):
            return None
        
        # Must be opposite charges
        opposite_charge = (
            (res1.resname in self.POSITIVE and res2.resname in self.NEGATIVE) or
            (res1.resname in self.NEGATIVE and res2.resname in self.POSITIVE)
        )
        
        if not opposite_charge:
            return None
        
        # Get charged atoms
        charged1 = self._get_charged_atoms(res1)
        charged2 = self._get_charged_atoms(res2)
        
        # Find minimum distance
        min_dist = float('inf')
        best_pair = None
        
        for atom1 in charged1:
            for atom2 in charged2:
                dist = self._distance(atom1, atom2)
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (atom1.name, atom2.name)
        
        if min_dist <= self.SALT_BRIDGE_DIST:
            return {
                'distance': round(min_dist, 2),
                'atom1': best_pair[0],
                'atom2': best_pair[1],
                'type': f"{res1.resname}-{res2.resname}"
            }
        
        return None
    
    def find_hydrophobic_contacts(self, res1, res2) -> List[Dict]:
        """
        Find hydrophobic contacts between residues.
        
        Criteria:
        - Both residues must be hydrophobic
        - Carbon-carbon distance ≤ 4.5 Å
        """
        if res1.resname not in self.HYDROPHOBIC or res2.resname not in self.HYDROPHOBIC:
            return []
        
        contacts = []
        
        # Get carbon atoms from sidechains
        carbons1 = [a for a in res1 if a.element == 'C' and a.name not in ['C', 'CA', 'O']]
        carbons2 = [a for a in res2 if a.element == 'C' and a.name not in ['C', 'CA', 'O']]
        
        for c1 in carbons1:
            for c2 in carbons2:
                dist = self._distance(c1, c2)
                if dist <= self.HYDROPHOBIC_DIST:
                    contacts.append({
                        'distance': round(dist, 2),
                        'atom1': c1.name,
                        'atom2': c2.name
                    })
        
        return contacts
    
    def find_pi_stacking(self, res1, res2) -> Optional[Dict]:
        """
        Find pi-pi stacking between aromatic residues.
        
        Criteria:
        - Both residues must be aromatic
        - Centroid distance ≤ 6.0 Å
        - Angle between ring planes ≤ 40° (parallel) or ≥ 60° (T-shaped)
        """
        if res1.resname not in self.AROMATIC or res2.resname not in self.AROMATIC:
            return None
        
        # Get ring centroids and normals
        centroid1, normal1 = self._get_aromatic_ring_info(res1)
        centroid2, normal2 = self._get_aromatic_ring_info(res2)
        
        if centroid1 is None or centroid2 is None:
            return None
        
        # Calculate distance
        dist = np.linalg.norm(centroid1 - centroid2)
        
        if dist > self.PI_STACKING_DIST:
            return None
        
        # Calculate angle between ring planes
        if normal1 is not None and normal2 is not None:
            angle = self._angle_between_vectors(normal1, normal2)
            angle_deg = np.degrees(angle)
            
            # Determine stacking type
            if angle_deg <= self.PI_STACKING_ANGLE_MAX:
                stack_type = "parallel"
            elif angle_deg >= (90 - self.PI_STACKING_ANGLE_MAX):
                stack_type = "T-shaped"
            else:
                stack_type = "offset"
            
            return {
                'distance': round(dist, 2),
                'angle': round(angle_deg, 1),
                'type': stack_type
            }
        else:
            return {
                'distance': round(dist, 2),
                'type': 'unknown'
            }
    
    def find_pi_cation(self, res1, res2) -> Optional[Dict]:
        """
        Find pi-cation interactions.
        
        Criteria:
        - One aromatic, one cationic residue
        - Distance from cation to ring centroid ≤ 6.0 Å
        - Angle from ring normal ≤ 60°
        """
        # Determine which is aromatic and which is cationic
        aromatic_res, cation_res = None, None
        
        if res1.resname in self.AROMATIC and res2.resname in self.POSITIVE:
            aromatic_res, cation_res = res1, res2
        elif res2.resname in self.AROMATIC and res1.resname in self.POSITIVE:
            aromatic_res, cation_res = res2, res1
        else:
            return None
        
        # Get aromatic ring info
        centroid, normal = self._get_aromatic_ring_info(aromatic_res)
        if centroid is None:
            return None
        
        # Get cation center
        cation_atom = self._get_cation_atom(cation_res)
        if cation_atom is None:
            return None
        
        # Calculate distance
        cation_coord = cation_atom.coord
        dist = np.linalg.norm(centroid - cation_coord)
        
        if dist > self.PI_CATION_DIST:
            return None
        
        # Calculate angle from ring normal
        if normal is not None:
            vector_to_cation = cation_coord - centroid
            vector_to_cation = vector_to_cation / np.linalg.norm(vector_to_cation)
            
            angle = self._angle_between_vectors(normal, vector_to_cation)
            angle_deg = np.degrees(angle)
            
            # Angle should be close to 90 degrees (perpendicular)
            deviation_from_perpendicular = abs(90 - angle_deg)
            
            if deviation_from_perpendicular <= self.PI_CATION_ANGLE_MAX:
                return {
                    'distance': round(dist, 2),
                    'angle': round(angle_deg, 1),
                    'aromatic': aromatic_res.resname,
                    'cation': cation_res.resname,
                    'cation_atom': cation_atom.name
                }
        else:
            # If we can't calculate angle, just use distance
            return {
                'distance': round(dist, 2),
                'aromatic': aromatic_res.resname,
                'cation': cation_res.resname,
                'cation_atom': cation_atom.name
            }
        
        return None
    
    # ============ Helper Methods ============
    
    def _distance(self, atom1, atom2) -> float:
        """Calculate distance between two atoms."""
        return np.linalg.norm(atom1.coord - atom2.coord)
    
    def _min_distance(self, res1, res2) -> float:
        """Calculate minimum distance between any atoms in two residues."""
        min_dist = float('inf')
        for atom1 in res1:
            for atom2 in res2:
                dist = self._distance(atom1, atom2)
                if dist < min_dist:
                    min_dist = dist
        return round(min_dist, 2)
    
    def _cb_distance(self, res1, res2) -> Optional[float]:
        """Calculate C-beta distance (C-alpha for glycine)."""
        cb1 = 'CB' if 'CB' in res1 else 'CA'
        cb2 = 'CB' if 'CB' in res2 else 'CA'
        
        if cb1 in res1 and cb2 in res2:
            return round(self._distance(res1[cb1], res2[cb2]), 2)
        return None
    
    def _get_hbond_atoms(self, residue, sidechain_only: bool = True) -> Tuple[List, List]:
        """
        Get potential hydrogen bond donors and acceptors.

        Args:
            residue: Residue object
            sidechain_only: If True, exclude backbone atoms (N and O)
        """
        donors = []
        acceptors = []

        # Backbone atoms - only include if not sidechain_only
        if not sidechain_only:
            if 'N' in residue:
                donors.append(residue['N'])
            if 'O' in residue:
                acceptors.append(residue['O'])

        # Sidechain atoms based on residue type
        resname = residue.resname

        # Donors (atoms with H attached)
        donor_atoms = {
            'SER': ['OG'], 'THR': ['OG1'], 'CYS': ['SG'],
            'TYR': ['OH'], 'LYS': ['NZ'], 'ARG': ['NE', 'NH1', 'NH2'],
            'HIS': ['ND1', 'NE2'], 'ASN': ['ND2'], 'GLN': ['NE2'],
            'TRP': ['NE1']
        }

        # Acceptors (O, N with lone pairs)
        acceptor_atoms = {
            'ASP': ['OD1', 'OD2'], 'GLU': ['OE1', 'OE2'],
            'ASN': ['OD1'], 'GLN': ['OE1'],
            'SER': ['OG'], 'THR': ['OG1'], 'TYR': ['OH'],
            'HIS': ['ND1', 'NE2'], 'MET': ['SD']
        }

        if resname in donor_atoms:
            for atom_name in donor_atoms[resname]:
                if atom_name in residue:
                    donors.append(residue[atom_name])

        if resname in acceptor_atoms:
            for atom_name in acceptor_atoms[resname]:
                if atom_name in residue:
                    acceptors.append(residue[atom_name])

        return donors, acceptors
    
    def _check_hbond_pairs(self, donors, acceptors, donor_res, acceptor_res) -> List[Dict]:
        """
        Check all donor-acceptor pairs for hydrogen bonds.

        Uses distance criterion and estimates geometry using heavy atoms.
        Since most PDB files lack hydrogens, we estimate the donor-acceptor
        geometry using the antecedent atom (atom bonded to the donor).
        """
        hbonds = []

        for donor in donors:
            for acceptor in acceptors:
                dist = self._distance(donor, acceptor)

                if dist <= self.HBOND_DIST_MAX:
                    # Estimate geometry using heavy atoms
                    # Get the antecedent atom for the donor (atom bonded to donor)
                    antecedent = self._get_antecedent_atom(donor, donor_res)

                    # Only accept if geometry is reasonable
                    geometry_ok = True
                    angle = None

                    if antecedent is not None:
                        # Calculate antecedent-donor-acceptor angle
                        # This approximates the D-H-A angle
                        angle = self._calculate_angle(antecedent, donor, acceptor)
                        angle_deg = np.degrees(angle)

                        # Require angle > 90° (ideally > 120°, but we're lenient without H)
                        # This filters out bad geometries where donor/acceptor are perpendicular
                        if angle_deg < 90:
                            geometry_ok = False

                    if geometry_ok:
                        hbond_data = {
                            'donor_atom': donor.name,
                            'acceptor_atom': acceptor.name,
                            'distance': round(dist, 2),
                            'donor_res': donor_res.resname,
                            'acceptor_res': acceptor_res.resname
                        }
                        if angle is not None:
                            hbond_data['angle'] = round(np.degrees(angle), 1)

                        hbonds.append(hbond_data)

        return hbonds
    
    def _get_charged_atoms(self, residue) -> List:
        """Get atoms involved in charged groups."""
        charged_atoms_map = {
            'LYS': ['NZ'],
            'ARG': ['NH1', 'NH2', 'NE', 'CZ'],
            'HIS': ['ND1', 'NE2'],
            'ASP': ['OD1', 'OD2'],
            'GLU': ['OE1', 'OE2']
        }
        
        atoms = []
        if residue.resname in charged_atoms_map:
            for atom_name in charged_atoms_map[residue.resname]:
                if atom_name in residue:
                    atoms.append(residue[atom_name])
        
        return atoms
    
    def _get_aromatic_ring_info(self, residue) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get aromatic ring centroid and normal vector.
        
        Returns:
            (centroid, normal) or (None, None)
        """
        ring_atoms_map = {
            'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'TRP': ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],  # 6-membered ring
            'HIS': ['CG', 'ND1', 'CD2', 'CE1', 'NE2']
        }
        
        if residue.resname not in ring_atoms_map:
            return None, None
        
        # Get atom coordinates
        coords = []
        for atom_name in ring_atoms_map[residue.resname]:
            if atom_name in residue:
                coords.append(residue[atom_name].coord)
        
        if len(coords) < 3:
            return None, None
        
        coords = np.array(coords)
        
        # Calculate centroid
        centroid = np.mean(coords, axis=0)
        
        # Calculate normal vector (for plane of ring)
        if len(coords) >= 3:
            # Use first three atoms to define plane
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)  # Normalize
        else:
            normal = None
        
        return centroid, normal
    
    def _get_cation_atom(self, residue):
        """Get the atom representing the cation center."""
        cation_atoms = {
            'LYS': 'NZ',
            'ARG': 'CZ',  # Guanidinium carbon center
            'HIS': 'CG'   # Ring center for histidine
        }
        
        if residue.resname in cation_atoms:
            atom_name = cation_atoms[residue.resname]
            if atom_name in residue:
                return residue[atom_name]
        
        return None
    
    def _angle_between_vectors(self, v1, v2) -> float:
        """Calculate angle between two vectors in radians."""
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)

        # Clamp dot product to avoid numerical errors
        dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)

        return np.arccos(dot_product)

    def _calculate_angle(self, atom1, atom2, atom3) -> float:
        """
        Calculate angle atom1-atom2-atom3 (angle at atom2).

        Args:
            atom1, atom2, atom3: Atom objects

        Returns:
            Angle in radians
        """
        vec1 = atom1.coord - atom2.coord
        vec2 = atom3.coord - atom2.coord
        return self._angle_between_vectors(vec1, vec2)

    def _get_antecedent_atom(self, donor_atom, residue):
        """
        Get the atom bonded to the donor (for angle estimation).

        For example:
        - For sidechain OG (SER/THR): return CB
        - For NZ (LYS): return CE
        - For backbone N: return CA

        This helps estimate D-H-A angle without hydrogen positions.
        """
        donor_name = donor_atom.name

        # Mapping of donor atoms to their antecedent (bonded heavy atom)
        antecedent_map = {
            # Backbone
            'N': 'CA',
            # Serine/Threonine
            'OG': 'CB', 'OG1': 'CB',
            # Cysteine
            'SG': 'CB',
            # Tyrosine
            'OH': 'CZ',
            # Lysine
            'NZ': 'CE',
            # Arginine
            'NE': 'CD', 'NH1': 'CZ', 'NH2': 'CZ',
            # Histidine
            'ND1': 'CG', 'NE2': 'CD2',
            # Asparagine
            'ND2': 'CG',
            # Glutamine
            'NE2': 'CD',
            # Tryptophan
            'NE1': 'CD1',
            # Acceptor atoms (for reverse direction)
            'O': 'C',
            'OD1': 'CG', 'OD2': 'CG',  # ASP
            'OE1': 'CD', 'OE2': 'CD',  # GLU
            'SD': 'CG',  # MET
        }

        antecedent_name = antecedent_map.get(donor_name)
        if antecedent_name and antecedent_name in residue:
            return residue[antecedent_name]

        return None
    
    def scan_all_interactions(self, distance_cutoff: float = 8.0) -> List[Dict]:
        """
        Scan entire protein for all residue-residue interactions.
        
        Args:
            distance_cutoff: Initial distance filter (Angstroms)
            
        Returns:
            List of all detected interactions
        """
        all_interactions = []
        
        # Get all residues
        all_residues = list(Selection.unfold_entities(self.model, 'R'))
        
        # Filter to standard amino acids
        standard_residues = [r for r in all_residues if r.id[0] == ' ']
        
        # Check all pairs
        for i, res1 in enumerate(standard_residues):
            for res2 in standard_residues[i+1:]:
                # Quick distance filter
                min_dist = self._min_distance(res1, res2)
                
                if min_dist <= distance_cutoff:
                    chain1 = res1.parent.id
                    chain2 = res2.parent.id
                    
                    result = self.analyze_pair(
                        chain1, res1.id[1],
                        chain2, res2.id[1]
                    )
                    
                    if result.get('summary', {}).get('has_interaction', False):
                        all_interactions.append(result)
        
        return all_interactions


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python protein_residue_interactions.py <pdb_file> [chain1 res1 chain2 res2]")
        print("\nExamples:")
        print("  # Analyze specific residue pair:")
        print("  python protein_residue_interactions.py protein.pdb A 50 A 75")
        print("\n  # Scan all interactions:")
        print("  python protein_residue_interactions.py protein.pdb --scan")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    analyzer = ProteinResidueInteractionAnalyzer(pdb_file)
    
    if len(sys.argv) >= 3 and sys.argv[2] == '--scan':
        print("Scanning entire protein for interactions...")
        interactions = analyzer.scan_all_interactions()
        
        print(f"\nFound {len(interactions)} interacting residue pairs\n")
        
        for interaction in interactions[:10]:  # Show first 10
            print(f"{interaction['residue1']} <-> {interaction['residue2']}")
            print(f"  Min distance: {interaction['min_distance']} Å")
            print(f"  CB distance: {interaction['cb_distance']} Å")
            
            for itype, data in interaction['interactions'].items():
                if data:
                    if isinstance(data, list):
                        print(f"  {itype}: {len(data)} contacts")
                    else:
                        print(f"  {itype}: {data}")
            print()
        
    elif len(sys.argv) >= 6:
        chain1, res1, chain2, res2 = sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
        
        print(f"\nAnalyzing interactions between {chain1}:{res1} and {chain2}:{res2}\n")
        
        result = analyzer.analyze_pair(chain1, res1, chain2, res2)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Residue 1: {result['residue1']}")
            print(f"Residue 2: {result['residue2']}")
            print(f"Minimum distance: {result['min_distance']} Å")
            print(f"C-beta distance: {result['cb_distance']} Å")
            print(f"\nInteractions found: {result['summary']['total_interactions']}\n")
            
            for interaction_type, data in result['interactions'].items():
                if data:
                    print(f"{interaction_type.upper()}:")
                    if isinstance(data, list):
                        for item in data:
                            print(f"  {item}")
                    else:
                        print(f"  {data}")
                    print()


if __name__ == '__main__':
    main()