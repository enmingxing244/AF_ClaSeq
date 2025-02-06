def reindex_pdb(input_pdb, chain_id, index_diff, default_pdb=None):
    """
    Reindex residue numbers in a PDB file for a specific chain by adding an offset.
    Only keeps protein atoms from the specified chain, removing HETATM, water, other chains and non-protein atoms.
    For NMR structures with multiple models, only uses the first model.
    Can compare against a reference PDB file to check residue numbering consistency.
    Removes residues with index <= 0 after applying offset.
    
    Args:
        input_pdb (str): Path to input PDB file
        chain_id (str): Chain ID to reindex and keep (other chains will be removed)
        index_diff (int): Number to add to existing residue indices (can be positive or negative)
        default_pdb (str, optional): Path to reference PDB file to check residue numbering
    """
    # Generate output filename with chain info
    output_pdb = input_pdb.replace('.pdb', f'_processed_chain{chain_id}.pdb')
    log_file = input_pdb.replace('.pdb', f'_processed_chain{chain_id}.log')
    
    # Get residue numbers and names from default PDB if provided
    default_residues = {}  # Map residue number to residue name
    if default_pdb:
        with open(default_pdb, 'r') as f:
            prev_resnum = None
            for line in f:
                if line.startswith('ATOM') and line[21] == chain_id:
                    resnum = int(line[22:26])
                    resname = line[17:20].strip()
                    default_residues[resnum] = resname
                    if prev_resnum and resnum - prev_resnum > 1:
                        print(f"Gap in default PDB between residues {prev_resnum} and {resnum}")
                    prev_resnum = resnum
    
    # Process input PDB
    reindexed_residues = {}  # Map new residue number to residue name
    with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out, open(log_file, 'w') as f_log:
        # Record command
        command = f"python ref_pdb_process.py {input_pdb} {chain_id} {index_diff}"
        if default_pdb:
            command += f" --default_pdb {default_pdb}"
        f_log.write(f"Command executed: {command}\n\n")
        
        prev_resnum = None
        last_atom_line = None
        in_first_model = True
        
        for line in f_in:
            # Track model boundaries
            if line.startswith('MODEL'):
                if line.strip().split()[-1] != '1':
                    in_first_model = False
                continue
            if line.startswith('ENDMDL'):
                in_first_model = False
                continue
                
            # Only process lines from first model
            if not in_first_model:
                continue
                
            if line.startswith('ATOM'):  # Only keep ATOM records (protein)
                # Only keep lines for the specified chain
                if line[21] == chain_id:
                    # Extract current residue number and name
                    curr_resnum = int(line[22:26])
                    curr_resname = line[17:20].strip()
                    # Calculate new residue number
                    new_resnum = curr_resnum + index_diff
                    
                    # Skip residues with index <= 0 after offset
                    if new_resnum <= 0:
                        continue
                        
                    reindexed_residues[new_resnum] = curr_resname
                    
                    # Check for gaps in numbering
                    if prev_resnum and new_resnum - prev_resnum > 1:
                        msg = f"Gap in processed PDB between residues {prev_resnum} and {new_resnum}"
                        print(msg)
                        f_log.write(msg + '\n')
                    prev_resnum = new_resnum
                    
                    # Format new residue number to maintain PDB format
                    new_resnum_str = str(new_resnum).rjust(4)
                    # Write modified line
                    new_line = line[:22] + new_resnum_str + line[26:]
                    f_out.write(new_line)
                    last_atom_line = new_line
        
        # Write TER line after last ATOM record
        if last_atom_line:
            # Extract info from last atom line to create TER line
            chain = last_atom_line[21]
            resname = last_atom_line[17:20]
            resnum = last_atom_line[22:26]
            ter_line = f"TER   {last_atom_line[6:11]}      {resname} {chain}{resnum}\n"
            f_out.write(ter_line)
    
    # Compare residue numbers and names if default PDB was provided
    if default_pdb:
        missing_in_reindex = set(default_residues.keys()) - set(reindexed_residues.keys())
        extra_in_reindex = set(reindexed_residues.keys()) - set(default_residues.keys())
        
        # Check residue name consistency
        common_residues = set(default_residues.keys()) & set(reindexed_residues.keys())
        inconsistent_residues = []
        for resnum in common_residues:
            if default_residues[resnum] != reindexed_residues[resnum]:
                inconsistent_residues.append((resnum, default_residues[resnum], reindexed_residues[resnum]))
        
        with open(log_file, 'a') as f_log:
            if inconsistent_residues:
                msg = "\nResidues with inconsistent names between default and processed PDB:"
                print(msg)
                f_log.write(msg + '\n')
                for resnum, def_name, reindex_name in inconsistent_residues:
                    msg = f"Residue {resnum}: {def_name} (default) vs {reindex_name} (processed)"
                    print(msg)
                    f_log.write(msg + '\n')
            else:
                msg = "\nAll common residues have consistent names between default and processed PDB"
                print(msg)
                f_log.write(msg + '\n')
            
            if missing_in_reindex:
                msg = f"\nResidues in default PDB but missing in processed PDB:"
                print(msg)
                f_log.write(msg + '\n')
                msg = str(sorted(missing_in_reindex))
                print(msg)
                f_log.write(msg + '\n')
                
            if extra_in_reindex:
                msg = f"\nResidues in processed PDB but not in default PDB:"
                print(msg)
                f_log.write(msg + '\n')
                msg = str(sorted(extra_in_reindex))
                print(msg)
                f_log.write(msg + '\n')
    
    msg = f"\nProcessed PDB saved as: {output_pdb}"
    print(msg)
    with open(log_file, 'a') as f_log:
        f_log.write(msg + '\n')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Reindex residue numbers in a PDB file')
    parser.add_argument('pdb_file', help='Input PDB file')
    parser.add_argument('chain_id', help='Chain ID to reindex')
    parser.add_argument('index_diff', type=int, help='Number to add to existing indices')
    parser.add_argument('--default_pdb', help='Reference PDB file to check residue numbering')
    
    args = parser.parse_args()
    
    reindex_pdb(args.pdb_file, args.chain_id, args.index_diff, args.default_pdb)
