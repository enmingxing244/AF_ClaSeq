#!/usr/bin/env python3
"""
Test script to debug cluster-based expansion header matching issues.

This script helps identify why good sequences are not being matched
to clusters in the MMseqs2 cluster file.
"""

import sys
import os
from pathlib import Path


from af_claseq.modules.cluster_based_expansion import ClusterBasedExpansion
from af_claseq.utils.sequence_processing import A3MParser


def test_header_cleaning():
    """Test the header cleaning function with various formats."""
    print("=" * 60)
    print("TESTING HEADER CLEANING")
    print("=" * 60)
    
    # Create instance without loading files to test cleaning method
    class TestExpansion(ClusterBasedExpansion):
        def __init__(self):
            self.logger = None  # Skip parent init
    
    expander = TestExpansion()
    
    test_headers = [
        ">A0A132BQ79\tOS=Bacteria OX=2",
        ">A0A132BQ79 OS=Bacteria OX=2", 
        ">A0A132BQ79",
        "A0A132BQ79\tOS=Bacteria OX=2",
        "A0A132BQ79 OS=Bacteria OX=2",
        "A0A132BQ79",
        ">sp|P12345|PROT_HUMAN Description here",
        "sp|P12345|PROT_HUMAN Description here",
        ">ERR1719502_1613376_127_0_283_5_449E-28_84_268_279",
        "ERR1719502_1613376_127_0_283_5_449E-28_84_268_279"
    ]
    
    print("Testing header cleaning:")
    for header in test_headers:
        clean = expander._clean_header(header)
        print(f"  '{header}' -> '{clean}'")
    print()


def analyze_cluster_file(cluster_file_path):
    """Analyze the cluster file format and content."""
    print("=" * 60)
    print("ANALYZING CLUSTER FILE")
    print("=" * 60)
    
    if not cluster_file_path.exists():
        print(f"❌ Cluster file not found: {cluster_file_path}")
        return None, None
    
    print(f"📁 Cluster file: {cluster_file_path}")
    
    representatives = set()
    members = set()
    sample_lines = []
    
    try:
        with open(cluster_file_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                    
                if i < 10:  # Store first 10 lines as samples
                    sample_lines.append(line)
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    rep = parts[0].strip()
                    member = parts[1].strip()
                    representatives.add(rep)
                    members.add(member)
                    
        print(f"📊 Found {len(representatives)} cluster representatives")
        print(f"📊 Found {len(members)} total cluster members")
        
        print("\n🔍 Sample cluster file lines:")
        for line in sample_lines:
            print(f"  {line}")
            
        print(f"\n🔍 Sample representatives: {list(representatives)[:5]}")
        print(f"🔍 Sample members: {list(members)[:5]}")
        
        return representatives, members
        
    except Exception as e:
        print(f"❌ Error reading cluster file: {e}")
        return None, None


def analyze_good_sequences(good_seq_file_path):
    """Analyze good sequences from structure analysis."""
    print("=" * 60)
    print("ANALYZING GOOD SEQUENCES")
    print("=" * 60)
    
    if not good_seq_file_path.exists():
        print(f"❌ Good sequences file not found: {good_seq_file_path}")
        # Try to find alternative files
        print("🔍 Looking for alternative good sequence files...")
        base_dir = good_seq_file_path.parent
        possible_files = [
            base_dir / "structure_analysis_results.json",
            base_dir / "filtered_sequences.a3m",
            base_dir / "good_sequences.json"
        ]
        
        for alt_file in possible_files:
            if alt_file.exists():
                print(f"📁 Found: {alt_file}")
        
        return None
    
    print(f"📁 Good sequences file: {good_seq_file_path}")
    
    try:
        parser = A3MParser(strict_validation=False)
        sequences = parser.parse_file(good_seq_file_path)
        
        print(f"📊 Found {len(sequences)} good sequences")
        
        # Show sample headers
        headers = list(sequences.keys())[:10]
        print(f"\n🔍 Sample good sequence headers:")
        for header in headers:
            print(f"  '{header}'")
            
        return sequences
        
    except Exception as e:
        print(f"❌ Error reading good sequences: {e}")
        return None


def test_header_matching(good_sequences, cluster_representatives, cluster_members):
    """Test if good sequence headers match cluster entries."""
    print("=" * 60)
    print("TESTING HEADER MATCHING")
    print("=" * 60)
    
    if not good_sequences or not cluster_members:
        print("❌ Cannot test matching - missing data")
        return
    
    # Create test expansion instance
    class TestExpansion(ClusterBasedExpansion):
        def __init__(self):
            pass
    
    expander = TestExpansion()
    
    matches_found = 0
    matches_after_cleaning = 0
    
    print("🔍 Testing header matching:")
    sample_headers = list(good_sequences.keys())[:20]  # Test first 20
    
    for header in sample_headers:
        clean_header = expander._clean_header(header)
        
        # Test direct matches
        in_members = header in cluster_members
        in_representatives = header in cluster_representatives
        
        # Test cleaned matches  
        clean_in_members = clean_header in cluster_members
        clean_in_representatives = clean_header in cluster_representatives
        
        if in_members or in_representatives:
            matches_found += 1
            print(f"  ✅ DIRECT MATCH: '{header}' found in clusters")
            
        if clean_in_members or clean_in_representatives:
            matches_after_cleaning += 1
            print(f"  ✅ CLEAN MATCH: '{header}' -> '{clean_header}' found in clusters")
        elif matches_found == 0 and matches_after_cleaning == 0:
            print(f"  ❌ NO MATCH: '{header}' -> '{clean_header}' not found")
            
    print(f"\n📊 Results:")
    print(f"  Direct matches: {matches_found}/{len(sample_headers)}")
    print(f"  Matches after cleaning: {matches_after_cleaning}/{len(sample_headers)}")
    
    # Show some cluster members vs good sequence examples
    print(f"\n🔍 Comparison examples:")
    print(f"  Cluster member example: '{list(cluster_members)[0] if cluster_members else 'None'}'")
    print(f"  Good sequence example: '{sample_headers[0] if sample_headers else 'None'}'")
    print(f"  Clean good seq example: '{expander._clean_header(sample_headers[0]) if sample_headers else 'None'}'")


def full_test_expansion(cluster_file_path, source_msa_path, good_sequences=None):
    """Test the full expansion process."""
    print("=" * 60)
    print("TESTING FULL EXPANSION")
    print("=" * 60)
    
    if not cluster_file_path.exists():
        print(f"❌ Cluster file missing: {cluster_file_path}")
        return
        
    if not source_msa_path.exists():
        print(f"❌ Source MSA missing: {source_msa_path}")
        return
    
    # Create test good sequences if not provided
    if good_sequences is None:
        print("🔧 Creating test good sequences...")
        try:
            parser = A3MParser(strict_validation=False)
            all_sequences = parser.parse_file(source_msa_path)
            # Take first 10 sequences as test "good" sequences
            good_sequences = dict(list(all_sequences.items())[:10])
            print(f"  Created {len(good_sequences)} test good sequences")
        except Exception as e:
            print(f"❌ Could not create test sequences: {e}")
            return
    
    print(f"🧪 Testing expansion with {len(good_sequences)} good sequences...")
    
    try:
        expander = ClusterBasedExpansion(cluster_file_path, source_msa_path)
        expanded = expander.expand_by_clusters(good_sequences)
        
        stats = expander.get_expansion_statistics()
        print(f"📊 Expansion stats: {stats}")
        
        if stats.get('clusters_found', 0) > 0:
            print("✅ Success! Clusters were found and expansion worked")
        else:
            print("❌ Problem: No clusters found - header matching failed")
            
        return expanded
        
    except Exception as e:
        print(f"❌ Expansion test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(
         cluster_file=Path("clustered_cluster.tsv"), 
         source_msa=Path("/fs/ess/PAA0203/xing244/AF_ClaSeq/results_updated/ABL1/mmseqs_default/ABL1_P00519.a3m"), 
         good_seq_file=Path("good_seq.a3m")):
    """
    Main test function.
    
    Args:
        base_dir: Base directory for hit_expand output (optional)
        cluster_file: Path to cluster TSV file (optional)
        source_msa: Path to source MSA file (optional) 
        good_seq_file: Path to good sequences file (optional)
    """
    print("🧪 CLUSTER EXPANSION DEBUG TEST")
    print("=" * 60)
    

    
   
    good_seq_file = Path(good_seq_file)
    

    print(f"📁 Cluster file: {cluster_file}")
    print(f"📁 Source MSA: {source_msa}")
    print(f"📁 Good sequences: {good_seq_file}")
    print()
    
    # Test 1: Header cleaning
    test_header_cleaning()
    
    # Test 2: Analyze cluster file
    representatives, members = analyze_cluster_file(cluster_file)
    
    # Test 3: Analyze good sequences
    good_sequences = None
    if good_seq_file:
        good_sequences = analyze_good_sequences(good_seq_file)
    
    # Test 4: Test matching
    if good_sequences and members:
        test_header_matching(good_sequences, representatives, members)
    
    # Test 5: Full expansion test
    full_test_expansion(cluster_file, source_msa, good_sequences)
    
    print("\n" + "=" * 60)
    print("🏁 DEBUG TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()