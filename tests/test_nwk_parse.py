def test_ete3_newick_parsing_and_clade_splitting():
    from ete3 import Tree
    from af_claseq.divide_and_conquer.nwk_parse import get_monophyletic_clades

    tree = Tree("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);")
    assert sorted(tree.get_leaf_names()) == ["A", "B", "C", "D"]

    clades = get_monophyletic_clades(tree, min_size=2, max_size=2)

    assert sorted(sorted(clade) for clade in clades) == [
        ["A", "B"],
        ["C", "D"],
    ]
