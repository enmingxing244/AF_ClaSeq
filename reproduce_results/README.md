# reproduce_results

Source data and figures for every protein case in the AF-ClaSeq paper — **one folder per
figure**. Open any folder to find the figure image, the CSV data behind each panel, and a
short `README` explaining what's what.

## What's in each folder

- 🖼️ the **figure** (`.png`)
- 📊 the **CSV files** that produced it (one per panel)
- 🧬 the **pure-sequence MSAs** (`.a3m`) that were folded for each prediction panel
- 📄 a **`README.md`** mapping each file back to how it was generated

> Numerical data, figures, and the pure-sequence `.a3m` MSAs only — no predicted PDB
> structures. Each `.a3m` is the exact sequence set that was folded to produce its panel
> (paired one-to-one with that panel's CSV); re-fold it with ColabFold — or rerun the
> pipeline in `af_claseq/` — to regenerate the structures from scratch.

## Main figures

| Folder | Figure | Protein | States |
|--------|:------:|---------|--------|
| `figure_3_adenylate_kinase` | 3 | Adenylate kinase | closed 1AKE / open 4AKE |
| `figure_4_abl1` | 4 | ABL1 kinase | active 6XR6 / inactive 6XRG |
| `figure_5_glp1r` | 5 | GLP-1 receptor | inactive / active |
| `figure_6_kaib` | 6 | KaiB | ground 2QKE / fold-switch 5JYT |
| `figure_7_gb98` | 7 | GB98 (designed protein) | 2LHC / 2LHD |

## Supplementary figures

| Folder | Figure | Protein | States |
|--------|:------:|---------|--------|
| `supple_figure_1_atar` | S1 | AtaR toxin | 6AJM / 6GTO |
| `supple_figure_2_hras` | S2 | human H-Ras | active 5P21 / inactive 4Q21 |
| `supple_figure_3_rfah` | S3 | RfaH | α-hairpin 5OND / β-barrel 2LCL |
| `supple_figure_4_xcl1` | S4 | XCL1 | monomer 2HDM / dimer 2JP1 |
| `supple_figure_5_calmodulin` | S5 | Calmodulin | apo 1CLL / bound 1CDL |
| `supple_figure_6_ppac` | S6 | inorganic pyrophosphatase | closed 2HAW / open 1K23 |
| `supple_figure_7_pfmate` | S7 | PfMATE | inward 6FHZ / outward 6HFB |
| `supple_figure_8_murj` | S8 | MurJ | inward 6NC7 / outward 6NC9 |
| `supple_figure_9_t1214` | S9 | T1214 / PqqU | apo / bound |

*(The GLP1R Supp 15 and KaiB Supp 17 panels live inside `figure_5_glp1r/` and
`figure_6_kaib/`.)*

## How the states are found

Each case separates two conformations of one protein by purifying its MSA — sub-sampling
sequences, folding them with ColabFold, scoring each against the two reference structures,
and letting the sequences "vote" for the state they encode. The winning sequence sets are
re-predicted and compared against a random-sequence control. See each folder's README for
the specifics.
