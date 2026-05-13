"""Run once to materialize tests/umap_voting/fixtures/synthetic/*.pdb."""
import numpy as np
from pathlib import Path

DIR = Path(__file__).parent / "synthetic"
DIR.mkdir(exist_ok=True)

N_RESIDUES = 30


def write_pdb(path: Path, coords: np.ndarray):
    with open(path, "w") as f:
        for i, (x, y, z) in enumerate(coords, 1):
            f.write(
                f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00           C\n"
            )
        f.write("END\n")


rng = np.random.default_rng(0)
for n in range(20):
    base = np.linspace(0, 30, N_RESIDUES).reshape(-1, 1) * np.array(
        [[1.0, 0, 0]]
    )
    noise = rng.normal(scale=0.5, size=(N_RESIDUES, 3))
    write_pdb(DIR / f"struct_{n + 1:04d}.pdb", base + noise)

ref_a = np.linspace(0, 30, N_RESIDUES).reshape(-1, 1) * np.array(
    [[1.0, 0, 0]]
)
ref_b = np.linspace(0, 30, N_RESIDUES).reshape(-1, 1) * np.array(
    [[0.0, 1.0, 0]]
)
write_pdb(DIR / "ref_A.pdb", ref_a)
write_pdb(DIR / "ref_B.pdb", ref_b)
print(f"wrote {20 + 2} PDBs to {DIR}")
