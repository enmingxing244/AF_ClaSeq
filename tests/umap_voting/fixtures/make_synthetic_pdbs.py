"""Generate synthetic PDB files for testing: 20 sampling + 2 reference PDBs."""

import numpy as np
from pathlib import Path

N_RESIDUES = 15
N_SAMPLING = 20


def _write_pdb(path: Path, coords: np.ndarray, chain: str = "A"):
    with open(path, "w") as f:
        for i, (x, y, z) in enumerate(coords, 1):
            f.write(
                f"ATOM  {i:5d}  CA  ALA {chain}{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        f.write("END\n")


def generate(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    # Reference A: line along X-axis
    ref_a = np.zeros((N_RESIDUES, 3), dtype=np.float32)
    ref_a[:, 0] = np.linspace(0, 14, N_RESIDUES)
    _write_pdb(out_dir / "ref_A.pdb", ref_a)

    # Reference B: line along Y-axis
    ref_b = np.zeros((N_RESIDUES, 3), dtype=np.float32)
    ref_b[:, 1] = np.linspace(0, 14, N_RESIDUES)
    _write_pdb(out_dir / "ref_B.pdb", ref_b)

    # Sampling: line backbone + Gaussian noise
    for i in range(N_SAMPLING):
        base = np.zeros((N_RESIDUES, 3), dtype=np.float32)
        base[:, 0] = np.linspace(0, 14, N_RESIDUES)
        noise = rng.randn(N_RESIDUES, 3).astype(np.float32) * 0.5
        _write_pdb(out_dir / f"sample_{i:02d}.pdb", base + noise)


if __name__ == "__main__":
    generate(Path(__file__).parent / "synthetic")
