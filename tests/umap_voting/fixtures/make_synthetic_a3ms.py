"""Run once to materialize tests/umap_voting/fixtures/a3ms/*.a3m."""
import random
from pathlib import Path

DIR = Path(__file__).parent / "a3ms"
DIR.mkdir(exist_ok=True)
random.seed(0)

QUERY = "MTEYKLVVVGAGGVGKSALTIQLIQNHFV"
COMMON_A = "A" * len(QUERY)
COMMON_B = "B" * len(QUERY)
COMMON_C = "C" * len(QUERY)

for i in range(10):
    path = DIR / f"a3m_{i:02d}.a3m"
    lines = [f"#{len(QUERY)}\t1", ">query", QUERY]
    seqs = [COMMON_A]
    if i < 7:
        seqs.append(COMMON_B)
    if i < 4:
        seqs.append(COMMON_C)
    while len(seqs) < 5:
        seqs.append(
            "".join(random.choice("DEFGHIK") for _ in range(len(QUERY)))
        )
    for s_i, s in enumerate(seqs):
        lines.append(f">seq_{i:02d}_{s_i}")
        lines.append(s)
    path.write_text("\n".join(lines) + "\n")

print(f"wrote 10 A3Ms to {DIR}")
