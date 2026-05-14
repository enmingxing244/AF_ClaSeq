"""Generate 10 A3M files with known frequency distributions for voting tests."""

from pathlib import Path

QUERY = "ACDEFGHIKLMNPQRST"
COMMON_A = "AAAAAAAAAAAAAAAAA"
COMMON_B = "BBBBBBBBBBBBBBBBB"
COMMON_C = "CCCCCCCCCCCCCCCCC"


def generate(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(10):
        path = out_dir / f"group_{i:02d}.a3m"
        with open(path, "w") as f:
            f.write(f">query\n{QUERY}\n")
            f.write(f">common_a\n{COMMON_A}\n")
            if i < 7:
                f.write(f">common_b\n{COMMON_B}\n")
            if i < 4:
                f.write(f">common_c\n{COMMON_C}\n")
            # Add unique sequences
            f.write(f">unique_{i}\nUNIQUE{i:011d}\n")


if __name__ == "__main__":
    generate(Path(__file__).parent / "a3ms")
