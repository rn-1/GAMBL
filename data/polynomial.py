#!/usr/bin/env python3
"""Generate a numeric polynomial dataset with train.py-compatible CSV output.

Each example is a string-to-string mapping such as:

    Input:  f(x)=2x^2+3x+1; x=4
    Output: 45

The generator uses a strict numeric format:
- decimal coefficients only
- fixed variable name x
- no extra spaces or decorative text

By default it writes a single CSV with columns compatible with
`data/csv_dataset.py` and `train.py --dataset csv`:

    word_one, word_two, word_three, word_four

Each generated row encodes one polynomial example as:
    word_one  = polynomial definition text
    word_two  = variable assignment text
    word_three= task cue text ("predict value")
    word_four = target output text

Example:
    python polynomial.py \
        --outdir data/poly_eval \
        --num_examples 20000 \
        --max_degree 4 \
        --coeff_min -9 \
        --coeff_max 9 \
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ALPHABET = string.ascii_lowercase


@dataclass(frozen=True)
class Example:
    input_text: str
    output_text: str
    metadata: Dict[str, object]


def roman_numeral(n: int) -> str:
    if n == 0:
        return "N"
    if n < 0:
        return "-" + roman_numeral(-n)
    vals = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    out = []
    x = n
    for v, s in vals:
        while x >= v:
            out.append(s)
            x -= v
    return "".join(out)


ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
]
TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
]


def number_to_words(n: int) -> str:
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        ten, one = divmod(n, 10)
        return TENS[ten] if one == 0 else f"{TENS[ten]}-{ONES[one]}"
    if n < 1000:
        hundred, rem = divmod(n, 100)
        if rem == 0:
            return f"{ONES[hundred]} hundred"
        return f"{ONES[hundred]} hundred {number_to_words(rem)}"
    return str(n)


def encode_int(n: int, mode: str) -> str:
    if mode != "decimal":
        raise ValueError(f"Only decimal mode is supported, got: {mode}")
    return str(n)


def random_identifier(rng: random.Random, min_len: int = 1, max_len: int = 3) -> str:
    length = rng.randint(min_len, max_len)
    first = rng.choice(ALPHABET)
    rest = "".join(rng.choice(ALPHABET + string.digits) for _ in range(length - 1))
    return first + rest


def random_var_name(rng: random.Random) -> str:
    candidates = ["x", "t", "u", "z", "q", "y"]
    if rng.random() < 0.7:
        return rng.choice(candidates)
    return random_identifier(rng, 1, 2)


def random_func_name(rng: random.Random) -> str:
    candidates = ["f", "g", "h", "p", "r", "s"]
    if rng.random() < 0.8:
        return rng.choice(candidates)
    return random_identifier(rng, 1, 2)


def sample_coefficients(rng: random.Random, degree: int, coeff_min: int, coeff_max: int, allow_zero: bool = True) -> List[int]:
    coeffs = [rng.randint(coeff_min, coeff_max) for _ in range(degree + 1)]
    if not allow_zero:
        coeffs = [c if c != 0 else rng.choice([i for i in range(coeff_min, coeff_max + 1) if i != 0]) for c in coeffs]
    # Ensure leading coefficient is nonzero.
    if coeffs[-1] == 0:
        choices = [i for i in range(coeff_min, coeff_max + 1) if i != 0]
        coeffs[-1] = rng.choice(choices)
    return coeffs


def evaluate_polynomial(coeffs: Sequence[int], x: int) -> int:
    total = 0
    power = 1
    for c in coeffs:
        total += c * power
        power *= x
    return total


def format_term(coeff: int, power: int, coeff_mode: str, first: bool) -> str:
    if coeff == 0:
        return ""

    abs_coeff = abs(coeff)
    coeff_text = encode_int(abs_coeff, coeff_mode)

    if power == 0:
        core = coeff_text
    elif power == 1:
        if abs_coeff == 1 and coeff_mode == "decimal":
            core = "x"
        else:
            core = f"{coeff_text}x"
    else:
        if abs_coeff == 1 and coeff_mode == "decimal":
            core = f"x^{power}"
        else:
            core = f"{coeff_text}x^{power}"

    if first:
        return core if coeff >= 0 else f"-{core}"
    return f"+{core}" if coeff >= 0 else f"-{core}"


def format_polynomial(
    coeffs: Sequence[int],
    coeff_mode: str,
    rng: random.Random,
    var_name: str,
    func_name: str,
) -> str:
    terms = []
    for power, coeff in enumerate(coeffs):
        term = format_term(coeff, power, coeff_mode=coeff_mode, first=(len(terms) == 0))
        if term:
            terms.append(term)

    if not terms:
        terms = ["0"]

    # Numeric expression only, no function wrapper or extra separators.
    return "".join(terms)


def format_query(
    poly_text: str,
    x_name: str,
    x_val: int,
    rng: random.Random,
    x_mode: str,
) -> str:
    x_enc = encode_int(x_val, x_mode)
    return f"{poly_text};{x_name}={x_enc}"


def make_example(
    rng: random.Random,
    max_degree: int,
    coeff_min: int,
    coeff_max: int,
    coeff_encodings: Sequence[str],
    x_encodings: Sequence[str],
) -> Example:
    degree = rng.randint(0, max_degree)
    coeffs = sample_coefficients(rng, degree, coeff_min, coeff_max)
    x_val = rng.randint(coeff_min, coeff_max)
    # Avoid overwhelming outputs from huge x values when degree is large.
    # The range is still enough to produce varied answers.
    x_val = max(min(x_val, 9), -9)

    func_name = "f"
    var_name = "x"

    coeff_mode = "decimal"
    x_mode = "decimal"

    poly_text = format_polynomial(coeffs, coeff_mode, rng, var_name=var_name, func_name=func_name)
    input_text = format_query(poly_text, var_name, x_val, rng, x_mode=x_mode)
    y = evaluate_polynomial(coeffs, x_val)

    output_text = str(y)

    metadata = {
        "degree": degree,
        "coeffs": coeffs,
        "x": x_val,
        "func_name": func_name,
        "var_name": var_name,
        "coeff_mode": coeff_mode,
        "x_mode": x_mode,
        "poly_text": poly_text,
        "x_assign": str(x_val),
    }
    return Example(input_text=input_text, output_text=output_text, metadata=metadata)


def split_examples(examples: Sequence[Example], train_frac: float, val_frac: float) -> Tuple[List[Example], List[Example], List[Example]]:
    n = len(examples)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train = list(examples[:n_train])
    val = list(examples[n_train:n_train + n_val])
    test = list(examples[n_train + n_val:])
    return train, val, test


def write_jsonl(path: Path, examples: Sequence[Example]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            row = {
                "input": ex.input_text,
                "output": ex.output_text,
                "metadata": ex.metadata,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_txt(path: Path, examples: Sequence[Example]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(f"Input: {ex.input_text}\n")
            f.write(f"Output: {ex.output_text}\n")
            f.write("\n")


def write_trainpy_csv(path: Path, examples: Sequence[Example]) -> None:
    """Write one CSV consumable by `train.py --dataset csv`.

    The loader requires columns: word_one, word_two, word_three, word_four.
    Additional columns are safe and ignored by the loader.
    """
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "category", "word_one", "word_two", "word_three", "word_four"])
        for i, ex in enumerate(examples):
            poly_text = ex.metadata.get("poly_text", ex.input_text)
            x_assign = ex.metadata.get("x_assign", "0")
            writer.writerow([
                i,
                "polynomial-eval",
                str(poly_text),
                str(x_assign),
                "0",
                ex.output_text,
            ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a text polynomial evaluation dataset.")
    p.add_argument("--outdir", type=str, default="poly_eval_dataset", help="Output directory.")
    p.add_argument("--num_examples", type=int, default=20000, help="Total number of examples.")
    p.add_argument("--max_degree", type=int, default=4, help="Maximum polynomial degree.")
    p.add_argument("--coeff_min", type=int, default=-9, help="Minimum coefficient value.")
    p.add_argument("--coeff_max", type=int, default=9, help="Maximum coefficient value.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--train_frac", type=float, default=0.9, help="Train split fraction.")
    p.add_argument("--val_frac", type=float, default=0.05, help="Validation split fraction.")
    p.add_argument("--csv_name", type=str, default="polynomial_analogies.csv",
                   help="CSV filename written to outdir for train.py --dataset csv.")
    p.add_argument("--jsonl", action="store_true", help="Also write JSONL files.")
    p.add_argument("--txt", action="store_true", help="Also write plain text files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    coeff_encodings = ["decimal"]
    x_encodings = ["decimal"]

    examples = [
        make_example(
            rng=rng,
            max_degree=args.max_degree,
            coeff_min=args.coeff_min,
            coeff_max=args.coeff_max,
            coeff_encodings=coeff_encodings,
            x_encodings=x_encodings,
        )
        for _ in range(args.num_examples)
    ]

    rng.shuffle(examples)
    train, val, test = split_examples(examples, args.train_frac, args.val_frac)

    all_examples = examples

    # Default output: one train.py-compatible CSV (train.py does its own split).
    write_trainpy_csv(outdir / args.csv_name, all_examples)

    # Optional split outputs for inspection/other workflows.
    if args.jsonl:
        write_jsonl(outdir / "train.jsonl", train)
        write_jsonl(outdir / "val.jsonl", val)
        write_jsonl(outdir / "test.jsonl", test)

    if args.txt:
        write_txt(outdir / "train.txt", train)
        write_txt(outdir / "val.txt", val)
        write_txt(outdir / "test.txt", test)

    # Save a small config file for reproducibility.
    config = {
        "num_examples": args.num_examples,
        "max_degree": args.max_degree,
        "coeff_min": args.coeff_min,
        "coeff_max": args.coeff_max,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "coeff_encodings": coeff_encodings,
        "x_encodings": x_encodings,
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(
        f"Wrote {len(all_examples)} CSV rows to {outdir / args.csv_name}. "
        f"Split counts: train={len(train)}, val={len(val)}, test={len(test)}"
    )


if __name__ == "__main__":
    main()
