"""README hygiene check: every headline number in README.md must match saved evidence.

This is a consistency check between prose and saved files. It is not a scientific validation. It
cannot tell whether an experiment was run correctly, only that the number printed in the README is
the number stored in the evidence file it points to (within the rounding the README prints).
It checks table cells, numbers quoted in the prose under the table, and the verdict words
(for example "26 points above" or "does not beat the floor"), which are derived from the saved
intervals so a stale verdict is caught when the evidence changes. The test-suite count is read from
`fast_suite.txt`, which `scripts/record_fast_suite.py` writes from a real pytest run.
It fails closed: a missing README row, a missing evidence file, a missing JSON path, a value that is
not a finite number or a mismatch gives exit code 1.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
VIEWS = "reports/results/privacy/v014/view_comparison/summary.json"
FLOOR = "reports/results/privacy/v0143/label_floor.json"
INTERVALS = "reports/results/privacy/v0143/view_intervals.json"
LIRA = "reports/results/privacy/v014/lira/{}/summary.json"
SUITE = "reports/results/privacy/v0143/fast_suite.txt"


@dataclass(frozen=True)
class Claim:
    id: str
    row: str            # start of the first cell of the README table row
    col: int            # 0-based cell index in that row
    file: str           # evidence file relative to the repo root
    path: str           # JSON path with / separators, integers index lists
    scale: float = 1.0  # evidence value times scale is compared with the README number
    minus_file: str = ""   # optional second evidence value, subtracted before scaling
    minus_path: str = ""


@dataclass(frozen=True)
class Ref:
    group: int          # regex group of the prose claim holding the printed number
    file: str
    path: str
    scale: float = 1.0
    minus_file: str = ""
    minus_path: str = ""


@dataclass(frozen=True)
class ProseClaim:
    id: str
    pattern: str        # regex that must match the README text, numbers captured in groups
    refs: tuple


@dataclass(frozen=True)
class Verdict:
    """The words in a table cell must agree with the sign of a paired interval in the evidence.
    If the lower end of the interval is above zero the cell must say 'above'. Otherwise it must say
    'does not beat' and must not say 'above'."""
    id: str
    row: str
    col: int
    file: str
    low_path: str


@dataclass(frozen=True)
class Phrase:
    id: str
    row: str
    col: int
    phrase: str


def _view_claims() -> list[Claim]:
    rows = [("Plaintext update", "plaintext_individual"), ("SecAgg+ aggregate only", "secagg_aggregate_maxw1000"),
            ("SecAgg+ with n-1 colluders", "secagg_collusion_n-1_maxw1000"), ("Opacus DP, sigma 0.2", "dp0.2_individual")]
    out = []
    for label, view in rows:
        out.append(Claim(f"inversion.{view}.mse", label, 1, VIEWS, f"views/{view}/victim_mse/0"))
        out.append(Claim(f"inversion.{view}.rate", label, 2, VIEWS, f"views/{view}/rate_mse_below_0_05/0", 100))
    for label, view in (("Plaintext update", "plaintext_individual"), ("SecAgg+ with n-1 colluders", "secagg_collusion_n-1_maxw1000")):
        out.append(Claim(f"inversion.{view}.points_above_floor", label, 3, VIEWS, f"views/{view}/rate_mse_below_0_05/0", 100,
                         FLOOR, "pooled/class_average_rate_mse_below_0_05"))
    out += [Claim("floor.label_only.mse", "Label-only floor", 1, FLOOR, "pooled/class_average_mse"),
            Claim("floor.label_only.rate", "Label-only floor", 2, FLOOR, "pooled/class_average_rate_mse_below_0_05", 100),
            Claim("floor.mean_image.mse", "Mean-image floor", 1, FLOOR, "pooled/mean_image_mse"),
            Claim("floor.mean_image.rate", "Mean-image floor", 2, FLOOR, "pooled/mean_image_rate_mse_below_0_05", 100)]
    return out


def _prose_claims() -> list[ProseClaim]:
    agg = "secagg_aggregate_maxw1000"
    sens = "pool_size_sensitivity/rows/{}/rate_mse_below_0_05"
    return [
        ProseClaim("prose.floor_counts_and_wilson",
                   r"\((\d+) of (\d+) images, Wilson 95% interval (\d+)% to (\d+)%\)",
                   (Ref(1, INTERVALS, "label_floor/recognisable/successes"), Ref(2, INTERVALS, "label_floor/recognisable/trials"),
                    Ref(3, INTERVALS, "label_floor/recognisable/wilson/0", 100), Ref(4, INTERVALS, "label_floor/recognisable/wilson/1", 100))),
        ProseClaim("prose.aggregate_against_floor",
                   r"on both error \(([\d.]+) against ([\d.]+)\) and recognisable share \((\d+)% against ([\d.]+)%\)",
                   (Ref(1, VIEWS, f"views/{agg}/victim_mse/0"), Ref(2, FLOOR, "pooled/class_average_mse"),
                    Ref(3, VIEWS, f"views/{agg}/rate_mse_below_0_05/0", 100), Ref(4, FLOOR, "pooled/class_average_rate_mse_below_0_05", 100))),
        ProseClaim("prose.dp_against_mean_image",
                   r"\(([\d.]+) is worse than the mean image at ([\d.]+)\)",
                   (Ref(1, VIEWS, "views/dp0.2_individual/victim_mse/0"), Ref(2, FLOOR, "pooled/mean_image_mse"))),
        ProseClaim("prose.maxw16_not_distinguishable",
                   r"the share is ([\d.]+)% and the paired difference to the floor is (-?[\d.]+) points with an interval of (-?[\d.]+) to \+?([\d.]+)",
                   (Ref(1, VIEWS, "views/secagg_aggregate_maxw16/rate_mse_below_0_05/0", 100),
                    Ref(2, INTERVALS, "views/secagg_aggregate_maxw16/paired_rate_minus_label_floor_bootstrap95/0", 100),
                    Ref(3, INTERVALS, "views/secagg_aggregate_maxw16/paired_rate_minus_label_floor_bootstrap95/1", 100),
                    Ref(4, INTERVALS, "views/secagg_aggregate_maxw16/paired_rate_minus_label_floor_bootstrap95/2", 100))),
        ProseClaim("prose.public_pool_size", r"(\d+)-image labelled public pool",
                   (Ref(1, FLOOR, "per_seed/0/public_pool_size"),)),
        ProseClaim("prose.pool_size_sensitivity",
                   r"pool of 20 images the floor is (\d+)% recognisable, with 50 images (\d+)%, with 100 images (\d+)% and with 300 images (\d+)%",
                   tuple(Ref(i + 1, FLOOR, "pool_size_sensitivity/rows/%d/rate_mse_below_0_05" % i, 100) for i in range(4))),
    ]


def _verdicts() -> list[Verdict]:
    key = "paired_rate_minus_label_floor_bootstrap95/1"
    return [Verdict("verdict.plaintext", "Plaintext update", 3, INTERVALS, f"views/plaintext_individual/{key}"),
            Verdict("verdict.aggregate", "SecAgg+ aggregate only", 3, INTERVALS, f"views/secagg_aggregate_maxw1000/{key}"),
            Verdict("verdict.collusion", "SecAgg+ with n-1 colluders", 3, INTERVALS, f"views/secagg_collusion_n-1_maxw1000/{key}")]


def _phrases() -> list[Phrase]:
    return [Phrase("verdict.dp_is_naive_only", "Opacus DP, sigma 0.2", 0, "naive attacker"),
            Phrase("verdict.dp_no_protection_claim", "Opacus DP, sigma 0.2", 3, "not a protection claim")]


def _lira_claims() -> list[Claim]:
    rows = [("Digits, overfit regime", "large_overfit"), ("Digits, moderate regime", "large_moderate"),
            ("MNIST, overfit regime", "large_mnist_overfit")]
    return [Claim(f"lira.{d}.mode{m}", label, m + 1, LIRA.format(d), f"modes/{m}/lira_auc/0")
            for label, d in rows for m in range(3)]


CLAIMS: list[Claim] = _view_claims() + _lira_claims()
PROSE: list[ProseClaim] = _prose_claims()
VERDICTS: list[Verdict] = _verdicts()
PHRASES: list[Phrase] = _phrases()
SUITE_README = re.compile(r"\((\d+) pass and (\d+) skip")
# The closing line of a pytest run, for example "195 passed, 4 skipped, 4 deselected, 2 warnings in 20.30s".
SUITE_EVIDENCE = re.compile(r"^(?:=+ )?((?:\d+ (?:passed|failed|skipped|deselected|xfailed|xpassed|errors?|warnings?)(?:, )?)+) in [\d.]+s")
SUITE_PART = re.compile(r"(\d+) (passed|failed|skipped|errors?)")


def table_cell(readme: str, row: str, col: int) -> str:
    for line in readme.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.lstrip().startswith("|") and cells and cells[0].startswith(row):
            if col >= len(cells):
                raise KeyError(f"row '{row}' has no column {col}")
            return cells[col]
    raise KeyError(f"no README table row starts with '{row}'")


NUMBER = re.compile(r"-?\d+(?:\.(\d+))?(?:[eE]([-+]?\d+))?")


def parse_number(cell: str) -> tuple[float, int]:
    """First number in a cell and how many decimals it prints. For an exponent such as 1e-3 the
    decimals are counted at the exponent, so 1e-3 prints three decimals and 1.5e-3 prints four."""
    m = NUMBER.search(cell)
    if not m:
        raise ValueError(f"no number in '{cell}'")
    return float(m.group(0)), len(m.group(1) or "") - int(m.group(2) or 0)


def json_value(data, path: str) -> float:
    for part in path.split("/"):
        data = data[int(part)] if isinstance(data, list) else data[part]
    if isinstance(data, bool) or not isinstance(data, (int, float)):
        raise TypeError(f"value at '{path}' is not a number")
    return float(data)


def evidence_value(root: Path, file: str, path: str, scale: float = 1.0, minus_file: str = "", minus_path: str = "") -> float:
    """Raises FileNotFoundError, KeyError, IndexError, ValueError or TypeError when the evidence is absent or
    malformed, and ValueError when the value is not finite (NaN and infinity never pass a comparison)."""
    def one(f: str, pth: str) -> float:
        fp = root / f
        if not fp.is_file():
            raise FileNotFoundError(f"evidence file {f}")
        v = json_value(json.loads(fp.read_text()), pth)
        if not math.isfinite(v):
            raise ValueError(f"value at '{pth}' in {f} is not finite ({v})")
        return v
    value = one(file, path)
    if minus_path:
        value -= one(minus_file or file, minus_path)
    return value * scale


def compare(printed: float, decimals: int, value: float) -> tuple[str, str]:
    if not math.isfinite(value) or not math.isfinite(printed):
        return "MISSING", f"not a finite number (README {printed}, evidence {value})"
    tol = 0.5 * 10 ** -decimals + 1e-9
    if not abs(printed - value) <= tol:
        return "MISMATCH", f"README {printed} but evidence {value:.6g} (tolerance {tol:g})"
    return "OK", f"README {printed} evidence {value:.6g}"


def check_claim(readme: str, root: Path, c: Claim) -> tuple[str, str]:
    try:
        printed, decimals = parse_number(table_cell(readme, c.row, c.col))
    except (KeyError, ValueError) as e:
        return "MISSING", f"README: {e}"
    try:
        value = evidence_value(root, c.file, c.path, c.scale, c.minus_file, c.minus_path)
    except FileNotFoundError as e:
        return "MISSING", str(e)
    except (KeyError, IndexError, ValueError, TypeError) as e:
        return "MISSING", f"evidence path {c.path} in {c.file}: {e!r}"
    return compare(printed, decimals, value)


def check_prose(readme: str, root: Path, c: ProseClaim) -> tuple[str, str]:
    m = re.search(c.pattern, readme)
    if not m:
        return "MISSING", f"README prose does not match /{c.pattern}/"
    details = []
    for ref in c.refs:
        printed, decimals = parse_number(m.group(ref.group))
        try:
            value = evidence_value(root, ref.file, ref.path, ref.scale, ref.minus_file, ref.minus_path)
        except FileNotFoundError as e:
            return "MISSING", str(e)
        except (KeyError, IndexError, ValueError, TypeError) as e:
            return "MISSING", f"evidence path {ref.path} in {ref.file}: {e!r}"
        status, detail = compare(printed, decimals, value)
        if status != "OK":
            return status, detail
        details.append(detail)
    return "OK", "; ".join(details)


def check_verdict(readme: str, root: Path, v: Verdict) -> tuple[str, str]:
    try:
        cell = table_cell(readme, v.row, v.col)
    except KeyError as e:
        return "MISSING", f"README: {e}"
    try:
        low = evidence_value(root, v.file, v.low_path)
    except FileNotFoundError as e:
        return "MISSING", str(e)
    except (KeyError, IndexError, ValueError, TypeError) as e:
        return "MISSING", f"evidence path {v.low_path} in {v.file}: {e!r}"
    says_above = re.search(r"\babove\b", cell) is not None
    says_not_beat = re.search(r"does not beat", cell) is not None
    if low > 0:
        ok, want = says_above and not says_not_beat, "above the floor"
    else:
        ok, want = says_not_beat and not says_above, "does not beat the floor"
    return ("OK", f"'{cell}' agrees with evidence ({want})") if ok else \
        ("MISMATCH", f"cell says '{cell}' but the paired interval (lower end {low:.4g}) means '{want}'")


def check_phrase(readme: str, p: Phrase) -> tuple[str, str]:
    try:
        cell = table_cell(readme, p.row, p.col)
    except KeyError as e:
        return "MISSING", f"README: {e}"
    return ("OK", f"'{p.phrase}' present") if p.phrase in cell else ("MISMATCH", f"cell '{cell}' lacks '{p.phrase}'")


def parse_suite_line(text: str) -> dict | None:
    """Counts from the last real pytest closing line in the text, or None. Other lines are ignored."""
    last = None
    for ln in text.splitlines():
        m = SUITE_EVIDENCE.match(ln.strip())
        if m:
            last = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
            for n, word in SUITE_PART.findall(m.group(1)):
                last["error" if word.startswith("error") else word] = int(n)
    return last


def check_suite_line(readme: str, root: Path) -> tuple[str, str]:
    m = SUITE_README.search(readme)
    if not m:
        return "MISSING", "README quick start has no '(N pass and M skip' line"
    f = root / SUITE
    if not f.is_file():
        return "MISSING", f"evidence file {SUITE} (write it with scripts/record_fast_suite.py)"
    got = parse_suite_line(f.read_text())
    if got is None:
        return "MISSING", f"no pytest closing line in {SUITE}"
    if got["failed"] or got["error"]:
        return "MISMATCH", f"saved pytest line has failures or errors: {got}"
    want = (int(m.group(1)), int(m.group(2)))
    have = (got["passed"], got["skipped"])
    return ("OK", f"{want}") if have == want else ("MISMATCH", f"README {want} but saved pytest line {have}")


def verify(readme_path: Path, root: Path, claims: list[Claim] | None = None, suite: bool = True,
           prose: list[ProseClaim] | None = None, verdicts: list[Verdict] | None = None,
           phrases: list[Phrase] | None = None) -> list[tuple[str, str, str]]:
    if not readme_path.is_file():
        return [("README", "MISSING", str(readme_path))]
    readme = readme_path.read_text()
    results = [(c.id, *check_claim(readme, root, c)) for c in (CLAIMS if claims is None else claims)]
    results += [(c.id, *check_prose(readme, root, c)) for c in (PROSE if prose is None else prose)]
    results += [(c.id, *check_verdict(readme, root, c)) for c in (VERDICTS if verdicts is None else verdicts)]
    results += [(c.id, *check_phrase(readme, c)) for c in (PHRASES if phrases is None else phrases)]
    if suite:
        results.append(("tests.fast_suite_count", *check_suite_line(readme, root)))
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--readme", type=Path, default=ROOT / "README.md")
    p.add_argument("--root", type=Path, default=ROOT)
    a = p.parse_args(argv)
    results = verify(a.readme, a.root)
    for cid, status, detail in results:
        print(f"{status:9s} {cid}: {detail}")
    bad = [r for r in results if r[1] != "OK"]
    print(f"{len(results) - len(bad)} of {len(results)} README claims verified")
    return 1 if bad or not results else 0


if __name__ == "__main__":
    sys.exit(main())
