"""
openfold.py — wrapper around OpenFold 3 that uses a *local* MMseqs2 search
against a user-provided custom database instead of the remote ColabFold MSA
server. Both this wrapper and OpenFold 3 itself are invoked through `uv`.

Workflow:
  1. Parse the input YAML.
  2. For each protein chain, run a configurable MMseqs2 search against the
     custom database, unpack its result database, and validate a plain a3m MSA.
  3. Lay the alignments out in OpenFold 3's per-chain directory structure.
  4. Build the OpenFold 3 query JSON.
  5. Write a runner.yml that registers the custom MSA file with MSASettings.
  6. Invoke `uv run run_openfold predict --use_msa_server=False ...`.

Usage (after `uv tool install /path/to/this/dir`):
    openfold --input_yaml in.yaml --output_dir out/
    uvx openfold --input_yaml in.yaml --output_dir out/      # ephemeral

Pointing at the OpenFold 3 environment:
    # If openfold3 is installed in a uv project at /opt/openfold3:
    openfold --input_yaml in.yaml --output_dir out/ \\
             --openfold_project /opt/openfold3

    # Or do an ephemeral install (heavy, includes torch + CUDA wheels):
    openfold --input_yaml in.yaml --output_dir out/ \\
             --openfold_with openfold3

    # If neither flag is given, the script just runs `uv run run_openfold ...`
    # and relies on whatever uv environment is active in $PWD.

Required system tools:
    - uv               (https://docs.astral.sh/uv/)
    - mmseqs           (executable on $PATH)
    - openfold3        (installed in the uv environment / project / tool)

Expected YAML structure:
    name: my_system
    database: /path/to/mmseqs2_custom_db        # prefix; <db>.dbtype must exist
    mmseqs:
      num_iterations: 1                         # macOS-safe default
      threads: 8                                # optional
    pairing: false                              # custom MSA is unpaired by default
    chains:
      - ids: ["A"]
        type: protein
        sequence: MSEL...
      - ids: ["B"]
        type: protein
        sequence: MESL...
        msa: false                              # optional: skip MSA generation
      - ids: ["C"]
        type: ligand
        sequence: CC(=O)O                       # SMILES, or a CCD code like "ATP"

Tip: for repeated runs against the same DB, pre-build the index once with
     `mmseqs createindex /path/to/mmseqs2_custom_db tmp` to avoid re-indexing.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# --- Tunable constants --------------------------------------------------------

# Filename used for MSAs produced by the custom database. Must match the key
# in runner.yml -> dataset_config_kwargs.msa.aln_order / max_seq_counts.
CUSTOM_MSA_NAME = "custom_db_hits"
MAX_SEQS = 100_000

# MMseqs2 release 18 can crash in multi-iteration profile searches on macOS
# ARM. Keep the macOS-safe default explicit and allow YAML inputs to override
# it for environments where profile iterations are known to be stable.
DEFAULT_MMSEQS_NUM_ITERATIONS = 1


# --- Helpers ------------------------------------------------------------------


def run(cmd: list, **kwargs) -> None:
    """Echo and run a subprocess; raise on non-zero exit."""
    print(f"[run] {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def write_query_fasta(seq: str, path: Path, name: str) -> None:
    path.write_text(f">{name}\n{seq}\n")


def read_mmseqs_settings(spec: dict) -> tuple[int, int | None]:
    """Read and validate optional MMseqs2 execution settings from input YAML."""
    cfg = spec.get("mmseqs", {})
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("mmseqs must be a mapping when provided")

    num_iterations = cfg.get("num_iterations", DEFAULT_MMSEQS_NUM_ITERATIONS)
    threads = cfg.get("threads")

    if isinstance(num_iterations, bool) or not isinstance(num_iterations, int):
        raise ValueError("mmseqs.num_iterations must be a positive integer")
    if num_iterations < 1:
        raise ValueError("mmseqs.num_iterations must be a positive integer")

    if threads is not None and (
        isinstance(threads, bool) or not isinstance(threads, int) or threads < 1
    ):
        raise ValueError("mmseqs.threads must be a positive integer")

    return num_iterations, threads


def append_threads(cmd: list[str], threads: int | None) -> list[str]:
    """Append an MMseqs2 thread limit when one was requested."""
    if threads is not None:
        cmd += ["--threads", str(threads)]
    return cmd


def validate_a3m(path: Path, query_sequence: str) -> None:
    """Validate that an unpacked A3M is plain text and aligned to the query."""
    data = path.read_bytes()
    if b"\0" in data:
        raise RuntimeError(f"A3M contains an MMseqs2 database terminator: {path}")

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"A3M is not valid UTF-8 text: {path}") from exc

    records: list[str] = []
    current: list[str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current is not None:
                records.append("".join(current))
            current = []
        elif current is None:
            raise RuntimeError(f"A3M sequence appears before its header: {path}")
        else:
            current.append(line)
    if current is not None:
        records.append("".join(current))

    if not records or any(not sequence for sequence in records):
        raise RuntimeError(f"A3M contains no complete sequence records: {path}")

    def aligned_sequence(sequence: str) -> str:
        return "".join(char for char in sequence if not char.islower() and char != ".")

    aligned = [aligned_sequence(sequence) for sequence in records]
    if aligned[0] != query_sequence:
        raise RuntimeError(f"A3M query sequence does not match the requested sequence: {path}")
    if any(len(sequence) != len(query_sequence) for sequence in aligned):
        raise RuntimeError(f"A3M rows do not share the query alignment width: {path}")


def run_mmseqs_for_chain(
    seq: str,
    db_path: Path,
    chain_dir: Path,
    tmp_root: Path,
    tag: str,
    num_iterations: int = DEFAULT_MMSEQS_NUM_ITERATIONS,
    threads: int | None = None,
) -> Path:
    """
    Run an MMseqs2 search for `seq` against `db_path`, unpack its result DB,
    and write a plain `<CUSTOM_MSA_NAME>.a3m` into `chain_dir`.
    """
    chain_dir.mkdir(parents=True, exist_ok=True)
    out_a3m = chain_dir / f"{CUSTOM_MSA_NAME}.a3m"
    out_a3m.unlink(missing_ok=True)

    work = tmp_root / f"mmseqs_{tag}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    query_fasta = work / "query.fasta"
    write_query_fasta(seq, query_fasta, name=tag)

    query_db = work / "queryDB"
    result_db = work / "resultDB"
    msa_db = work / "msaDB"
    unpack_dir = work / "unpacked_msa"
    search_tmp = work / "search_tmp"
    search_tmp.mkdir()

    run(["mmseqs", "createdb", str(query_fasta), str(query_db)])

    run(
        append_threads(
            [
                "mmseqs",
                "search",
                str(query_db),
                str(db_path),
                str(result_db),
                str(search_tmp),
                "--num-iterations",
                str(num_iterations),
                "-a",
            ],
            threads,
        )
    )

    run(
        append_threads(
            [
                "mmseqs",
                "result2msa",
                str(query_db),
                str(db_path),
                str(result_db),
                str(msa_db),
                "--msa-format-mode",
                "5",  # a3m, query first
            ],
            threads,
        )
    )

    unpack_dir.mkdir()
    run(
        append_threads(
            [
                "mmseqs",
                "unpackdb",
                str(msa_db),
                str(unpack_dir),
                "--unpack-name-mode",
                "0",
                "--unpack-suffix",
                ".a3m",
            ],
            threads,
        )
    )

    unpacked = sorted(unpack_dir.glob("*.a3m"))
    if len(unpacked) != 1:
        raise RuntimeError(
            f"Expected one unpacked A3M for {tag}, found {len(unpacked)} in {unpack_dir}"
        )

    validate_a3m(unpacked[0], query_sequence=seq)
    unpacked[0].replace(out_a3m)

    return out_a3m


def detect_ligand_type(value: str) -> str:
    """Heuristic: classify a ligand 'sequence' field as CCD code or SMILES."""
    v = value.strip()
    if 1 <= len(v) <= 5 and v.isalnum() and v.isupper():
        return "ccd"
    return "smiles"


def normalize_ids(raw) -> list:
    if isinstance(raw, str):
        return [raw]
    return list(raw)


def should_skip_msa(chain_cfg: dict) -> bool:
    """Check if MSA should be skipped for this chain (msa: false)."""
    return chain_cfg.get("msa", True) is False


def build_chain_entry(chain_cfg: dict, alignments_root: Path) -> dict:
    """Convert one YAML chain block to an OpenFold 3 chain JSON entry."""
    ids = normalize_ids(chain_cfg["ids"])
    chain_ids = ",".join(ids)
    ctype = chain_cfg["type"].lower()
    seq = chain_cfg["sequence"]

    if ctype == "protein":
        chain_dir = alignments_root / f"chain_{'_'.join(ids)}"
        entry = {
            "molecule_type": "protein",
            "chain_ids": chain_ids,
            "sequence": seq,
        }
        # Only include main_msa_file_paths if MSA was generated
        if not should_skip_msa(chain_cfg):
            entry["main_msa_file_paths"] = str(chain_dir.resolve())
        return entry

    if ctype in ("rna", "dna"):
        return {
            "molecule_type": ctype,
            "chain_ids": chain_ids,
            "sequence": seq,
        }

    if ctype == "ligand":
        entry = {"molecule_type": "ligand", "chain_ids": chain_ids}
        if detect_ligand_type(seq) == "ccd":
            entry["ccd_codes"] = [seq.strip()]
        else:
            entry["smiles"] = seq.strip()
        return entry

    raise ValueError(f"Unsupported chain type: {ctype!r}")


def write_runner_yaml(path: Path, pairing: bool = False) -> None:
    cfg = {
        "dataset_config_kwargs": {
            "msa": {
                "max_seq_counts": {CUSTOM_MSA_NAME: MAX_SEQS},
                "msas_to_pair": [CUSTOM_MSA_NAME] if pairing else [],
                "aln_order": [CUSTOM_MSA_NAME],
            }
        },
        # MLX attention for Apple Silicon (openfold-3-mlx fork).
        "model_update": {
            "presets": ["predict", "pae_enabled"],
            "custom": {
                "settings": {
                    "memory": {
                        "eval": {
                            "use_deepspeed_evo_attention": False,
                            "use_lma": False,
                            "use_mlx_attention": True,
                            "use_mlx_triangle_kernels": True,
                            "use_mlx_activation_functions": True,
                        }
                    }
                }
            },
        },
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def validate_prediction_outputs(predictions_dir: Path, expected_query_names: list[str]) -> None:
    """Reject semantically failed OpenFold runs even when their exit code is zero."""
    summary_path = predictions_dir / "summary.txt"
    if not summary_path.is_file():
        raise RuntimeError(f"OpenFold produced no prediction summary: {summary_path}")

    summary = summary_path.read_text()

    def read_count(label: str) -> int:
        match = re.search(rf"{re.escape(label)}:\s*(\d+)", summary)
        if match is None:
            raise RuntimeError(f"OpenFold summary is missing {label!r}: {summary_path}")
        return int(match.group(1))

    total = read_count("Total Queries Processed")
    successful = read_count("Successful Queries")
    failed = read_count("Failed Queries")
    expected = len(expected_query_names)
    if total != expected or successful != expected or failed != 0:
        raise RuntimeError("OpenFold reported unsuccessful predictions:\n" + summary.strip())

    missing_models = [
        query_name
        for query_name in expected_query_names
        if not any((predictions_dir / query_name).rglob("*_model.cif"))
    ]
    if missing_models:
        raise RuntimeError(
            "OpenFold reported success but produced no mmCIF model for: "
            + ", ".join(missing_models)
        )


def build_uv_run_prefix(
    openfold_project: Path | None,
    openfold_with: list,
) -> list:
    """Return the `uv run [...]` prefix used to invoke run_openfold."""
    cmd: list = ["uv", "run"]
    if openfold_project is not None:
        cmd += ["--project", str(openfold_project)]
    for s in openfold_with:
        cmd += ["--with", s]
    return cmd


# --- Main ---------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input_yaml", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument(
        "--inference_ckpt_path",
        default=None,
        type=Path,
        help="Override the OpenFold 3 checkpoint path. "
        "If omitted, the default cached weights are used.",
    )
    ap.add_argument(
        "--openfold_project",
        default=None,
        type=Path,
        help="Path to a uv project (directory containing pyproject.toml) "
        "where openfold3 is installed. Forwarded as `uv run --project`.",
    )
    ap.add_argument(
        "--openfold_with",
        default=[],
        action="append",
        metavar="SPEC",
        help="Package spec(s) to install ephemerally for the openfold run. "
        "May be passed multiple times. Forwarded as `uv run --with`.",
    )
    args = ap.parse_args()

    # ---- Load input
    spec = yaml.safe_load(args.input_yaml.read_text())
    name = spec["name"]
    db_path = Path(spec["database"]).expanduser().resolve()
    chains_cfg = spec["chains"]
    num_iterations, mmseqs_threads = read_mmseqs_settings(spec)
    pairing = spec.get("pairing", False)
    if not isinstance(pairing, bool):
        raise ValueError("pairing must be true or false")

    out = args.output_dir.expanduser().resolve()
    alignments_root = out / "alignments"
    tmp_root = out / "tmp"
    predictions_dir = out / "predictions"
    for d in (out, alignments_root, tmp_root, predictions_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- Sanity-check the MMseqs2 DB
    if not Path(f"{db_path}.dbtype").exists():
        sys.exit(
            f"[error] MMseqs2 DB not found: expected {db_path}.dbtype to exist. "
            f"Did you point `database:` at a valid MMseqs2 DB prefix?"
        )

    # ---- 1) Run MMseqs2 for each protein chain, build chain entries
    chain_entries = []
    for chain_cfg in chains_cfg:
        ctype = chain_cfg["type"].lower()
        ids = normalize_ids(chain_cfg["ids"])

        if ctype == "protein":
            if should_skip_msa(chain_cfg):
                print(f"[mmseqs] skipping MSA for chain(s) {ids} (msa: false)")
            else:
                chain_dir = alignments_root / f"chain_{'_'.join(ids)}"
                tag = "_".join(ids)
                print(f"[mmseqs] building MSA for chain(s) {ids}")
                run_mmseqs_for_chain(
                    seq=chain_cfg["sequence"],
                    db_path=db_path,
                    chain_dir=chain_dir,
                    tmp_root=tmp_root,
                    tag=tag,
                    num_iterations=num_iterations,
                    threads=mmseqs_threads,
                )

        chain_entries.append(build_chain_entry(chain_cfg, alignments_root))

    # ---- 2) Write the query JSON
    query_json = {
        "queries": {
            name: {
                "chains": chain_entries,
                "use_paired_msas": pairing,
                "use_main_msas": True,
            }
        }
    }
    query_json_path = out / f"{name}_query.json"
    query_json_path.write_text(json.dumps(query_json, indent=2))
    print(f"[query] wrote {query_json_path}")

    # ---- 3) Write the runner YAML
    runner_yaml_path = out / "runner.yml"
    write_runner_yaml(runner_yaml_path, pairing=pairing)
    print(f"[runner] wrote {runner_yaml_path}")

    # ---- 4) Invoke OpenFold 3 through uv
    uv_prefix = build_uv_run_prefix(
        openfold_project=args.openfold_project,
        openfold_with=args.openfold_with,
    )
    cmd = [
        *uv_prefix,
        "run_openfold",
        "predict",
        f"--query_json={query_json_path}",
        "--use_msa_server=False",
        f"--output_dir={predictions_dir}",
        f"--runner_yaml={runner_yaml_path}",
    ]
    if args.inference_ckpt_path is not None:
        cmd.append(f"--inference_ckpt_path={args.inference_ckpt_path}")

    run(cmd)
    validate_prediction_outputs(predictions_dir, expected_query_names=[name])
    print(f"[done] predictions written under {predictions_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
