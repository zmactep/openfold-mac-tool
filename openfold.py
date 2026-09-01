"""
openfold.py — wrapper around OpenFold 3 that uses a *local* MMseqs2 search
against a user-provided custom database instead of the remote ColabFold MSA
server. Both this wrapper and OpenFold 3 itself are invoked through `uv`.

Workflow:
  1. Parse the input YAML.
  2. For each protein chain, run an MMseqs2 3-iteration profile search
     against the configured custom database and emit an a3m MSA.
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

# MMseqs2 search settings (ColabFold-style: 3 profile iterations).
MMSEQS_NUM_ITERATIONS = 3


# --- Helpers ------------------------------------------------------------------

def run(cmd: list, **kwargs) -> None:
    """Echo and run a subprocess; raise on non-zero exit."""
    print(f"[run] {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def write_query_fasta(seq: str, path: Path, name: str) -> None:
    path.write_text(f">{name}\n{seq}\n")


def run_mmseqs_for_chain(
    seq: str,
    db_path: Path,
    chain_dir: Path,
    tmp_root: Path,
    tag: str,
) -> Path:
    """
    Run a 3-iteration MMseqs2 profile search for `seq` against `db_path` and
    write `<CUSTOM_MSA_NAME>.a3m` into `chain_dir`. Returns the a3m path.
    """
    chain_dir.mkdir(parents=True, exist_ok=True)
    out_a3m = chain_dir / f"{CUSTOM_MSA_NAME}.a3m"

    work = tmp_root / f"mmseqs_{tag}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    query_fasta = work / "query.fasta"
    write_query_fasta(seq, query_fasta, name=tag)

    query_db = work / "queryDB"
    result_db = work / "resultDB"
    search_tmp = work / "search_tmp"
    search_tmp.mkdir()

    run(["mmseqs", "createdb", str(query_fasta), str(query_db)])

    run([
        "mmseqs", "search",
        str(query_db),
        str(db_path),
        str(result_db),
        str(search_tmp),
        "--num-iterations", str(MMSEQS_NUM_ITERATIONS),
        "-a",
    ])

    run([
        "mmseqs", "result2msa",
        str(query_db),
        str(db_path),
        str(result_db),
        str(out_a3m),
        "--msa-format-mode", "5",   # a3m, query first
    ])

    if not out_a3m.exists() or out_a3m.stat().st_size == 0:
        raise RuntimeError(f"MMseqs2 produced no MSA for {tag}: {out_a3m}")

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


def write_runner_yaml(path: Path) -> None:
    cfg = {
        "dataset_config_kwargs": {
            "msa": {
                "max_seq_counts": {CUSTOM_MSA_NAME: MAX_SEQS},
                "msas_to_pair": [],          # no online pairing for a custom DB
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
                )

        chain_entries.append(build_chain_entry(chain_cfg, alignments_root))

    # ---- 2) Write the query JSON
    query_json = {"queries": {name: {"chains": chain_entries}}}
    query_json_path = out / f"{name}_query.json"
    query_json_path.write_text(json.dumps(query_json, indent=2))
    print(f"[query] wrote {query_json_path}")

    # ---- 3) Write the runner YAML
    runner_yaml_path = out / "runner.yml"
    write_runner_yaml(runner_yaml_path)
    print(f"[runner] wrote {runner_yaml_path}")

    # ---- 4) Invoke OpenFold 3 through uv
    uv_prefix = build_uv_run_prefix(
        openfold_project=args.openfold_project,
        openfold_with=args.openfold_with,
    )
    cmd = uv_prefix + [
        "run_openfold", "predict",
        f"--query_json={query_json_path}",
        "--use_msa_server=False",
        f"--output_dir={predictions_dir}",
        f"--runner_yaml={runner_yaml_path}",
    ]
    if args.inference_ckpt_path is not None:
        cmd.append(f"--inference_ckpt_path={args.inference_ckpt_path}")

    run(cmd)
    print(f"[done] predictions written under {predictions_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
