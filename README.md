# OpenFold Mac Tool

Local MMseqs2 + MLX-powered OpenFold 3 wrapper for Apple Silicon. Run protein structure prediction entirely offline — no ColabFold MSA server, no CUDA.

## Overview

`openfold-mac-tool` is a lightweight wrapper that combines:

1. **Local MMseqs2** — searches a custom sequence database to build multiple sequence alignments (MSAs) without any remote server.
2. **[openfold-3-mlx](https://github.com/latent-spacecraft/openfold-3-mlx)** — Apple Silicon fork of OpenFold 3 using MLX attention kernels instead of CUDA/DeepSpeed.

The wrapper takes a declarative YAML input, runs the MSA search locally, and hands the results to OpenFold 3 for prediction — all through `uv`.

```
Input YAML  →  MMseqs2 search  →  unpacked MSA (.a3m)  →  OpenFold 3 (MLX)  →  predictions/
```

## Requirements

| Component | Details |
|-----------|---------|
| **OS** | macOS 14 (Sonoma) or later |
| **Hardware** | Apple Silicon — M1, M2, M3, M4 |
| **RAM** | 16 GB minimum, 32+ GB recommended for large proteins |
| **Homebrew** | Package manager (used for `mmseqs2` and `uv`) |
| **uv** | Fast Python package manager (used for everything Python) |

## Installation

### 1. Install Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install MMseqs2

```bash
brew install brewsci/bio/mmseqs2
```

Verify:

```bash
mmseqs version
```

### 3. Install uv

```bash
brew install uv
```

Verify:

```bash
uv --version
```

### 4. Install OpenFold 3 (MLX fork)

```bash
uv tool install https://github.com/latent-spacecraft/openfold-3-mlx.git
```

This provides the `run_openfold` command on your `PATH`.

### 5. Install this wrapper

```bash
git clone https://github.com/zmactep/openfold-mac-tool.git
cd openfold-mac-tool
uv tool install .
```

Verify:

```bash
openfold --help
```

## Preparing a custom MMseqs2 database

The wrapper searches a local MMseqs2 database. Prepare one from a FASTA file:

```bash
# Build the database
mmseqs createdb /path/to/sequences.fasta /path/to/my_db

# Pre-index for fast search (optional, recommended for repeated runs)
mmseqs createindex /path/to/my_db tmp_index
```

The database path used in your YAML is the **prefix** — e.g. `/path/to/my_db`. The tool expects `/path/to/my_db.dbtype` to exist.

> **Tip:** The ColabFold databases (`uniref30`, `colabfold_envdb`) work well. Download them from the [ColabFold wiki](https://github.com/sokrypton/ColabFold/wiki) and index with `mmseqs createindex`.

## Input YAML format

```yaml
name: my_complex                      # output name (used for filenames)
database: /path/to/mmseqs_db          # MMseqs2 DB prefix
mmseqs:
  num_iterations: 1                   # default: 1 (safe on macOS ARM)
  threads: 8                          # optional; MMseqs2 default if omitted
pairing: false                        # true: pre-pair best Swiss-Prot hits by OX taxon
pairing_taxonomy: /path/to/extra.tsv  # optional accession<TAB>taxon_id sidecar
chains:
  - ids: ["A"]                        # chain ID(s)
    type: protein                     # protein | rna | dna | ligand
    sequence: MKLFVTNS...             # amino acid sequence
  - ids: ["B", "C"]
    type: ligand
    sequence: ATP                     # CCD code (uppercase, ≤5 chars) or SMILES
  - ids: ["D"]
    type: protein
    sequence: PEPTIDESEQUENCE
    msa: false                        # query-only MSA, useful for a bound peptide
```

- **`database`**: path prefix to the MMseqs2 database.
- **`mmseqs.num_iterations`**: number of profile-search iterations. The default
  is `1` because MMseqs2 releases 17 and 18 can crash in multi-iteration profile
  searches on macOS ARM. Use `3` only after verifying it is stable with your DB.
- **`mmseqs.threads`**: optional thread limit forwarded to MMseqs2 search and MSA
  export commands.
- **`pairing`**: when `true`, the wrapper reads Swiss-Prot `OX=` taxonomy IDs
  from the MMseqs header database, selects the closest hit per taxon and chain,
  and writes row-aligned precomputed paired MSAs for OpenFold. Non-Swiss-Prot
  records without an `OX=` identifier remain available in the main unpaired MSA
  but are not guessed into pairs.
- **`pairing_taxonomy`**: optional TSV sidecar for curated database records whose
  FASTA headers lack `OX=`. Its first two columns must be the exact A3M accession
  and numeric NCBI taxonomy ID. Blank lines, `#` comments and a header beginning
  with `accession` are accepted.
- **`chains`**: one block per chain. Multiple IDs on one entry mean identical sequences (useful for homomers).
- **`type: protein`**: triggers the MMseqs2 MSA search (required for at least one chain). Non-protein chains are passed directly to OpenFold 3.
- **`msa: false`**: skips MMseqs for that protein but still writes the query-only
  A3M required by OpenFold. This is appropriate for a short bound peptide.
- **`type: ligand`**: short uppercase strings are treated as CCD codes, longer strings as SMILES.

## Usage

```bash
# Basic run
openfold --input_yaml input.yaml --output_dir results/

# Override the checkpoint
openfold --input_yaml input.yaml --output_dir results/ \
    --inference_ckpt_path /path/to/model.pt
```

The wrapper requires `run_openfold` to be available. By default it runs `uv run run_openfold predict ...` from the current directory. If OpenFold 3 is installed elsewhere, use one of:

```bash
# Point to a uv project where openfold3 is installed
openfold ... --openfold_project /opt/openfold3

# Ephemeral install (e.g. from PyPI or git)
openfold ... --openfold_with openfold3
openfold ... --openfold_with git+https://github.com/latent-spacecraft/openfold-3-mlx.git
```

### Output structure

```
results/
├── alignments/                  # per-chain a3m files from MMseqs2
│   └── chain_A/
│       └── custom_db_hits.a3m        # unpacked and validated plain-text A3M
├── tmp/                         # MMseqs2 temporary files
├── my_complex_query.json        # OpenFold 3 query JSON
├── runner.yml                   # runner config (MLX settings)
└── predictions/                 # OpenFold 3 output (mmCIF, confidence, PAE)
```

## Full example

```bash
# 1. Prepare a database
mmseqs createdb uniref30.fasta uniref30
mmseqs createindex uniref30 tmp_idx

# 2. Create an input YAML
cat > my_protein.yaml << 'EOF'
name: af2_sample
database: /Users/me/dbs/uniref30
mmseqs:
  num_iterations: 1
pairing: false
chains:
  - ids: ["A"]
    type: protein
    sequence: MKLFVTNSGYLLIAQALPDSVKEGQIELDQLLKQLGITVPEQVSSLTS
EOF

# 3. Run
openfold --input_yaml my_protein.yaml --output_dir results/

# 4. Check results
ls results/predictions/
```

The command exits with an error if OpenFold writes a failed `summary.txt`,
reports fewer successful queries than requested, or produces no mmCIF model.

## MSA export and pairing

`mmseqs result2msa` writes an MMseqs2 result database, not a standalone file,
even when its output prefix ends in `.a3m`. The wrapper keeps that database in
`tmp/`, runs `mmseqs unpackdb`, validates the resulting plain-text A3M, and only
then exposes it to OpenFold.

For heteromers, `pairing: false` still uses every custom MSA as a main unpaired
alignment. With `pairing: true`, the wrapper keeps that main MSA and additionally
creates one precomputed paired A3M per MSA-enabled protein chain. Pairing is
restricted to taxa present in every such chain and uses the best query match in
each taxon. The MMseqs database must therefore retain its `<prefix>_h` header DB
with Swiss-Prot-style headers containing `OX=` taxonomy identifiers.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `mmseqs: command not found` | Run `brew install brewsci/bio/mmseqs2` and restart your shell. |
| `MMseqs2 DB not found` | Verify the path in your YAML `database:` field — the `.dbtype` file must exist at that prefix. |
| `RuntimeError: MMseqs2 produced no MSA` | Your query sequence has no homologs in the database. Try a larger/unfiltered DB like UniRef30. |
| `Bus error: 10` or `Prefilter died` with multiple iterations on Apple Silicon | Set `mmseqs.num_iterations: 1`. Multi-iteration MMseqs2 profile search has known macOS ARM crashes. |
| `run_openfold: command not found` | Install it with `uv tool install https://github.com/latent-spacecraft/openfold-3-mlx.git`. |
| OpenFold exits normally but `summary.txt` reports failed queries | The wrapper now treats this as an error. Read the OpenFold traceback above the summary; no successful structure was produced. |
| Out of memory (OOM) | Close other applications, reduce `MAX_SEQS` in `openfold.py`, or use a smaller protein. |
| MLX attention errors | Ensure you are on macOS ≥14 and Apple Silicon. Intel Macs are not supported. |

## License

MIT © 2026 Pavel Yakovlev
