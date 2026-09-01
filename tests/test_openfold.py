from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import openfold
import yaml


class OpenFoldWrapperTests(unittest.TestCase):
    def test_mmseqs_result_database_is_unpacked_to_plain_a3m(self) -> None:
        commands: list[list[str]] = []

        def fake_run(cmd: list[str], **_kwargs) -> None:
            commands.append(cmd)
            if cmd[1] == "result2msa":
                msa_db = Path(cmd[5])
                msa_db.write_bytes(b">A\nACD\n\0")
                Path(f"{msa_db}.dbtype").write_bytes(b"\x0c\x00\x00\x00")
                Path(f"{msa_db}.index").write_text("0\t0\t8\n")
            elif cmd[1] == "unpackdb":
                unpack_dir = Path(cmd[3])
                unpack_dir.mkdir(parents=True, exist_ok=True)
                (unpack_dir / "0.a3m").write_text(">A\nACD\n>hit\nACD\n")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chain_dir = root / "alignments" / "chain_A"
            with mock.patch.object(openfold, "run", side_effect=fake_run):
                result = openfold.run_mmseqs_for_chain(
                    seq="ACD",
                    db_path=root / "target_db",
                    chain_dir=chain_dir,
                    tmp_root=root / "tmp",
                    tag="A",
                    num_iterations=1,
                    threads=4,
                )

            self.assertEqual(result, chain_dir / "custom_db_hits.a3m")
            self.assertEqual(result.read_bytes(), b">A\nACD\n>hit\nACD\n")
            self.assertFalse(Path(f"{result}.dbtype").exists())
            self.assertFalse(Path(f"{result}.index").exists())

        search_cmd = next(cmd for cmd in commands if cmd[1] == "search")
        self.assertEqual(search_cmd[search_cmd.index("--num-iterations") + 1], "1")
        self.assertEqual(search_cmd[search_cmd.index("--threads") + 1], "4")
        self.assertTrue(any(cmd[1] == "unpackdb" for cmd in commands))

    def test_main_disables_pairing_and_forwards_mmseqs_settings(self) -> None:
        captured_mmseqs: list[dict] = []

        def fake_mmseqs(**kwargs) -> Path:
            captured_mmseqs.append(kwargs)
            chain_dir = kwargs["chain_dir"]
            chain_dir.mkdir(parents=True, exist_ok=True)
            msa_path = chain_dir / "custom_db_hits.a3m"
            msa_path.write_text(">A\nACD\n")
            return msa_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_yaml = root / "input.yml"
            output_dir = root / "output"
            db_path = root / "target_db"
            Path(f"{db_path}.dbtype").write_bytes(b"\x00")
            input_yaml.write_text(
                yaml.safe_dump(
                    {
                        "name": "test_complex",
                        "database": str(db_path),
                        "mmseqs": {"num_iterations": 1, "threads": 3},
                        "pairing": False,
                        "chains": [
                            {
                                "ids": ["A"],
                                "type": "protein",
                                "sequence": "ACD",
                            }
                        ],
                    }
                )
            )

            with (
                mock.patch.object(openfold, "run_mmseqs_for_chain", side_effect=fake_mmseqs),
                mock.patch.object(openfold, "run", side_effect=self._write_successful_prediction),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "openfold",
                        "--input_yaml",
                        str(input_yaml),
                        "--output_dir",
                        str(output_dir),
                    ],
                ),
            ):
                self.assertEqual(openfold.main(), 0)

            query = json.loads((output_dir / "test_complex_query.json").read_text())
            query_cfg = query["queries"]["test_complex"]
            self.assertIs(query_cfg["use_paired_msas"], False)
            self.assertIs(query_cfg["use_main_msas"], True)

            runner = yaml.safe_load((output_dir / "runner.yml").read_text())
            self.assertEqual(runner["dataset_config_kwargs"]["msa"]["msas_to_pair"], [])

        self.assertEqual(len(captured_mmseqs), 1)
        self.assertEqual(captured_mmseqs[0]["num_iterations"], 1)
        self.assertEqual(captured_mmseqs[0]["threads"], 3)

    def test_main_builds_precomputed_taxonomy_pairing_and_query_only_peptide(self) -> None:
        def fake_mmseqs(**kwargs) -> Path:
            tag = kwargs["tag"]
            chain_dir = kwargs["chain_dir"]
            chain_dir.mkdir(parents=True, exist_ok=True)
            msa_path = chain_dir / "custom_db_hits.a3m"
            if tag == "A":
                msa_path.write_text(">A\nACD\n>P_A\nACD\n>P_A2\nA-D\n")
            else:
                msa_path.write_text(">B\nEFG\n>P_B\nEFG\n>P_B2\nE-G\n")
            return msa_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_yaml = root / "input.yml"
            output_dir = root / "output"
            db_path = root / "target_db"
            Path(f"{db_path}.dbtype").write_bytes(b"\x00")
            Path(f"{db_path}_h").write_bytes(
                b"sp|P_A|A_SPEC Protein A OX=9606\n\0"
                b"sp|P_A2|A_MOUSE Protein A OX=10090\n\0"
                b"sp|P_B|B_SPEC Protein B OX=9606\n\0"
                b"sp|P_B2|B_MOUSE Protein B OX=10090\n\0"
            )
            input_yaml.write_text(
                yaml.safe_dump(
                    {
                        "name": "test_complex",
                        "database": str(db_path),
                        "pairing": True,
                        "chains": [
                            {"ids": ["A"], "type": "protein", "sequence": "ACD"},
                            {"ids": ["B"], "type": "protein", "sequence": "EFG"},
                            {
                                "ids": ["C"],
                                "type": "protein",
                                "sequence": "HIK",
                                "msa": False,
                            },
                        ],
                    }
                )
            )

            with (
                mock.patch.object(openfold, "run_mmseqs_for_chain", side_effect=fake_mmseqs),
                mock.patch.object(openfold, "run", side_effect=self._write_successful_prediction),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "openfold",
                        "--input_yaml",
                        str(input_yaml),
                        "--output_dir",
                        str(output_dir),
                    ],
                ),
            ):
                self.assertEqual(openfold.main(), 0)

            query = json.loads((output_dir / "test_complex_query.json").read_text())
            query_cfg = query["queries"]["test_complex"]
            self.assertIs(query_cfg["use_paired_msas"], True)
            chains = {chain["chain_ids"]: chain for chain in query_cfg["chains"]}
            self.assertIn("paired_msa_file_paths", chains["A"])
            self.assertIn("paired_msa_file_paths", chains["B"])
            self.assertIn("paired_msa_file_paths", chains["C"])
            self.assertTrue(chains["C"]["main_msa_file_paths"].endswith("query_only.a3m"))

            paired_a = Path(chains["A"]["paired_msa_file_paths"]).read_text()
            paired_b = Path(chains["B"]["paired_msa_file_paths"]).read_text()
            paired_c = Path(chains["C"]["paired_msa_file_paths"]).read_text()
            self.assertEqual(paired_a.count(">"), 3)
            self.assertEqual(paired_b.count(">"), 3)
            self.assertEqual(paired_c.count(">"), 3)
            self.assertIn("OX9606", paired_a)
            self.assertIn("OX10090", paired_b)
            self.assertEqual(
                paired_c,
                ">C\nHIK\n"
                ">pair_00001_OX9606\n---\n"
                ">pair_00002_OX10090\n---\n",
            )

            runner = yaml.safe_load((output_dir / "runner.yml").read_text())
            msa = runner["dataset_config_kwargs"]["msa"]
            self.assertEqual(msa["msas_to_pair"], [])
            self.assertEqual(msa["paired_msa_order"], ["custom_db_paired"])

    def test_main_rejects_zero_exit_prediction_with_failed_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_yaml = root / "input.yml"
            output_dir = root / "output"
            db_path = root / "target_db"
            Path(f"{db_path}.dbtype").write_bytes(b"\x00")
            input_yaml.write_text(
                yaml.safe_dump(
                    {
                        "name": "failed_complex",
                        "database": str(db_path),
                        "chains": [
                            {
                                "ids": ["A"],
                                "type": "protein",
                                "sequence": "ACD",
                                "msa": False,
                            }
                        ],
                    }
                )
            )

            def write_failed_prediction(cmd: list[str], **_kwargs) -> None:
                output_arg = next(arg for arg in cmd if arg.startswith("--output_dir="))
                predictions_dir = Path(output_arg.split("=", 1)[1])
                predictions_dir.mkdir(parents=True, exist_ok=True)
                (predictions_dir / "summary.txt").write_text(
                    "Total Queries Processed: 1\n"
                    "  - Successful Queries:  0\n"
                    "  - Failed Queries:      1\n"
                    "\nFailed Queries: failed_complex\n"
                )

            with (
                mock.patch.object(openfold, "run", side_effect=write_failed_prediction),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "openfold",
                        "--input_yaml",
                        str(input_yaml),
                        "--output_dir",
                        str(output_dir),
                    ],
                ),
                self.assertRaisesRegex(RuntimeError, "Failed Queries"),
            ):
                openfold.main()

    @staticmethod
    def _write_successful_prediction(cmd: list[str], **_kwargs) -> None:
        output_arg = next(arg for arg in cmd if arg.startswith("--output_dir="))
        predictions_dir = Path(output_arg.split("=", 1)[1])
        model_dir = predictions_dir / "test_complex" / "seed_42"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "test_complex_seed_42_sample_1_model.cif").write_text("data_test\n")
        (predictions_dir / "summary.txt").write_text(
            "Total Queries Processed: 1\n  - Successful Queries:  1\n  - Failed Queries:      0\n"
        )


if __name__ == "__main__":
    unittest.main()
