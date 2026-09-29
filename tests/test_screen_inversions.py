"""Small end-to-end and identifiability checks for the two screen inversions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True  # This repository tracks an older generated .pyc.

import torch

from generate_turpy_datasets import generate_turpy_trajectory, trajectory_to_one_step_examples
from invert_screens_split_step import final_intensity, make_propagator
from inversion_evaluation import begin_test_set_run, numeric_summary
from turpy import make_turpy_simulator
from utilities import FNO2d


ROOT = Path(__file__).resolve().parents[1]


class ScreenInversionTests(unittest.TestCase):
    def test_aggregate_statistics_and_resume_protection(self) -> None:
        self.assertEqual(numeric_summary([2.0])["median"], 2.0)
        self.assertEqual(numeric_summary([1.0, 3.0])["p25"], 1.5)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            rows, _ = begin_test_set_run(output, {"max_its": 2}, resume=False)
            self.assertEqual(rows, [])
            with self.assertRaises(FileExistsError):
                begin_test_set_run(output, {"max_its": 2}, resume=False)
            with self.assertRaises(ValueError):
                begin_test_set_run(output, {"max_its": 3}, resume=True)

    def test_split_step_reproduces_generator_and_last_screen_is_invisible(self) -> None:
        params, simulator = make_turpy_simulator(grid_size=8, dx=0.25, device="cpu")
        trajectory = generate_turpy_trajectory(
            simulator, params, n_z=4, total_distance=300.0, beam_type="gaussian",
            initial_phase="flat", zero_padding=True, padding_factor=2, seed=13,
        )
        rho0 = trajectory["intensities"][0]
        observed = trajectory["intensities"][-1]
        phase = trajectory["delta_n"] * (2 * torch.pi / params["wavelength"]) * 100.0
        propagate = make_propagator(simulator, 8, 8, 100.0, True, 2, torch.device("cpu"))
        predicted = final_intensity(rho0, phase, propagate)
        relative = (predicted - observed).norm() / observed.norm()
        self.assertLess(float(relative), 1e-4)
        changed = phase.clone()
        changed[-1] += torch.randn_like(changed[-1])
        final_changed = final_intensity(rho0, changed, propagate)
        self.assertLess(float((final_changed - predicted).norm() / observed.norm()), 1e-5)

    def test_both_scripts_write_separate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            data_dir = temp / "chunks"
            data_dir.mkdir()
            checkpoint_dir = temp / "checkpoint"
            checkpoint_dir.mkdir()
            params, simulator = make_turpy_simulator(grid_size=8, dx=0.25, device="cpu")
            trajectory = generate_turpy_trajectory(
                simulator, params, n_z=4, total_distance=300.0,
                beam_type="gaussian", initial_phase="flat", zero_padding=True,
                padding_factor=2, seed=17,
            )
            x, y = trajectory_to_one_step_examples(trajectory)
            torch.save({
                "X": x, "Y": y, "path_ids": torch.zeros(3, dtype=torch.long),
                "n_z": 4, "total_distance": 300.0, "wavelength": params["wavelength"],
                "n0": params["n"], "dx": params["dx"], "initial_phase": "flat",
            }, data_dir / "chunk_000.pt")
            trajectory_second = generate_turpy_trajectory(
                simulator, params, n_z=4, total_distance=300.0,
                beam_type="gaussian", initial_phase="flat", zero_padding=True,
                padding_factor=2, seed=19,
            )
            x_second, y_second = trajectory_to_one_step_examples(trajectory_second)
            torch.save({
                "X": x_second, "Y": y_second,
                "path_ids": torch.ones(3, dtype=torch.long),
                "n_z": 4, "total_distance": 300.0, "wavelength": params["wavelength"],
                "n0": params["n"], "dx": params["dx"], "initial_phase": "flat",
            }, data_dir / "chunk_001.pt")
            (checkpoint_dir / "split_manifest.json").write_text(
                json.dumps({"test_path_ids": [0, 1]}), encoding="utf-8"
            )
            model_kwargs = {"input_channels": 5, "modes_y": 2, "modes_x": 2,
                            "width": 4, "layers": 1}
            model = FNO2d(**model_kwargs)
            torch.save({"model_kwargs": model_kwargs, "model_state_dict": model.state_dict(),
                        "normalization": {"intensity_mean": 0.0, "intensity_std": 1.0,
                                          "delta_n_rms": 1e-9}}, checkpoint_dir / "best.pt")
            for script, output in (("invert_screens_fno.py", temp / "fno"),
                                   ("invert_screens_split_step.py", temp / "split")):
                command = [sys.executable, str(ROOT / script), "--data-dir", str(data_dir),
                           "--output-dir", str(output), "--max-its", "2",
                           "--device", "cpu", "--print-every", "1"]
                if script == "invert_screens_fno.py":
                    command += ["--checkpoint", str(checkpoint_dir / "best.pt")]
                else:
                    command += ["--manifest", str(checkpoint_dir / "split_manifest.json")]
                subprocess.run(
                    command, cwd=ROOT, check=True, capture_output=True, text=True,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                         "MPLCONFIGDIR": str(temp / "matplotlib_cache")},
                )
                self.assertTrue((output / "final_intensity.png").is_file())
                self.assertTrue((output / "phase_screens.png").is_file())
                self.assertTrue((output / "optimization.png").is_file())
                summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                self.assertFalse(summary["last_screen_identifiable_from_rhoZ"])
                self.assertEqual(len(summary["screen_metrics_evaluation_only"]), 3)
                if script == "invert_screens_split_step.py":
                    self.assertLess(summary["oracle_split_step_image_metrics_evaluation_only"]["relative_l2"], 1e-4)

            for script, output in (("invert_screens_fno.py", temp / "fno_batch"),
                                   ("invert_screens_split_step.py", temp / "split_batch"),
                                   ("run_turpy_inversion.py", temp / "rho0_batch")):
                command = [sys.executable, str(ROOT / script), "--data-dir", str(data_dir),
                           "--output-dir", str(output), "--all-test-paths",
                           "--example-plots", "0", "--max-its", "2", "--device", "cpu"]
                if script == "invert_screens_split_step.py":
                    command += ["--manifest", str(checkpoint_dir / "split_manifest.json")]
                else:
                    command += ["--checkpoint", str(checkpoint_dir / "best.pt")]
                environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                               "MPLCONFIGDIR": str(temp / "matplotlib_cache")}
                subprocess.run(command + ["--max-paths", "1"], cwd=ROOT, check=True, capture_output=True,
                               text=True, env=environment)
                pilot = json.loads((output / "test_set_summary.json").read_text(encoding="utf-8"))
                self.assertEqual(pilot["test_path_count"], 1)
                subprocess.run(command + ["--resume"], cwd=ROOT, check=True,
                               capture_output=True, text=True, env=environment)
                summary = json.loads((output / "test_set_summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["test_path_count"], 2)
                self.assertTrue((output / "test_set_summary.png").is_file())
                rows = (output / "path_metrics.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertEqual({json.loads(row)["path_id"] for row in rows}, {0, 1})
                self.assertFalse((output / "examples").exists())
                if script != "run_turpy_inversion.py":
                    self.assertEqual(len(summary["per_screen_evaluation_only"]), 2)
                else:
                    self.assertEqual(summary["rho0_relative_l2"]["count"], 2)
                self.assertEqual(len((output / "path_metrics.jsonl").read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
