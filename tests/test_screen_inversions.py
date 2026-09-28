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
from turpy import make_turpy_simulator
from utilities import FNO2d


ROOT = Path(__file__).resolve().parents[1]


class ScreenInversionTests(unittest.TestCase):
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
            (checkpoint_dir / "split_manifest.json").write_text(
                json.dumps({"test_path_ids": [0]}), encoding="utf-8"
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


if __name__ == "__main__":
    unittest.main()
