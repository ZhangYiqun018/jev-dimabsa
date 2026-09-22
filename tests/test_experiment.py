import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

import numpy as np

from jev.client import Answer, JevClient, JevError, Response
from jev.data import load_jsonl, write_jsonl
from jev.fewshot import Example, ExampleSet
from runners import run_st1, run_all_st1
from tools import calibrate_st1 as cal


class ExperimentTests(unittest.TestCase):
    def test_transient_errors_retry_but_auth_errors_do_not(self):
        success = {"model": "jev-1.13.0", "answers": {}, "usage": {"input_tokens": 7}}
        overload = urllib.error.HTTPError("https://example.invalid", 529, "overloaded", {}, io.BytesIO(b"busy"))
        with patch("jev.client.load_api_key", return_value="test-key"), \
             patch("jev.client.time.sleep") as sleep, \
             patch("jev.client.urllib.request.urlopen", side_effect=[
                 overload, TimeoutError("timeout"), io.BytesIO(json.dumps(success).encode())]) as urlopen:
            response = JevClient().ask("text", {})
            self.assertEqual(response.attempts, 3)
            self.assertEqual(response.usage["input_tokens"], 7)
            self.assertEqual(urlopen.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
        auth = urllib.error.HTTPError("https://example.invalid", 401, "unauthorized", {}, io.BytesIO(b"bad"))
        with patch("jev.client.load_api_key", return_value="test-key"), \
             patch("jev.client.time.sleep") as sleep, \
             patch("jev.client.urllib.request.urlopen", side_effect=auth) as urlopen:
            with self.assertRaises(JevError):
                JevClient().ask("text", {})
            self.assertEqual(urlopen.call_count, 1)
            sleep.assert_not_called()

    def test_resume_retains_completed_rows_and_counts_usage_once(self):
        calls = []
        def ask(state, questions):
            calls.append(state)
            if state == "second" and calls.count(state) == 1:
                raise JevError("persistent overload")
            return Response("jev-1.13.0", {name: Answer(name, "score", {
                "type": "score", "score": 5.125, "confidence": .9,
                "probabilities": {"5": .875, "6": .125}}) for name in questions},
                {"input_tokens": 11})
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "eng_restaurant_dev_task1.jsonl"
            out = Path(directory) / "pred.jsonl"
            write_jsonl(source, [{"ID": name, "Text": name,
                                 "Aspect_VA": [{"Aspect": "food"}]} for name in ["first", "second"]])
            argv = ["run_st1", "--data", str(source), "--out", str(out), "--shots", "0", "--quiet"]
            with patch.object(run_st1, "JevClient") as client, patch("sys.argv", argv), \
                 contextlib.redirect_stdout(io.StringIO()):
                client.return_value.model = "jev-1.13.0"
                client.return_value.ask.side_effect = ask
                self.assertEqual(run_st1.main(), 1)
                first_line = out.read_text()
                # Simulate losing end-of-run accounting: only early config survived.
                meta_path = Path(str(out) + ".meta.json")
                config = json.loads(meta_path.read_text())["config"]
                meta_path.write_text(json.dumps({"config": config}))
                self.assertEqual(run_st1.main(), 0)
                self.assertEqual(run_st1.main(), 0)
            self.assertTrue(out.read_text().startswith(first_line))
            self.assertEqual(calls.count("first"), 1)
            self.assertEqual(calls.count("second"), 2)
            self.assertEqual(len(load_jsonl(out)), 2)
            meta = json.loads(meta_path.read_text())
            self.assertEqual(meta["usage_total"]["input_tokens"], 22)
            self.assertEqual(load_jsonl(out)[0]["_jev"]["answers"]["v0"]["score"], 5.125)
            batch_pred = Path(directory) / "pred_st1_eng_restaurant_dev_s0.jsonl"
            batch_pred.write_text(out.read_text())
            with patch.object(run_all_st1, "REPORTS", Path(directory)), \
                 patch.object(run_all_st1.subprocess, "run", return_value=SimpleNamespace(
                     returncode=1, stdout="incomplete", stderr="")) as process:
                failed = run_all_st1.run_one("eng", "restaurant", "dev", 1, 0, "first-k")
                self.assertIn("error", failed)
                self.assertEqual(failed["input_tokens"], 22)
                self.assertEqual(process.call_count, 3)
                for call in process.call_args_list:
                    self.assertNotIn("--restart", call.args[0])
            with patch.object(run_all_st1, "CORPORA", [("eng", "restaurant")]), \
                 patch.object(run_all_st1, "REPORTS", Path(directory)), \
                 patch.object(run_all_st1, "run_one", return_value={"corpus": "eng_restaurant", "error": "missing ID", "input_tokens": 22}), \
                 patch("sys.argv", ["run_all"]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_all_st1.main(), 1)

    def test_calibration_math_and_group_isolation(self):
        x = np.column_stack([np.arange(1., 8.), np.arange(1., 8.)])
        y = 1 + x * [.8, .6]
        x = np.concatenate([x, x[-1:]])
        y = np.concatenate([y, y[-1:]])
        groups = np.array(["a", "b", "c", "d", "e", "f", "g", "g"])
        fitted = cal.fit(x, y, groups)
        np.testing.assert_allclose(cal.apply(x, fitted["linear"]), y, atol=1e-12)
        np.testing.assert_allclose(fitted["shrink"]["slope"], [.75, .5])
        np.testing.assert_allclose(fitted["mean"]["intercept"], y.mean(axis=0))
        np.testing.assert_allclose(fitted["offset"]["intercept"], (y - x).mean(axis=0))
        flat = cal.coefficients(np.ones_like(x), y, "linear")
        np.testing.assert_allclose(flat["slope"], [0, 0])
        np.testing.assert_allclose(cal.apply(np.array([[-5, 20]]), fitted["raw"]), [[1, 9]])
        with patch("tools.analyze_st1_design.subprocess.run", return_value=SimpleNamespace(
            returncode=0, stdout="Final Results: {'RMSE_VA': np.float64(1.25), 'PCC_A': np.float64(nan)}",
            stderr="constant input")):
            self.assertEqual(cal.official_check("gold.jsonl", [("a", "food")], [[5., 5.]]), 1.25)
        train = [{"ID": str(i), "Text": f"review {i}", "Aspect_VA": [
            {"Aspect": "food", "VA": "6.00#5.00"}]} for i in range(260)]
        train.append({**train[2], "ID": "duplicate"})
        example = ExampleSet("eng_restaurant", [Example("0", "review 0", "food", 6, 5)])
        with tempfile.TemporaryDirectory() as directory:
            def read(path):
                return train if str(path) == "train" else [{"ID": "different-ID", "Text": "review 1"}]
            with patch.object(cal, "CORPORA", [("eng", "restaurant")]), \
                 patch.object(cal, "train_path", return_value="train"), \
                 patch.object(cal, "split_path", return_value="heldout"), \
                 patch.object(cal, "load_jsonl", side_effect=read), \
                 patch.object(cal, "load_examples", return_value=example):
                manifest = cal.prepare(Path(directory))["corpora"]["eng_restaurant"]
            self.assertNotIn("0", manifest["ids"])
            self.assertNotIn("1", manifest["ids"])
            self.assertEqual(len(set(manifest["groups"].values())), 256)
            self.assertEqual("2" in manifest["ids"], "duplicate" in manifest["ids"])
            if "2" in manifest["ids"]:
                self.assertEqual(manifest["groups"]["2"], manifest["groups"]["duplicate"])


if __name__ == "__main__":
    unittest.main()
