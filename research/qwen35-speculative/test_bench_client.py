"""Offline tests for the speculative-decoding benchmark client; no model/GPU needed."""
import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

CLIENT = pathlib.Path(__file__).with_name("bench_client.py")
spec = importlib.util.spec_from_file_location("spec_bench_client", CLIENT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

class FakeResponse:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self, *_args, **_kwargs):
        return self.payload
    def __init__(self, data):
        self.payload = json.dumps(data).encode("utf-8")

class BenchmarkTests(unittest.TestCase):
    def test_rejects_nonlocal_hosts(self):
        for url in ["https://127.0.0.1:8080", "http://example.com",
                    "http://192.168.0.5:8080", "http://admin:password@localhost:8080"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                bench.localhost_api(url)
        self.assertEqual(bench.localhost_api("http://127.0.0.1:8080"),
                         "http://127.0.0.1:8080/completion")

    def test_complete_records_distinct_wall_and_server_timings(self):
        payload = {"content": "three tokens", "tokens_predicted": 3,
                   "timings": {"predicted_per_second": 21.5, "predicted_ms": 135.0}}
        with mock.patch.object(bench.urllib.request, "urlopen",
                               return_value=FakeResponse(payload)) as req:
            item = bench.complete("http://127.0.0.1:8080/completion", "hi", 3, 1)
        self.assertEqual(item["server_predicted_tokens_per_second"], 21.5)
        self.assertGreaterEqual(item["client_wall_ms"], 0)
        self.assertEqual(item["tokens_predicted"], 3)
        self.assertEqual(len(item["content_sha256"]), 64)
        request_payload = json.loads(req.call_args.args[0].data)
        self.assertEqual(request_payload["temperature"], 0)
        self.assertFalse(request_payload["stream"])

    def test_mock_session_writes_metadata_and_all_samples(self):
        payload = {"content": "deterministic", "tokens_predicted": 5,
                   "timings": {"predicted_per_second": 12.3}}
        with tempfile.TemporaryDirectory() as folder:
            argv = ["bench_client.py", "--variant", "vanilla", "--build", "abc123",
                    "--session", "one", "--rounds", "2", "--warmups", "1",
                    "--output-dir", folder]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                    bench.urllib.request, "urlopen",
                    side_effect=lambda *a, **k: FakeResponse(payload)) as req:
                with contextlib.redirect_stdout(io.StringIO()):
                    code = bench.main()
            self.assertEqual(code, 0)
            self.assertEqual(req.call_count, 7)  # one warmup + three prompts x two rounds
            paths = list(pathlib.Path(folder).glob("*.jsonl"))
            self.assertEqual(len(paths), 1)
            records = [json.loads(line) for line in paths[0].read_text().splitlines()]
            self.assertEqual(records[0]["kind"], "metadata")
            samples = records[1:]
            self.assertEqual(len(samples), 6)
            self.assertEqual({r["prompt_index"] for r in samples}, {0, 1, 2})
            self.assertTrue(all(r["server_predicted_tokens_per_second"] == 12.3 for r in samples))

if __name__ == "__main__":
    unittest.main()
