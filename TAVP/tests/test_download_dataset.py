"""Offline regression tests for recoverable downloads and CSV identity merges."""
import csv
import hashlib
import http.server
import importlib.util
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dataset_builder", ROOT / "download_dataset.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.payload = bytes(range(256)) * 80
        self.mode, self.calls, self.ranges = "normal", 0, []
        fixture = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                fixture.calls += 1
                header = self.headers.get("Range")
                fixture.ranges.append(header)
                if fixture.mode == "retry" and fixture.calls == 1:
                    self.send_error(503)
                    return
                start = int(header.split("=")[1].split("-")[0]) if header else 0
                if fixture.mode == "ignore_range":
                    start = 0
                self.send_response(206 if start else 200)
                if start:
                    self.send_header("Content-Range", "bytes %d-%d/%d" % (start, len(fixture.payload) - 1, len(fixture.payload)))
                self.send_header("Content-Length", str(len(fixture.payload) - start))
                self.end_headers()
                data = fixture.payload[start:]
                if fixture.mode == "truncate" and fixture.calls == 1:
                    data = data[:len(data) // 2]
                    self.close_connection = True
                self.wfile.write(data)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.record = {
            "video_id": "test", "expected_file_size_bytes": len(self.payload),
            "repository_blob_sha1": hashlib.sha1(("blob %d\0" % len(self.payload)).encode() + self.payload).hexdigest(),
            "original_video_url": "http://127.0.0.1:%d/video.mp4" % self.server.server_port,
        }
        self.path = self.folder / "video.mp4"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def check_download(self):
        result = builder.download(self.path, self.record, timeout=2, retries=1)
        self.assertEqual(self.path.read_bytes(), self.payload)
        self.assertTrue(builder.original_integrity(self.path, self.record))
        self.assertFalse(self.path.with_suffix(".mp4.part").exists())
        return result

    def test_partial_resume(self):
        self.path.with_suffix(".mp4.part").write_bytes(self.payload[:3333])
        self.assertEqual(self.check_download(), "resumed")
        self.assertEqual(self.ranges, ["bytes=3333-"])

    def test_server_ignoring_range_restarts(self):
        self.mode = "ignore_range"
        self.path.with_suffix(".mp4.part").write_bytes(self.payload[:3333])
        self.assertEqual(self.check_download(), "downloaded")

    def test_transient_http_failure_retries(self):
        self.mode = "retry"
        self.check_download()
        self.assertEqual(self.calls, 2)

    def test_truncated_transfer_resumes_on_retry(self):
        self.mode = "truncate"
        self.assertEqual(self.check_download(), "resumed")
        self.assertEqual(self.ranges, [None, "bytes=%d-" % (len(self.payload) // 2)])

    def test_corrupt_final_is_replaced_then_verified_skip(self):
        self.path.write_bytes(b"x" * len(self.payload))
        self.check_download()
        calls = self.calls
        self.assertEqual(self.check_download(), "skipped_verified")
        self.assertEqual(self.calls, calls)

    def test_complete_partial_promoted_without_network(self):
        self.path.with_suffix(".mp4.part").write_bytes(self.payload)
        self.assertEqual(self.check_download(), "resumed")
        self.assertEqual(self.calls, 0)


class DiscoveryTests(unittest.TestCase):
    def test_csv_row_order_and_extra_columns_do_not_change_identity(self):
        source = ROOT / "sources/tavp_real_23_with_usable_task_prompts.csv"
        original = builder.discover(ROOT, [source], timeout=2, retries=0)
        with source.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        with tempfile.TemporaryDirectory() as temp:
            shuffled = Path(temp) / "shuffled.csv"
            extra = Path(temp) / "supplement.csv"
            with shuffled.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(reversed(rows))
            with extra.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["video_url", "custom_note"])
                writer.writeheader()
                writer.writerow({"video_url": rows[7]["video_url"], "custom_note": "preserved useful metadata"})
            merged = builder.discover(ROOT, [shuffled, extra], timeout=2, retries=0)
        fields = ["video_id", "task_instruction", "prompt_provenance", "original_repository_path", "relative_folder_path"]
        self.assertEqual([[r[k] for k in fields] for r in original], [[r[k] for k in fields] for r in merged])
        self.assertEqual(merged[7]["source_csv_records"][1]["columns"]["custom_note"], "preserved useful metadata")


if __name__ == "__main__":
    unittest.main()
