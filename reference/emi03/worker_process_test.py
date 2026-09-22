"""Exercise the real persistent worker, including isolation after rejected jobs."""

import argparse
import json
from pathlib import Path
import selectors
import struct
import subprocess
import tempfile
import unittest


WORKER = None


class WorkerProcessTest(unittest.TestCase):
    def test_persistence_isolation_failure_recovery_and_existing_output_preservation(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="emi03-worker-test-") as temporary:
            directory = Path(temporary)
            with (directory / "diagnostics.log").open("wb") as log:
                process = subprocess.Popen(
                    [str(WORKER), "--worker"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                )
                try:
                    selector = selectors.DefaultSelector()
                    selector.register(process.stdout, selectors.EVENT_READ)

                    def send(line):
                        process.stdin.write(line.encode() + b"\n")
                        process.stdin.flush()
                        self.assertTrue(
                            selector.select(timeout=30), "worker failed to reply"
                        )
                        return json.loads(process.stdout.readline())

                    def run(name, voltage):
                        source, raw, stats = [
                            directory / (name + suffix)
                            for suffix in (".cir", ".raw", ".json")
                        ]
                        source.write_text(
                            f"* {name}\nVinput out 0 {voltage}\nRload out 0 1k\n.op\n.save v(out)\n.end\n"
                        )
                        response = send("\t".join(map(str, (source, raw, stats))))
                        self.assertEqual(response["status"], "complete", response)
                        self.assertEqual(response["input"], str(source))
                        self.assertEqual(response["exit_code"], 0)
                        header, payload = raw.read_bytes().split(b"Binary:\n", 1)
                        self.assertIn(f"Title: {name}\n".encode(), header)
                        self.assertEqual(struct.unpack("<dd", payload), (0, voltage))
                        self.assertEqual(
                            json.loads(stats.read_text())["status"], "complete"
                        )
                        return source, raw, stats

                    first = run("first", 2)
                    first_bytes = [path.read_bytes() for path in first]
                    run("second", 7)
                    self.assertIsNone(process.poll())
                    self.assertEqual(send("malformed")["status"], "typed_failure")
                    missing = directory / "missing.cir"
                    rejected = send(
                        "\t".join(
                            map(
                                str,
                                (
                                    missing,
                                    directory / "missing.raw",
                                    directory / "missing.json",
                                ),
                            )
                        )
                    )
                    self.assertEqual(rejected["status"], "typed_failure")
                    self.assertEqual(rejected["input"], str(missing))
                    self.assertFalse((directory / "missing.raw").exists())
                    self.assertFalse((directory / "missing.json").exists())
                    duplicate = send("\t".join(map(str, first)))
                    self.assertEqual(duplicate["status"], "typed_failure")
                    self.assertEqual([path.read_bytes() for path in first], first_bytes)
                    run("after-rejection", -3)
                    process.stdin.close()
                    self.assertEqual(process.wait(timeout=30), 0)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    process.stdout.close()
                    if not process.stdin.closed:
                        process.stdin.close()
                    selector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    args, remaining = parser.parse_known_args()
    WORKER = Path(args.worker).resolve()
    unittest.main(argv=[__file__] + remaining)
