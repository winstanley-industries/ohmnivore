"""Exercise actual concurrent resident channels, identities and typed failures."""

import argparse
import concurrent.futures
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest

from reference.emi03 import ensemble


WORKER = None


class GpuWorkerProcessTest(unittest.TestCase):
    def test_private_owners_survive_independent_failure_and_preserve_output(self):
        with tempfile.TemporaryDirectory(prefix="emi03-gpu-owners-") as temporary:
            root = Path(temporary)
            limits = ensemble.study.selected_manifest("emi01-v2")["limits"]
            pool = ensemble.PersistentPool(WORKER, root / "pool", limits, True, 16)
            records = []

            def run(owner, phase, rejected=False):
                directory = root / f"p{phase}-o{owner}"
                directory.mkdir()
                source, raw, stats = [
                    directory / name
                    for name in ("input.cir", "output.raw", "stats.json")
                ]
                voltage = owner + 2 if phase == 0 else -(owner + 3)
                source.write_text(
                    "* rejected\nBinvalid x 0\n.end\n"
                    if rejected
                    else f"* private owner {owner}\nVinput out 0 {voltage}\n"
                    "Bload out 0 I={v(out)/1000}\nCload out 0 1n\n"
                    ".tran 1n 10n\n.save v(out)\n.end\n"
                )
                worker = pool.workers[owner]
                if rejected:
                    with self.assertRaisesRegex(ValueError, "unsupported_input"):
                        worker.request(source, raw, stats, threading.Event())
                    self.assertFalse(raw.exists())
                    self.assertFalse(stats.exists())
                    process = worker.last_process
                else:
                    process = worker.request(source, raw, stats, threading.Event())
                    _, payload = raw.read_bytes().split(b"Binary:\n", 1)
                    values = struct.unpack("<" + "d" * (len(payload) // 8), payload)
                    self.assertGreaterEqual(len(values), 4)
                    for value in values[1::2]:
                        self.assertAlmostEqual(value, voltage, places=10)
                native = json.loads(Path(str(stats) + ".gpu.json").read_text())
                for name in (
                    "gpu_fallbacks",
                    "allocation_failures",
                    "cleanup_failures",
                    "outstanding_device_bytes",
                ):
                    self.assertEqual(native[name], 0)
                with pool.device_lock:
                    key = ensemble.worker_key(worker)
                    pool.device_peaks[key] = max(
                        pool.device_peaks.get(key, 0), native["peak_device_bytes"]
                    )
                    records.append({"process": process, "gpu": native})

            try:
                environment = dict(
                    field.split(b"=", 1)
                    for field in Path(f"/proc/{pool.shared.process.pid}/environ")
                    .read_bytes()
                    .split(b"\0")
                    if field
                )
                self.assertEqual(environment[b"CUDA_DEVICE_MAX_CONNECTIONS"], b"32")
                self.assertEqual(
                    environment[b"CUDA_DEVICE_MAX_COPY_CONNECTIONS"], b"32"
                )
                for phase in (0, 1):
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=16
                    ) as executor:
                        futures = [
                            executor.submit(
                                run, owner, phase, phase == 0 and owner == 7
                            )
                            for owner in range(16)
                        ]
                        for future in futures:
                            future.result()
                self.assertEqual(
                    len({worker.process.pid for worker in pool.workers}), 1
                )
                self.assertEqual(len({worker.thread_id for worker in pool.workers}), 16)
                for owner, worker in enumerate(pool.workers):
                    self.assertEqual(
                        worker.observed_affinity, [ensemble.CPU_AFFINITY[owner]]
                    )
                original = root / "p0-o0"
                paths = [
                    original / name
                    for name in ("input.cir", "output.raw", "stats.json")
                ]
                old = [path.read_bytes() for path in paths]
                with self.assertRaises(ValueError):
                    pool.workers[0].request(*paths, threading.Event())
                self.assertEqual([path.read_bytes() for path in paths], old)
                records.append({"process": pool.workers[0].last_process})
                self.assertIn(
                    "unsupported", (root / "pool/worker-7/worker.log").read_text()
                )
                self.assertEqual((root / "pool/worker-6/worker.log").read_text(), "")
            finally:
                workers = pool.close()
                resources = ensemble.pool_resources(pool)
            self.assertEqual(len(records), 33)
            self.assertTrue(ensemble.audit_resources(resources, records, workers, True))
            ensemble.audit_worker_affinity(workers, True, 16)

    def test_dead_shared_process_rejects_every_owner_without_retry(self):
        with tempfile.TemporaryDirectory(prefix="emi03-dead-pool-") as temporary:
            root = Path(temporary)
            limits = ensemble.study.selected_manifest("emi01-v2")["limits"]
            pool = ensemble.PersistentPool(WORKER, root / "pool", limits, True, 4)
            pid = pool.shared.process.pid
            pool.shared.stop()
            try:
                for owner, worker in enumerate(pool.workers):
                    with self.assertRaisesRegex(
                        ValueError, "persistent worker unavailable"
                    ):
                        worker.request(
                            root / f"input{owner}",
                            root / f"raw{owner}",
                            root / f"stats{owner}",
                            threading.Event(),
                        )
                    self.assertEqual(worker.last_process["worker_pid"], pid)
                    self.assertEqual(worker.last_process["worker_owner"], owner)
                    self.assertEqual(worker.last_process["worker_request"], 1)
                    self.assertFalse((root / f"raw{owner}").exists())
            finally:
                pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    args, remaining = parser.parse_known_args()
    WORKER = Path(args.worker).resolve()
    unittest.main(argv=[__file__] + remaining)
