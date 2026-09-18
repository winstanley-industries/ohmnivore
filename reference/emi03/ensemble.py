"""Bounded EMI-03 qualification and persistent whole-circuit replay experiment.

The harness never enables dispatch. Failed accuracy, resource or completion gates
are retained as negative evidence, and cannot become performance observations.
"""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import resource
import selectors
import shutil
import signal
import statistics
import subprocess
import tempfile
import threading
import time

from reference.emi01 import study
from reference.emi01 import adapter, circuits, metrics, signals
from reference.emi02 import importer, qualification


SCHEMA = "emi03-ensemble-v1"
CPU_WORKERS = (1, 4, 16)
GPU_WORKERS = (4,)
CPU_AFFINITY = (4, 6, 20, 22, 0, 2, 8, 10, 12, 14, 16, 18, 24, 26, 28, 30)
ENSEMBLES = (9, 36)
WARMUPS = 1
MEASURED = 5
HOST_BYTES = 16 * 1024**3
DEVICE_BYTES = 4 * 1024**3
DEVICE_WORKER_BYTES = 256 * 1024**2
FAILURES = study.FAILURES | {"parse_failure", "compile_failure"}
_ARCHIVE_LOCK = threading.Lock()
_INVOCATION_MONITOR = None
SOURCE_FILES = (
    "reference/emi03/ensemble.py",
    "reference/emi03/README.md",
    "reference/emi03/BUILD.bazel",
    "docs/adr/ADR-008-emi03-transient-ensembles.md",
    "docs/emi01-gpu-experiment-contract.md",
    "cpp/benchmarks/emi03_worker.cc",
    "cuda/BUILD.bazel",
    "cuda/emi03_real_solver.cu",
    "cuda/emi03_real_solver.h",
    "cuda/emi03_expression.cu",
    "cuda/emi03_expression.h",
    "cuda/emi03_cuda_internal.h",
)


def fail(status, message):
    raise ValueError(status + ": " + message)


def identities():
    result = qualification.identities()
    for name in SOURCE_FILES:
        path = qualification.ROOT / name
        if not path.is_file():
            fail("provenance_mismatch", "mandatory EMI-03 source unavailable")
        result[name] = study.sha(path.read_bytes())
    return result


def accelerator_executable():
    executable = shutil.which("nvidia-smi")
    if executable is None and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    if executable is None:
        fail("unsupported_input", "accelerator identity tool unavailable")
    return executable


def cpu_identity():
    model = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    topology = []
    for cpu in CPU_AFFINITY:
        directory = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        topology.append(
            {
                "logical_cpu": cpu,
                "package": int((directory / "physical_package_id").read_text()),
                "core": int((directory / "core_id").read_text()),
            }
        )
    if (
        "9950X3D" not in model
        or len({(row["package"], row["core"]) for row in topology}) != 16
    ):
        fail("unsupported_input", "frozen 9950X3D physical-core mapping required")
    return {"model": model, "physical_core_mapping": topology}


def accelerator_identity():
    executable = accelerator_executable()
    result = subprocess.run(
        [
            executable,
            "--id=0",
            "--query-gpu=name,uuid,driver_version,memory.total,clocks.current.sm,clocks.current.memory,power.limit",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        timeout=10,
        check=False,
    )
    if result.returncode or not result.stdout or len(result.stdout) > 64 * 1024:
        fail("unsupported_input", "accelerator identity unavailable")
    return {
        "fields": "name,uuid,driver_version,memory.total_MiB,sm_MHz,memory_MHz,power_limit_W",
        "value": result.stdout.decode("utf-8").strip(),
        "diagnostic_executable": executable,
        "policy": "observed clocks and configured power limit; no clock or power changes",
    }


def device_resident_bytes(executable):
    result = subprocess.run(
        [
            executable,
            "--id=0",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        timeout=10,
        check=False,
    )
    try:
        value = float(result.stdout.decode().strip())
    except (ValueError, UnicodeError):
        fail("resource_limit", "device memory telemetry unavailable")
    if result.returncode or not math.isfinite(value) or value < 0:
        fail("resource_limit", "device memory telemetry unavailable")
    return math.ceil(value * 1024 * 1024)


def replay_specs(manifest, count, label):
    if type(count) is not int or count not in ENSEMBLES:
        fail("unsupported_input", "only nine or thirty-six complete jobs")
    return [
        {
            **spec,
            "replica": replica,
            "reference_id": spec["id"],
            "id": spec["id"].replace("q2-", f"{label}-r{replica}-", 1),
        }
        for replica in range(count // 9)
        for spec in study.make_specs(manifest, {}, "q2", "ensemble", 2)
    ]


def process_usage(pid):
    """Linux process CPU and resident bytes; CUDA virtual reservation is excluded."""
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")") + 2 :].split()
    ticks = int(fields[11]) + int(fields[12])
    return ticks / os.sysconf("SC_CLK_TCK"), int(fields[21]) * os.sysconf(
        "SC_PAGE_SIZE"
    )


class InvocationHostMonitor:
    """Account the complete descendant tree, including fresh external oracles."""

    def __init__(self):
        self.finished = threading.Event()
        self.exhausted = threading.Event()
        self.started = time.perf_counter()
        self.peak_bytes = 0
        self.samples = 0
        self.errors = 0
        self.peaks = []
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _sample(self):
        pending = [os.getpid()]
        visited = set()
        resident = 0
        while pending:
            pid = pending.pop()
            if pid in visited:
                continue
            visited.add(pid)
            try:
                status = Path(f"/proc/{pid}/status").read_text().splitlines()
                values = {
                    line.split(":", 1)[0]: line.split()[1]
                    for line in status
                    if line.startswith(("VmHWM:", "VmRSS:"))
                }
                resident += int(values.get("VmHWM", values.get("VmRSS", "0"))) * 1024
                for task in Path(f"/proc/{pid}/task").iterdir():
                    pending.extend(map(int, (task / "children").read_text().split()))
            except FileNotFoundError:
                continue  # A child can exit between enumeration and sampling.
        with self.lock:
            self.samples += 1
            if resident > self.peak_bytes:
                self.peak_bytes = resident
                self.peaks.append(
                    {
                        "elapsed_s": time.perf_counter() - self.started,
                        "bytes": resident,
                        "pids": sorted(visited),
                    }
                )
        if resident > HOST_BYTES:
            self.exhausted.set()

    def _monitor(self):
        while not self.finished.is_set():
            try:
                self._sample()
            except (OSError, ValueError):
                with self.lock:
                    self.errors += 1
                self.exhausted.set()
            self.finished.wait(0.02)

    def snapshot(self):
        with self.lock:
            return {
                "resource_pass": not self.exhausted.is_set()
                and self.errors == 0
                and self.peak_bytes <= HOST_BYTES,
                "peak_host_bytes": self.peak_bytes,
                "samples": self.samples,
                "errors": self.errors,
                "peak_observations": list(self.peaks),
                "scope": "harness and recursively enumerated live descendants, VmHWM sum",
                "sampling_interval_s": 0.02,
            }

    def close(self):
        self.finished.set()
        self.thread.join()
        return self.snapshot()


def _child_limits(limits, gpu, core=None):
    if core is not None:
        os.sched_setaffinity(0, {core})
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits["file_bytes"],) * 2)
    if not gpu:
        resource.setrlimit(resource.RLIMIT_AS, (limits["address_bytes"],) * 2)
    # RLIMIT_CPU is cumulative over a persistent process. The parent enforces
    # each request's 110-second CPU delta, including unsuccessful requests.


class Worker:
    """One real persistent process, one private transient solve at a time."""

    def __init__(self, binary, directory, limits, gpu, core):
        self.limits = limits
        self.directory = directory
        self.core = core
        self.log = (directory / "worker.log").open("wb")
        self.process = subprocess.Popen(
            [str(binary), "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            env={
                "PATH": "",
                "LC_ALL": "C",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
            start_new_session=True,
            preexec_fn=lambda: _child_limits(limits, gpu, core),
        )
        self.observed_affinity = sorted(os.sched_getaffinity(self.process.pid))
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b""
        self.requests = 0

    def stop(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
        self.process.wait()

    def request(self, source, raw, stats, exhausted):
        begin = time.perf_counter()
        try:
            initial_cpu = process_usage(self.process.pid)[0]
        except OSError:
            initial_cpu = 0
        self.last_cpu = initial_cpu
        self.requests += 1
        self.last_process = {
            "status": "internal_failure",
            "worker_pid": self.process.pid,
            "worker_request": self.requests,
            "input": str(source.resolve()),
        }
        try:
            self.last_process = self._request(source, raw, stats, exhausted)
            return self.last_process
        except (ValueError, OSError) as exc:
            try:
                final_cpu = process_usage(self.process.pid)[0]
            except OSError:
                final_cpu = initial_cpu
            self.last_process.update(
                status=str(exc).split(":", 1)[0],
                wall_s=time.perf_counter() - begin,
                cpu_s=max(0, max(final_cpu, self.last_cpu) - initial_cpu),
                exit_code=self.process.poll(),
            )
            raise

    def _request(self, source, raw, stats, exhausted):
        if self.process.poll() is not None:
            fail("resource_limit", "persistent worker unavailable; zero retry")
        paths = [str(path.resolve()) for path in (source, raw, stats)]
        if any("\n" in path or "\t" in path for path in paths):
            fail("unsupported_input", "worker path delimiter")
        begin = time.perf_counter()
        cpu_start, _ = process_usage(self.process.pid)
        log_start = self.log.tell()
        self.process.stdin.write(("\t".join(paths) + "\n").encode())
        self.process.stdin.flush()
        while b"\n" not in self.buffer:
            if exhausted.is_set():
                self.stop()
                fail("resource_limit", "aggregate host-memory limit")
            if time.perf_counter() - begin > self.limits["wall_s"]:
                self.stop()
                fail("timeout", "per-job wall budget")
            try:
                cpu, _ = process_usage(self.process.pid)
            except OSError:
                cpu = cpu_start
            self.last_cpu = cpu
            if cpu - cpu_start > self.limits["cpu_s"]:
                self.stop()
                fail("resource_limit", "per-job CPU budget")
            if self.selector.select(timeout=0.02):
                block = os.read(self.process.stdout.fileno(), 8192)
                if not block:
                    code = self.process.poll()
                    fail(
                        "resource_limit"
                        if code in {-signal.SIGXCPU, -signal.SIGXFSZ, -signal.SIGKILL}
                        else "numerical_failure",
                        "persistent worker closed protocol",
                    )
                self.buffer += block
                if len(self.buffer) > 64 * 1024:
                    self.stop()
                    fail("resource_limit", "worker response budget")
        line, self.buffer = self.buffer.split(b"\n", 1)
        cpu, _ = process_usage(self.process.pid)
        self.last_cpu = cpu
        if time.perf_counter() - begin > self.limits["wall_s"]:
            self.stop()
            fail("timeout", "completed request exceeded wall budget")
        if cpu - cpu_start > self.limits["cpu_s"] or exhausted.is_set():
            self.stop()
            fail("resource_limit", "completed request exceeded CPU or memory budget")
        try:
            response = json.loads(line)
        except (ValueError, UnicodeError):
            fail("malformed_output", "worker response JSON")
        if not isinstance(response, dict) or response.get("input") != paths[0]:
            fail("provenance_mismatch", "crossed worker job identity")
        if type(response.get("exit_code")) is not int or (
            response.get("status") == "complete" and response["exit_code"] != 0
        ):
            fail("malformed_output", "worker response exit status")
        if response.get("status") != "complete":
            self.last_process["input"] = paths[0]
            self.last_process["reply_exit_code"] = response["exit_code"]
            status = response.get("status")
            if status == "typed_failure":
                with (self.directory / "worker.log").open("rb") as stream:
                    stream.seek(log_start)
                    diagnostic = stream.read(64 * 1024)
                status, diagnostic_tag = qualification.cpu_failure_status(
                    diagnostic, "numerical_failure"
                )
                self.last_process["diagnostic_tag"] = diagnostic_tag
                self.last_process["diagnostic_sha256"] = study.sha(diagnostic)
                self.last_process["diagnostic_prefix_bytes"] = len(diagnostic)
            fail(
                status if status in FAILURES else "numerical_failure",
                "worker rejected job",
            )
        return {
            "status": "ok",
            "input": paths[0],
            "exit_code": 0,
            "wall_s": time.perf_counter() - begin,
            "cpu_s": max(0, cpu - cpu_start),
            "worker_pid": self.process.pid,
            "worker_request": self.requests,
        }

    def close(self):
        self.stop()
        self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()
        data = (self.directory / "worker.log").read_bytes()
        return {
            "pid": self.process.pid,
            "requests": self.requests,
            "cpu_affinity": self.observed_affinity,
            "diagnostic_sha256": study.sha(data),
            "diagnostic_bytes": len(data),
        }


class PersistentPool:
    def __init__(self, binary, directory, limits, gpu, count):
        if count not in CPU_WORKERS or type(count) is not int:
            fail("unsupported_input", "worker count must be one, four or sixteen")
        if not set(CPU_AFFINITY[:count]).issubset(os.sched_getaffinity(0)):
            fail("unsupported_input", "frozen physical-core affinity unavailable")
        self.workers = []
        self.available = queue.Queue()
        self.exhausted = threading.Event()
        self.finished = threading.Event()
        self.peak_host_bytes = 0
        self.device_peaks = {}
        self.device_lock = threading.Lock()
        self.device_sampled_peak_bytes = 0
        self.device_samples = 0
        self.device_baseline_bytes = 0
        self.device_observations = []
        self.monitor_errors = 0
        self.gpu = gpu
        self.device_tool = accelerator_executable() if gpu else None
        start = time.perf_counter()
        if gpu:
            self.device_sampled_peak_bytes = device_resident_bytes(self.device_tool)
            self.device_baseline_bytes = self.device_sampled_peak_bytes
            self.device_samples = 1
            self.device_observations.append(self.device_baseline_bytes)
        try:
            for index in range(count):
                worker_directory = directory / f"worker-{index}"
                worker_directory.mkdir(parents=True)
                worker = Worker(
                    binary, worker_directory, limits, gpu, CPU_AFFINITY[index]
                )
                self.workers.append(worker)
                self.available.put(worker)
        except BaseException:
            for worker in self.workers:
                worker.close()
            raise
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=count)
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()
        self.setup_s = time.perf_counter() - start

    def _monitor(self):
        next_device_sample = 0
        while not self.finished.is_set():
            if (
                _INVOCATION_MONITOR is not None
                and _INVOCATION_MONITOR.exhausted.is_set()
            ):
                self.exhausted.set()
            resident = 0
            for pid in [os.getpid()] + [w.process.pid for w in self.workers]:
                try:
                    resident += process_usage(pid)[1]
                    highwater = [
                        line
                        for line in Path(f"/proc/{pid}/status").read_text().splitlines()
                        if line.startswith("VmHWM:")
                    ]
                    if highwater:
                        resident += max(
                            0,
                            int(highwater[0].split()[1]) * 1024 - process_usage(pid)[1],
                        )
                except OSError:
                    pass
            self.peak_host_bytes = max(self.peak_host_bytes, resident)
            if resident > HOST_BYTES:
                self.exhausted.set()
            if self.gpu and time.perf_counter() >= next_device_sample:
                try:
                    device = device_resident_bytes(self.device_tool)
                    self.device_sampled_peak_bytes = max(
                        self.device_sampled_peak_bytes, device
                    )
                    self.device_samples += 1
                    self.device_observations.append(device)
                    if max(0, device - self.device_baseline_bytes) > DEVICE_BYTES:
                        self.exhausted.set()
                except (ValueError, OSError, subprocess.TimeoutExpired):
                    self.monitor_errors += 1
                    self.exhausted.set()
                next_device_sample = time.perf_counter() + 0.1
            self.finished.wait(0.02)

    def close(self):
        self.executor.shutdown(wait=True)
        self.finished.set()
        self.monitor.join()
        return [worker.close() for worker in self.workers]


def pool_resources(pool):
    return {
        "resource_pass": not pool.exhausted.is_set() and pool.monitor_errors == 0,
        "peak_host_bytes": pool.peak_host_bytes,
        "peak_device_bytes_upper_bound": sum(pool.device_peaks.values()),
        "device_baseline_bytes": pool.device_baseline_bytes,
        "device_sampled_peak_bytes": pool.device_sampled_peak_bytes,
        "device_incremental_peak_bytes": max(
            0, pool.device_sampled_peak_bytes - pool.device_baseline_bytes
        ),
        "device_samples": pool.device_samples,
        "device_observations_bytes": pool.device_observations,
        "monitor_errors": pool.monitor_errors,
    }


def validate_gpu_telemetry(value, expected_input=None):
    if (
        not isinstance(value, dict)
        or value.get("schema") != "emi03-cuda-v1"
        or value.get("backend") != "cuda-fp64"
        or not isinstance(value.get("job_id"), str)
        or not Path(value["job_id"]).is_absolute()
        or expected_input is not None
        and value["job_id"] != expected_input
        or type(value.get("gpu_fallbacks")) is not int
        or value["gpu_fallbacks"] != 0
        or type(value.get("peak_device_bytes")) is not int
        or not 0 < value["peak_device_bytes"] <= DEVICE_WORKER_BYTES
        or type(value.get("successful_solves")) is not int
        or value["successful_solves"] < 1
        or type(value.get("expression_batches")) is not int
        or value["expression_batches"] < 1
        or type(value.get("maximum_device_bytes")) is not int
        or value["maximum_device_bytes"] != DEVICE_WORKER_BYTES
        or any(
            type(value.get(key)) is not int or value[key] != 0
            for key in (
                "current_device_bytes",
                "outstanding_device_bytes",
                "live_factorizations",
                "cleanup_failures",
                "allocation_failures",
            )
        )
    ):
        fail("resource_limit", "missing GPU execution or memory/fallback gate")
    study.encoded(value)


def compare_pair(table, record, reference_table, reference):
    if record["fixture"] != reference["fixture"]:
        fail("provenance_mismatch", "fixture differs from immutable CPU reference")
    keys = (
        "candidate",
        "corner",
        "level",
        "max_step_s",
        "sample_step_s",
        "reference_version",
        "deck_sha256",
        "import",
    )
    if any(record.get(key) != reference.get(key) for key in keys):
        fail("provenance_mismatch", "physical, model or numerical input changed")
    if record["fixture"] == "dpt":
        result = study.dpt_compare(
            record["metrics"], reference["metrics"], record["id"]
        )
        grid, _ = signals.resample(
            reference_table[:, 0],
            reference_table[:, 1],
            0,
            16e-6,
            record["sample_step_s"],
        )
        waves = {}
        for index, name in enumerate(circuits.DPT_NAMES[1:], 1):
            compared = signals.waveform_comparison(
                study.np.interp(grid, table[:, 0], table[:, index]),
                study.np.interp(grid, reference_table[:, 0], reference_table[:, index]),
                absolute=10.0 if name == "v(d)" else 0.5 if name == "v(g)" else 0.02,
                relative=0 if name.startswith("v(") else 0.02,
            )
            waves[name] = compared
            result["pass"] &= compared["passed"]
        result["waveforms"] = waves
    else:
        result = metrics.compare(
            table,
            reference_table,
            circuits.STUDY_NAMES,
            record["sample_step_s"],
            record["corner"]["bus_v"],
        )
    result["classification_agrees"] = record["status"] == reference["status"]
    result["pass"] = bool(result["pass"] and result["classification_agrees"])
    return result


def run_job(spec, binary_sha, archive, out, pool, gpu, references, reference_out):
    submitted = spec.pop("submitted")
    record = {
        **spec,
        "status": "internal_failure",
        "attempts": 1,
        "binary_sha256": binary_sha,
        "backend": "cuda-fp64" if gpu else "cpu-klu",
    }
    dest = out / "jobs" / spec["id"]
    dest.mkdir(parents=True)
    begin = time.perf_counter()
    record["queue_s"] = begin - submitted
    worker = pool.available.get()
    request_attempted = False
    try:
        if pool.exhausted.is_set():
            fail("resource_limit", "aggregate host-memory gate")
        with tempfile.TemporaryDirectory(prefix="emi03-job-") as temporary:
            directory = Path(temporary)
            source = qualification.source_deck(spec)
            flat, provenance = importer.import_archive(archive, source)
            (directory / "input.cir").write_text(flat)
            record["request_input"] = str(directory / "input.cir")
            record.update(
                deck_sha256=study.sha(source.encode()),
                flattened_sha256=study.sha(flat.encode()),
                **{"import": provenance},
            )
            study.write_json(dest / "import.json", provenance)
            raw_path, stats_path = (
                directory / "output.raw",
                directory / "statistics.json",
            )
            try:
                request_attempted = True
                record["process"] = worker.request(
                    directory / "input.cir", raw_path, stats_path, pool.exhausted
                )
                study.write_json(dest / "process.json", record["process"])
                if not raw_path.is_file():
                    fail("missing_output", "no waveform")
                if raw_path.stat().st_size > pool.workers[0].limits["file_bytes"]:
                    fail("resource_limit", "raw byte budget")
                raw = raw_path.read_bytes()
                if not raw.startswith(
                    ("Title: " + source.splitlines()[0].lower() + "\n").encode()
                ):
                    fail("provenance_mismatch", "waveform belongs to another job")
                names = (
                    circuits.DPT_NAMES
                    if spec["fixture"] == "dpt"
                    else circuits.STUDY_NAMES
                )
                table = signals.parse_raw(
                    raw,
                    names,
                    16e-6 if spec["fixture"] == "dpt" else 200e-6,
                    spec["max_step_s"],
                )
                with _ARCHIVE_LOCK:
                    header, index = study.store_raw(out, raw)
                (dest / "raw.header").write_bytes(header)
                study.write_json(dest / "raw.json", index)
                record.update(raw_sha256=index["raw_sha256"], raw_points=len(table))
                stats = qualification.read_json_bounded(stats_path)
                qualification.validate_statistics(
                    stats, len(table), len(names), provenance["mna_unknowns"], len(raw)
                )
                record["statistics"] = stats
                study.write_json(dest / "statistics.json", stats)
                if gpu:
                    telemetry = qualification.read_json_bounded(
                        Path(str(stats_path) + ".gpu.json")
                    )
                    record["gpu"] = telemetry
                    study.write_json(dest / "gpu.json", telemetry)
                    with pool.device_lock:
                        peak = (
                            telemetry.get("peak_device_bytes")
                            if isinstance(telemetry, dict)
                            else None
                        )
                        if type(peak) is int and peak >= 0:
                            pool.device_peaks[worker.process.pid] = max(
                                pool.device_peaks.get(worker.process.pid, 0), peak
                            )
                        if sum(pool.device_peaks.values()) > DEVICE_BYTES:
                            pool.exhausted.set()
                            fail("resource_limit", "aggregate device allocation limit")
                    validate_gpu_telemetry(telemetry, record["request_input"])
                if spec["fixture"] == "dpt":
                    record["metrics"] = metrics.dpt(table, spec["sample_step_s"])
                    record["status"] = (
                        "qualified" if record["metrics"]["pass"] else "accuracy_failure"
                    )
                else:
                    record["metrics"], spectrum = metrics.evaluate(
                        table, spec["candidate"], spec["corner"], spec["sample_step_s"]
                    )
                    record["status"] = record["metrics"]["status"]
                if references is not None:
                    validation_start = time.perf_counter()
                    reference = references.get(spec.get("reference_id", spec["id"]))
                    if reference is None or reference["status"] not in study.VALID | {
                        "qualified"
                    }:
                        fail("missing_output", "fresh CPU oracle unavailable")
                    reference_table = study.restore(reference_out, reference)
                    record["cpu_validation"] = compare_pair(
                        table, record, reference_table, reference
                    )
                    record["cpu_validation_s"] = time.perf_counter() - validation_start
                    record["reference_raw_sha256"] = reference["raw_sha256"]
                    if not record["cpu_validation"]["pass"]:
                        fail(
                            "accuracy_failure", "fresh per-result CPU differential gate"
                        )
                if spec["fixture"] == "ensemble":
                    (dest / "spectra.f64").write_bytes(spectrum.astype("<f8").tobytes())
                    study.write_json(
                        dest / "spectra.json",
                        {
                            "shape": list(spectrum.shape),
                            "sha256": study.sha((dest / "spectra.f64").read_bytes()),
                        },
                    )
            finally:
                # Even rejected/non-finite/truncated output remains evidence. Never
                # retain the flattened proprietary model or model-bearing stderr.
                if raw_path.is_file() and "raw_sha256" not in record:
                    size = raw_path.stat().st_size
                    record["failed_raw_bytes"] = size
                    if size <= pool.workers[0].limits["file_bytes"]:
                        data = raw_path.read_bytes()
                        record["failed_raw_sha256"] = study.sha(data)
                        (dest / "invalid.raw.gz").write_bytes(
                            gzip.compress(data, compresslevel=1, mtime=0)
                        )
                if gpu and "gpu" not in record:
                    sidecar = Path(str(stats_path) + ".gpu.json")
                    if sidecar.is_file():
                        try:
                            telemetry = qualification.read_json_bounded(sidecar)
                            record["gpu"] = telemetry
                            study.write_json(dest / "gpu.json", telemetry)
                            peak = (
                                telemetry.get("peak_device_bytes")
                                if isinstance(telemetry, dict)
                                else None
                            )
                            if type(peak) is int and peak >= 0:
                                with pool.device_lock:
                                    pool.device_peaks[worker.process.pid] = max(
                                        pool.device_peaks.get(worker.process.pid, 0),
                                        peak,
                                    )
                            else:
                                record["gpu_telemetry_incomplete"] = True
                                pool.exhausted.set()
                        except ValueError:
                            record["gpu_telemetry_incomplete"] = True
                            pool.exhausted.set()
                    else:
                        record["gpu_telemetry_missing"] = True
                        pool.exhausted.set()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        status = str(exc).split(":", 1)[0]
        record["status"] = status if status in FAILURES else "internal_failure"
        record["error"] = record["status"] + ": complete job rejected"
        if "metrics" in record:
            record["rejected_metrics"] = record.pop("metrics")
        if (
            request_attempted
            and "process" not in record
            and hasattr(worker, "last_process")
        ):
            record["process"] = worker.last_process
    finally:
        pool.available.put(worker)
    record["elapsed_s"] = time.perf_counter() - begin
    record["request_attempted"] = request_attempted
    if "process" in record:
        study.write_json(dest / "process.json", record["process"])
    record["completion_latency_s"] = time.perf_counter() - submitted
    record["files"] = {
        path.name: study.sha(path.read_bytes())
        for path in sorted(dest.iterdir())
        if path.is_file()
    }
    study.write_json(dest / "result.json", record)
    return record


def failed_completion(spec, directory, status, binary_sha, gpu):
    """Finalize a failure without erasing already closed evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        **spec,
        "status": status,
        "attempts": 1,
        "binary_sha256": binary_sha,
        "backend": "cuda-fp64" if gpu else "cpu-klu",
        "error": status + ": executor did not return the scheduled terminal job",
    }
    for filename, field in (
        ("import.json", "import"),
        ("process.json", "process"),
        ("statistics.json", "statistics"),
        ("gpu.json", "gpu"),
    ):
        if (directory / filename).is_file():
            record[field] = qualification.read_json_bounded(directory / filename)
    if "import" in record:
        record["deck_sha256"] = record["import"]["source_deck_sha256"]
        record["flattened_sha256"] = record["import"]["flat_deck_sha256"]
    if "process" in record:
        record["request_input"] = record["process"].get("input")
    if "gpu" in record:
        record.setdefault("request_input", record["gpu"].get("job_id"))
    if (directory / "raw.json").is_file():
        index = qualification.read_json_bounded(directory / "raw.json")
        record["raw_sha256"] = index["raw_sha256"]
        record["raw_points"] = len(study.restore(directory.parent.parent, record))
    record["files"] = {
        path.name: study.sha(path.read_bytes())
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "result.json"
    }
    study.write_json(directory / "result.json", record)
    return record


def complete_study(
    specs, binary_sha, archive, out, pool, gpu, references, reference_out
):
    out.mkdir(parents=True)
    started = time.perf_counter()
    pending = {
        pool.executor.submit(
            run_job,
            {**spec, "submitted": time.perf_counter()},
            binary_sha,
            archive,
            out,
            pool,
            gpu,
            references,
            reference_out,
        ): spec
        for spec in specs
    }
    done = {}
    for future in concurrent.futures.as_completed(pending):
        spec = pending[future]
        try:
            record = future.result()
        except Exception:
            # Infrastructure failure cannot drop a job from the denominator.
            record = failed_completion(
                spec, out / "jobs" / spec["id"], "internal_failure", binary_sha, gpu
            )
        if record.get("id") != spec["id"]:
            record = failed_completion(
                spec, out / "jobs" / spec["id"], "provenance_mismatch", binary_sha, gpu
            )
        done[spec["id"]] = record
        study.write_json(
            out / "progress.json", [done[s["id"]] for s in specs if s["id"] in done]
        )
        print(json.dumps({"job": spec["id"], "status": record["status"]}), flush=True)
    records = [done[spec["id"]] for spec in specs]
    if gpu:
        try:
            device = device_resident_bytes(pool.device_tool)
            pool.device_sampled_peak_bytes = max(pool.device_sampled_peak_bytes, device)
            pool.device_samples += 1
            pool.device_observations.append(device)
            if max(0, device - pool.device_baseline_bytes) > DEVICE_BYTES:
                pool.exhausted.set()
        except (ValueError, OSError, subprocess.TimeoutExpired):
            pool.monitor_errors += 1
            pool.exhausted.set()
    success = (
        len(records) == len(specs)
        and not pool.exhausted.is_set()
        and all(row["status"] in study.VALID | {"qualified"} for row in records)
        and (
            references is None
            or all(row.get("cpu_validation", {}).get("pass") is True for row in records)
        )
    )
    result = {
        "schema": SCHEMA,
        "pass": bool(success),
        "jobs": records,
        "expected_jobs": len(specs),
        "completed_jobs": len(records),
        "peak_host_bytes": pool.peak_host_bytes,
    }
    if gpu:
        result.update(
            device_sampled_peak_bytes=pool.device_sampled_peak_bytes,
            device_baseline_bytes=pool.device_baseline_bytes,
            device_incremental_peak_bytes=max(
                0, pool.device_sampled_peak_bytes - pool.device_baseline_bytes
            ),
            device_samples=pool.device_samples,
            peak_device_bytes_upper_bound=sum(pool.device_peaks.values()),
        )
    # The observation ends after required raw/spectral/terminal output closes.
    # The scalar clock observation itself is retained separately for audit.
    study.write_json(out / "study.json", result)
    result["complete_study_s"] = time.perf_counter() - started
    study.write_json(
        out / "timing.json", {"complete_study_s": result["complete_study_s"]}
    )
    return result


def nearest_rank(values, percentile=0.95):
    if not values or any(
        type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values
    ):
        fail("non_finite", "positive finite complete-study observations required")
    return sorted(values)[math.ceil(percentile * len(values)) - 1]


def performance_gate(
    modes, qualification_s, cpu_qualification_s=None, common_setup_s=0
):
    """Finite replay evidence only; failures and warmups cannot be filtered out."""
    expected = {
        (engine, workers, size)
        for engine, counts in (("cpu", CPU_WORKERS), ("gpu", GPU_WORKERS))
        for workers in counts
        for size in ENSEMBLES
    }
    keys = [(row.get("engine"), row.get("workers"), row.get("size")) for row in modes]
    complete = len(keys) == len(expected) and set(keys) == expected
    valid = complete and all(
        len(row.get("studies", [])) == WARMUPS + MEASURED
        and row.get("peak_host_bytes", HOST_BYTES + 1) <= HOST_BYTES
        and (
            row["engine"] == "cpu"
            or row.get("peak_device_bytes_upper_bound", DEVICE_BYTES + 1)
            <= DEVICE_BYTES
        )
        and row.get("resource_pass") is True
        and (
            row["engine"] == "cpu"
            or row.get("device_incremental_peak_bytes", DEVICE_BYTES + 1)
            <= DEVICE_BYTES
        )
        and all(
            s.get("pass") is True
            and s.get("completed_jobs") == row["size"]
            and s.get("expected_jobs") == row["size"]
            for s in row["studies"]
        )
        for row in modes
    )
    if not valid:
        return {
            "pass": False,
            "status": "incomplete_or_failed_study",
            "comparisons": [],
            "dispatch_authorized": False,
        }
    rows = []
    for row in modes:
        times = [s["complete_study_s"] for s in row["studies"]]
        nearest_rank(times)
        measured = times[WARMUPS:]
        rows.append(
            {
                **{key: row[key] for key in ("engine", "workers", "size")},
                "median_s": statistics.median(measured),
                "empirical_p95_s": nearest_rank(measured),
                "cold_s": (
                    qualification_s
                    if row["engine"] == "gpu" or cpu_qualification_s is None
                    else cpu_qualification_s
                )
                + common_setup_s
                + row["setup_s"]
                + times[0],
            }
        )
    comparisons = []
    for size in ENSEMBLES:
        cpus = [row for row in rows if row["engine"] == "cpu" and row["size"] == size]
        best_median = min(cpus, key=lambda row: row["median_s"])
        best_p95 = min(cpus, key=lambda row: row["empirical_p95_s"])
        best_cold = min(cpus, key=lambda row: row["cold_s"])
        for gpu in [
            row for row in rows if row["engine"] == "gpu" and row["size"] == size
        ]:
            median = best_median["median_s"] / gpu["median_s"]
            p95 = best_p95["empirical_p95_s"] / gpu["empirical_p95_s"]
            comparisons.append(
                {
                    "size": size,
                    "gpu_workers": gpu["workers"],
                    "cpu_median_workers": best_median["workers"],
                    "cpu_p95_workers": best_p95["workers"],
                    "cpu_cold_workers": best_cold["workers"],
                    "median_speedup": median,
                    "empirical_p95_speedup": p95,
                    "cold_speedup": best_cold["cold_s"] / gpu["cold_s"],
                    "pass": median >= 2
                    and p95 >= 1.5
                    and gpu["cold_s"] <= best_cold["cold_s"],
                }
            )
    return {
        "pass": all(
            any(row["pass"] for row in comparisons if row["size"] == size)
            for size in ENSEMBLES
        ),
        "status": "complete",
        "comparisons": comparisons,
        "summaries": rows,
        "dispatch_authorized": False,
    }


def audit_gpu_records(
    out, records, specs, archive, binary_sha, references, reference_out, gpu=True
):
    """Reconstruct each accepted job, including every raw chunk and spectral bin."""
    if (
        not isinstance(records, list)
        or len(records) != len(specs)
        or [r.get("id") for r in records] != [s["id"] for s in specs]
    ):
        fail("missing_output", "complete ordered job records required")
    allowed = {
        "import.json",
        "raw.header",
        "raw.json",
        "statistics.json",
        "process.json",
        "gpu.json",
        "spectra.f64",
        "spectra.json",
        "invalid.raw.gz",
    }
    for record, spec in zip(records, specs, strict=True):
        if (
            any(record.get(key) != value for key, value in spec.items())
            or record.get("attempts") != 1
        ):
            fail("provenance_mismatch", "changed scheduled job identity")
        directory = out / "jobs" / spec["id"]
        if (
            directory.is_symlink()
            or qualification.read_json_bounded(directory / "result.json") != record
        ):
            fail("provenance_mismatch", "terminal record changed")
        files = record.get("files", {})
        if not isinstance(files, dict) or set(files) - allowed:
            fail("malformed_output", "unknown evidence file")
        if {
            path.name for path in directory.iterdir() if path.name != "result.json"
        } != set(files):
            fail("missing_output", "missing or unrecorded evidence")
        for name, digest in files.items():
            path = directory / name
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size > signals.MAX_RAW_BYTES
            ):
                fail("resource_limit", "invalid evidence path or byte count")
            value = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    value.update(block)
            if value.hexdigest() != digest:
                fail("provenance_mismatch", "evidence hash changed")
        successful = record.get("status") in study.VALID | {"qualified"}
        if not successful and record.get("status") not in FAILURES:
            fail("malformed_output", "unknown terminal status")
        if "raw.json" in files:
            table = study.restore(out, record)
            index = qualification.read_json_bounded(directory / "raw.json")
            if index["raw_sha256"] != record.get("raw_sha256"):
                fail("provenance_mismatch", "raw identity changed")
        if not successful:
            audit_failed_record(
                out,
                directory,
                record,
                spec,
                files,
                archive,
                binary_sha,
                references,
                reference_out,
                gpu,
            )
            continue
        process = record.get("process")
        if qualification.read_json_bounded(directory / "process.json") != process:
            fail("provenance_mismatch", "accepted process accounting changed")
        limits = study.selected_manifest("emi01-v2")["limits"]
        if (
            not isinstance(process, dict)
            or process.get("status") != "ok"
            or type(process.get("exit_code")) is not int
            or process["exit_code"] != 0
            or any(
                type(process.get(key)) not in (int, float)
                or not math.isfinite(process[key])
                or not 0 <= process[key] <= limits[limit]
                for key, limit in (("wall_s", "wall_s"), ("cpu_s", "cpu_s"))
            )
            or any(
                type(process.get(key)) is not int or process[key] < 1
                for key in ("worker_pid", "worker_request")
            )
            or record.get("backend") != ("cuda-fp64" if gpu else "cpu-klu")
        ):
            fail("malformed_output", "accepted process accounting")
        source = qualification.source_deck(spec)
        flat, provenance = importer.import_archive(archive, source)
        if (
            record.get("binary_sha256") != binary_sha
            or record.get("deck_sha256") != study.sha(source.encode())
            or record.get("flattened_sha256") != study.sha(flat.encode())
            or record.get("import") != provenance
            or qualification.read_json_bounded(directory / "import.json") != provenance
            or "raw.json" not in files
            or record.get("raw_points") != len(table)
        ):
            fail("provenance_mismatch", "accepted job provenance changed")
        header = (directory / "raw.header").read_bytes()
        if not header.startswith(
            ("Title: " + source.splitlines()[0].lower() + "\n").encode()
        ):
            fail("provenance_mismatch", "accepted waveform title changed")
        stats = qualification.read_json_bounded(directory / "statistics.json")
        qualification.validate_statistics(
            stats,
            len(table),
            table.shape[1],
            provenance["mna_unknowns"],
            len(header) + index["payload_bytes"],
        )
        if record.get("statistics") != stats:
            fail("provenance_mismatch", "solver statistics changed")
        if gpu:
            telemetry = qualification.read_json_bounded(directory / "gpu.json")
            validate_gpu_telemetry(telemetry, record.get("request_input"))
            if telemetry != record.get("gpu"):
                fail("provenance_mismatch", "GPU execution telemetry changed")
        if spec["fixture"] == "dpt":
            measured = metrics.dpt(table, spec["sample_step_s"])
            status = "qualified" if measured["pass"] else "accuracy_failure"
        else:
            measured, spectrum = metrics.evaluate(
                table, spec["candidate"], spec["corner"], spec["sample_step_s"]
            )
            status = measured["status"]
            spectrum_bytes = spectrum.astype("<f8").tobytes()
            if (
                directory / "spectra.f64"
            ).read_bytes() != spectrum_bytes or qualification.read_json_bounded(
                directory / "spectra.json"
            ) != {"shape": list(spectrum.shape), "sha256": study.sha(spectrum_bytes)}:
                fail("provenance_mismatch", "spectral recomputation changed")
        if record.get("metrics") != measured or record["status"] != status:
            fail("provenance_mismatch", "measurement recomputation changed")
        reference = references[spec.get("reference_id", spec["id"])]
        comparison = compare_pair(
            table, record, study.restore(reference_out, reference), reference
        )
        if (
            record.get("cpu_validation") != comparison
            or comparison["pass"] is not True
            or record.get("reference_raw_sha256") != reference["raw_sha256"]
        ):
            fail("provenance_mismatch", "fresh CPU differential recomputation changed")
    return records


def audit_failed_record(
    out,
    directory,
    record,
    spec,
    files,
    archive,
    binary_sha,
    references,
    reference_out,
    gpu,
):
    """Failure evidence remains bound to its inputs and available raw telemetry."""
    if record.get("binary_sha256") != binary_sha or record.get("backend") != (
        "cuda-fp64" if gpu else "cpu-klu"
    ):
        fail("provenance_mismatch", "failed job binary/backend changed")
    source = qualification.source_deck(spec)
    if "process.json" in files and qualification.read_json_bounded(
        directory / "process.json"
    ) != record.get("process"):
        fail("provenance_mismatch", "failed process accounting changed")
    if "import.json" in files or "import" in record:
        flat, provenance = importer.import_archive(archive, source)
        if (
            record.get("import") != provenance
            or "import.json" not in files
            or qualification.read_json_bounded(directory / "import.json") != provenance
            or record.get("deck_sha256") != study.sha(source.encode())
            or record.get("flattened_sha256") != study.sha(flat.encode())
        ):
            fail("provenance_mismatch", "failed job physical/model input changed")
    if ("gpu" in record) != ("gpu.json" in files):
        fail("provenance_mismatch", "failed GPU telemetry missing")
    if "gpu.json" in files:
        telemetry = qualification.read_json_bounded(directory / "gpu.json")
        if (
            not isinstance(telemetry, dict)
            or telemetry != record["gpu"]
            or telemetry.get("schema") != "emi03-cuda-v1"
            or telemetry.get("backend") != "cuda-fp64"
            or telemetry.get("job_id") != record.get("request_input")
            or type(telemetry.get("peak_device_bytes")) is not int
            or telemetry["peak_device_bytes"] < 0
        ):
            fail("provenance_mismatch", "failed GPU telemetry identity changed")
    if "invalid.raw.gz" in files:
        with gzip.open(directory / "invalid.raw.gz", "rb") as stream:
            raw = stream.read(signals.MAX_RAW_BYTES + 1)
        if len(raw) > signals.MAX_RAW_BYTES:
            fail("resource_limit", "failed raw reconstruction byte budget")
        if len(raw) != record.get("failed_raw_bytes") or study.sha(raw) != record.get(
            "failed_raw_sha256"
        ):
            fail("provenance_mismatch", "failed raw identity changed")
    if "raw.json" not in files:
        if "rejected_metrics" in record or "cpu_validation" in record:
            fail("missing_output", "rejected measurement raw unavailable")
        return
    table = study.restore(out, record)
    header = (directory / "raw.header").read_bytes()
    index = qualification.read_json_bounded(directory / "raw.json")
    if not header.startswith(
        ("Title: " + source.splitlines()[0].lower() + "\n").encode()
    ) or record.get("raw_points") != len(table):
        fail("provenance_mismatch", "failed waveform title or point count changed")
    if "statistics.json" in files:
        stats = qualification.read_json_bounded(directory / "statistics.json")
        if stats != record.get("statistics"):
            fail("provenance_mismatch", "failed job statistics changed")
        qualification.validate_statistics(
            stats,
            len(table),
            table.shape[1],
            record["import"]["mna_unknowns"],
            len(header) + index["payload_bytes"],
        )
    if "rejected_metrics" not in record:
        if record["status"] == "accuracy_failure":
            fail("missing_output", "accuracy rejection measurements missing")
        return
    if spec["fixture"] == "dpt":
        measured = metrics.dpt(table, spec["sample_step_s"])
        status = "qualified" if measured["pass"] else "accuracy_failure"
    else:
        measured, _ = metrics.evaluate(
            table, spec["candidate"], spec["corner"], spec["sample_step_s"]
        )
        status = measured["status"]
    if record["rejected_metrics"] != measured:
        fail("provenance_mismatch", "rejected measurements changed")
    if "cpu_validation" in record:
        reference = references[spec.get("reference_id", spec["id"])]
        compared = compare_pair(
            table,
            {**record, "metrics": measured, "status": status},
            study.restore(reference_out, reference),
            reference,
        )
        if (
            record["cpu_validation"] != compared
            or record.get("reference_raw_sha256") != reference["raw_sha256"]
        ):
            fail("provenance_mismatch", "rejected differential changed")


def audit_resources(resource_record, records, workers, gpu):
    device_peaks = {}
    requests = {}
    for record in records:
        process = record.get("process", {})
        pid = process.get("worker_pid")
        if pid is not None:
            requests.setdefault(pid, []).append(process.get("worker_request"))
            if "gpu" in record:
                value = (
                    record["gpu"].get("peak_device_bytes")
                    if isinstance(record["gpu"], dict)
                    else None
                )
                if type(value) is int and value >= 0:
                    device_peaks[pid] = max(device_peaks.get(pid, 0), value)
    if sum(device_peaks.values()) != resource_record.get(
        "peak_device_bytes_upper_bound"
    ):
        fail("provenance_mismatch", "aggregate native device allocation accounting")
    if not isinstance(workers, list) or len(
        {worker.get("pid") for worker in workers}
    ) != len(workers):
        fail("provenance_mismatch", "persistent worker identities")
    if set(requests) - {worker["pid"] for worker in workers}:
        fail("provenance_mismatch", "unknown worker process")
    for worker in workers:
        observed = requests.get(worker["pid"], [])
        if sorted(observed) != list(range(1, worker["requests"] + 1)):
            fail("provenance_mismatch", "persistent request completion sequence")
    observations = resource_record.get("device_observations_bytes")
    if (
        not isinstance(observations, list)
        or any(type(value) is not int or value < 0 for value in observations)
        or resource_record.get("device_samples") != len(observations)
    ):
        fail("malformed_output", "device residency observations")
    baseline = observations[0] if observations else 0
    peak = max(observations, default=0)
    if (
        gpu
        and not observations
        or resource_record.get("device_baseline_bytes") != baseline
        or resource_record.get("device_sampled_peak_bytes") != peak
        or resource_record.get("device_incremental_peak_bytes")
        != max(0, peak - baseline)
    ):
        fail("provenance_mismatch", "sampled device residency accounting")
    expected = (
        resource_record["peak_host_bytes"] <= HOST_BYTES
        and sum(device_peaks.values()) <= DEVICE_BYTES
        and max(0, peak - baseline) <= DEVICE_BYTES
        and resource_record["monitor_errors"] == 0
    )
    if resource_record.get("resource_pass") is True and not expected:
        fail("provenance_mismatch", "resource acceptance gate")
    return resource_record.get("resource_pass") is True


def audit_host_resources(value):
    if (
        not isinstance(value, dict)
        or type(value.get("samples")) is not int
        or value["samples"] < 1
        or type(value.get("errors")) is not int
        or value["errors"] < 0
        or type(value.get("peak_host_bytes")) is not int
        or value["peak_host_bytes"] < 1
        or not isinstance(value.get("peak_observations"), list)
        or not value["peak_observations"]
    ):
        fail("malformed_output", "whole-invocation host memory observations")
    previous_bytes, previous_time = 0, 0
    for observation in value["peak_observations"]:
        if (
            type(observation.get("bytes")) is not int
            or observation["bytes"] <= previous_bytes
            or type(observation.get("elapsed_s")) not in (int, float)
            or not math.isfinite(observation["elapsed_s"])
            or observation["elapsed_s"] < previous_time
            or not isinstance(observation.get("pids"), list)
            or not observation["pids"]
            or any(type(pid) is not int or pid < 1 for pid in observation["pids"])
        ):
            fail("malformed_output", "whole-invocation peak evidence")
        previous_bytes, previous_time = observation["bytes"], observation["elapsed_s"]
    expected_pass = previous_bytes <= HOST_BYTES and value["errors"] == 0
    if (
        value["peak_host_bytes"] != previous_bytes
        or value.get("resource_pass") is not expected_pass
    ):
        fail("provenance_mismatch", "whole-invocation memory gate")
    return expected_pass


def oracle_budget_check(cpu, external, limits):
    checks = []
    for lane, records in (("cpu", cpu), ("external", external)):
        for record in records:
            wall = record.get("process", {}).get("wall_s")
            checks.append(
                {
                    "engine": lane,
                    "id": record["id"],
                    "wall_s": wall,
                    "pass": type(wall) in (int, float)
                    and math.isfinite(wall)
                    and 0 <= wall <= limits["wall_s"],
                }
            )
    return {
        "pass": len(checks) == 60 and all(check["pass"] for check in checks),
        "checks": checks,
    }


def audit(out, archive, cpu_binary, gpu_binary, oracle_binary):
    invocation = qualification.read_json_bounded(out / "invocation.json")
    result = qualification.read_json_bounded(
        out / "qualification.json", 32 * 1024 * 1024
    )
    summary = qualification.read_json_bounded(out / "summary.json", 32 * 1024 * 1024)
    manifest, manifest_sha = study.load_manifest(
        study.manifest_path("emi01-v2"), "emi01-v2"
    )
    if (
        invocation.get("schema") != SCHEMA
        or invocation.get("sources") != identities()
        or invocation.get("manifest_sha256") != manifest_sha
        or invocation.get("cpu_sha256") != study.sha(cpu_binary.read_bytes())
        or invocation.get("gpu_sha256") != study.sha(gpu_binary.read_bytes())
        or invocation.get("ngspice_sha256") != study.check_elf(oracle_binary)
        or invocation.get("cpu_workers") != list(CPU_WORKERS)
        or invocation.get("gpu_workers") != list(GPU_WORKERS)
        or invocation.get("ensemble_sizes") != list(ENSEMBLES)
        or invocation.get("warmups") != WARMUPS
        or invocation.get("measured") != MEASURED
        or invocation.get("retries") != 0
        or invocation.get("job_limits") != manifest["limits"]
        or invocation.get("persistent_worker_cores") != list(CPU_AFFINITY)
        or invocation.get("qualification_cpu_affinity") != list(CPU_AFFINITY[:4])
    ):
        fail("provenance_mismatch", "frozen invocation inputs changed")
    with tempfile.TemporaryDirectory(prefix="emi03-audit-") as temporary:
        model = adapter.adapt_archive(archive, Path(temporary) / "model.lib")
    if invocation.get("model") != model:
        fail("provenance_mismatch", "model identity changed")
    baseline = out / "qualification"
    cpu = qualification.audit_records(
        baseline / "cpu", result["cpu_jobs"], manifest, archive, True
    )
    external = qualification.audit_records(
        baseline / "reference", result["reference_jobs"], manifest, archive, False
    )
    references = {record["id"]: record for record in cpu}
    gpu = audit_gpu_records(
        baseline / "gpu",
        result["gpu_jobs"],
        qualification.expected_specs(manifest),
        archive,
        invocation["gpu_sha256"],
        references,
        baseline / "cpu",
    )
    checks = {
        "oracle_budget": oracle_budget_check(cpu, external, manifest["limits"]),
        "cpu_refinement": study.qualify(
            baseline / "cpu", cpu, write=False, manifest=manifest
        ),
        "external_refinement": study.qualify(
            baseline / "reference", external, write=False, manifest=manifest
        ),
        "external_comparison": qualification.compare_jobs(
            baseline / "cpu", baseline / "reference", cpu, external
        ),
        "gpu_refinement": study.qualify(
            baseline / "gpu", gpu, write=False, manifest=manifest
        ),
    }
    finest = [
        record
        for record in gpu
        if record["fixture"] == "ensemble" and record["level"] == 2
    ]
    checks["reference_cases"] = study.reference_cases(finest, manifest)
    qualification_workers = qualification.read_json_bounded(
        baseline / "gpu-workers.json"
    )
    if [worker.get("cpu_affinity") for worker in qualification_workers] != [
        [cpu] for cpu in CPU_AFFINITY[:4]
    ]:
        fail("provenance_mismatch", "GPU qualification workers or affinity")
    resource_pass = audit_resources(
        result["gpu_resources"],
        gpu,
        qualification_workers,
        True,
    )
    passed = (
        audit_host_resources(result.get("invocation_host_resources"))
        and resource_pass
        and all(check["pass"] for check in checks.values())
        and all(
            row["status"] in study.VALID | {"qualified"}
            and row.get("cpu_validation", {}).get("pass") is True
            for row in gpu
        )
    )
    final_host_pass = audit_host_resources(summary.get("invocation_host_resources"))
    if (
        any(result.get(key) != value for key, value in checks.items())
        or result.get("pass") != passed
        or result.get("complete") is not True
        or result.get("ranking") != study.ranking(finest, passed, manifest)
        or summary.get("qualification_pass") != (passed and final_host_pass)
    ):
        fail("provenance_mismatch", "qualification recomputation changed")
    modes = summary.get("modes")
    if not isinstance(modes, list) or (not passed and modes):
        fail("provenance_mismatch", "performance after failed qualification")
    for mode in modes:
        label = f"{mode['engine']}-n{mode['size']}-w{mode['workers']}"
        if (
            qualification.read_json_bounded(
                out / "performance" / (label + "-mode.json")
            )
            != mode
        ):
            fail("provenance_mismatch", "mode accounting changed")
        mode_records = []
        for repeat, observed in enumerate(mode["studies"]):
            study_label = f"{label}-s{repeat}"
            directory = out / "performance" / study_label
            retained = qualification.read_json_bounded(
                directory / "study.json", 32 * 1024 * 1024
            )
            specs = replay_specs(manifest, mode["size"], study_label)
            records = audit_gpu_records(
                directory,
                retained["jobs"],
                specs,
                archive,
                invocation[mode["engine"] + "_sha256"],
                references,
                baseline / "cpu",
                mode["engine"] == "gpu",
            )
            mode_records.extend(records)
            timing = qualification.read_json_bounded(directory / "timing.json")
            if timing.get("complete_study_s") != observed.get("complete_study_s"):
                fail("provenance_mismatch", "closed-output study timing changed")
            if (
                retained.get("pass") != observed.get("pass")
                or retained.get("expected_jobs") != len(specs)
                or retained.get("completed_jobs") != len(records)
                or retained.get("pass") is True
                and any(row["status"] not in study.VALID for row in records)
            ):
                fail("provenance_mismatch", "performance completion accounting changed")
        audit_resources(
            mode, mode_records, mode["worker_records"], mode["engine"] == "gpu"
        )
        if [worker.get("cpu_affinity") for worker in mode["worker_records"]] != [
            [cpu] for cpu in CPU_AFFINITY[: mode["workers"]]
        ]:
            fail("provenance_mismatch", "actual persistent worker affinity")
    expected_gate = (
        performance_gate(
            modes,
            result["qualification_s"],
            result["cpu_qualification_s"],
            result["common_setup_s"],
        )
        if modes
        else {
            "pass": False,
            "status": "qualification_only" if passed else "qualification_failed",
            "comparisons": [],
            "dispatch_authorized": False,
        }
    )
    if not final_host_pass:
        expected_gate = {
            "pass": False,
            "status": "resource_limit",
            "comparisons": [],
            "dispatch_authorized": False,
        }
    if (
        summary.get("performance") != expected_gate
        or summary.get("dispatch_authorized") is not False
    ):
        fail("provenance_mismatch", "performance gate recomputation changed")
    return {
        "audit_pass": True,
        "qualification_pass": passed and final_host_pass,
        "performance_pass": expected_gate["pass"],
        "cpu_jobs": len(cpu),
        "external_jobs": len(external),
        "gpu_jobs": len(gpu),
    }


def main():
    global _INVOCATION_MONITOR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--ngspice", required=True)
    parser.add_argument("--model-archive", required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--out")
    destination.add_argument("--audit")
    parser.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args()
    if args.audit:
        if args.qualify_only:
            parser.error("--audit cannot be combined with --qualify-only")
        result = audit(
            Path(args.audit).resolve(),
            Path(args.model_archive).resolve(),
            Path(args.cpu).resolve(),
            Path(args.gpu).resolve(),
            Path(args.ngspice).resolve(),
        )
        print(json.dumps(result), flush=True)
        return 0
    out = Path(args.out).resolve()
    if out.exists():
        parser.error("output must be new; evidence cannot be selectively repaired")
    out.mkdir(parents=True)
    invocation_start = time.perf_counter()
    _INVOCATION_MONITOR = InvocationHostMonitor()
    manifest, manifest_sha = study.load_manifest(
        study.manifest_path("emi01-v2"), "emi01-v2"
    )
    sources = identities()
    archive = Path(args.model_archive).resolve()
    invocation = {
        "schema": SCHEMA,
        "sources": sources,
        "manifest_sha256": manifest_sha,
        "cpu_workers": CPU_WORKERS,
        "gpu_workers": GPU_WORKERS,
        "ensemble_sizes": ENSEMBLES,
        "warmups": WARMUPS,
        "measured": MEASURED,
        "host": platform.uname()._asdict(),
        "logical_cpus": os.cpu_count(),
        "cpu_topology": cpu_identity(),
        "accelerator": accelerator_identity(),
        "parent_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "persistent_worker_cores": CPU_AFFINITY,
        "qualification_cpu_affinity": CPU_AFFINITY[:4],
        "aggregate_host_limit_bytes": HOST_BYTES,
        "device_limit_bytes": DEVICE_BYTES,
        "job_limits": manifest["limits"],
        "retries": 0,
        "qualify_only": args.qualify_only,
        "reuse": "persistent process/context; immutable fresh CPU trajectories; no numerical state",
        "dispatch_authorized": False,
    }
    with tempfile.TemporaryDirectory(prefix="emi03-invocation-") as temporary:
        scratch = Path(temporary)
        binaries = {}
        for lane, path in (("cpu", args.cpu), ("gpu", args.gpu)):
            source = Path(path).resolve()
            binary = scratch / (lane + "-runner")
            data = source.read_bytes()
            binary.write_bytes(data)
            binary.chmod(0o500)
            binaries[lane] = binary
            invocation[lane + "_sha256"] = study.sha(data)
        oracle, invocation["ngspice_sha256"] = study.snapshot_oracle(
            Path(args.ngspice).resolve(), scratch
        )
        invocation["model"] = adapter.adapt_archive(archive, scratch / "model.lib")
        study.write_json(out / "invocation.json", invocation)
        common = {
            "limits": manifest["limits"],
            "binary": str(oracle),
            "model": str(scratch / "model.lib"),
        }
        specs = [{**spec, **common} for spec in qualification.expected_specs(manifest)]
        baseline = out / "qualification"
        (baseline / "cpu").mkdir(parents=True)
        (baseline / "reference").mkdir()
        reference_start = time.perf_counter()
        common_setup_s = reference_start - invocation_start
        previous_affinity = os.sched_getaffinity(0)
        if not set(CPU_AFFINITY).issubset(previous_affinity):
            fail("unsupported_input", "recorded sixteen-core affinity unavailable")
        try:
            os.sched_setaffinity(0, set(CPU_AFFINITY[:4]))
            cpu, external = qualification.run_jobs(
                specs, binaries["cpu"], archive, baseline
            )
        finally:
            os.sched_setaffinity(0, previous_affinity)
        fresh_cpu_external_s = time.perf_counter() - reference_start
        cpu_refinement = study.qualify(baseline / "cpu", cpu, manifest=manifest)
        external_refinement = study.qualify(
            baseline / "reference", external, manifest=manifest
        )
        external_comparison = qualification.compare_jobs(
            baseline / "cpu", baseline / "reference", cpu, external
        )
        oracle_budget = oracle_budget_check(cpu, external, manifest["limits"])
        cpu_qualification_s = time.perf_counter() - reference_start
        references = {record["id"]: record for record in cpu}
        gpu_pool = PersistentPool(
            binaries["gpu"],
            scratch / "qualification-workers",
            manifest["limits"],
            True,
            4,
        )
        try:
            gpu_study = complete_study(
                qualification.expected_specs(manifest),
                invocation["gpu_sha256"],
                archive,
                baseline / "gpu",
                gpu_pool,
                True,
                references,
                baseline / "cpu",
            )
        finally:
            worker_records = gpu_pool.close()
            study.write_json(baseline / "gpu-workers.json", worker_records)
        gpu = gpu_study["jobs"]
        gpu_resources = pool_resources(gpu_pool)
        gpu_refinement = study.qualify(baseline / "gpu", gpu, manifest=manifest)
        qualification_s = time.perf_counter() - reference_start
        qualified = bool(
            oracle_budget["pass"]
            and cpu_refinement["pass"]
            and external_refinement["pass"]
            and external_comparison["pass"]
            and gpu_study["pass"]
            and gpu_refinement["pass"]
            and gpu_resources["resource_pass"]
            and _INVOCATION_MONITOR.snapshot()["resource_pass"]
        )
        finest = [
            record
            for record in gpu
            if record["fixture"] == "ensemble" and record["level"] == 2
        ]
        cases = study.reference_cases(finest, manifest)
        qualified &= cases["pass"]
        result = {
            "schema": SCHEMA,
            "pass": qualified,
            "complete": len(cpu) == len(external) == len(gpu) == 30,
            "cpu_refinement": cpu_refinement,
            "oracle_budget": oracle_budget,
            "external_refinement": external_refinement,
            "external_comparison": external_comparison,
            "gpu_refinement": gpu_refinement,
            "reference_cases": cases,
            "ranking": study.ranking(finest, qualified, manifest),
            "cpu_jobs": cpu,
            "reference_jobs": external,
            "gpu_jobs": gpu,
            "fresh_cpu_external_s": fresh_cpu_external_s,
            "qualification_s": qualification_s,
            "cpu_qualification_s": cpu_qualification_s,
            "common_setup_s": common_setup_s,
            "gpu_resources": gpu_resources,
            "invocation_host_resources": _INVOCATION_MONITOR.snapshot(),
            "dispatch_authorized": False,
        }
        study.write_json(out / "qualification.json", result)
        modes = []
        if qualified and not args.qualify_only:
            for engine, counts in (("cpu", CPU_WORKERS), ("gpu", GPU_WORKERS)):
                for size in ENSEMBLES:
                    for count in counts:
                        label = f"{engine}-n{size}-w{count}"
                        pool = PersistentPool(
                            binaries[engine],
                            scratch / label,
                            manifest["limits"],
                            engine == "gpu",
                            count,
                        )
                        mode = {
                            "engine": engine,
                            "workers": count,
                            "size": size,
                            "setup_s": pool.setup_s,
                            "studies": [],
                        }
                        try:
                            for repeat in range(WARMUPS + MEASURED):
                                study_label = f"{label}-s{repeat}"
                                completed = complete_study(
                                    replay_specs(manifest, size, study_label),
                                    invocation[engine + "_sha256"],
                                    archive,
                                    out / "performance" / study_label,
                                    pool,
                                    engine == "gpu",
                                    references,
                                    baseline / "cpu",
                                )
                                mode["studies"].append(
                                    {
                                        key: completed[key]
                                        for key in (
                                            "pass",
                                            "expected_jobs",
                                            "completed_jobs",
                                            "complete_study_s",
                                        )
                                    }
                                )
                        finally:
                            mode["worker_records"] = pool.close()
                        mode.update(pool_resources(pool))
                        study.write_json(
                            out / "performance" / (label + "-mode.json"), mode
                        )
                        modes.append(mode)
                        study.write_json(out / "performance-progress.json", modes)
        gate = (
            performance_gate(
                modes, qualification_s, cpu_qualification_s, common_setup_s
            )
            if modes
            else {
                "pass": False,
                "status": "qualification_only" if qualified else "qualification_failed",
                "comparisons": [],
                "dispatch_authorized": False,
            }
        )
        unchanged = identities() == sources
        invocation_host_resources = _INVOCATION_MONITOR.close()
        if not invocation_host_resources["resource_pass"]:
            gate = {
                "pass": False,
                "status": "resource_limit",
                "comparisons": [],
                "dispatch_authorized": False,
            }
        if not unchanged:
            gate = {
                "pass": False,
                "status": "provenance_mismatch",
                "comparisons": [],
                "dispatch_authorized": False,
            }
        summary = {
            "schema": SCHEMA,
            "qualification_pass": qualified
            and unchanged
            and invocation_host_resources["resource_pass"],
            "performance": gate,
            "modes": modes,
            "fresh_cpu_external_s": fresh_cpu_external_s,
            "qualification_s": qualification_s,
            "whole_invocation_s": time.perf_counter() - invocation_start,
            "source_identity_unchanged": unchanged,
            "invocation_host_resources": invocation_host_resources,
            "independent_invocations_required": 2,
            "claim_scope": "finite known-input replay only; no discovery or novel-candidate throughput",
            "dispatch_authorized": False,
        }
        study.write_json(out / "summary.json", summary)
        print(
            json.dumps(
                {
                    "qualification_pass": summary["qualification_pass"],
                    "performance_pass": gate["pass"],
                    "output": str(out),
                }
            ),
            flush=True,
        )
        return (
            0
            if summary["qualification_pass"] and (args.qualify_only or gate["pass"])
            else 1
        )


if __name__ == "__main__":
    raise SystemExit(main())
