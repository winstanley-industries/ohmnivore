"""Runtime linkage check for the Linux x86_64 EMI-01 reference dependency."""

from pathlib import Path
import sys
import unittest

from third_party.emi_python.runtime import load_numpy


class RuntimeTest(unittest.TestCase):
    def test_numpy_fft_and_runtime_linkage(self):
        numpy = load_numpy()
        self.assertEqual(sys.version_info[:3], (3, 12, 13))
        self.assertEqual(numpy.__version__, "2.4.3")
        self.assertEqual(numpy.fft.rfft(numpy.ones(8)).tolist(), [8, 0, 0, 0, 0])
        glibc = {
            "ld-linux-x86-64.so.2",
            "libc.so.6",
            "libdl.so.2",
            "libm.so.6",
            "libpthread.so.0",
            "librt.so.1",
            "libutil.so.1",
        }
        for line in Path("/proc/self/maps").read_text().splitlines():
            path = line.split()[-1]
            if path.startswith(("/usr/lib/", "/lib/", "/lib64/")) and ".so" in path:
                self.assertIn(Path(path).name, glibc, path)


if __name__ == "__main__":
    unittest.main()
