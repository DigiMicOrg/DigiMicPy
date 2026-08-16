import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


class PackageImportTests(unittest.TestCase):
    def test_import_has_no_random_plotting_or_filesystem_side_effects(self):
        source_root = Path(__file__).resolve().parents[1] / "src"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(source_root), environment.get("PYTHONPATH")))
        )
        script = textwrap.dedent(
            """
            import os
            import sys

            import numpy as np

            np.random.seed(17)
            expected = np.random.random()
            np.random.seed(17)
            files_before = set(os.listdir())

            import digimicpy

            assert np.random.random() == expected
            assert set(os.listdir()) == files_before
            assert "matplotlib" not in sys.modules
            assert hasattr(digimicpy, "MiCRMParameters")
            """
        )

        with tempfile.TemporaryDirectory() as working_directory:
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=working_directory,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()