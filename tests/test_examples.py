import os
from pathlib import Path
import subprocess
import sys
import unittest


class ExampleTests(unittest.TestCase):
    def test_basic_simulation_runs_headlessly(self):
        project_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["MPLBACKEND"] = "Agg"
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(project_root / "src"), environment.get("PYTHONPATH")))
        )

        result = subprocess.run(
            [sys.executable, str(project_root / "examples" / "basic_simulation.py")],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()