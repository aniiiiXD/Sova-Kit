"""Distribution checks use temporary destinations and local artifacts only."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_packaged_skills_match_canonical_source(self):
        canonical = (ROOT / 'skills/switchit/SKILL.md').read_bytes()
        for relative in ('npm/skills/switchit/SKILL.md', 'harness-js/skills/switchit/SKILL.md'):
            with self.subTest(path=relative):
                self.assertEqual((ROOT / relative).read_bytes(), canonical)

    @unittest.skipUnless(shutil.which('sh'), 'shell installer requires a POSIX shell')
    def test_shell_installer_uses_explicit_isolated_destination(self):
        with tempfile.TemporaryDirectory(prefix='switchit-skill-install-') as folder:
            destination = Path(folder) / 'skills with spaces' / 'switchit'
            command = ['sh', str(ROOT / 'scripts/install-skill.sh'), '--dest', str(destination)]
            first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual((destination / 'SKILL.md').read_bytes(), (ROOT / 'skills/switchit/SKILL.md').read_bytes())
            second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn('already up to date', second.stdout)
            self.assertEqual(sorted(p.name for p in destination.iterdir()), ['SKILL.md'])

            invalid = subprocess.run(['sh', str(ROOT / 'scripts/install-skill.sh'), '--dest', '--help'],
                                     cwd=folder, capture_output=True, text=True, timeout=10)
            self.assertEqual(invalid.returncode, 2)
            self.assertFalse((Path(folder) / '--help').exists())

    @unittest.skipUnless(shutil.which('node') and shutil.which('npm'), 'Node/npm distribution checks need Node and npm')
    def test_node_bridge_and_packed_contents(self):
        self._node_suite('harness-js/test', '*.test.js')

    @unittest.skipUnless(shutil.which('node') and shutil.which('npm'), 'Node/npm distribution checks need Node and npm')
    def test_node_skill_installer_and_packed_contents(self):
        self._node_suite('npm/test', '*.test.cjs')

    def _node_suite(self, directory, pattern):
        files = sorted((ROOT / directory).glob(pattern))
        self.assertTrue(files, 'distribution test suite is missing')
        environment = {**os.environ, 'SWITCHIT_TEST_PYTHON': sys.executable}
        completed = subprocess.run([shutil.which('node'), '--test', *map(str, files)], cwd=ROOT,
                                   env=environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == '__main__':
    unittest.main()
