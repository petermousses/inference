import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from materialize_deepseek_revision import MAX_REVISION_BYTES, materialize


class DeepSeekRevisionMaterializationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "metadata/SOURCE_REVISION"
        self.target = self.root / "model/SOURCE_REVISION"
        self.source.parent.mkdir()
        self.target.parent.mkdir()
        self.source.write_text("repository=deepseek-ai/model\nrevision=abc123\n")

    def tearDown(self):
        self.temporary.cleanup()

    def test_creates_regular_read_only_copy(self):
        self.assertTrue(materialize(self.source, self.target))
        self.assertTrue(self.target.is_file())
        self.assertFalse(self.target.is_symlink())
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o440)

    def test_creates_missing_target_parent(self):
        self.target.parent.rmdir()
        self.assertTrue(materialize(self.source, self.target))
        self.assertTrue(self.target.is_file())
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())

    def test_replaces_only_symlink_to_the_tracked_source(self):
        self.target.symlink_to(self.source)
        self.assertTrue(materialize(self.source, self.target))
        self.assertFalse(self.target.is_symlink())
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())

    def test_keeps_matching_regular_copy(self):
        self.target.write_bytes(self.source.read_bytes())
        self.target.chmod(0o644)
        self.assertFalse(materialize(self.source, self.target))
        self.assertFalse(self.target.is_symlink())
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o440)

    def test_refuses_mismatched_file_and_unrelated_symlink(self):
        self.target.write_text("wrong revision\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            materialize(self.source, self.target)
        self.target.unlink()
        unrelated = self.root / "other/SOURCE_REVISION"
        unrelated.parent.mkdir()
        unrelated.write_text("different\n")
        self.target.symlink_to(unrelated)
        with self.assertRaisesRegex(ValueError, "does not point"):
            materialize(self.source, self.target)
        self.assertTrue(self.target.is_symlink())

    def test_refuses_oversized_existing_regular_file_without_reading_it(self):
        with self.target.open("wb") as output:
            output.truncate(MAX_REVISION_BYTES + 1)
        original_read_bytes = Path.read_bytes

        def reject_target_read(path):
            if path == self.target:
                raise AssertionError("oversized target must not be read")
            return original_read_bytes(path)

        with patch.object(Path, "read_bytes", autospec=True, side_effect=reject_target_read):
            with self.assertRaisesRegex(ValueError, "differs"):
                materialize(self.source, self.target)

    def test_refuses_symlink_source(self):
        linked_source = self.root / "linked-source"
        linked_source.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "small regular file"):
            materialize(linked_source, self.target)


if __name__ == "__main__":
    unittest.main()
