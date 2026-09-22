import contextlib
import io
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_bvh_to_csv as exporter


class BatchBvhToCsvTest(unittest.TestCase):
    def test_planned_recycle_batch_boundary(self):
        self.assertEqual(exporter.PLANNED_RECYCLE_EXIT_CODE, 75)
        cases = [
            # Unlimited mode never recycles.
            ((4, 0, 32, 40), False),
            # The configured number of batches has not completed yet.
            ((3, 4, 24, 40), False),
            # Recycle exactly at the boundary when initial pending work remains.
            ((4, 4, 32, 40), True),
            # Do not recycle after the final pending batch.
            ((4, 4, 32, 32), False),
        ]
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(exporter._planned_recycle_due(*arguments), expected)

    def test_max_batches_cli_default_and_validation(self):
        required = ["--import-root", "/input", "--export-root", "/output"]
        parser = exporter.build_parser()
        self.assertEqual(parser.parse_args(required).max_batches_per_process, 0)
        self.assertEqual(
            parser.parse_args([*required, "--max-batches-per-process", "4"])
            .max_batches_per_process,
            4,
        )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([*required, "--max-batches-per-process", "-1"])

    def test_main_returns_planned_code_after_committed_batch_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            import_root = root / "input"
            export_root = root / "output"
            import_root.mkdir()
            paths = [import_root / f"clip_{index}.bvh" for index in range(3)]
            importer = mock.Mock()
            importer.create_skeleton.return_value = (object(), None)

            with (
                mock.patch.object(
                    exporter,
                    "_collect_paths",
                    return_value=(paths, 0, [], len(paths)),
                ),
                mock.patch.object(
                    exporter.bvh_utils, "BVHImporter", return_value=importer
                ),
                mock.patch.object(exporter, "SpaceConverter") as converter,
                mock.patch.object(
                    exporter, "get_facing_direction_type_from_str", return_value=object()
                ),
                mock.patch.object(exporter.wp, "transform_identity", return_value=object()),
                mock.patch.object(exporter.wp, "ScopedDevice", return_value=nullcontext()),
                mock.patch.object(exporter.csv_utils, "get_csv_config", return_value=object()),
                mock.patch.object(
                    exporter.newton_pipeline, "NewtonPipeline", return_value=object()
                ),
                mock.patch.object(
                    exporter,
                    "_load_batch",
                    side_effect=lambda batch, _skeleton: (batch, [object()] * len(batch)),
                ),
                mock.patch.object(
                    exporter,
                    "_retarget_with_split",
                    side_effect=lambda _pipeline, batch, *_args: len(batch),
                ) as retarget,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                converter.return_value.transform.return_value = object()
                exit_code = exporter.main(
                    [
                        "--import-root",
                        str(import_root),
                        "--export-root",
                        str(export_root),
                        "--batch-size",
                        "1",
                        "--max-batches-per-process",
                        "2",
                        "--device",
                        "cpu",
                    ]
                )

            self.assertEqual(exit_code, exporter.PLANNED_RECYCLE_EXIT_CODE)
            self.assertEqual(retarget.call_count, 2)
            progress = root / "logs" / "batch_progress.csv"
            self.assertEqual(len(progress.read_text(encoding="utf-8").splitlines()), 3)
            self.assertIn("[RECYCLE] completed_batches=2", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
