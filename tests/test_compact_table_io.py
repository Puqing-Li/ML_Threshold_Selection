import gzip
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from rebuild_revision_results import _read_table


def test_csv_gz_round_trip_preserves_binary_floats(tmp_path):
    source = pd.DataFrame({
        "Volume3d (mm^3) ": [0.0013500000000000001, 0.0055080000000000004],
        "EigenVal1": [1.2345678901234567, 9.876543210987654],
        "Sample": ["LE01", "LE01"],
    })
    path = tmp_path / "Quantity_LE01.csv.gz"
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as stream:
            source.to_csv(stream, index=False, lineterminator="\n", float_format="%.17g")

    restored = _read_table(path)

    pd.testing.assert_frame_equal(source, restored, check_dtype=False, check_exact=True)
