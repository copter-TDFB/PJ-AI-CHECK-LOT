"""Full-training notebook must train straight from the reference dataset."""

import json

from services import notebook_generator


def _full_nb_source() -> str:
    nb_bytes = notebook_generator.build_full_notebook(
        packaging_key="testpack", output_folder_id="out-folder",
    )
    nb = json.loads(nb_bytes)
    return "".join("".join(c["source"]) for c in nb["cells"])


def test_full_notebook_has_no_bundle_or_merge_logic():
    src = _full_nb_source()
    assert "gdown" not in src
    assert "bundle" not in src
    assert "addition" not in src
    assert "offset" not in src


def test_full_notebook_trains_from_reference_dataset():
    src = _full_nb_source()
    assert "data check lot" in src
    assert "data classify check lot" in src
    assert "full_detector.pt" in src
    assert "full_classifier.pt" in src
    assert "OUTPUT_FOLDER_ID = 'out-folder'" in src or '"out-folder"' in src


def test_seed_notebook_unchanged_still_uses_bundle():
    nb_bytes = notebook_generator.build_seed_notebook(
        packaging_key="x", bundle_file_id="bid", output_folder_id="oid",
    )
    src = "".join("".join(c["source"]) for c in json.loads(nb_bytes)["cells"])
    assert "gdown" in src and "bundle" in src
