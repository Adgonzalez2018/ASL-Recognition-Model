"""Synthetic ASL Citizen fixtures.

Builds a miniature dataset on disk — split CSVs plus real encoded video files —
so the parser and audit are exercised against actual files rather than mocks.
The full dataset is never required for the ordinary test suite.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

GLOSSES = ["APPLE", "BOOK", "CAT"]


def write_video(path: Path, frames: int = 20, width: int = 64, height: int = 48, fps: int = 25):
    """Encode a small real video file."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for index in range(frames):
            # A moving band, so frame order is recoverable from pixel content.
            array = np.zeros((height, width, 3), dtype=np.uint8)
            band = int((index / max(frames - 1, 1)) * (height - 4))
            array[band : band + 4, :, 0] = 255
            array[:, :, 1] = index * (255 // max(frames, 1))

            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            container.mux(stream.encode(frame))

        container.mux(stream.encode())
    return path


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    """A miniature, signer-independent ASL Citizen lookalike.

    Layout mirrors the real release: split CSVs at the root, videos in videos/.
    """
    root = tmp_path_factory.mktemp("asl_citizen_synthetic")
    videos = root / "videos"

    plan = {
        "train": ["signer01", "signer02"],
        "val": ["signer03"],
        "test": ["signer04"],
    }

    counter = 0
    for split, signers in plan.items():
        rows = []
        for signer in signers:
            for gloss in GLOSSES:
                counter += 1
                filename = f"clip{counter:03d}.mp4"
                write_video(videos / filename, frames=20)
                rows.append(
                    {
                        "Participant ID": signer,
                        "Video file": filename,
                        "Gloss": gloss,
                        "ASL-LEX Code": f"lex_{gloss.lower()}",
                    }
                )

        path = root / f"{split}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["Participant ID", "Video file", "Gloss", "ASL-LEX Code"]
            )
            writer.writeheader()
            writer.writerows(rows)

    return root
