"""Object masks (`object` mode): white = the object, black = everything else.

When the object is turned over, the table and room no longer agree with it, so camera poses
must come from the object alone: COLMAP reads these masks when extracting features and again
when fusing depth maps. COLMAP's naming: the mask for `p1_000001.jpg` is `p1_000001.jpg.png`.

Model: BiRefNet (MIT) through rembg (MIT), on the GPU when onnxruntime can use it.
"""

import functools
from pathlib import Path

import numpy as np

from castlive.recon import PipelineError

MODEL = "birefnet-general-lite"
# How far to move the mask edge, as a fraction of the image's long side.
GROW_FOR_POSES = 0.015  # out: keep the object's edges and corners, where joining features are
GROW_FOR_FUSION = -0.003  # in: edge pixels are part background
# In the ground the outline only empties space round the object, so it errs outwards: a
# shrunk outline shaved the walls above the ground and left a lip at their foot.
GROW_FOR_CARVE = 0.003


@functools.cache
def session():
    try:
        import onnxruntime

        if hasattr(onnxruntime, "preload_dlls"):
            onnxruntime.preload_dlls()  # CUDA/cuDNN from the nvidia-* pip packages
        from rembg import new_session

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return new_session(MODEL, providers=providers)
    except Exception as e:  # noqa: BLE001 - model missing or runtime broken: an operator problem
        raise PipelineError(f"The masking model could not be loaded: {e}") from None


def make_masks(images: Path, out: Path, grow: float, sess=None, progress=None) -> dict:
    import cv2
    from PIL import Image
    from rembg import remove

    sess = sess or session()
    out.mkdir(parents=True, exist_ok=True)
    coverage = []
    paths = sorted(images.glob("*.jpg"))
    for i, path in enumerate(paths, 1):
        im = Image.open(path).convert("RGB")
        mask = np.asarray(remove(im, session=sess, only_mask=True).convert("L")) > 127
        k = max(1, round(abs(grow) * max(im.size)))
        shift = cv2.dilate if grow > 0 else cv2.erode
        mask = shift(mask.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
        Image.fromarray((mask * 255).astype(np.uint8)).save(out / (path.name + ".png"))
        coverage.append(float(mask.mean()))
        if progress:
            progress(i / len(paths))
    if not coverage:
        raise PipelineError("No frames to mask.")
    median = float(np.median(coverage))
    if median < 0.01:
        raise PipelineError("The object could not be found in the frames. Fill more of the "
                            "picture with it, on a plain surface of a different colour.")
    return {"model": MODEL, "images": len(coverage), "coverage_median": round(median, 3),
            "empty": int(sum(c < 0.005 for c in coverage))}
