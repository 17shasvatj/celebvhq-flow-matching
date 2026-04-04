import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

def extract_frames(video_path, target_fps=8, target_size=256):
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        cap.release()
        return None
    frame_skip = max(1, round(src_fps / target_fps))
    frames = []
    idx = 0
    while cap.isOpened():
        ret, bgr = cap.read()
        if not ret:
            break
        if idx % frame_skip == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            s = min(h, w)
            y0, x0 = (h-s) // 2, (w - s) // 2
            cropped = rgb[y0:y0+s, x0:x0+s]
            resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
            frames.append(resized)
        idx += 1
    cap.release()
    return np.stack(frames) if len(frames) >= 16 else None


def process_dataset(video_dir, output_dir, info_path, target_fps=8,
                    target_size=256, min_frames=16):
    """Process all clips, save as .npz, build manifest."""
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(info_path) as f:
        info = json.load(f)

    manifest = []

    for clip_id in tqdm(info["clips"], desc="Processing"):
        # Find the video file — naming may vary
        video_path = video_dir / f"{clip_id}.mp4"
        if not video_path.exists():
            continue

        frames = extract_frames(video_path, target_fps, target_size)
        if frames is None:
            continue

        # Save frames
        out_path = output_dir / f"{clip_id}.npz"
        np.savez_compressed(str(out_path), video=frames.astype(np.uint8))

        # Extract metadata
        meta = info["clips"][clip_id]
        manifest.append({
            "clip_id": clip_id,
            "path": str(out_path),
            "num_frames": len(frames),
            "attributes": meta.get("attributes", {}),
        })

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    print(f"Processed {len(manifest)} clips")
    print(f"Manifest saved to {manifest_path}")
    return manifest

# Run it
manifest = process_dataset(
    video_dir="./celebvhq_raw/videos",
    output_dir="./celebvhq_processed",
    info_path="./celebvhq_raw/celebvhq_info.json",
    target_fps=8,
    target_size=256,
)

# Verify a few processed clips
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

for entry in manifest[:3]:
    data = np.load(entry["path"])
    video = data["video"]  # (T, 256, 256, 3)
    print(f"{entry['clip_id']}: {video.shape}, "
          f"range [{video.min()}, {video.max()}]")

    # Save a frame strip
    strip = np.concatenate(
        [video[i] for i in range(0, min(16, len(video)), 2)],
        axis=1
    )
    plt.figure(figsize=(20, 3))
    plt.imshow(strip)
    plt.axis("off")
    plt.savefig(f"check_{entry['clip_id']}.png",
                bbox_inches="tight", dpi=100)
    plt.close()
    print(f"  Saved check_{entry['clip_id']}.png")









