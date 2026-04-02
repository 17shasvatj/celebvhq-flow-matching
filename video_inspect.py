import os, json, cv2

with open("celebvhq_raw/celebvhq_info.json") as f:
    info = json.load(f)

clip_ids = list(info["clips"].keys())
print(f"Total clips: {len(clip_ids)}")
print(f"\nFirst clip: {clip_ids[0]}")
print(json.dumps(info["clips"][clip_ids[0]], indent=2))

# Inspect attributes
first_clip = info["clips"][clip_ids[0]]
for key in first_clip:
    print(f"  {key}: {type(first_clip[key])}")

# Inspect a video clip
video_dir = "./celebvhq_raw/videos"  # adjust to wherever your videos are
videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
print(f"Found {len(videos)} video files")

# Look at the first one
cap = cv2.VideoCapture(os.path.join(video_dir, videos[0]))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {videos[0]}")
print(f"Resolution: {w}x{h}, FPS: {fps}, Frames: {n_frames}")

# Save a sample frame to visually inspect
ret, frame = cap.read()
cv2.imwrite("sample_frame.png", frame)
cap.release()
print("Saved sample_frame.png — go look at it")