import numpy as np, os, sys, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")

np.random.seed(42)
t0 = time.time()
raw = np.fromfile(os.path.join(DATASET_DIR, "data_batch_1.bin"), dtype=np.uint8)
n_imgs = min(len(raw) // 3073, 500)
raw = raw[:n_imgs * 3073].reshape(n_imgs, 3073)
labels = raw[:, 0].copy()
r = raw[:, 1:1025].reshape(n_imgs, 32, 32).astype(np.float32) / 255.0
g = raw[:, 1025:2049].reshape(n_imgs, 32, 32).astype(np.float32) / 255.0
b = raw[:, 2049:3073].reshape(n_imgs, 32, 32).astype(np.float32) / 255.0
imgs = np.stack([r, g, b], axis=-1)
print(f"Loaded {len(imgs)} images in {time.time()-t0:.1f}s")
sys.stdout.flush()

# Build objects
bg = imgs[42].copy()
sz = 14
cx0 = 16 - sz // 2
objs = []
for cat in range(10):
    idxs = np.where(labels == cat)[0]
    for k in range(min(2, len(idxs))):
        ii = int(idxs[k * len(idxs) // min(2, len(idxs))])
        objs.append(imgs[ii, cx0:cx0 + sz, cx0:cx0 + sz, :].copy())
print(f"{len(objs)} foreground objects ({sz}x{sz})")
sys.stdout.flush()

# Generate 60 frames (2 segments)
n = 60
mlen = 12
slen = 18  # shorter for testing
period = mlen + slen
frames = np.zeros((n, 32, 32, 3), dtype=np.float32)
truth = np.zeros(n, dtype=np.int32)
idx = 0
seg = 0
while idx < n:
    fg = objs[seg % len(objs)]
    sx = np.random.randint(0, 33 - sz)
    sy = np.random.randint(0, 33 - sz)
    ex = np.random.randint(0, 33 - sz)
    ey = np.random.randint(0, 33 - sz)
    while abs(ex - sx) + abs(ey - sy) < 12:
        ex = np.random.randint(sz + 1, 31 - sz)
        ey = np.random.randint(sz + 1, 31 - sz)
    for i in range(min(period, n - idx)):
        fi = idx + i
        frame = np.clip(bg + np.random.uniform(-0.005, 0.005, bg.shape), 0.0, 1.0)
        if i < mlen:
            frac = i / max(mlen - 1, 1)
            cx = int(sx + (ex - sx) * frac)
            cy = int(sy + (ey - sy) * frac)
            truth[fi] = 1
        else:
            cx, cy = ex, ey
            truth[fi] = 0
        cx = max(0, min(cx, 32 - sz))
        cy = max(0, min(cy, 32 - sz))
        frame[cy:cy + sz, cx:cx + sz] = fg
        frames[fi] = frame
    idx += min(period, n - idx)
    seg += 1
print(f"Generated {n} frames")
sys.stdout.flush()

# SAD
sads = []
for i in range(1, n):
    sads.append(float(np.sum(np.abs(frames[i] - frames[i - 1]))))

motion_sads = [sads[i - 1] for i in range(1, n) if truth[i] == 1]
static_sads = [sads[i - 1] for i in range(1, n) if truth[i] == 0]
if motion_sads and static_sads:
    print(f"Motion SAD: mean={np.mean(motion_sads):.1f} min={np.min(motion_sads):.1f} max={np.max(motion_sads):.1f}")
    print(f"Static SAD: mean={np.mean(static_sads):.1f} min={np.min(static_sads):.1f} max={np.max(static_sads):.1f}")
    print(f"Ratio: {np.mean(motion_sads)/np.mean(static_sads):.1f}x")
else:
    print("No motion or static frames found")
print("TEST DONE")
