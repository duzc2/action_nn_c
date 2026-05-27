"""Quick SAD diagnostic — no matplotlib, just numbers."""
import numpy as np, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")

np.random.seed(42)
raw = np.fromfile(os.path.join(DATASET_DIR, "data_batch_1.bin"), dtype=np.uint8)
n_imgs = min(len(raw) // 3073, 500)
raw = raw[:n_imgs * 3073].reshape(n_imgs, 3073)
labels = raw[:, 0].copy()
r = raw[:, 1:1025].reshape(n_imgs, 32, 32).astype(np.float32) / 255.0
g = raw[:, 1025:2049].reshape(n_imgs, 32, 32).astype(np.float32) / 255.0
b = raw[:, 2049:3073].reshape(n_imgs, 32, 32).astype(np.float32) / 255.0
imgs = np.stack([r, g, b], axis=-1)
print(f"Loaded {len(imgs)} images")

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

# Generate frames
n = 200
mlen = 12
slen = 30
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

# Compute SAD per frame pair
sads = []
for i in range(1, len(frames)):
    sads.append(float(np.sum(np.abs(frames[i] - frames[i - 1]))))

# Categorize
motion_sads = [sads[i - 1] for i in range(1, n) if truth[i] == 1]
static_sads = [sads[i - 1] for i in range(1, n) if truth[i] == 0]

print(f"\n=== SAD Analysis ({n} frames, {sz}x{sz} objects, mlen={mlen}, slen={slen}) ===")
print(f"Motion frames: {len(motion_sads)}")
print(f"  SAD: min={np.min(motion_sads):.1f}  mean={np.mean(motion_sads):.1f}  max={np.max(motion_sads):.1f}")
print(f"Static frames: {len(static_sads)}")
print(f"  SAD: min={np.min(static_sads):.1f}  mean={np.mean(static_sads):.1f}  max={np.max(static_sads):.1f}")
print(f"\nRatio: motion/static = {np.mean(motion_sads)/np.mean(static_sads):.1f}x")

# Test with adaptive threshold (base=30, factor=2.0, alpha=0.1)
class AdaptiveThr:
    def __init__(self, base=30, factor=2.0, alpha=0.1):
        self.base = base; self.factor = factor; self.alpha = alpha
        self.mean = 0; self.std = 0; self.init = False
    def update(self, sad):
        if not self.init:
            self.mean = sad; self.std = 0; self.init = True
        else:
            d = sad - self.mean
            self.mean += self.alpha * d
            self.std += self.alpha * (abs(d) - self.std)
        return self.base + self.factor * self.std if self.init else self.base

thr = AdaptiveThr()
detected = 0
total_motion = 0
total_static = 0
correct_motion = 0
correct_static = 0

for i in range(1, n):
    sad = sads[i - 1]
    t = thr.update(sad)
    raw = 1 if sad > t else 0
    if raw: detected += 1
    if truth[i]:
        total_motion += 1
        if raw: correct_motion += 1
    else:
        total_static += 1
        if not raw: correct_static += 1

print(f"\n=== Detection Performance ===")
print(f"Raw detections: {detected}/{n - 1}")
print(f"Motion recall: {correct_motion}/{total_motion} = {correct_motion/total_motion*100:.0f}%")
print(f"Static precision: {correct_static}/{total_static} = {correct_static/total_static*100:.0f}%")

if total_motion > 0:
    precision = correct_motion / detected if detected > 0 else 0
    recall = correct_motion / total_motion
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")

print("\nDone. Motion detection with real CIFAR-10 images is:",
      "VIABLE" if np.min(motion_sads) > 30 else "NEEDS TUNING")
