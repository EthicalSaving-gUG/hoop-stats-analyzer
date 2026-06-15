import numpy as np
import time
import math
import torch

def original_method(labels, bbox, frame_shape):
    h, w = labels.shape
    dense_points = []
    for y in range(0, h, 6):
        for x in range(0, w, 6):
            label_val = int(labels[y, x])
            if label_val > 0:
                norm_x = (bbox[0] + (x / w) * (bbox[2] - bbox[0])) / frame_shape[1]
                norm_y = (bbox[1] + (y / h) * (bbox[3] - bbox[1])) / frame_shape[0]
                z = 0.05 * math.sin(label_val)
                dense_points.append([round(norm_x, 4), round(norm_y, 4), round(z, 4), label_val])
    return dense_points

def optimized_method(labels, bbox, frame_shape):
    h, w = labels.shape

    # Ensure labels is a numpy array for efficient slicing and masking
    if hasattr(labels, 'numpy'):
        labels_np = labels.numpy()
    else:
        labels_np = labels

    labels_downsampled = labels_np[::6, ::6]
    mask = labels_downsampled > 0

    y_indices = np.arange(0, h, 6)
    x_indices = np.arange(0, w, 6)
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)

    valid_x = x_grid[mask]
    valid_y = y_grid[mask]
    valid_labels = labels_downsampled[mask].astype(int)

    norm_x = (bbox[0] + (valid_x / w) * (bbox[2] - bbox[0])) / frame_shape[1]
    norm_y = (bbox[1] + (valid_y / h) * (bbox[3] - bbox[1])) / frame_shape[0]
    z = 0.05 * np.sin(valid_labels)

    dense_points = [
        [round(float(nx), 4), round(float(ny), 4), round(float(nz), 4), int(l)]
        for nx, ny, nz, l in zip(norm_x, norm_y, z, valid_labels)
    ]
    return dense_points


h, w = 480, 640
labels_np = np.random.randint(0, 5, size=(h, w)).astype(np.float32)
labels_torch = torch.from_numpy(labels_np)
bbox = [100.0, 50.0, 300.0, 400.0]
frame_shape = (720, 1280, 3)

# Test correctness
print("Testing correctness...")
res_orig = original_method(labels_torch, bbox, frame_shape)
res_opt = optimized_method(labels_torch, bbox, frame_shape)

assert len(res_orig) == len(res_opt), "Lengths differ!"
for i in range(len(res_orig)):
    assert res_orig[i] == res_opt[i], f"Mismatch at {i}: {res_orig[i]} vs {res_opt[i]}"
print("Correctness verified!")

# Benchmark
n_iters = 100

start = time.time()
for _ in range(n_iters):
    original_method(labels_torch, bbox, frame_shape)
orig_time = time.time() - start

start = time.time()
for _ in range(n_iters):
    optimized_method(labels_torch, bbox, frame_shape)
opt_time = time.time() - start

print(f"Original: {orig_time:.4f} s")
print(f"Optimized: {opt_time:.4f} s")
print(f"Speedup: {orig_time/opt_time:.2f}x")
