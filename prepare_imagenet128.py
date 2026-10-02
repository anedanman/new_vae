"""Build 128x128 center-cropped ImageNet arrays (ADM preprocessing).

Output (in --out_dir):
    {split}_images.npy   uint8 (N, 128, 128, 3), memory-mappable
    {split}_labels.npy   int16 (N,)
    {split}_valid.npy    bool  (N,)   rows whose JPEG failed to decode are False
    meta.json            counts and checksums, to verify hosts built identical data

Sources:
    --packed_dir  dir with {split}.bin + {split}_index.npy (offset, length, label)
    --parquet_dir HF ILSVRC/imagenet-1k dir with data/{split}-*.parquet
The packed files were built from the parquet shards in order, so after dropping
invalid rows both sources yield the same image sequence.
"""

import argparse
import glob
import hashlib
import io
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image


def center_crop_arr(pil_image, image_size):
    """ADM center crop (guided-diffusion/improved-diffusion)."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def decode(jpeg_bytes, size):
    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        img.load()
        return center_crop_arr(img.convert("RGB"), size)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# workers write directly into the shared output memmap
# ---------------------------------------------------------------------------

def _write_rows(fd, header, row_bytes, first_row, rows):
    """Write consecutive uint8 rows with one positioned write (no shared mmap)."""
    buf = np.ascontiguousarray(np.stack(rows)).tobytes()
    os.pwrite(fd, buf, header + first_row * row_bytes)


def _packed_worker(job):
    bin_path, index_path, out_path, header, lo, hi, size = job
    index = np.load(index_path, mmap_mode="r")
    row_bytes = size * size * 3
    zero = np.zeros((size, size, 3), dtype=np.uint8)
    valid = np.ones(hi - lo, dtype=bool)
    rows = []
    with open(bin_path, "rb") as f:
        for i in range(lo, hi):
            off, length, _ = index[i]
            f.seek(int(off))
            arr = decode(f.read(int(length)), size)
            if arr is None:
                valid[i - lo] = False
                arr = zero
            rows.append(arr)
    fd = os.open(out_path, os.O_WRONLY)
    try:
        _write_rows(fd, header, row_bytes, lo, rows)
    finally:
        os.close(fd)
    return lo, valid


def _parquet_worker(job):
    import pyarrow.parquet as pq
    path, out_path, header, row_start, size = job
    row_bytes = size * size * 3
    zero = np.zeros((size, size, 3), dtype=np.uint8)
    pf = pq.ParquetFile(path)
    valid = []
    labels = []
    row = row_start
    fd = os.open(out_path, os.O_WRONLY)
    try:
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["image", "label"])
            images = table.column("image").to_pylist()
            labels.extend(table.column("label").to_pylist())
            rows = []
            for item in images:
                arr = decode(item["bytes"], size)
                valid.append(arr is not None)
                rows.append(zero if arr is None else arr)
            _write_rows(fd, header, row_bytes, row, rows)
            row += len(rows)
    finally:
        os.close(fd)
    return row_start, np.array(valid, dtype=bool), np.array(labels, dtype=np.int16)


def _create_output(out_path, n, size):
    """Create the .npy (header + sparse data region) and return the data offset."""
    np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(n, size, size, 3)).flush()
    return np.load(out_path, mmap_mode="r").offset


def build_packed(args, split):
    bin_path = os.path.join(args.packed_dir, f"{split}.bin")
    index_path = os.path.join(args.packed_dir, f"{split}_index.npy")
    index = np.load(index_path, mmap_mode="r")
    n = len(index)
    labels = np.asarray(index["label"], dtype=np.int16)
    out_path = os.path.join(args.out_dir, f"{split}_images.npy")
    header = _create_output(out_path, n, args.size)

    chunk = 1000
    jobs = [(bin_path, index_path, out_path, header, lo, min(lo + chunk, n), args.size)
            for lo in range(0, n, chunk)]
    valid = np.ones(n, dtype=bool)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, (lo, v) in enumerate(pool.imap_unordered(_packed_worker, jobs)):
            valid[lo: lo + len(v)] = v
            if k % 100 == 0:
                print(f"[{split}] {k + 1}/{len(jobs)} chunks, {time.time() - t0:.0f}s", flush=True)
    return labels, valid


def build_parquet(args, split):
    import pyarrow.parquet as pq
    hf_split = "validation" if split == "val" else split
    files = sorted(glob.glob(os.path.join(args.parquet_dir, "data", f"{hf_split}-*.parquet")))
    counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).tolist()
    n = int(sum(counts))
    out_path = os.path.join(args.out_dir, f"{split}_images.npy")
    header = _create_output(out_path, n, args.size)

    jobs = [(f, out_path, header, s, args.size) for f, s in zip(files, starts)]
    valid = np.ones(n, dtype=bool)
    labels = np.zeros(n, dtype=np.int16)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, (lo, v, lab) in enumerate(pool.imap_unordered(_parquet_worker, jobs)):
            valid[lo: lo + len(v)] = v
            labels[lo: lo + len(lab)] = lab
            print(f"[{split}] {k + 1}/{len(jobs)} files, {time.time() - t0:.0f}s", flush=True)
    return labels, valid


def checksum(images, labels, valid):
    """Order-sensitive digest over valid rows: labels plus a strided image sample."""
    idx = np.flatnonzero(valid)
    h = hashlib.md5(labels[idx].tobytes())
    for i in idx[:: max(1, len(idx) // 2000)]:
        h.update(np.ascontiguousarray(images[i]).tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packed_dir", default=None)
    p.add_argument("--parquet_dir", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--splits", nargs="+", default=["val", "train"])
    args = p.parse_args()
    assert (args.packed_dir is None) != (args.parquet_dir is None), "pass exactly one source"
    os.makedirs(args.out_dir, exist_ok=True)

    meta_path = os.path.join(args.out_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for split in args.splits:
        t0 = time.time()
        if args.packed_dir:
            labels, valid = build_packed(args, split)
        else:
            labels, valid = build_parquet(args, split)
        np.save(os.path.join(args.out_dir, f"{split}_labels.npy"), labels)
        np.save(os.path.join(args.out_dir, f"{split}_valid.npy"), valid)
        images = np.load(os.path.join(args.out_dir, f"{split}_images.npy"), mmap_mode="r")
        meta[split] = {
            "rows": int(len(valid)),
            "valid": int(valid.sum()),
            "invalid_rows": np.flatnonzero(~valid).tolist(),
            "checksum": checksum(images, labels, valid),
            "seconds": round(time.time() - t0, 1),
            "source": args.packed_dir or args.parquet_dir,
        }
        print(f"[{split}] done: {meta[split]}", flush=True)
        json.dump(meta, open(meta_path, "w"), indent=2)


if __name__ == "__main__":
    main()
