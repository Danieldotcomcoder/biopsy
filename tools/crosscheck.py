"""Cross-check biopsy against independent readers (optional; not part of cargo verify).

GGUF:        llama.cpp's own gguf-py (GGUFReader + gguf.quants.dequantize).
safetensors: the format implemented directly with numpy.

Checks tensor names, types, shapes, byte ranges and metadata exactly, every
decoded value bit for bit, and biopsy's per-tensor summary against numpy.

    pip install numpy gguf
    cargo build --release
    cargo run --release --example dump -- models/x.gguf target/dump [scalar|auto]
    python tools/crosscheck.py models/x.gguf target/dump target/release/biopsy
"""
import json, os, subprocess, sys
import numpy as np

model, dumpdir, exe = sys.argv[1:4]
exe = os.path.abspath(exe)
if not os.path.exists(exe) and os.path.exists(exe + ".exe"):
    exe += ".exe"  # Windows
manifest = json.load(open(f"{dumpdir}/manifest.json"))
U32 = np.uint32

def check_values(m, ref):
    ours = np.fromfile(f"{dumpdir}/{m['file']}", dtype="<f4")
    ref = np.ascontiguousarray(ref, dtype=np.float32).reshape(-1)
    assert ours.shape == ref.shape, (m["name"], ours.shape, ref.shape)
    both_nan = np.isnan(ours) & np.isnan(ref)
    bad = (ours.view(U32) != ref.view(U32)) & ~both_nan
    assert not bad.any(), f"{m['name']}: {bad.sum()} values differ, first at {np.argmax(bad)}"
    s = m["summary"]
    fin = ref[np.isfinite(ref)].astype(np.float64)
    assert s["count"] == ref.size and s["finite"] == fin.size
    assert s["nan"] == int(np.isnan(ref).sum()) and s["zeros"] == int((ref == 0).sum())
    if fin.size:
        assert s["min"] == float(ref[np.isfinite(ref)].min()) and s["max"] == float(ref[np.isfinite(ref)].max())
        mean, std = fin.mean(), fin.std()
        assert abs(s["mean"] - mean) <= 1e-9 * max(abs(mean), fin.std(), 1e-30), (m["name"], s["mean"], mean)
        assert abs(s["std"] - std) <= 1e-9 * max(std, 1e-30), (m["name"], s["std"], std)
    return ref.size

decoded = elements = 0
if open(model, "rb").read(4) == b"GGUF":
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(model)
    assert len(manifest) == len(r.tensors), (len(manifest), len(r.tensors))
    for m, t in zip(manifest, r.tensors):
        dims = [int(x) for x in t.shape]
        assert m["name"] == t.name, (m["name"], t.name)
        assert m["type"] == t.tensor_type.name, (m["name"], m["type"], t.tensor_type.name)
        assert m["shape"] == dims[::-1], (m["name"], m["shape"], dims)
        assert m["elements"] == int(t.n_elements)
        assert m["file_offset"] == int(t.data_offset), (m["name"], m["file_offset"], t.data_offset)
        assert m["byte_len"] == int(t.n_bytes)
        if m["decoded"]:
            elements += check_values(m, dequantize(t.data, t.tensor_type))
            decoded += 1
    # metadata: count, alignment and a few values, against `biopsy inspect`
    text = subprocess.run([exe, "inspect", model], capture_output=True, text=True, check=True).stdout
    keys = [k for k in r.fields if not k.startswith("GGUF.")]
    assert f"Metadata ({len(keys)} entries):" in text, "metadata count differs"
    assert f"alignment {r.alignment} bytes" in text
    assert f"tensor data begins at byte {r.data_offset}" in text, r.data_offset
    tokens = r.fields["tokenizer.ggml.tokens"]
    assert f"tokenizer.ggml.tokens: array = [string x {len(tokens.data)}]" in text
    print(f"metadata OK: {len(keys)} keys, alignment {r.alignment}, data at {r.data_offset}, {len(tokens.data)} tokens")
else:
    raw = open(model, "rb").read()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + n])
    start = 8 + n
    entries = sorted((k, v) for k, v in header.items() if k != "__metadata__")
    assert len(entries) == len(manifest)
    for m, (name, v) in zip(manifest, entries):
        b, e = v["data_offsets"]
        assert (m["name"], m["type"], m["shape"]) == (name, v["dtype"], v["shape"]), name
        assert (m["file_offset"], m["byte_len"]) == (start + b, e - b), name
        chunk = np.frombuffer(raw, dtype=np.uint8, count=e - b, offset=start + b)
        ref = {"F32": lambda c: c.view("<f4"),
               "F16": lambda c: c.view("<f2").astype(np.float32),
               "BF16": lambda c: (c.view("<u2").astype(U32) << 16).view(np.float32)}.get(v["dtype"])
        if m["decoded"]:
            elements += check_values(m, ref(chunk))
            decoded += 1
types = sorted({m["type"] for m in manifest if m["decoded"]})
print(f"OK {model}: {len(manifest)} tensors, {decoded} decoded ({', '.join(types)}), {elements} values bit-identical")
