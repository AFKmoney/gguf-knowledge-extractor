"""Create a minimal valid GGUF file for testing the extractor."""
import struct
import numpy as np
import sys
from pathlib import Path

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_VERSION = 3

# Value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# Quantization types (just what we need)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def write_string(buf, s):
    b = s.encode("utf-8")
    buf.extend(struct.pack("<Q", len(b)))
    buf.extend(b)


def write_scalar(buf, type_id, value):
    buf.extend(struct.pack("<I", type_id))
    if type_id == GGUF_TYPE_UINT32:
        buf.extend(struct.pack("<I", value))
    elif type_id == GGUF_TYPE_INT32:
        buf.extend(struct.pack("<i", value))
    elif type_id == GGUF_TYPE_FLOAT32:
        buf.extend(struct.pack("<f", value))
    elif type_id == GGUF_TYPE_BOOL:
        buf.extend(struct.pack("<B", 1 if value else 0))
    elif type_id == GGUF_TYPE_STRING:
        # Already wrote the type above; write the string
        b = value.encode("utf-8")
        buf.extend(struct.pack("<Q", len(b)))
        buf.extend(b)
    elif type_id == GGUF_TYPE_UINT64:
        buf.extend(struct.pack("<Q", value))
    else:
        raise ValueError(f"Unsupported type {type_id}")


def write_kv(buf, key, type_id, value):
    write_string(buf, key)
    write_scalar(buf, type_id, value)


def write_array(buf, key, elem_type, values):
    write_string(buf, key)
    buf.extend(struct.pack("<I", GGUF_TYPE_ARRAY))   # type = ARRAY
    buf.extend(struct.pack("<I", elem_type))          # element type
    buf.extend(struct.pack("<Q", len(values)))         # length
    for v in values:
        if elem_type == GGUF_TYPE_UINT32:
            buf.extend(struct.pack("<I", v))
        elif elem_type == GGUF_TYPE_INT32:
            buf.extend(struct.pack("<i", v))
        elif elem_type == GGUF_TYPE_FLOAT32:
            buf.extend(struct.pack("<f", v))
        elif elem_type == GGUF_TYPE_STRING:
            b = v.encode("utf-8")
            buf.extend(struct.pack("<Q", len(b)))
            buf.extend(b)
        else:
            raise ValueError(f"Unsupported array element type {elem_type}")


def write_tensor_info(buf, name, n_dims, dims, type_id, offset):
    write_string(buf, name)
    buf.extend(struct.pack("<I", n_dims))
    for d in dims:
        buf.extend(struct.pack("<Q", d))
    buf.extend(struct.pack("<I", type_id))
    buf.extend(struct.pack("<Q", offset))


def make_test_gguf(out_path, vocab_size=64, embed_dim=32, n_layers=2):
    """Make a tiny GGUF file with a small embedding + transformer-like layers.

    Each layer has: attn_q, attn_k, attn_v, attn_output, ffn_up, ffn_down.
    """
    buf = bytearray()

    n_heads = 4
    head_dim = embed_dim // n_heads
    # ---- Header ----
    buf.extend(struct.pack("<I", GGUF_MAGIC))
    buf.extend(struct.pack("<I", GGUF_VERSION))
    n_tensors = 1 + (6 * n_layers)  # token_embd + per layer: attn_q/k/v/output, ffn_up, ffn_down
    buf.extend(struct.pack("<Q", n_tensors))  # tensor_count
    n_kv = 14  # 12 scalars + 2 arrays (tokens + scores)
    buf.extend(struct.pack("<Q", n_kv))  # kv_count

    # ---- KV pairs ----
    write_kv(buf, "general.architecture", GGUF_TYPE_STRING, "llama")
    write_kv(buf, "general.name", GGUF_TYPE_STRING, "test-tiny-model")
    write_kv(buf, "general.description", GGUF_TYPE_STRING, "Tiny synthetic model for testing the extractor")
    write_kv(buf, "general.license", GGUF_TYPE_STRING, "MIT")
    write_kv(buf, "general.organization", GGUF_TYPE_STRING, "TestLab")
    write_kv(buf, "llama.vocab_size", GGUF_TYPE_UINT32, vocab_size)
    write_kv(buf, "llama.context_length", GGUF_TYPE_UINT32, 512)
    write_kv(buf, "llama.embedding_length", GGUF_TYPE_UINT32, embed_dim)
    write_kv(buf, "llama.block_count", GGUF_TYPE_UINT32, n_layers)
    write_kv(buf, "llama.feed_forward_length", GGUF_TYPE_UINT32, embed_dim * 4)
    write_kv(buf, "llama.attention.head_count", GGUF_TYPE_UINT32, n_heads)
    write_kv(buf, "llama.attention.head_count_kv", GGUF_TYPE_UINT32, n_heads)

    # Tokenizer
    tokens = [f"<tok_{i}>" for i in range(vocab_size)]
    write_array(buf, "tokenizer.ggml.tokens", GGUF_TYPE_STRING, tokens)
    write_array(buf, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, [0.0] * vocab_size)

    # ---- Tensor info ----
    align = 32
    tensor_names_dims = []
    tensor_names_dims.append(("token_embd.weight", [vocab_size, embed_dim], GGML_TYPE_F32))
    for layer in range(n_layers):
        # Attention: Q, K, V all [embed_dim, embed_dim] (n_heads * head_dim = embed_dim)
        tensor_names_dims.append((f"blk.{layer}.attn_q.weight", [embed_dim, embed_dim], GGML_TYPE_F16))
        tensor_names_dims.append((f"blk.{layer}.attn_k.weight", [embed_dim, embed_dim], GGML_TYPE_F16))
        tensor_names_dims.append((f"blk.{layer}.attn_v.weight", [embed_dim, embed_dim], GGML_TYPE_F16))
        tensor_names_dims.append((f"blk.{layer}.attn_output.weight", [embed_dim, embed_dim], GGML_TYPE_F16))
        # FFN
        tensor_names_dims.append((f"blk.{layer}.ffn_up.weight", [embed_dim * 4, embed_dim], GGML_TYPE_F16))
        tensor_names_dims.append((f"blk.{layer}.ffn_down.weight", [embed_dim, embed_dim * 4], GGML_TYPE_F16))

    def info_size(name, n_dims):
        name_bytes = len(name.encode("utf-8"))
        return 8 + name_bytes + 4 + (8 * n_dims) + 4 + 8

    total_info_size = sum(info_size(n, len(d)) for n, d, _ in tensor_names_dims)
    data_start = len(buf) + total_info_size
    data_start = (data_start + align - 1) // align * align

    offset = 0
    for name, dims, ttype in tensor_names_dims:
        write_tensor_info(buf, name, len(dims), dims, ttype, offset)
        n_elem = 1
        for d in dims:
            n_elem *= d
        if ttype == GGML_TYPE_F32:
            sz = n_elem * 4
        elif ttype == GGML_TYPE_F16:
            sz = n_elem * 2
        else:
            sz = n_elem * 4
        offset += (sz + align - 1) // align * align

    while len(buf) < data_start:
        buf.append(0)

    for name, dims, ttype in tensor_names_dims:
        n_elem = 1
        for d in dims:
            n_elem *= d
        if ttype == GGML_TYPE_F32:
            arr = np.random.randn(n_elem).astype(np.float32)
            buf.extend(arr.tobytes())
        else:
            arr = np.random.randn(n_elem).astype(np.float16)
            buf.extend(arr.tobytes())
        while len(buf) % align != 0:
            buf.append(0)

    with open(out_path, "wb") as f:
        f.write(buf)
    return out_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/z/my-project/download/test_tiny.gguf"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    make_test_gguf(out, vocab_size=128, embed_dim=64, n_layers=3)
    print(f"Wrote {out} ({len(open(out,'rb').read())} bytes)")
