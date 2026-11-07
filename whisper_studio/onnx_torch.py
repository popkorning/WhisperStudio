"""Run the speaker embedding model (WeSpeaker ResNet293, embedding.onnx) with torch on the GPU.

The main venv has onnxruntime for the CPU only, but torch with CUDA: this reads the ONNX graph
itself (a minimal protobuf reader, no `onnx` package) and executes its few operators with
torch. The convolutions run in fp16, channels-last (RTX 2060, 10 s input: 48 ms against
109 ms in fp32, onnxruntime-gpu fp32 121 ms; cosine to fp32 0.99998), the statistics pooling
and the last layer in fp32, so the means and variances over 1000 frames don't lose precision.
Activations are freed after their last use. Only the operators of this model are implemented;
anything else raises NotImplementedError and the caller falls back to onnxruntime.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_FLOAT, _INT64 = 1, 7


def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def _fields(b):
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        f, w = key >> 3, key & 7
        if w == 0:
            v, i = _varint(b, i)
        elif w == 1:
            v, i = b[i:i + 8], i + 8
        elif w == 2:
            ln, i = _varint(b, i)
            v, i = b[i:i + ln], i + ln
        elif w == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"protobuf wire type {w}")
        yield f, w, v


def _ints(w, v) -> list[int]:
    if w == 0:
        return [v if v < 1 << 63 else v - (1 << 64)]
    out, i = [], 0
    while i < len(v):
        x, i = _varint(v, i)
        out.append(x if x < 1 << 63 else x - (1 << 64))
    return out


def _tensor(b) -> tuple[str, np.ndarray]:
    dims, dtype, raw, name, fl, il = [], _FLOAT, None, "", [], []
    for f, w, v in _fields(b):
        if f == 1:
            dims += _ints(w, v)
        elif f == 2:
            dtype = v
        elif f == 4:
            fl += list(np.frombuffer(bytes(v), "<f4")) if w == 2 else [np.frombuffer(bytes(v), "<f4")[0]]
        elif f == 7:
            il += _ints(w, v)
        elif f == 8:
            name = bytes(v).decode()
        elif f == 9:
            raw = bytes(v)
    np_t = {_FLOAT: "<f4", _INT64: "<i8"}.get(dtype)
    if np_t is None:
        raise NotImplementedError(f"tensor type {dtype}")
    if raw is not None:
        a = np.frombuffer(raw, np_t).copy()
    else:
        a = np.array(fl if dtype == _FLOAT else il, dtype=np_t)
    return name, a.reshape(dims)


def _attr(b):
    name, val = "", None
    ints = []
    for f, w, v in _fields(b):
        if f == 1:
            name = bytes(v).decode()
        elif f == 2:
            val = float(np.frombuffer(bytes(v), "<f4")[0])
        elif f == 3:
            val = _ints(0, v)[0]
        elif f == 5:
            val = _tensor(v)[1]
        elif f == 8:
            ints += _ints(w, v)
    return name, ints if ints else val


def load_graph(path: Path):
    data = memoryview(Path(path).read_bytes())
    graph = next(v for f, _w, v in _fields(data) if f == 7)
    nodes, inits, inputs, outputs = [], {}, [], []
    for f, _w, v in _fields(graph):
        if f == 1:
            node = dict(op=None, inputs=[], outputs=[], attrs={})
            for g, _x, u in _fields(v):
                if g == 1:
                    node["inputs"].append(bytes(u).decode())
                elif g == 2:
                    node["outputs"].append(bytes(u).decode())
                elif g == 4:
                    node["op"] = bytes(u).decode()
                elif g == 5:
                    k, a = _attr(u)
                    node["attrs"][k] = a
            nodes.append(node)
        elif f == 5:
            k, a = _tensor(v)
            inits[k] = a
        elif f in (11, 12):
            name = next(bytes(u).decode() for g, _x, u in _fields(v) if g == 1)
            (inputs if f == 11 else outputs).append(name)
    inputs = [i for i in inputs if i not in inits]
    return nodes, inits, inputs, outputs


class TorchModel:
    """embedding.onnx on a torch device; __call__(feats [B, T, 80] float32 numpy) -> [B, 256]."""

    FP32_OPS = {"ReduceMean", "ReduceProd", "Shape", "Gather", "Cast", "Sub", "Mul", "Div", "Sqrt",
                "Flatten", "Concat", "Gemm"}

    def __init__(self, path: Path, device: str = "cuda", half: bool = True, channels_last: bool = False):
        import torch

        self.torch = torch
        self.device = device
        self.cl = channels_last
        self.half = half
        nodes, inits, self.inputs, self.outputs = load_graph(path)
        known = {"Conv", "Relu", "Add", "Constant", "Transpose", "Unsqueeze"} | self.FP32_OPS
        unknown = {n["op"] for n in nodes} - known
        if unknown:
            raise NotImplementedError(f"operators {sorted(unknown)}")
        self.nodes = nodes
        conv_w = {n["inputs"][1] for n in nodes if n["op"] == "Conv"} | {n["inputs"][2] for n in nodes
                                                                          if n["op"] == "Conv" and len(n["inputs"]) > 2}
        self.w = {}
        for k, a in inits.items():
            t = torch.from_numpy(a).to(device)
            if half and k in conv_w:
                t = t.half()
            if channels_last and t.dim() == 4:
                t = t.contiguous(memory_format=torch.channels_last)
            self.w[k] = t
        for n in nodes:
            if n["op"] == "Conv":
                p = n["attrs"].get("pads") or [0, 0, 0, 0]
                if p[:2] != p[2:]:
                    raise NotImplementedError(f"asymmetric pads {p}")
        # activations are dropped after their last use: 300 layers of a 10 s input kept
        # alive would need ~6 GB
        last = {}
        for k, n in enumerate(nodes):
            for i in n["inputs"]:
                last[i] = k
        self.drop = [[i for i in dict.fromkeys(n["inputs"]) if i and last[i] == k and i not in self.w
                      and i not in self.outputs] for k, n in enumerate(nodes)]

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        torch = self.torch
        F = torch.nn.functional
        env = dict(self.w)
        env[self.inputs[0]] = torch.from_numpy(np.ascontiguousarray(feats, np.float32)).to(self.device)
        with torch.inference_mode():
            for n, drop in zip(self.nodes, self.drop):
                op, a = n["op"], n["attrs"]
                xs = [env[i] if i else None for i in n["inputs"]]
                if op in self.FP32_OPS:
                    xs = [x.float() if x is not None and x.is_floating_point() else x for x in xs]
                if op == "Conv":
                    x = xs[0].half() if self.half else xs[0]
                    if self.cl and x.dim() == 4:
                        x = x.contiguous(memory_format=torch.channels_last)
                    p = a.get("pads") or [0, 0, 0, 0]
                    y = F.conv2d(x, xs[1], xs[2] if len(xs) > 2 else None, stride=tuple(a.get("strides", [1, 1])),
                                 padding=tuple(p[:2]), dilation=tuple(a.get("dilations", [1, 1])),
                                 groups=a.get("group", 1))
                elif op == "Relu":
                    y = torch.relu(xs[0])
                elif op == "Add":
                    y = xs[0] + xs[1]
                elif op == "Constant":
                    y = torch.from_numpy(np.asarray(a["value"])).to(self.device)
                elif op == "Transpose":
                    y = xs[0].permute(*a["perm"])
                elif op == "Unsqueeze":
                    axes = a.get("axes") or [int(v) for v in xs[1].reshape(-1).tolist()]
                    y = xs[0]
                    for ax in sorted(axes):
                        y = y.unsqueeze(ax)
                elif op == "ReduceMean":
                    y = xs[0].mean(dim=tuple(a["axes"]), keepdim=bool(a.get("keepdims", 1)))
                elif op == "Shape":
                    y = torch.tensor(list(xs[0].shape), dtype=torch.int64, device=self.device)
                elif op == "Gather":
                    ax = a.get("axis", 0)
                    idx = xs[1].to(xs[0].device).long()
                    idx = torch.where(idx < 0, idx + xs[0].shape[ax], idx)  # ONNX allows negative indices
                    y = torch.index_select(xs[0], ax, idx.reshape(-1)).reshape(
                        xs[0].shape[:ax] + idx.shape + xs[0].shape[ax + 1:])
                elif op == "ReduceProd":
                    y = xs[0].prod()
                    if a.get("keepdims", 1):
                        y = y.reshape([1] * xs[0].dim())
                elif op == "Cast":
                    y = xs[0].to({1: torch.float32, 7: torch.int64}[a["to"]])
                elif op == "Sub":
                    y = xs[0] - xs[1]
                elif op == "Mul":
                    y = xs[0] * xs[1]
                elif op == "Div":
                    y = xs[0] / xs[1]
                elif op == "Sqrt":
                    y = torch.sqrt(xs[0])
                elif op == "Flatten":
                    ax = a.get("axis", 1)
                    y = xs[0].reshape(int(np.prod(xs[0].shape[:ax])), -1)
                elif op == "Concat":
                    y = torch.cat(xs, dim=a["axis"])
                elif op == "Gemm":
                    b = xs[1].t() if a.get("transB") else xs[1]
                    y = a.get("alpha", 1.0) * (xs[0] @ b)
                    if len(xs) > 2:
                        y = y + a.get("beta", 1.0) * xs[2]
                env[n["outputs"][0]] = y
                del xs
                for i in drop:
                    env.pop(i, None)
            out =env[self.outputs[0]].float().cpu().numpy()
        return out
