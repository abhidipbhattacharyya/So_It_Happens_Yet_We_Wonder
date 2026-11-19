#!/usr/bin/env python3
"""
Generate a layer-aligned Graphviz DOT for GPT-style graphs (Option A):
- Columns are: ResidStartCol, layers 0..(n_layers-1), ResidEndCol
- "Resid Start" is placed in its own column BEFORE layer 0
- "Resid End" is placed in its own column AFTER layer (n_layers-1)
Then runs Graphviz `dot` to render (svg/png/pdf).

Input JSON format:
{
  "SRC->DST": score_float,
  ...
}

Node naming conventions supported:
- "MLP 12"                 (layer = 12)
- "A17.13"                 (layer = 17, head = 13)
- "A17.13.V" / ".K" / ".Q" (layer = 17, head = 13)
- "Resid Start", "Resid End"
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List


# ---------- Parsing ----------

@dataclass(frozen=True)
class NodeInfo:
    raw: str
    kind: str              # "resid", "mlp", "attn", "other"
    layer: Optional[int]   # for optionA: resid nodes have layer None (they live in special columns)
    head: Optional[int]
    suffix: Optional[str]  # "Q"/"K"/"V"


_MLP_RE = re.compile(r"^MLP\s+(\d+)$")
_ATTN_RE = re.compile(r"^A(\d+)\.(\d+)(?:\.(Q|K|V))?$")
_RESID_START_RE = re.compile(r"^Resid Start$")
_RESID_END_RE = re.compile(r"^Resid End$")


def parse_node(node: str, n_layers: int, n_heads: int) -> NodeInfo:
    node = node.strip()

    if _RESID_START_RE.match(node):
        return NodeInfo(raw=node, kind="resid", layer=None, head=None, suffix=None)
    if _RESID_END_RE.match(node):
        return NodeInfo(raw=node, kind="resid", layer=None, head=None, suffix=None)

    m = _MLP_RE.match(node)
    if m:
        layer = int(m.group(1))
        if 0 <= layer < n_layers:
            return NodeInfo(raw=node, kind="mlp", layer=layer, head=None, suffix=None)
        return NodeInfo(raw=node, kind="mlp", layer=None, head=None, suffix=None)

    m = _ATTN_RE.match(node)
    if m:
        layer = int(m.group(1))
        head = int(m.group(2))
        suffix = m.group(3)
        ok_layer = 0 <= layer < n_layers
        ok_head = 0 <= head < n_heads
        return NodeInfo(
            raw=node,
            kind="attn",
            layer=layer if ok_layer else None,
            head=head if ok_head else None,
            suffix=suffix,
        )

    return NodeInfo(raw=node, kind="other", layer=None, head=None, suffix=None)


def parse_edge_key(edge_key: str) -> Tuple[str, str]:
    if "->" not in edge_key:
        raise ValueError(f"Bad edge key (missing '->'): {edge_key}")
    src, dst = edge_key.split("->", 1)
    return src.strip(), dst.strip()


# ---------- DOT generation (Option A) ----------

def _dot_node_id(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", label)
    return f"n_{safe}"


def generate_dot_option_a(
    json_path: str | Path,
    n_layers: int,
    n_heads: int,
    out_dir: str | Path | None = None,
    graph_name: str = "GPT_Graph_OptionA",
) -> Path:
    """
    Reads edges from json_path, generates a layer-aligned DOT (Option A),
    writes it to disk, and returns the DOT file path.
    """
    json_path = Path(json_path)
    if out_dir is None:
        out_dir = json_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        edges: Dict[str, float] = json.load(f)

    # Collect nodes + parsed edges
    nodes: Dict[str, NodeInfo] = {}
    parsed_edges: List[Tuple[str, str, float]] = []
    for k, score in edges.items():
        src, dst = parse_edge_key(k)
        parsed_edges.append((src, dst, float(score)))
        if src not in nodes:
            nodes[src] = parse_node(src, n_layers, n_heads)
        if dst not in nodes:
            nodes[dst] = parse_node(dst, n_layers, n_heads)

    # Group nodes by layer plus special columns
    by_layer: Dict[int, List[str]] = {i: [] for i in range(n_layers)}
    resid_start_nodes: List[str] = []
    resid_end_nodes: List[str] = []
    unlayered: List[str] = []

    for label, info in nodes.items():
        if label == "Resid Start":
            resid_start_nodes.append(label)
        elif label == "Resid End":
            resid_end_nodes.append(label)
        elif info.layer is None:
            unlayered.append(label)
        else:
            by_layer[info.layer].append(label)

    for i in range(n_layers):
        by_layer[i].sort(key=lambda s: (nodes[s].kind, s))
    resid_start_nodes.sort()
    resid_end_nodes.sort()
    unlayered.sort()

    def node_style(label: str) -> str:
        info = nodes[label]
        if info.kind == "resid":
            if label == "Resid Start":
                return 'fillcolor="#e8f0ff"'
            if label == "Resid End":
                return 'fillcolor="#ffe8e8"'
            return 'fillcolor="#eeeeee"'
        if info.kind == "mlp":
            return 'fillcolor="#eaffea"'
        if info.kind == "attn":
            return 'fillcolor="#fff7e6"'
        return 'fillcolor="#f7f7f7"'

    dot_lines: List[str] = []
    dot_lines.append(f"digraph {graph_name} {{")
    dot_lines.append("  rankdir=LR;")
    dot_lines.append("  splines=true;")
    dot_lines.append("  nodesep=0.35;")
    dot_lines.append("  ranksep=1.15;")
    dot_lines.append('  bgcolor="white";')
    dot_lines.append('  graph [fontname="Helvetica"];')
    dot_lines.append('  node  [fontname="Helvetica", shape=box, style="rounded,filled", fillcolor="#f7f7f7", color="#444444"];')
    dot_lines.append('  edge  [fontname="Helvetica", color="#444444", arrowsize=0.7];')
    dot_lines.append("")

    # Column anchors
    dot_lines.append('  { rank=same; Lm1 [label="Resid Start", shape=plaintext]; }')
    for i in range(n_layers):
        dot_lines.append(f'  {{ rank=same; L{i} [label="Layer {i}", shape=plaintext]; }}')
    dot_lines.append('  { rank=same; Lp1 [label="Resid End", shape=plaintext]; }')
    dot_lines.append("")

    # Keep columns ordered
    dot_lines.append("  Lm1 -> L0 [style=invis];")
    for i in range(n_layers - 1):
        dot_lines.append(f"  L{i} -> L{i+1} [style=invis];")
    dot_lines.append(f"  L{n_layers-1} -> Lp1 [style=invis];")
    dot_lines.append("")

    # Resid Start column nodes
    if resid_start_nodes:
        dot_lines.append("  { rank=same; Lm1;")
        for label in resid_start_nodes:
            nid = _dot_node_id(label)
            dot_lines.append(f'    {nid} [label="{label}", {node_style(label)}];')
        dot_lines.append("  }")
        dot_lines.append("")

    # Layer columns nodes
    for i in range(n_layers):
        layer_nodes = by_layer[i]
        if not layer_nodes:
            continue
        dot_lines.append(f"  {{ rank=same; L{i};")
        for label in layer_nodes:
            nid = _dot_node_id(label)
            dot_lines.append(f'    {nid} [label="{label}", {node_style(label)}];')
        dot_lines.append("  }")
        dot_lines.append("")

    # Resid End column nodes
    if resid_end_nodes:
        dot_lines.append("  { rank=same; Lp1;")
        for label in resid_end_nodes:
            nid = _dot_node_id(label)
            dot_lines.append(f'    {nid} [label="{label}", {node_style(label)}];')
        dot_lines.append("  }")
        dot_lines.append("")

    # Unlayered nodes
    if unlayered:
        dot_lines.append("  // Unlayered nodes (could not infer layer)")
        dot_lines.append("  { rank=sink;")
        for label in unlayered:
            nid = _dot_node_id(label)
            dot_lines.append(f'    {nid} [label="{label}", {node_style(label)}];')
        dot_lines.append("  }")
        dot_lines.append("")

    # Edges
    dot_lines.append("  // Edges (only from JSON)")
    for src, dst, score in parsed_edges:
        dot_lines.append(f'  {_dot_node_id(src)} -> {_dot_node_id(dst)} [label="{score:.3f}"];')

    dot_lines.append("}")

    dot_path = out_dir / f"{json_path.stem}.optionA.dot"
    dot_path.write_text("\n".join(dot_lines), encoding="utf-8")
    return dot_path


# ---------- Rendering ----------

def run_dot(dot_path: str | Path, fmt: str = "svg", dpi: int = 200) -> Path:
    """
    Runs Graphviz dot to render dot_path into fmt (svg/png/pdf).
    Returns output path.
    """
    dot_path = Path(dot_path)
    fmt = fmt.lower()
    if fmt not in {"svg", "png", "pdf"}:
        raise ValueError("fmt must be one of: svg, png, pdf")

    out_path = dot_path.with_suffix(f".{fmt}")

    cmd = ["dot", f"-T{fmt}", str(dot_path), "-o", str(out_path)]
    if fmt == "png":
        cmd.insert(2, f"-Gdpi={dpi}")

    subprocess.run(cmd, check=True)
    return out_path


# ---------- CLI ----------
# python generate_graph_optionA.py edges.json --layers 24 --heads 16 --fmt svg
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("json_path", help="Path to JSON edge dict")
    p.add_argument("--layers", type=int, default=24, help="Number of layers (e.g., 24 for gpt2-medium)")
    p.add_argument("--heads", type=int, default=16, help="Attention heads per layer (e.g., 16 for gpt2-medium)")
    p.add_argument("--out-dir", default=None, help="Directory to write DOT and rendered output")
    p.add_argument("--fmt", default="svg", choices=["svg", "png", "pdf"], help="Output format")
    p.add_argument("--dpi", type=int, default=300, help="DPI for PNG")
    args = p.parse_args()

    dot_file = generate_dot_option_a(args.json_path, args.layers, args.heads, out_dir=args.out_dir)
    print(f"Wrote DOT: {dot_file}")

    try:
        out_img = run_dot(dot_file, fmt=args.fmt, dpi=args.dpi)
        print(f"Rendered: {out_img}")
        dot_file.unlink(missing_ok=True)  # delete DOT from disk to save storage
    except FileNotFoundError:
        raise SystemExit(
            "Graphviz `dot` not found. Install Graphviz and ensure `dot` is on PATH.\n"
            "macOS: brew install graphviz\n"
            "Ubuntu/Debian: sudo apt-get install graphviz\n"
            "Windows: winget install Graphviz.Graphviz"
        )