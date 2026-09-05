"""External validation against the Elliptic Bitcoin dataset.

Why this exists
---------------
Every number this project produces comes from a simulator we wrote. Separating
events from features removed the crudest circularity, but what counts as
fraudulent behaviour is still our model of it. Only real labelled data opens
that circle, and Elliptic is the one public dataset that has all three
properties at once: real transactions, real fraud labels, and graph structure.

203 769 transaction nodes, 234 355 edges, 49 coarse time steps. About 46 000
nodes carry a label — roughly 4 500 illicit (ransomware, darknet markets,
Ponzi schemes) against 42 000 licit; the rest are unlabelled.

What transfers and what does not
--------------------------------
The transfer is partial, and pretending otherwise would be the same mistake
this project keeps finding in itself.

**Transfers.** Structural features. Elliptic is a payment graph, so density,
convergence, divergence and relay structure are all computable on a node's
neighbourhood.

**Does not transfer.** Two things, for concrete reasons rather than
convenience:

- *Amounts are not exposed.* The 165 columns are anonymised, so value-weighted
  variants are unavailable. Structural features are computed **unweighted**
  here — by edge count rather than by value — and the two are not the same
  quantity. A result on one is evidence about the other, not proof.
- *Time resolution is 49 coarse steps*, each covering roughly two weeks.
  Nothing in the sequence branch survives that: ``burst_ratio_90s`` cannot be
  evaluated on a series whose finest tick is a fortnight.

So this module tests one claim: **does relay structure carry signal on real
fraud data?** It cannot test the speed hypothesis, and it says so.

Data is downloaded on demand into ``data/elliptic`` and is git-ignored; the
features file alone is about 690 MB.
"""

from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

PYG_MIRROR = "https://data.pyg.org/datasets/elliptic"
DEFAULT_DATA_DIR = Path("data") / "elliptic"

EDGE_FILE = "elliptic_txs_edgelist.csv"
CLASS_FILE = "elliptic_txs_classes.csv"
FEATURE_FILE = "elliptic_txs_features.csv"

# Labels in the raw file: "1" illicit, "2" licit, "unknown" unlabelled.
LABEL_ILLICIT = "1"
LABEL_LICIT = "2"


@dataclass(frozen=True)
class EllipticGraph:
    graph: nx.DiGraph
    labels: dict[str, int]          # node -> 1 illicit, 0 licit
    time_steps: dict[str, int]      # node -> 1..49

    def labelled_nodes(self) -> list[str]:
        return [node for node in self.labels if node in self.graph]

    def summary(self) -> dict[str, int]:
        illicit = sum(1 for value in self.labels.values() if value == 1)
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "labelled": len(self.labels),
            "illicit": illicit,
            "licit": len(self.labels) - illicit,
        }


def download_if_missing(data_dir: Path = DEFAULT_DATA_DIR, *, with_features: bool = False) -> None:
    """Fetch the raw CSVs from the public PyG mirror.

    ``with_features`` is off by default: the feature matrix is ~690 MB and
    nothing here reads it except the time-step column, which this module gets
    from a streaming pass rather than a full load.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    wanted = [EDGE_FILE, CLASS_FILE] + ([FEATURE_FILE] if with_features else [])
    for name in wanted:
        target = data_dir / name
        if target.exists():
            continue
        with urllib.request.urlopen(f"{PYG_MIRROR}/{name}.zip", timeout=600) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(data_dir)


def _read_time_steps(path: Path) -> dict[str, int]:
    """Read only the first two columns of the feature file.

    The file is ~690 MB and 167 columns wide; a full parse costs minutes and
    gigabytes for two columns. Streaming and slicing each row costs seconds.
    """
    steps: dict[str, int] = {}
    if not path.exists():
        return steps
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            try:
                steps[row[0]] = int(float(row[1]))
            except ValueError:
                continue  # header row
    return steps


def load_elliptic(data_dir: Path = DEFAULT_DATA_DIR) -> EllipticGraph:
    """Load the graph, its labels and (if present) the coarse time steps."""
    edge_path = data_dir / EDGE_FILE
    class_path = data_dir / CLASS_FILE
    if not edge_path.exists() or not class_path.exists():
        raise FileNotFoundError(
            f"Elliptic files not found in {data_dir}. "
            "Call download_if_missing() first."
        )

    graph = nx.DiGraph()
    with open(edge_path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                graph.add_edge(row[0], row[1])

    labels: dict[str, int] = {}
    with open(class_path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            if row[1] == LABEL_ILLICIT:
                labels[row[0]] = 1
            elif row[1] == LABEL_LICIT:
                labels[row[0]] = 0

    return EllipticGraph(
        graph=graph,
        labels=labels,
        time_steps=_read_time_steps(data_dir / FEATURE_FILE),
    )


# ==========================================================================
# Neighbourhood extraction — the unit of analysis
# ==========================================================================


def neighbourhood(graph: nx.DiGraph, node: str, *, hops: int = 2, cap: int = 400) -> nx.DiGraph:
    """The k-hop neighbourhood of a node, treated as one case.

    Direction is ignored while expanding: a relay structure is only visible
    if both the accounts that fed a node and the ones it fed are included.
    Expansion stops at ``cap`` nodes so a hub with tens of thousands of
    neighbours cannot dominate the run.
    """
    seen = {node}
    frontier = {node}
    for _ in range(hops):
        nxt: set[str] = set()
        for current in frontier:
            nxt.update(graph.successors(current))
            nxt.update(graph.predecessors(current))
            if len(seen) + len(nxt) > cap:
                break
        frontier = nxt - seen
        seen |= nxt
        if len(seen) >= cap:
            break
    return graph.subgraph(list(seen)[:cap]).copy()


# ==========================================================================
# Unweighted structural features — the same shapes, counted not valued
# ==========================================================================

STRUCTURAL_FEATURE_NAMES: tuple[str, ...] = (
    "density",
    "hub_share",
    "fanout_share",
    "relay_share",
    "reciprocity",
)


def structural_features(subgraph: nx.DiGraph) -> dict[str, float]:
    """Topology-only counterparts of the simulator's graph features.

    Unweighted, because Elliptic exposes no amounts. Each name matches the
    quantity it measures; where a value-weighted definition was used on
    simulated data, the counting version is used here and the difference is
    stated rather than glossed over.
    """
    empty = {name: 0.0 for name in STRUCTURAL_FEATURE_NAMES}
    if subgraph.number_of_nodes() < 3 or subgraph.number_of_edges() < 2:
        return empty

    in_degrees = dict(subgraph.in_degree())
    out_degrees = dict(subgraph.out_degree())
    total_in = sum(in_degrees.values())
    total_out = sum(out_degrees.values())

    hub_share = (max(in_degrees.values()) / total_in) if total_in else 0.0
    fanout_share = (max(out_degrees.values()) / total_out) if total_out else 0.0

    sink = max(in_degrees, key=lambda n: in_degrees[n])
    source = max(out_degrees, key=lambda n: out_degrees[n])
    relayed = 0
    if source != sink:
        for intermediary in subgraph.successors(source):
            if intermediary != sink and subgraph.has_edge(intermediary, sink):
                relayed += 1
    relay = relayed / subgraph.number_of_edges()

    return {
        "density": float(nx.density(subgraph)),
        "hub_share": float(hub_share),
        "fanout_share": float(fanout_share),
        "relay_share": float(relay),
        "reciprocity": float(nx.reciprocity(subgraph) or 0.0),
    }
