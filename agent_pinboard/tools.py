"""Read-side graph tools exposed to the LLM agent.

``make_graph_tools(hooks=...)`` returns a list of LangChain ``BaseTool``
instances for the agent to call. They are the *only* read API for the
graph from inside an agent — the LLM uses them to navigate facts,
events, and the tool-call log.

Phase 1 ships:

* ``explore`` — subgraph around an entity, ``skip_events=True`` by default
* ``timeline`` — events involving an entity, chronological
* ``graph_summary`` — known types from the registry + per-type top-N facts
* ``search_nodes`` — listing/glob filter, EventNodes hidden by default
* ``what_have_i_done`` — tool-call log filter

``find_path`` and ``get_evidence`` are deferred (Phase 2 / Phase 3).

Output style: compact plain text. Edge ``description`` from ``node()``
is always shown, since field names alone are not a contract for the LLM.
"""

from __future__ import annotations

import fnmatch

import networkx as nx
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolRuntime

from pinboard import store as store_io
from pinboard.enums import Direction
from pinboard.exceptions import PinBoardConfigError
from pinboard.graph import FactGraph
from pinboard.hooks import PinBoardHooks
from pinboard.models import EVENT_NODE_TYPE, EventNode, FactEdge, FactNode
from pinboard.registry import known_entities
from pinboard.session import get_or_load_session, lock_for, thread_id_from

# --------------------------------------------------------------------------- #
# Public factory.                                                             #
# --------------------------------------------------------------------------- #

def make_graph_tools(hooks: PinBoardHooks | None = None) -> list[BaseTool]:
    """Build the per-session set of graph-read tools.

    ``hooks`` is accepted for symmetry with ``@fact``, but the read tools
    do not currently mutate the graph and so do not invoke them.
    """

    @tool
    def explore(
        node_type: str,
        value: str,
        depth: int = 2,
        direction: str = Direction.BOTH.value,
        skip_events: bool = True,
        max_nodes: int = 30,
        runtime: ToolRuntime = None,  # type: ignore[assignment]
    ) -> str:
        """Show the subgraph around a fact node.

        Default behaviour treats event nodes as transparent — the result
        is the set of facts that share at least one event with the start.
        Set ``skip_events=False`` to see event nodes themselves.
        """
        graph = _graph(runtime)
        start_id = graph.find_by_value(node_type, value)
        if start_id is None:
            return _no_node(node_type, value)
        return _render_explore(
            graph, start_id, depth, Direction(direction), skip_events, max_nodes
        )

    @tool
    def find_path(
        from_type: str,
        from_value: str,
        to_type: str,
        to_value: str,
        max_depth: int = 6,
        skip_events: bool = True,
        top: int = 1,
        runtime: ToolRuntime = None,  # type: ignore[assignment]
    ) -> str:
        """Shortest path(s) between two fact nodes.

        With ``skip_events=True`` (default) the search runs on the
        fact-only projection: two facts are adjacent iff they share an
        event, so a returned path of length N means "N hops between
        facts". With ``skip_events=False`` the underlying
        ``MultiDiGraph`` is walked undirected, and EventNodes consume
        hops.

        ``top`` — return at most this many distinct paths in
        non-decreasing length order (Yen's algorithm via networkx
        ``shortest_simple_paths``). Default 1 = single shortest path.

        ``max_depth`` — paths longer than this are discarded.

        Returns a text rendering. If no path exists within the bounds,
        returns a hint message instead of raising.
        """
        graph = _graph(runtime)
        start_id = graph.find_by_value(from_type, from_value)
        end_id = graph.find_by_value(to_type, to_value)
        if start_id is None:
            return _no_node(from_type, from_value)
        if end_id is None:
            return _no_node(to_type, to_value)
        if start_id == end_id:
            return (
                f"find_path: {from_type}={from_value!r} and "
                f"{to_type}={to_value!r} are the same node — already there."
            )
        return _render_find_path(
            graph, start_id, end_id, max_depth, skip_events, max(1, top),
        )

    @tool
    def timeline(
        node_type: str,
        value: str,
        limit: int = 50,
        rank: bool = False,
        runtime: ToolRuntime = None,  # type: ignore[assignment]
    ) -> str:
        """List the events in which this entity appeared.

        Default — chronological (oldest first). Set ``rank=True`` to sort
        by the AriGraph relevance score instead:
        ``score = n_i / max(N_i, 1) * log2(max(N_i, 1) + 1)`` where
        ``n_i`` is the number of facts the event touched that are *also*
        graph-neighbours of this entity, and ``N_i`` is the total number
        of facts the event touched. The log factor downweights events
        that drag in a long tail of unrelated facts (e.g. a bulk
        enrichment), per Anokhin et al. (IJCAI 2025).
        """
        graph = _graph(runtime)
        start_id = graph.find_by_value(node_type, value)
        if start_id is None:
            return _no_node(node_type, value)
        node = graph.get(start_id)
        assert isinstance(node, FactNode)

        events: list[EventNode] = []
        for eid in node.source_events:
            ev = graph.get(eid)
            if isinstance(ev, EventNode):
                events.append(ev)

        scores: dict[str, float] = {}
        if rank:
            scores = _arigraph_score_events(graph, start_id, events)
            events.sort(key=lambda e: (-scores.get(e.id, 0.0), e.timestamp))
        else:
            events.sort(key=lambda e: e.timestamp)
        events = events[:limit]
        if not events:
            return f"timeline({node_type}={value!r}): no events recorded"

        order = "by relevance" if rank else "oldest first"
        lines = [f"timeline({node_type}={value!r}, {len(events)} events, {order}):"]
        for ev in events:
            score_part = f"  [score={scores[ev.id]:.3f}]" if rank else ""
            lines.append(f"  {ev.timestamp.isoformat()} via {ev.source_tool}{score_part}")
            if ev.properties:
                lines.append(f"    properties: {ev.properties}")
        return "\n".join(lines)

    @tool
    def graph_summary(
        top_per_type: int = 5,
        runtime: ToolRuntime = None,  # type: ignore[assignment]
    ) -> str:
        """List node types known to this session with counts and top entities.

        Includes types declared via ``node()`` even if no instance has been
        ingested yet — gives the LLM a map of the territory before any
        tool calls.
        """
        graph = _graph(runtime)
        registry = known_entities()
        present_types = {
            t for t in graph.nodes_by_type if t != EVENT_NODE_TYPE
        }
        all_types = sorted(set(registry) | present_types)
        if not all_types:
            return (
                "graph_summary: no entity types declared yet. "
                "Apply @fact(model=...) to a tool to register types."
            )
        lines = ["graph_summary:"]
        for t in all_types:
            ids = graph.nodes_by_type.get(t, set())
            entity = registry.get(t)
            desc = f" — {entity.description}" if entity else ""
            lines.append(f"  {t} ({len(ids)} in graph){desc}")
            top = _top_facts(graph, t, top_per_type)
            for f in top:
                lines.append(
                    f"    {f.value}  (in {len(f.source_events)} events, "
                    f"via {sorted(f.source_tools)})"
                )
        return "\n".join(lines)

    @tool
    def search_nodes(
        node_type: str | None = None,
        value_pattern: str | None = None,
        include_events: bool = False,
        limit: int = 50,
        runtime: ToolRuntime = None,  # type: ignore[assignment]
    ) -> str:
        """Find fact nodes by type and/or value glob (``fnmatchcase``)."""
        graph = _graph(runtime)
        hits: list[FactNode | EventNode] = []
        for ntype, ids in graph.nodes_by_type.items():
            if node_type and ntype != node_type:
                continue
            if not include_events and ntype == EVENT_NODE_TYPE:
                continue
            for nid in ids:
                node = graph.get(nid)
                if not isinstance(node, (FactNode, EventNode)):
                    continue
                cmp_value = (
                    node.canonical_value if isinstance(node, FactNode) else node.id
                )
                if value_pattern and not fnmatch.fnmatchcase(cmp_value, value_pattern):
                    continue
                hits.append(node)
                if len(hits) >= limit:
                    break
            if len(hits) >= limit:
                break
        if not hits:
            tip = ""
            if not include_events:
                tip = (
                    "  (events hidden — pass include_events=True if "
                    "you are searching for tool-call records)"
                )
            return (
                f"search_nodes(node_type={node_type!r}, "
                f"pattern={value_pattern!r}): no matches\n{tip}"
            )
        lines = [
            f"search_nodes(node_type={node_type!r}, pattern={value_pattern!r}):"
        ]
        for n in hits:
            if isinstance(n, EventNode):
                lines.append(f"  Event: id={n.id} via {n.source_tool} at {n.timestamp.isoformat()}")
            else:
                lines.append(
                    f"  {n.node_type}: {n.value}  "
                    f"(in {len(n.source_events)} events, "
                    f"via {sorted(n.source_tools)})"
                )
        return "\n".join(lines)

    @tool
    def get_evidence(
        event_id: str,
        runtime: ToolRuntime = None,  # type: ignore[assignment]
    ) -> str:
        """Return the raw tool return that produced ``event_id``.

        Only available if the producing tool was decorated with
        ``@fact(store_raw=True)``. Otherwise returns a hint pointing at
        the EventNode's structured properties (which carry the non-node
        scalar fields from the parsed model).
        """
        graph = _graph(runtime)
        store, thread_id = _store_and_thread(runtime)
        ev = graph.get(event_id)
        if not isinstance(ev, EventNode):
            return (
                f"get_evidence: no event with id={event_id!r}. "
                "Use timeline() or what_have_i_done() to look up event ids."
            )
        raw = store_io.load_raw_event(store, thread_id, event_id)
        if raw is None:
            return (
                f"get_evidence: raw payload for {event_id!r} was not stored. "
                f"Add `store_raw=True` to @fact on tool {ev.source_tool!r} to "
                "capture full returns. EventNode properties (parsed scalars):\n"
                f"  {ev.properties}"
            )
        import json as _json
        try:
            pretty = _json.dumps(raw, indent=2, default=str)
        except (TypeError, ValueError):
            pretty = repr(raw)
        return (
            f"get_evidence({event_id!r}) — tool={ev.source_tool}, "
            f"timestamp={ev.timestamp.isoformat()}:\n{pretty}"
        )

    @tool
    def what_have_i_done(
        tool_name: str | None = None,
        node_type: str | None = None,
        value: str | None = None,
        limit: int = 50,
        runtime: ToolRuntime = None,  # type: ignore[assignment]
    ) -> str:
        """List tool calls made in this session, optionally filtered.

        Filters combine with AND. ``value`` requires ``node_type`` (a value
        without a type is ambiguous — ``"1"`` could be many things).
        """
        if value is not None and node_type is None:
            return (
                "what_have_i_done: `value` requires `node_type` "
                "(otherwise the filter is ambiguous)"
            )
        graph = _graph(runtime)
        store, thread_id = _store_and_thread(runtime)
        records = store_io.load_tool_calls(store, thread_id)

        # Build the set of event_ids matching the entity filter, if any.
        # If a value is given, normalise it via the type's Entity.normalizer
        # (if one exists) so callers can search by either raw or canonical form.
        entity_event_ids: set[str] | None = None
        if node_type is not None:
            entity_event_ids = set()
            target_canonical: str | None = None
            if value is not None:
                entity = known_entities().get(node_type)
                target_canonical = (
                    entity.normalizer(value) if entity and entity.normalizer
                    else str(value)
                )
            ids_of_type = graph.nodes_by_type.get(node_type, set())
            for nid in ids_of_type:
                fact = graph.get(nid)
                if not isinstance(fact, FactNode):
                    continue
                if target_canonical is not None and fact.canonical_value != target_canonical:
                    continue
                entity_event_ids.update(fact.source_events)

        filtered = []
        for r in records:
            if tool_name is not None and r.tool_name != tool_name:
                continue
            if entity_event_ids is not None and (
                r.event_id is None or r.event_id not in entity_event_ids
            ):
                continue
            filtered.append(r)
        filtered = filtered[-limit:]

        if not filtered:
            return "what_have_i_done: no matching tool calls"
        lines = [f"what_have_i_done ({len(filtered)} of {len(records)} records):"]
        for r in filtered:
            lines.append(
                f"  {r.timestamp.isoformat()} {r.tool_name}({r.args_repr}) "
                f"-> {r.summary} ({r.duration_ms}ms)"
            )
        return "\n".join(lines)

    return [
        explore, find_path, timeline,
        graph_summary, search_nodes,
        get_evidence, what_have_i_done,
    ]


# --------------------------------------------------------------------------- #
# Helpers.                                                                    #
# --------------------------------------------------------------------------- #

def _graph(runtime: ToolRuntime | None) -> FactGraph:
    store, thread_id = _store_and_thread(runtime)
    with lock_for(thread_id):
        return get_or_load_session(store, thread_id)


def _store_and_thread(runtime: ToolRuntime | None) -> tuple[object, str]:
    if runtime is None:
        raise PinBoardConfigError("graph tool invoked without ToolRuntime")
    store = getattr(runtime, "store", None)
    if store is None:
        raise PinBoardConfigError(
            "graph must be compiled with .compile(store=...) to use graph tools"
        )
    return store, thread_id_from(runtime)


def _no_node(node_type: str, value: str) -> str:
    return (
        f"No node found: {node_type} = {value}\n"
        f"Try: search_nodes(node_type={node_type!r}) to list all "
        f"nodes of this type."
    )


def _arigraph_score_events(
    graph: FactGraph, start_id: str, events: list[EventNode]
) -> dict[str, float]:
    """Compute the AriGraph relevance score per event w.r.t. ``start_id``.

    score(e) = n_i / max(N_i, 1) * log2(max(N_i, 1) + 1)

    where N_i is the total number of facts touched by event ``e`` and
    n_i is how many of those facts are graph-neighbours of the start
    entity (i.e., facts that share at least one *other* event with it).
    """
    import math

    # The start node's neighbour-set: facts that share at least one event
    # with `start_id`. We exclude the start itself.
    start_node = graph.get(start_id)
    if not isinstance(start_node, FactNode):
        return {}
    neighbours: set[str] = set()
    for eid in start_node.source_events:
        for edge in graph.edges_for_event(eid):
            if edge.target_id != start_id:
                neighbours.add(edge.target_id)

    scores: dict[str, float] = {}
    for ev in events:
        targets = {e.target_id for e in graph.edges_for_event(ev.id)}
        n_total = len(targets)
        n_overlap = len(targets & neighbours)
        if n_total == 0:
            scores[ev.id] = 0.0
        else:
            scores[ev.id] = (n_overlap / n_total) * math.log2(n_total + 1)
    return scores


def _top_facts(graph: FactGraph, node_type: str, n: int) -> list[FactNode]:
    facts: list[FactNode] = []
    for nid in graph.nodes_by_type.get(node_type, set()):
        node = graph.get(nid)
        if isinstance(node, FactNode):
            facts.append(node)
    facts.sort(key=lambda f: -len(f.source_events))
    return facts[:n]


def _render_explore(
    graph: FactGraph,
    start_id: str,
    depth: int,
    direction: Direction,
    skip_events: bool,
    max_nodes: int,
) -> str:
    """BFS the subgraph around ``start_id`` to ``depth`` hops in ``direction``.

    With ``skip_events=True`` the BFS treats EventNodes as transparent —
    one hop = "from a FactNode through any shared event to another
    FactNode". With ``skip_events=False`` the underlying MultiDiGraph
    edges are followed directly, so EventNodes consume hops.

    ``direction``: ``OUT`` follows edges in their natural direction
    (Event → FactNode), ``IN`` follows them backwards, ``BOTH`` follows
    both. EventNodes have only outbound edges in our schema, so for
    fact-only queries the practical effect of ``OUT`` vs ``IN`` differs
    only when ``skip_events=False``.
    """
    start = graph.get(start_id)
    assert isinstance(start, FactNode)

    visited: dict[str, tuple[str, str, str | None]] = {start_id: ("", "", None)}
    # entry: node_id -> (edge_description, source_tool, timestamp_iso) of the
    # *first* edge that reached it. Used purely for rendering, not traversal.

    frontier: set[str] = {start_id}
    overflow_remaining = False

    for _ in range(max(0, depth)):
        next_frontier: set[str] = set()
        for nid in frontier:
            for target_id, desc, src_tool, ts_iso in _neighbours(
                graph, nid, direction, skip_events
            ):
                if target_id in visited:
                    continue
                visited[target_id] = (desc, src_tool, ts_iso)
                next_frontier.add(target_id)
                if len(visited) - 1 >= max_nodes:  # -1 to exclude the start
                    overflow_remaining = True
                    break
            if overflow_remaining:
                break
        if overflow_remaining:
            break
        if not next_frontier:
            break
        frontier = next_frontier

    visited.pop(start_id, None)
    rendered = list(visited.items())
    overflow = 1 if overflow_remaining else 0

    lines = [
        f"explore({start.node_type}={start.value!r}, depth={depth}, "
        f"direction={direction.value}, skip_events={skip_events}):",
        f"  {start.node_type}: {start.value}  (in {len(start.source_events)} events)",
    ]
    if not rendered:
        lines.append("  (no related facts within the requested depth)")
        return "\n".join(lines)

    lines.append("  Related facts:")
    for nid, (desc, src_tool, ts_iso) in rendered:
        node = graph.get(nid)
        if node is None:
            continue
        type_label = node.node_type
        value = (
            node.value if isinstance(node, FactNode)
            else f"id={node.id} via {node.source_tool}"
        )
        desc_part = f" — {desc!r}" if desc else ""
        lines.append(f"    {type_label}: {value}")
        if src_tool:
            ts = f"@{ts_iso}" if ts_iso else ""
            lines.append(f"      via {src_tool}{ts}{desc_part}")
    if overflow:
        lines.append(
            "  ... more nodes elided (raise max_nodes to see them)"
        )
    return "\n".join(lines)


def _neighbours(
    graph: FactGraph,
    node_id: str,
    direction: Direction,
    skip_events: bool,
):
    """Yield ``(target_id, edge_description, source_tool, timestamp_iso)``.

    With ``skip_events=True`` and the start being a FactNode, neighbours
    are the FactNodes that share an event with it; the source_tool /
    timestamp returned describe the linking event.
    """
    g = graph.g
    src_node = graph.get(node_id)

    if skip_events and isinstance(src_node, FactNode):
        # FactNode → walk through every event it touches → other FactNodes.
        # Direction filters which side of that event we look at:
        # OUT — events in src_node.source_events, then their targets.
        # IN  — same set of events (FactNodes don't have inbound edges from
        #       anything else), so direction collapses on a star topology.
        for ev_id in src_node.source_events:
            ev = graph.get(ev_id)
            if not isinstance(ev, EventNode):
                continue
            for edge in graph.edges_for_event(ev_id):
                if edge.target_id == node_id:
                    continue
                yield (edge.target_id, edge.description, ev.source_tool, ev.timestamp.isoformat())
        return

    if skip_events and isinstance(src_node, EventNode):
        # Walking from an event with skip_events=True is unusual but
        # well-defined: just return its targets.
        for edge in graph.edges_for_event(node_id):
            yield (edge.target_id, edge.description, src_node.source_tool, src_node.timestamp.isoformat())
        return

    # skip_events=False: use the underlying MultiDiGraph directly.
    if direction in (Direction.OUT, Direction.BOTH):
        for _, target_id, data in g.out_edges(node_id, data=True):
            edge = data.get("obj")
            desc = edge.description if isinstance(edge, FactEdge) else ""
            tool, ts = _edge_provenance(graph, edge)
            yield (target_id, desc, tool, ts)
    if direction in (Direction.IN, Direction.BOTH):
        for source_id, _, data in g.in_edges(node_id, data=True):
            edge = data.get("obj")
            desc = edge.description if isinstance(edge, FactEdge) else ""
            tool, ts = _edge_provenance(graph, edge)
            yield (source_id, desc, tool, ts)


def _edge_provenance(graph: FactGraph, edge: FactEdge | None) -> tuple[str, str | None]:
    if edge is None:
        return "", None
    ev = graph.get(edge.event_id)
    if isinstance(ev, EventNode):
        return ev.source_tool, ev.timestamp.isoformat()
    return "", None


def _render_find_path(
    graph: FactGraph,
    start_id: str,
    end_id: str,
    max_depth: int,
    skip_events: bool,
    top: int,
) -> str:
    """Compute and render up to ``top`` shortest simple paths."""
    nx_graph, edge_metadata = _build_path_graph(graph, skip_events)
    if start_id not in nx_graph or end_id not in nx_graph:
        # Either node has no edges in the projected/full graph.
        return _format_no_path(graph, start_id, end_id, max_depth, skip_events)

    paths: list[list[str]] = []
    try:
        for path in nx.shortest_simple_paths(nx_graph, start_id, end_id):
            if len(path) - 1 > max_depth:
                break
            paths.append(path)
            if len(paths) >= top:
                break
    except nx.NetworkXNoPath:
        return _format_no_path(graph, start_id, end_id, max_depth, skip_events)

    if not paths:
        return _format_no_path(graph, start_id, end_id, max_depth, skip_events)

    start = graph.get(start_id)
    end = graph.get(end_id)
    assert isinstance(start, FactNode) and isinstance(end, FactNode)

    lines = [
        f"find_path({start.node_type}={start.value!r} → "
        f"{end.node_type}={end.value!r}, top={top}, "
        f"max_depth={max_depth}, skip_events={skip_events}): "
        f"found {len(paths)} path(s)"
    ]
    for i, path in enumerate(paths, 1):
        lines.append(f"  Path {i} ({len(path) - 1} hop{'s' if len(path) != 2 else ''}):")
        for j, nid in enumerate(path):
            node = graph.get(nid)
            if node is None:
                continue
            label = _path_node_label(node)
            if j == 0:
                lines.append(f"    {label}")
            else:
                via = edge_metadata.get((path[j - 1], nid)) or edge_metadata.get((nid, path[j - 1]))
                via_part = f"  via {via}" if via else ""
                lines.append(f"      ↓{via_part}")
                lines.append(f"    {label}")
    return "\n".join(lines)


def _path_node_label(node: FactNode | EventNode) -> str:
    if isinstance(node, EventNode):
        return f"Event: {node.source_tool}@{node.timestamp.isoformat()}"
    return f"{node.node_type}: {node.value}"


def _format_no_path(
    graph: FactGraph,
    start_id: str,
    end_id: str,
    max_depth: int,
    skip_events: bool,
) -> str:
    start = graph.get(start_id)
    end = graph.get(end_id)
    assert isinstance(start, FactNode) and isinstance(end, FactNode)
    return (
        f"find_path: no path from {start.node_type}={start.value!r} to "
        f"{end.node_type}={end.value!r} within {max_depth} hops "
        f"(skip_events={skip_events}).\n"
        f"Try raising max_depth, flipping skip_events, or use "
        f"explore() on either endpoint to see what's reachable."
    )


def _build_path_graph(
    graph: FactGraph, skip_events: bool
) -> tuple[nx.Graph, dict[tuple[str, str], str]]:
    """Build the search graph and a (a, b) → "via …" annotation map.

    With ``skip_events=True`` the fact-only projection is built: two
    facts are adjacent iff they share at least one event. The annotation
    cites the linking event(s).

    With ``skip_events=False`` the full topology is exposed as an
    undirected ``Graph`` (collapsing parallel edges from MultiDiGraph).
    """
    proj = nx.Graph()
    annotations: dict[tuple[str, str], str] = {}

    if skip_events:
        # Fact-only projection.
        for fact in graph.all_facts():
            proj.add_node(fact.id)
        for ev_id in graph.nodes_by_type.get(EVENT_NODE_TYPE, ()):
            ev = graph.get(ev_id)
            edges = graph.edges_for_event(ev_id)
            fact_ids = [e.target_id for e in edges if graph.get(e.target_id) is not None]
            event_label = (
                f"{ev.source_tool}@{ev.timestamp.isoformat()}"
                if isinstance(ev, EventNode)
                else "<unknown event>"
            )
            for i, a in enumerate(fact_ids):
                for b in fact_ids[i + 1:]:
                    proj.add_edge(a, b)
                    key = (a, b) if (a, b) not in annotations else (a, b)
                    if key not in annotations:
                        annotations[key] = event_label
        return proj, annotations

    # Full topology, undirected. Edges already go EventNode → FactNode.
    for nid in graph.g.nodes:
        proj.add_node(nid)
    for src, dst, data in graph.g.edges(data=True):
        proj.add_edge(src, dst)
        edge = data.get("obj")
        if edge is not None and isinstance(edge, FactEdge):
            ev = graph.get(edge.event_id)
            tool_name = ev.source_tool if isinstance(ev, EventNode) else "?"
            annotations[(src, dst)] = f"edge {edge.edge_type!r} from {tool_name}"
    return proj, annotations


__all__ = ["make_graph_tools"]
