# Graph tools

`make_graph_tools()` returns five LangChain tools for the LLM agent to
read the graph. They are stateless and stable across calls — register
them alongside your `@fact`-decorated tools and the agent uses them
naturally.

```python
from pinboard import make_graph_tools

tools = [my_fetch_tool, *make_graph_tools()]
```

| Tool | Phase | Purpose |
|---|---|---|
| `graph_summary` | 1 | Map of all known entity types + counts + top entities |
| `search_nodes` | 1 | Listing / glob-filter for fact nodes |
| `explore` | 1 | Subgraph around an entity, configurable depth + direction |
| `timeline` | 1 | Chronological events in which an entity participated |
| `what_have_i_done` | 1 | Filter the tool-call log of this session |
| `find_path` | 2 | Top-N shortest paths between two entities |
| `get_evidence` | 3 | Raw return JSON for a specific event (requires `@fact(store_raw=True)`) |

## Suggested LLM workflow

1. **Discover** — `graph_summary()` first, before any reasoning. Returns
   every type the registered tools can produce, with how many instances
   are currently in the graph.
2. **Locate** — `search_nodes(node_type="IP", value_pattern="185.220.*")`
   to find candidates by glob.
3. **Investigate** — `explore("IP", "185.220.101.42")` to see what's
   connected. Default `skip_events=True` shows facts that share an event
   without making the LLM think about EventNode hops.
4. **Reconstruct** — `timeline("IP", "185.220.101.42")` for chronology.
5. **Self-introspect** — `what_have_i_done(node_type="IP", value="...")`
   when the LLM is unsure whether it already gathered something.

## `graph_summary(top_per_type=5)`

Lists every entity type known to the session — both types declared
through `@fact(model=...)` and types currently present in the graph.
Counts come from the live graph; types with `0 in graph` mean
"the agent could populate this if it tried".

```text
graph_summary:
  Action (2 in graph) — Performed API action
    AssumeRole  (in 1 events, via ['fetch'])
    ListBuckets  (in 1 events, via ['fetch'])
  IP (3 in graph) — IPv4 or IPv6 network address
    185.220.101.42  (in 3 events, via ['fetch', 'vt'])
    45.77.0.1  (in 1 events, via ['vt'])
  User (1 in graph) — Acting principal
    arn:aws:iam::123:user/admin  (in 2 events, via ['fetch'])
```

`top_per_type` caps how many top facts to surface per type. Top is by
"how many events touched this fact" (descending).

## `search_nodes`

```python
search_nodes(
    node_type=None,         # exact match on FactNode.node_type
    value_pattern=None,     # fnmatchcase glob over canonical_value
    include_events=False,   # set True to find specific tool calls
    limit=50,               # global cap on matches
)
```

`fnmatch.fnmatchcase` means the pattern is **case-sensitive** on every
OS, and uses standard Unix globs (`*`, `?`, `[abc]`). For
case-insensitive search, set up your normalizer to lower-case the
canonical form, then pass the pattern in lower case.

EventNodes are hidden by default. Switch `include_events=True` if the
agent specifically wants to look up a tool call.

## `explore`

```python
explore(
    node_type, value,
    depth=2,
    direction=Direction.BOTH,
    skip_events=True,
    max_nodes=30,
)
```

Walks the subgraph around the entity to `depth` hops. With
`skip_events=True` (default), one hop = "from a FactNode through any
shared event to another FactNode" — events are transparent connectors.
With `skip_events=False`, the underlying `MultiDiGraph` is walked
directly, and EventNodes consume hops.

`direction` only matters when `skip_events=False`, because the star
topology means FactNodes only have **inbound** edges from EventNodes.
Concretely:

- `Direction.IN` + `skip_events=False` finds the EventNodes a fact came from.
- `Direction.OUT` + `skip_events=False` finds nothing for a FactNode (no outbound edges).
- `Direction.BOTH` (default) covers both.

## `find_path`

```python
find_path(
    from_type, from_value,
    to_type, to_value,
    max_depth=6,
    skip_events=True,
    top=1,
)
```

Returns the top-N shortest simple paths between two fact nodes (Yen's
algorithm via `networkx.shortest_simple_paths`). With `skip_events=True`
(default) the search runs on the fact-only projection — two facts are
adjacent iff they share an event, so a length-N path means "N hops
between facts". With `skip_events=False` the underlying MultiDiGraph
is walked undirected and EventNodes consume hops.

`top` defaults to 1 (just the shortest). Set to 5 to get up to 5
shortest paths in non-decreasing length order — useful when there are
multiple plausible chains and you want the LLM to compare them.

`max_depth` truncates: paths longer than this are not returned. Default
6 — a sane upper bound on session-scope graphs.

If no path exists within the bounds, returns a hint message instead
of raising.

```text
find_path(IP='185.220.101.42' → IP='8.8.8.8', top=1, max_depth=6, skip_events=True): found 1 path(s)
  Path 1 (1 hop):
    IP: 185.220.101.42
      ↓  via vt_lookup@2026-04-15T12:00:00+00:00
    IP: 8.8.8.8
```

## `get_evidence`

```python
get_evidence(event_id: str)
```

Returns the raw tool return that produced ``event_id`` — only available
if the producing tool was decorated with ``@fact(store_raw=True)``.
Otherwise returns a hint pointing at the EventNode's structured
``properties`` (the non-node scalar fields from the parsed model).

```text
get_evidence('e-1234') — tool=fetch_cloudtrail, timestamp=2026-04-15T14:22:01+00:00:
{
  "src_ip": "185.220.101.42",
  "actor": {"user_arn": "arn:aws:iam::123:user/admin"},
  "action_name": "AssumeRole",
  ...
}
```

`store_raw` is opt-in: most agents don't need full raw payloads, only
forensic / compliance use cases do.

## `timeline`

Lists the events a specific fact participated in. Default order is
chronological (oldest first); pass ``rank=True`` to sort by AriGraph
relevance score instead — events whose other facts overlap with the
entity's neighbourhood rank higher, downweighted by a log factor that
penalises bulk events that drag in unrelated facts.

Each entry shows timestamp, source tool, optional score, and the
event's properties (non-node scalar fields from the model).

```text
timeline(IP='185.220.101.42', 3 events):
  2026-04-15T14:22:01+00:00 via fetch_cloudtrail
    properties: {'action': 'AssumeRole', 'event_time': '...'}
  2026-04-15T14:23:15+00:00 via fetch_cloudtrail
    properties: {'action': 'ListBuckets', 'event_time': '...'}
  2026-04-15T14:24:00+00:00 via vt_lookup
    properties: {}
```

## `what_have_i_done`

Filters the per-session tool-call log. Filters AND together.

```python
what_have_i_done(
    tool_name=None,     # only calls to this tool
    node_type=None,     # only calls that produced a node of this type
    value=None,         # only calls that produced this exact entity
    limit=50,
)
```

`value` requires `node_type` (a value without a type is ambiguous —
`"1"` could be an IP, a user ID, an order number). `value` is
normalised through `Entity.normalizer` before the filter, so callers
can search by either raw or canonical form.

```text
what_have_i_done (3 of 7 records):
  2026-04-15T14:22:01+00:00 fetch({"user": "admin"}) -> +2 nodes, +0 linked, +2 edges (5ms)
  2026-04-15T14:23:15+00:00 fetch({"user": "admin"}) -> duplicate (skipped) (0ms)
  2026-04-15T14:24:00+00:00 vt({"value": "185.220.101.42"}) -> +1 nodes, +1 linked, +2 edges (12ms)
```

## Rendering format

All tools return plain text, not JSON. Format is intentionally stable
and a contract — your agent prompts can rely on the structure. The
edge `description` from `node()` is **always** rendered next to the
related fact, since field names alone are not a contract for the LLM.

## Empty results — never raise

When a tool can't find anything, it returns a text hint, not an
exception. Example:

```text
No node found: IP = 9.9.9.9
Try: search_nodes(node_type='IP') to list all nodes of this type.
```

This keeps the LLM's reasoning chain unbroken — the missing-node case
is just data, not an error.
