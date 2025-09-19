"""
Reusable multi-agent topology executors (star and ring).

Goals:
- Provide simple, fast-to-wire star and ring orchestrations without forcing LangGraph usage.
- Keep state shape flexible: messages + arbitrary context.
- Allow async agent callables that read/update shared state.

Contracts:
- AgentCallable: async def agent(state) -> dict | None
  - Read/Write: agents can read state["messages"] and state["context"].
  - Return a partial dict to merge into state (e.g., {"messages": [...]} or {"context": {...}}) or None.
- Hub/Planner for star: async def plan(state) -> list[str]
  - Return agent ids to fan-out to.
- Aggregate: async def aggregate(state, results) -> dict | None
  - results is a dict[str, dict | None] keyed by agent id; may append messages or synthesize a final answer.

Error modes:
- Any agent exception is captured and stored under results[agent_id] = {"error": str(e)}; execution continues.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TypedDict


class TopologyState(TypedDict, total=False):
    messages: List[Any]
    context: Dict[str, Any]
    loop_count: int
    current_agent: str


AgentCallable = Callable[[TopologyState], Awaitable[Optional[Dict[str, Any]]]]
PlannerCallable = Callable[[TopologyState], Awaitable[List[str]]]
AggregateCallable = Callable[[TopologyState, Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]
StopCondition = Callable[[TopologyState], bool]


def _merge(state: TopologyState, update: Optional[Dict[str, Any]]):
    if not update:
        return
    # Shallow merge top-level keys; nested dicts for context are merged shallowly as well
    for k, v in update.items():
        if k == "context" and isinstance(v, dict):
            state.setdefault("context", {})
            state["context"].update(v)
        elif k == "messages" and isinstance(v, list):
            state.setdefault("messages", [])
            state["messages"].extend(v)
        else:
            state[k] = v  # type: ignore[index]


async def run_star(
    state: TopologyState,
    planner: PlannerCallable,
    agents: Dict[str, AgentCallable],
    aggregate: Optional[AggregateCallable] = None,
) -> TopologyState:
    """Execute a star topology: hub/planner selects N agents, run them (concurrently), then aggregate.

    Args:
        state: shared state (messages/context/etc.).
        planner: async function returning a list of agent ids to execute.
        agents: mapping agent_id -> async callable.
        aggregate: optional aggregator to combine results.
    Returns:
        Updated state after aggregation.
    """
    selected = await planner(state)
    # Deduplicate and keep only known agents
    selected = [a for i, a in enumerate(selected) if a in agents and a not in selected[:i]]

    async def _run_one(aid: str):
        try:
            return aid, await agents[aid](state)
        except Exception as e:  # noqa: BLE001
            return aid, {"error": str(e)}

    results: Dict[str, Any] = {}
    if selected:
        pairs = await asyncio.gather(*[_run_one(a) for a in selected])
        results = {k: v for k, v in pairs}

    if aggregate:
        agg_update = await aggregate(state, results)
        _merge(state, agg_update)
    else:
        # Default: append a simple summary into context
        _merge(state, {"context": {"star_results": results}})

    return state


async def run_ring(
    state: TopologyState,
    agents_in_order: List[Tuple[str, AgentCallable]],
    rounds: int = 1,
    stop_condition: Optional[StopCondition] = None,
) -> TopologyState:
    """Execute a ring topology: iterate agents in fixed order for a number of rounds,
    optionally stopping early when stop_condition(state) is True.

    Args:
        state: shared state.
        agents_in_order: [(agent_id, agent_callable), ...].
        rounds: number of full cycles to run (>=1).
        stop_condition: optional predicate to break early.
    Returns:
        Updated state after the loop.
    """
    if rounds < 1:
        rounds = 1
    state["loop_count"] = 0
    for r in range(rounds):
        state["loop_count"] = r
        for aid, fn in agents_in_order:
            state["current_agent"] = aid
            try:
                update = await fn(state)
                _merge(state, update)
            except Exception as e:  # noqa: BLE001
                _merge(state, {"context": {f"{aid}_error": str(e)}})
            if stop_condition and stop_condition(state):
                return state
    return state
