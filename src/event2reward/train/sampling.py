"""Query planning for annotated trajectories (the `trajectories` data format)."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import random

from event2reward.inference import causal_history

# target_source of a query -> modeling.labels.ProgressTargetKind
KIND = {"relation": 0, "expert": 1, "stall_success": 2, "completion": 3}


# Event type -> (relation kind, margin), as in the LIBERO indices. Member
# frames lie inside the event; for regressions (recovered or not) they start
# at the loss frame or, without one, 0.5 s after the anchor (5 frames at
# 10 Hz in the LIBERO indices). Regression groups also hold the recovery
# frame as a second reference.
EVENTS = {"progress_stall": (5, 0.02), "no_further_progress": (3, 0.0),
          "unrecovered_regression": (4, 0.1), "regression_and_recovery": (6, 0.1)}


def corrected_time_progress(record, frame):
    """Progress of a successful trajectory with progress stalls: stall time does not count."""
    times, completion = record["times"], record["C"]
    if completion is None:
        raise ValueError("failed trajectories have no absolute progress target")
    def active(t):
        return t - sum(max(0, min(t, times[e["end_frame"]]) - times[e["anchor_frame"]])
                       for e in record["events"] if e["type"] == "progress_stall")
    total = active(times[completion])
    if total <= 0:
        raise ValueError("no positive effective completion time")
    return max(0.0, min(1.0, active(min(times[frame], times[completion])) / total))


def member_frames(record, event):
    a, b = event["anchor_frame"], event["end_frame"]
    if event["type"] == "progress_stall":
        return list(range(a + 1, b))
    if event["type"] == "no_further_progress":
        return list(range(a + 1, b + 1))
    first = event.get("loss_frame") or bisect_left(record["times"], record["times"][a] + 0.5 - 1e-9)
    return list(range(first, b if event["type"] == "regression_and_recovery" else b + 1))


def stall_success(record):
    """Successful, with progress stalls and no other events: its absolute progress is known everywhere."""
    return record["C"] is not None and bool(record["events"]) and all(e["type"] == "progress_stall" for e in record["events"])


def event_inventory(records, config):
    """Usable events per task, and the events left out because they are too short."""
    k = config["data"]["trajectories"]["sampling"]["relations"]["members"]
    usable, skipped = defaultdict(list), []
    for r in records:
        for e in r["events"]:
            needed = k - 1 if e["type"] == "regression_and_recovery" else k
            frames = len(member_frames(r, e))
            if needed >= 1 and frames >= needed:
                usable[r["task"]].append((r, e))
            else:
                reason = ("regression groups need relations.members >= 2" if needed < 1 else
                          f"needs {needed} member frames, has {frames}")
                skipped.append({"uid": r["uid"], "type": e["type"], "anchor_frame": e["anchor_frame"], "reason": reason})
    return usable, skipped


def strata(record):
    end, times = record["C"], record["times"]
    bins = [[] for _ in range(16)]
    for f in range(1, end):
        bins[min(15, int(16 * times[f] / times[end]))].append(f)
    for phase, frames in enumerate(bins):
        if not frames:
            frames.append(max(0, min(end, bisect_right(times, (phase + 0.5) * times[end] / 16) - 1)))
    return [[0], *bins, [end]]


def build_plan(records, config):
    """Plan every microbatch: expert, successful-event and completion queries plus event groups.

    Tasks alternate between microbatches. Goals always come from an independent
    expert of the same task. A task without (usable) events, or without
    successful trajectories with progress stalls, fills those queries with expert frames.
    """
    t, inp = config["training"], config["data"]["trajectories"]
    p, rel = inp["sampling"]["progress"], inp["sampling"]["relations"]
    rng, proxy_rng, history_rng = (random.Random(t["seed"] + i) for i in (0, 171, 271))
    tasks = sorted({r["task"] for r in records})
    lookup = {r["uid"]: r for r in records}
    usable, _ = event_inventory(records, config)
    pools = {}
    orders, event_orders, type_orders = defaultdict(list), defaultdict(list), defaultdict(list)
    visits, turns = Counter(), Counter()
    absolute_budget = sum(p.values())
    for task in tasks:
        rs = [r for r in records if r["task"] == task]
        experts = [r for r in rs if r["role"] == "expert"]
        success = [r for r in rs if r["C"] is not None]
        hesitant = [r for r in rs if stall_success(r)]
        if rel["balance"] == "trajectory":
            # Event trajectories take equal turns; each visit takes its next event.
            by_uid = defaultdict(list)
            for r, e in usable[task]:
                by_uid[r["uid"]].append(e)
            events = [(lookup[uid], es) for uid, es in by_uid.items()]
        else:
            # Event types take equal turns; events of one type take equal turns.
            events = defaultdict(list)
            for r, e in usable[task]:
                events[e["type"]].append((r, e))
        pools[task] = experts, events, success, hesitant
    bins = {r["uid"]: strata(r) for r in records if r["role"] == "expert"}

    def query(r, f, g, target=0.0, source="relation"):
        return {"uid": r["uid"], "frame": f, "goal_uid": g["uid"], "goal_frame": g["C"],
                "target_progress": target, "progress_valid": source != "relation",
                "trend_valid": False, "target_source": source}

    def expert_query(r, experts):
        order = orders[r["uid"]]
        if not order:
            order.extend((phase, gi) for phase in range(18) for gi, g in enumerate(experts) if g["uid"] != r["uid"])
            rng.shuffle(order)
        phase, gi = order.pop()
        f = rng.choice(bins[r["uid"]][phase])
        return query(r, f, experts[gi], r["times"][f] / r["times"][r["C"]], "expert")

    def next_event(task, events):
        if rel["balance"] == "trajectory":
            if not event_orders[task]:
                event_orders[task] = list(range(len(events)))
                rng.shuffle(event_orders[task])
            r, es = events[event_orders[task].pop()]
            e = es[visits[r["uid"]] % len(es)]; visits[r["uid"]] += 1
            return r, e
        types = sorted(events)
        kind = types[turns[task] % len(types)]; turns[task] += 1
        order = type_orders[task, kind]
        if not order:
            order.extend(range(len(events[kind])))
            rng.shuffle(order)
        return events[kind][order.pop()]

    # As for LIBERO experts: an expert frame whose whole history lies inside the
    # trajectory is labeled trend "growth". Only used when loss.trend > 0.
    label_trend = config["loss"]["trend"] > 0
    k = rel["members"]
    plan = []
    for i in range(t["steps"] * t["accumulation"]):
        task = tasks[i % len(tasks)]
        local = i // len(tasks)
        experts, events, success, hesitant = pools[task]
        n_expert = absolute_budget + (0 if events else rel["groups"] * (1 + k))
        clips = [expert_query(experts[(local * absolute_budget + j) % len(experts)], experts)
                 for j in range(n_expert)]
        relations = []
        for group in range(rel["groups"] if events else 0):
            r, e = next_event(task, events)
            kind, margin = EVENTS[e["type"]]
            candidates = member_frames(r, e)
            n = k - 1 if e["type"] == "regression_and_recovery" else k
            members = [rng.choice(candidates[j * len(candidates) // n:(j + 1) * len(candidates) // n]) for j in range(n)]
            start = len(clips); g = experts[(local + group) % len(experts)]
            references = [e["anchor_frame"]] + ([e["end_frame"]] if e["type"] == "regression_and_recovery" else [])
            clips.extend(query(r, f, g) for f in [*references, *members])
            relations.append({"kind": kind, "anchor_index": start,
                              "secondary_anchor_index": start + 1 if e["type"] == "regression_and_recovery" else -1,
                              "member_indices": list(range(start + len(references), start + len(references) + n)),
                              "margin": margin})
        # Rotate replacement slots so every expert keeps equal opportunities.
        for j in range(p["completion"]):
            r = success[(local * p["completion"] + j) % len(success)]
            goals = [g for g in experts if g["uid"] != r["uid"]]
            g = goals[(local // len(success) + j) % len(goals)]
            clips[(local + j) % absolute_budget] = query(r, r["C"], g, 1.0, "completion")
        for j in range(p["stall_success"] if hesitant else 0):
            r = hesitant[(local // max(1, absolute_budget) + j) % len(hesitant)]
            f = proxy_rng.randrange(r["C"] + 1)
            g = experts[(local + j) % len(experts)]
            slot = (local + p["completion"] + j) % absolute_budget
            clips[slot] = query(r, f, g, corrected_time_progress(r, f), "stall_success")
        for q in clips:
            r = lookup[q["uid"]]
            scales = inp["training_history_scales"]
            scale = history_rng.choice(scales) if len(scales) > 1 else scales[0]
            offsets = [o * scale for o in inp["history_offsets"]]
            q["history"] = causal_history(r["times"], q["frame"], offsets)
            if label_trend and q["target_source"] == "expert":
                q["trend_valid"] = r["times"][q["frame"]] >= max(offsets) - 1e-9
            if q["uid"] == q["goal_uid"] or lookup[q["goal_uid"]]["task"] != r["task"]:
                raise ValueError("goal must be an independent same-task training expert")
        plan.append({"microbatch": i, "task": task, "clips": clips, "relations": relations})
    return plan


def supervision(batch, device, terminal_weight=1.0):
    import torch
    qs = batch["clips"]
    tensor = lambda values, dtype: torch.tensor(values, dtype=dtype, device=device)
    result = {"target_progress": tensor([q["target_progress"] for q in qs], torch.float32),
              "progress_valid": tensor([q["progress_valid"] for q in qs], torch.bool),
              "target_trend": tensor([0] * len(qs), torch.long),
              "trend_valid": tensor([q["trend_valid"] for q in qs], torch.bool),
              "progress_target_kind": tensor([KIND[q["target_source"]] for q in qs], torch.long)}
    rs = batch["relations"]
    if rs:
        width = max(len(r["member_indices"]) for r in rs)
        result["relation"] = {
            "anchor_index": tensor([r["anchor_index"] for r in rs], torch.long),
            "secondary_anchor_index": tensor([r["secondary_anchor_index"] for r in rs], torch.long),
            # Regression groups hold one member fewer (their second reference); pad to a rectangle.
            "member_indices": tensor([r["member_indices"] + [0] * (width - len(r["member_indices"])) for r in rs], torch.long),
            "member_valid": tensor([[True] * len(r["member_indices"]) + [False] * (width - len(r["member_indices"]))
                                    for r in rs], torch.bool),
            "relation_kind": tensor([r["kind"] for r in rs], torch.long),
            "margin": tensor([r["margin"] for r in rs], torch.float32),
            "relation_valid": tensor([True] * len(rs), torch.bool),
            "relation_detach_anchor": tensor([True] * len(rs), torch.bool),
            "relation_loss_weight": tensor([terminal_weight if r["kind"] in (3, 4) else 1.0 for r in rs], torch.float32),
            "relation_normalizer_weight": tensor([1.0] * len(rs), torch.float32),
        }
    return result
