from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from util import stable_id


@dataclass
class Task:
    id: str
    family: str
    difficulty: str
    seed: int
    prompt: str
    expected: str
    prolog_source: str
    tools: dict[str, int]
    metadata: dict[str, Any]


DIFFICULTY = {
    "easy": {"depth": 3, "distractors": 4, "events": 6},
    "medium": {"depth": 8, "distractors": 18, "events": 18},
    "hard": {"depth": 16, "distractors": 60, "events": 45},
}

STARINTEL_FAMILIES = [
    "starintel_relation_path",
    "starintel_provenance",
    "starintel_actor_loop",
    "starintel_document_policy",
]


def make_task(family: str, difficulty: str, seed: int) -> Task:
    if difficulty not in DIFFICULTY:
        raise ValueError(difficulty)
    fn = {
        "reachability": _reachability,
        "constraints": _constraints,
        "state": _state,
        "authorization": _authorization,
        "starintel_relation_path": _starintel_relation_path,
        "starintel_provenance": _starintel_provenance,
        "starintel_actor_loop": _starintel_actor_loop,
        "starintel_document_policy": _starintel_document_policy,
    }[family]
    return fn(difficulty, seed)


def _reachability(difficulty: str, seed: int) -> Task:
    cfg = DIFFICULTY[difficulty]
    rng = random.Random(seed)
    depth = cfg["depth"]
    edges = [(f"n{i}", f"n{i+1}") for i in range(depth)]
    for _ in range(cfg["distractors"]):
        a = f"d{rng.randrange(max(2, depth * 3))}"
        b = f"d{rng.randrange(max(2, depth * 3))}"
        if a != b:
            edges.append((a, b))
    target = f"n{depth}"
    lines = ["EDGES:"] + [f"  {a} -> {b}" for a, b in edges]
    prompt = "\n".join(lines) + f"\n\nQUESTION: Is {target} reachable from n0? Answer YES or NO."
    facts = "\n".join(f"edge({a},{b})." for a, b in edges)
    prolog = facts + "\nreachable(A,B):-edge(A,B).\nreachable(A,B):-edge(A,C),reachable(C,B).\n"
    return Task(
        id=stable_id("reachability", difficulty, seed),
        family="reachability",
        difficulty=difficulty,
        seed=seed,
        prompt=prompt,
        expected="YES",
        prolog_source=prolog,
        tools={"reachable": 2},
        metadata={"logical_depth": depth, "distractors": cfg["distractors"]},
    )


def _constraints(difficulty: str, seed: int) -> Task:
    cfg = DIFFICULTY[difficulty]
    rng = random.Random(seed)
    size = min(6, max(3, cfg["depth"] // 2))
    values = list(range(1, size + 1))
    forbidden = sorted({(rng.randrange(size), rng.randrange(size)) for _ in range(size + cfg["distractors"] // 4)})
    forbidden = [(a, b) for a, b in forbidden if a != b]
    target = values[:]
    rng.shuffle(target)
    constraints = []
    for a, b in forbidden:
        constraints.append((a, b, target[a] != target[b]))
    clauses = [f"X{i}\\=X{j}" for i in range(size) for j in range(i + 1, size)]
    order_pairs = []
    for i in range(size - 1):
        a, b = i, i + 1
        op = "<" if target[a] < target[b] else ">"
        order_pairs.append((a, b, op))
        clauses.append(f"X{a}{op}X{b}")
    vars_ = ",".join(f"X{i}" for i in range(size))
    domains = ",".join(f"between(1,{size},X{i})" for i in range(size))
    prolog = f"solution([{vars_}]):-{domains},{','.join(clauses)}.\nexists_solution:-solution(_).\n"
    human = [f"Assign each of A0..A{size-1} a distinct integer from 1..{size}."]
    for a, b, op in order_pairs:
        human.append(f"A{a} must be {'less than' if op == '<' else 'greater than'} A{b}.")
    human.append("QUESTION: Does at least one valid assignment exist? Answer YES or NO.")
    return Task(
        id=stable_id("constraints", difficulty, seed),
        family="constraints",
        difficulty=difficulty,
        seed=seed,
        prompt="\n".join(human),
        expected="YES",
        prolog_source=prolog,
        tools={"exists_solution": 0},
        metadata={"variables": size, "constraint_count": len(clauses)},
    )


def _state(difficulty: str, seed: int) -> Task:
    cfg = DIFFICULTY[difficulty]
    rng = random.Random(seed)
    value = rng.randrange(-10, 11)
    start = value
    events: list[tuple[int, str, int]] = []
    for t in range(1, cfg["events"] + 1):
        if rng.random() < 0.18:
            n = rng.randrange(-20, 21)
            value = n
            events.append((t, "set", n))
        else:
            n = rng.randrange(-6, 7)
            value += n
            events.append((t, "add", n))
    facts = [f"initial({start})."]
    human = [f"Initial counter value: {start}."]
    for t, kind, n in events:
        facts.append(f"event({t},{kind},{n}).")
        human.append(f"t={t}: {kind.upper()} {n}")
    prolog = "\n".join(facts) + "\n" + r'''
state(0,V):-initial(V).
state(T,V):-T>0,T0 is T-1,state(T0,V0),(event(T,set,N)->V=N;event(T,add,N)->V is V0+N;V=V0).
final_value(V):-findall(T,event(T,_,_),Ts),max_list([0|Ts],M),state(M,V).
'''
    human.append("QUESTION: What is the final counter value? Return only the integer.")
    return Task(
        id=stable_id("state", difficulty, seed),
        family="state",
        difficulty=difficulty,
        seed=seed,
        prompt="\n".join(human),
        expected=str(value),
        prolog_source=prolog,
        tools={"final_value": 1, "state": 2},
        metadata={"events": len(events)},
    )


def _authorization(difficulty: str, seed: int) -> Task:
    cfg = DIFFICULTY[difficulty]
    rng = random.Random(seed)
    users = [f"u{i}" for i in range(max(4, cfg["depth"] // 2))]
    resources = [f"r{i}" for i in range(4)]
    roles = ["analyst", "admin", "guest"]
    user_role = {u: rng.choice(roles) for u in users}
    role_access = {"analyst": {"r0", "r1"}, "admin": set(resources), "guest": {"r0"}}
    blocked = {(rng.choice(users), rng.choice(resources)) for _ in range(max(1, cfg["distractors"] // 8))}
    target_user = users[0]
    target_resource = resources[-1]
    if seed % 2 == 0:
        user_role[target_user] = "admin"
        blocked.discard((target_user, target_resource))
        expected = "YES"
    else:
        user_role[target_user] = "guest"
        expected = "NO"
    human = ["ROLE ASSIGNMENTS:"] + [f"  {u}: {r}" for u, r in user_role.items()]
    human += ["ROLE ACCESS:"] + [f"  {r}: {', '.join(sorted(rs))}" for r, rs in role_access.items()]
    if blocked:
        human += ["BLOCKED PAIRS:"] + [f"  {u} -> {r}" for u, r in sorted(blocked)]
    human.append(f"QUESTION: May {target_user} access {target_resource}? Answer YES or NO.")
    facts = []
    for u, role in user_role.items():
        facts.append(f"role({u},{role}).")
    for role, rs in role_access.items():
        for resource in rs:
            facts.append(f"role_access({role},{resource}).")
    for u, resource in blocked:
        facts.append(f"blocked({u},{resource}).")
    prolog = "\n".join(facts) + "\npermitted(U,R):-role(U,Role),role_access(Role,R),\\+blocked(U,R).\n"
    return Task(
        id=stable_id("authorization", difficulty, seed),
        family="authorization",
        difficulty=difficulty,
        seed=seed,
        prompt="\n".join(human),
        expected=expected,
        prolog_source=prolog,
        tools={"permitted": 2},
        metadata={"users": len(users), "blocked_pairs": len(blocked)},
    )


def _starintel_relation_path(difficulty: str, seed: int) -> Task:
    """Directed StarIntel relation traversal using realistic predicate names in the prompt."""
    cfg = DIFFICULTY[difficulty]
    rng = random.Random(seed)
    depth = cfg["depth"]
    edges: list[tuple[str, str, str]] = []
    predicates = ["derived-from", "extracted-from", "observed-by", "links-to"]
    for i in range(depth):
        edges.append((f"doc{i}", f"doc{i+1}", predicates[i % len(predicates)]))
    for _ in range(cfg["distractors"]):
        a = f"noise{rng.randrange(max(3, depth * 3))}"
        b = f"noise{rng.randrange(max(3, depth * 3))}"
        if a != b:
            edges.append((a, b, rng.choice(predicates)))
    target = f"doc{depth}"
    human = ["STARINTEL RELATIONS (source --predicate--> target):"]
    human.extend(f"  {a} --{p}--> {b}" for a, b, p in edges)
    human.append(f"QUESTION: Is there a directed relation path from doc0 to {target}? Answer YES or NO.")
    facts = "\n".join(f"relation({a},{b})." for a, b, _ in edges)
    prolog = facts + "\nrelation_path(A,B):-relation(A,B).\nrelation_path(A,B):-relation(A,C),relation_path(C,B).\n"
    return Task(
        id=stable_id("starintel_relation_path", difficulty, seed),
        family="starintel_relation_path",
        difficulty=difficulty,
        seed=seed,
        prompt="\n".join(human),
        expected="YES",
        prolog_source=prolog,
        tools={"relation_path": 2},
        metadata={"starintel": True, "relations": len(edges), "logical_depth": depth},
    )


def _starintel_provenance(difficulty: str, seed: int) -> Task:
    """Resolve recursive provenance to the root document."""
    cfg = DIFFICULTY[difficulty]
    rng = random.Random(seed)
    depth = cfg["depth"]
    chain = [(f"doc{i}", f"doc{i+1}") for i in range(depth)]
    distractors: list[tuple[str, str]] = []
    for _ in range(cfg["distractors"]):
        a = f"aux{rng.randrange(max(3, depth * 3))}"
        b = f"aux{rng.randrange(max(3, depth * 3))}"
        if a != b:
            distractors.append((a, b))
    final_doc = f"doc{depth}"
    human = [
        "STARINTEL RECURSIVE PROVENANCE:",
        "Each edge parent -> child corresponds to a derived output and the child's parent_id.",
    ]
    human.extend(f"  {a} -> {b}" for a, b in chain + distractors)
    human.append(f"QUESTION: Is doc0 an ancestor/root source of {final_doc}? Answer YES or NO.")
    facts = "\n".join(f"parent({b},{a})." for a, b in chain + distractors)
    prolog = facts + "\nprovenance_ancestor(A,D):-parent(D,A).\nprovenance_ancestor(A,D):-parent(D,P),provenance_ancestor(A,P).\n"
    return Task(
        id=stable_id("starintel_provenance", difficulty, seed),
        family="starintel_provenance",
        difficulty=difficulty,
        seed=seed,
        prompt="\n".join(human),
        expected="YES",
        prolog_source=prolog,
        tools={"provenance_ancestor": 2},
        metadata={"starintel": True, "depth": depth, "distractors": len(distractors)},
    )


def _starintel_actor_loop(difficulty: str, seed: int) -> Task:
    """Detect a repeated actor in recursive actor_path metadata."""
    cfg = DIFFICULTY[difficulty]
    rng = random.Random(seed)
    count = max(4, cfg["depth"])
    actors = [f"actor{i}" for i in range(count)]
    expected = "YES" if seed % 2 == 0 else "NO"
    if expected == "YES":
        source = rng.randrange(max(1, count // 2))
        actors[-1] = actors[source]
    human = ["STARINTEL extensions.star_server.actor_path:"]
    human.append("  [" + ", ".join(actors) + "]")
    human.append("A repeated actor means the recursive dataflow has re-entered an actor already on this path.")
    human.append("QUESTION: Does this actor_path contain a recursion loop? Answer YES or NO.")
    facts = "\n".join(f"actor_hop(trace0,{i},{actor})." for i, actor in enumerate(actors))
    prolog = facts + "\nloop_detected(T):-actor_hop(T,I,A),actor_hop(T,J,A),I<J.\n"
    return Task(
        id=stable_id("starintel_actor_loop", difficulty, seed),
        family="starintel_actor_loop",
        difficulty=difficulty,
        seed=seed,
        prompt="\n".join(human),
        expected=expected,
        prolog_source=prolog,
        tools={"loop_detected": 1},
        metadata={"starintel": True, "actor_hops": len(actors)},
    )


def _starintel_document_policy(difficulty: str, seed: int) -> Task:
    """Check canonical StarIntel dtype policy versus explicitly rejected aliases."""
    canonical = ["person", "org", "relation", "target", "domain", "host", "network", "url", "email", "message"]
    rejected = ["organization", "organisation", "investigation_target", "social_media_post", "email_message"]
    rng = random.Random(seed)
    if seed % 2 == 0:
        dtype = rng.choice(canonical)
        expected = "YES"
    else:
        dtype = rng.choice(rejected)
        expected = "NO"
    noise_count = DIFFICULTY[difficulty]["distractors"]
    noise = [rng.choice(canonical + rejected) for _ in range(noise_count)]
    human = [
        "STARINTEL 0.9 DOCUMENT POLICY:",
        "Canonical dtypes are accepted; legacy/rejected aliases are not silently normalized.",
        "Reference examples: " + ", ".join(noise),
        f"Candidate dtype: {dtype}",
        "QUESTION: Is the candidate dtype canonical/accepted by this policy? Answer YES or NO.",
    ]
    facts = "\n".join(f"canonical_dtype({name})." for name in canonical)
    prolog = facts + "\nvalid_dtype(D):-canonical_dtype(D).\n"
    return Task(
        id=stable_id("starintel_document_policy", difficulty, seed),
        family="starintel_document_policy",
        difficulty=difficulty,
        seed=seed,
        prompt="\n".join(human),
        expected=expected,
        prolog_source=prolog,
        tools={"valid_dtype": 1},
        metadata={"starintel": True, "candidate_dtype": dtype, "reference_items": noise_count},
    )
