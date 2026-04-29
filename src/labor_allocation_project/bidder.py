"""
Electrical contractor bidding MILP (single-timestep formulation).

Decision variables:
    S[p, t] in {0, 1}      : project p starts at time t
    F[p, t, s] in Z_+      : core workers of skill s on project p at time t

Active demand (derived, linear in S):
    D_active[p, t, s] = sum_{tau=0}^{H_p - 1} D[p, tau, s] * S[p, t - tau]

Objective:
    min  sum_{t,s} U[t,s]  -  sum_{p, t in [E_p, L_p]} A_p * P_p * S[p,t]
"""

from __future__ import annotations

import argparse
from src.labor_allocation_project.synthetic_instances import *
from typing import Dict, Tuple
import pulp

# ------------------------------------------------------------
# --- Helper functions ---
# ------------------------------------------------------------

def active_demand_expr(proj: Project, t: int, s: str,
                       S: Dict[Tuple[str, int], pulp.LpVariable]) -> pulp.LpAffineExpression:
    """
    Build the linear expression D_active[p, t, s] = sum_tau D[p,tau,s] * S[p, t-tau].

    Only terms where (t - tau) is a feasible start time for p contribute,
    because S[p, t'] exists only for t' in [E_p, L_p].
    """
    terms = []
    for tau in range(proj.duration):
        t_start = t - tau
        if t_start < proj.earliest_start_week or t_start > proj.latest_start_week:
            continue
        d = proj.worker_demand.get((tau, s), 0.0)
        if d == 0.0:
            continue
        terms.append(d * S[(proj.pid, t_start)])
    if not terms:
        return pulp.LpAffineExpression()   # the zero expression
    return pulp.lpSum(terms)


# ---------------------------------------------------------------------------
# --- Model ---
# ---------------------------------------------------------------------------

def solve_bidding_mip(inst: Instance,
                      solver: pulp.LpSolver | None = None,
                      verbose: bool = False) -> Dict:
    # Create the model container.
    model = pulp.LpProblem("contractor_bidding", pulp.LpMinimize)

    T = range(inst.horizon)
    skills = inst.skills

    # --- Decision variables ---

    # Binary indicator for whether project p starts at time t
    # S[p, t] only for t in [E_p, L_p]  (constraint 5 enforced by construction)
    S: Dict[Tuple[str, int], pulp.LpVariable] = {}
    for p in inst.projects:
        for t in range(p.earliest_start_week, p.latest_start_week + 1):
            S[(p.pid, t)] = pulp.LpVariable(f"S_{p.pid}_{t}", cat=pulp.LpBinary)

    # Number of core workers of skill s on project p at time t
    # F[p, t, s] integer >= 0
    F: Dict[Tuple[str, int, str], pulp.LpVariable] = {}
    for p in inst.projects:
        for t in T:
            for s in skills:
                F[(p.pid, t, s)] = pulp.LpVariable(
                    f"F_{p.pid}_{t}_{s}", lowBound=0, cat=pulp.LpInteger
                )

    # --- Objective ---

    # Unused core labor: U[t,s] = C[t,s] - sum_p F[p,t,s]
    unused = pulp.lpSum(
        inst.employee_availability[(t, s)] - pulp.lpSum(F[(p.pid, t, s)] for p in inst.projects)
        for t in T for s in skills
    )

    # Expected profit term
    profit = pulp.lpSum(
        p.probability_of_bid_winning * p.profit * S[(p.pid, t)]
        for p in inst.projects
        for t in range(p.earliest_start_week, p.latest_start_week + 1)
    )

    # NOTE: play with the weighting of the objective function components?
    LAMBDA = 1.0
    model += unused - LAMBDA * profit, "objective"

    # --- Constraints ---

    # (1) Core capacity:  sum_p F[p,t,s] <= C[t,s]
    for t in T:
        for s in skills:
            model += (
                pulp.lpSum(F[(p.pid, t, s)] for p in inst.projects) <= inst.employee_availability[(t, s)],
                f"cap_t{t}_s{s}",
            )

    # (2) Core labor <= active demand:  F[p,t,s] <= D_active[p,t,s]
    # (3) Core ratio: sum_s F[p,t,s] >= R_p * sum_s D_active[p,t,s]
    for p in inst.projects:
        for t in T:
            total_active = pulp.LpAffineExpression()
            for s in skills:
                d_active = active_demand_expr(p, t, s, S)
                model += (
                    F[(p.pid, t, s)] <= d_active,
                    f"core_le_demand_{p.pid}_t{t}_s{s}",
                )
                total_active += d_active
            model += (
                pulp.lpSum(F[(p.pid, t, s)] for s in skills) >= p.core_to_total_ratio * total_active,
                f"core_ratio_{p.pid}_t{t}",
            )

    # (4) At most one start time per project
    for p in inst.projects:
        model += (
            pulp.lpSum(S[(p.pid, t)] for t in range(p.earliest_start_week, p.latest_start_week + 1)) <= 1,
            f"one_start_{p.pid}",
        )

    # --- Solve ---

    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0)
    status = model.solve(solver)

    # --- Extract solution ---

    result = {
        "status": pulp.LpStatus[status],
        "objective": pulp.value(model.objective),
        "bids": {},            # pid -> chosen start week, or None if not bidding
        "F": {},               # (pid, t, s) -> core workers
        "active_demand": {},   # (pid, t, s) -> D_active value
        "unused_core": {},     # (t, s) -> unused core workers
        "expected_profit": 0.0,
    }

    for p in inst.projects:
        chosen = None
        for t in range(p.earliest_start_week, p.latest_start_week + 1):
            v = pulp.value(S[(p.pid, t)])
            if v is not None and v > 0.5:
                chosen = t
                result["expected_profit"] += p.probability_of_bid_winning * p.profit
                break
        result["bids"][p.pid] = chosen

    for (pid, t, s), var in F.items():
        val = pulp.value(var) or 0.0
        if val > 1e-9:
            result["F"][(pid, t, s)] = val

    for p in inst.projects:
        for t in T:
            for s in skills:
                d_val = pulp.value(active_demand_expr(p, t, s, S)) or 0.0
                if d_val > 1e-9:
                    result["active_demand"][(p.pid, t, s)] = d_val

    for t in T:
        for s in skills:
            used = sum(
                (pulp.value(F[(p.pid, t, s)]) or 0.0) for p in inst.projects
            )
            result["unused_core"][(t, s)] = inst.employee_availability[(t, s)] - used

    return result

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # with options: simple_three_project_instance, oversubscribed_seven_project_instance
    argparser.add_argument("--instance", type=str, required=True, choices=["two_projects_not_enough_capacity", "two_projects_hire_noncore_workers_instance", "simple_three_project_instance"])
    args = argparser.parse_args()
    match args.instance:
        case "simple_three_project_instance":
            inst = simple_three_project_instance()
        case "two_projects_not_enough_capacity":
            inst = two_projects_not_enough_capacity()
        case "two_projects_hire_noncore_workers_instance":
            inst = two_projects_hire_noncore_workers_instance()
        case _:
            raise ValueError(f"Invalid instance: {args.instance}")

    res = solve_bidding_mip(inst, verbose=False)

    print(f"Solver status:     {res['status']}")
    print(f"Objective value:   {res['objective']:.2f}")
    print(f"Expected profit:   {res['expected_profit']:.2f}")
    print()
    print("Bid decisions:")
    for pid, start in res["bids"].items():
        print(f"  project {pid}: " + (f"bid, start week {start}" if start is not None else "no bid"))
    print()
    print("Core worker assignments F[p,t,s] (nonzero only):")
    for (pid, t, s), v in sorted(res["F"].items()):
        print(f"  F[{pid}, t={t}, {s}] = {v:g}")
    print()
    print("Unused core labor U[t,s]:")
    for (t, s), v in sorted(res["unused_core"].items()):
        print(f"  U[t={t}, {s}] = {v:g}")