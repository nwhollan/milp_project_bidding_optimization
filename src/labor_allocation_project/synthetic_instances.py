from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

# --- Project ---
@dataclass
class Project:
    pid: str
    duration: int                               # duration in weeks
    core_to_total_ratio: float                  # required core-to-total ratio in [0, 1]
    probability_of_bid_winning: float           # P(bid accepted) in [0, 1] (estimated probability of winning the bid)
    profit: float                               # expected profit if won
    earliest_start_week: int                    # earliest start week (inclusive)
    latest_start_week: int                      # latest start week (inclusive)
    worker_demand: Dict[Tuple[int, str], float] # D[(tau, skill)] -> demand, tau in [0, H-1]: demand for skill s at time tau weeks from the start of the project

# --- Instance ---
# A collection of projects and their constraints
@dataclass
class Instance:
    skills: List[str]
    horizon: int                                         # number of weeks in planning horizon
    employee_availability: Dict[Tuple[int, str], float]  # C[(t, skill)] -> available core workers
    projects: List[Project]

def two_projects_not_enough_capacity() -> Instance:
    """
    Test 1: Two jobs, fully overlapping, high core ratio (1.0).
    3 electricians and 2 helpers -- enough for one but not two.
 
    Project A: E[profit] = 0.8 * 12000 = $9,600
    Project B: E[profit] = 0.7 * 20000 = $14,000
 
    Expected: bid on B only (higher expected profit).
    """
    skills = ["electrician", "helper"]
    horizon = 5
    employee_availability = {(t, "electrician"): 3 for t in range(horizon)}
    employee_availability.update({(t, "helper"): 2 for t in range(horizon)})
 
    projA = Project(
        pid="A", duration=3, core_to_total_ratio=1.0,
        probability_of_bid_winning=0.8, profit=12000,
        earliest_start_week=0, latest_start_week=1,
        worker_demand={(0, "electrician"): 3, (1, "electrician"): 3, (2, "electrician"): 2,
                       (0, "helper"): 2,      (1, "helper"): 2,      (2, "helper"): 1},
    )
    projB = Project(
        pid="B", duration=3, core_to_total_ratio=1.0,
        probability_of_bid_winning=0.7, profit=20000,
        earliest_start_week=0, latest_start_week=1,
        worker_demand={(0, "electrician"): 3, (1, "electrician"): 3, (2, "electrician"): 2,
                       (0, "helper"): 2,      (1, "helper"): 2,      (2, "helper"): 1},
    )
 
    return Instance(skills=skills, horizon=horizon,
                    employee_availability=employee_availability,
                    projects=[projA, projB])

def two_projects_hire_noncore_workers_instance() -> Instance:
    """
    Test 2: Two overlapping jobs that both should be bid on.
    Only 2 electricians and 1 helper on staff, but each project demands
    more.  Core ratio is low (0.3) so subcontractors fill the gap.
 
    Expected: bid on both, zero idle core workers, heavy subcontracting.
    """
    skills = ["electrician", "helper"]
    horizon = 6
    employee_availability = {(t, "electrician"): 2 for t in range(horizon)}
    employee_availability.update({(t, "helper"): 1 for t in range(horizon)})
 
    projA = Project(
        pid="A", duration=3, core_to_total_ratio=0.3,
        probability_of_bid_winning=0.9, profit=8000,
        earliest_start_week=0, latest_start_week=1,
        worker_demand={(0, "electrician"): 3, (1, "electrician"): 3, (2, "electrician"): 2,
                       (0, "helper"): 2,      (1, "helper"): 2,      (2, "helper"): 1},
    )
    projB = Project(
        pid="B", duration=3, core_to_total_ratio=0.3,
        probability_of_bid_winning=0.85, profit=7000,
        earliest_start_week=0, latest_start_week=1,
        worker_demand={(0, "electrician"): 2, (1, "electrician"): 3, (2, "electrician"): 2,
                       (0, "helper"): 1,      (1, "helper"): 2,      (2, "helper"): 1},
    )
 
    return Instance(skills=skills, horizon=horizon,
                    employee_availability=employee_availability,
                    projects=[projA, projB])

def simple_three_project_instance() -> Instance:
    skills = ["electrician", "helper"]
    # 8 weeks in the planning horizon
    horizon = 8

    # 3 core electricians and 2 helpers available every week (employee availability)
    employee_availability = {(t, "electrician"): 3 for t in range(horizon)}
    employee_availability.update({(t, "helper"): 2 for t in range(horizon)})

    # Project A: 3 weeks, flat demand, must start in weeks 0-2, high profit
    projA = Project(
        pid="A", duration=3, core_to_total_ratio=0.5, probability_of_bid_winning=0.7, profit=10000, earliest_start_week=0, latest_start_week=2,
        worker_demand={(0, "electrician"): 2, (1, "electrician"): 2, (2, "electrician"): 1,
           (0, "helper"): 1,      (1, "helper"): 1,      (2, "helper"): 1},
    )
    # Project B: 2 weeks, must start 2-4, medium profit
    projB = Project(
        pid="B", duration=2, core_to_total_ratio=0.6, probability_of_bid_winning=0.5, profit=6000, earliest_start_week=2, latest_start_week=4,
        worker_demand={(0, "electrician"): 1, (1, "electrician"): 1,
           (0, "helper"): 1,      (1, "helper"): 0},
    )
    # Project C: 4 weeks, must start 1-3, low acceptance prob
    projC = Project(
        pid="C", duration=4, core_to_total_ratio=0.4, probability_of_bid_winning=0.3, profit=15000, earliest_start_week=1, latest_start_week=3,
        worker_demand={(tau, "electrician"): 2 for tau in range(4)}
         | {(tau, "helper"): 1 for tau in range(4)},
    )

    return Instance(skills=skills, horizon=horizon, employee_availability=employee_availability, projects=[projA, projB, projC])