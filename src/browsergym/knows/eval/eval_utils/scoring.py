import math
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any


def calculate_percentage_score(success_count: int, total_count: int, max_points: int = 10) -> int:
    """Calculate score based on percentage, floored to nearest 10%.

    Args:
        success_count: Number of successful items.
        total_count: Total number of items.
        max_points: Maximum points available (default 10).

    Returns:
        Score as an integer, scaled to max_points.
    """
    if total_count == 0:
        return 0
    percentage = success_count / total_count
    floored_percentage = math.floor(percentage * max_points) / max_points
    return int(floored_percentage * max_points)


class StepCategory:
    """Mechanism-based categories for evaluation steps.

    Each step records the mechanism that decided its outcome: on success, the
    check that accepted the artifact; on failure, the check that rejected it.
    For tiered/fallback checks (e.g. exact -> perceptual hash -> VLM), the
    category is the last mechanism that ran, since it made the final call.
    """
    DETERMINISTIC = "deterministic"  # Exact/normalized text match, pixel-exact image match, exact numeric comparison
    FUZZY_MATCH = "fuzzy_match"  # Tolerance-based non-geometric matching: fuzzy text, perceptual hash, numeric/color tolerance
    LLM_VLM_JUDGEMENT = "llm_vlm_judgement"  # An LLM/VLM verdict decided the step
    SPATIAL = "spatial"  # Geometric/rendered-coordinate check: bbox, OCR location, area, pixel visibility
    STRUCTURAL = "structural"  # Document-structure check: element counts, order, position-in-structure, styles
    WEB_VISIT = "web_visit"  # String/regex match over the agent's browsing history
    DEPENDENCY_NOT_EVALUATED = "dependency_not_evaluated"  # Skipped because a prerequisite step failed (failure-only)
    EXECUTION_ERROR = "execution_error"  # Exception/missing data prevented the check from running (failure-only)
    VACUOUS_PASS = "vacuous_pass"  # Success recorded although no check actually ran (success-only)

    VALID = {
        DETERMINISTIC,
        FUZZY_MATCH,
        LLM_VLM_JUDGEMENT,
        SPATIAL,
        STRUCTURAL,
        WEB_VISIT,
        DEPENDENCY_NOT_EVALUATED,
        EXECUTION_ERROR,
        VACUOUS_PASS,
    }

    # Precedence for tie-breaking in aggregate(): most-suspect mechanism first.
    AGGREGATE_PRECEDENCE = [
        LLM_VLM_JUDGEMENT,
        FUZZY_MATCH,
        SPATIAL,
        STRUCTURAL,
        WEB_VISIT,
        DETERMINISTIC,
        VACUOUS_PASS,
        DEPENDENCY_NOT_EVALUATED,
        EXECUTION_ERROR,
    ]

    # Maps match_image_tiered()'s match_method return values to categories.
    MATCH_METHOD_CATEGORIES = {
        "exact": DETERMINISTIC,
        "perceptual_hash": FUZZY_MATCH,
        "vlm": LLM_VLM_JUDGEMENT,
    }

    @classmethod
    def from_match_method(cls, match_method: str) -> str:
        """Map a match_image_tiered() match_method to a step category.

        Args:
            match_method (str): One of "exact", "perceptual_hash", "vlm".

        Returns:
            str: The corresponding category; LLM_VLM_JUDGEMENT for unknown
                methods (the VLM tier is the final arbiter of tiered matching).
        """
        return cls.MATCH_METHOD_CATEGORIES.get(match_method, cls.LLM_VLM_JUDGEMENT)

    @classmethod
    def aggregate(cls, items: List[tuple]) -> str:
        """Derive one category for a step that aggregates many sub-item checks.

        Rule: if any item failed, the step category is the majority category
        among the FAILING items (the step's failure is attributed to the
        mechanism that rejected most items). If all items passed, it is the
        majority category among all items. Ties are broken by
        AGGREGATE_PRECEDENCE (most-suspect mechanism first).

        Args:
            items: List of (category, success) tuples, one per sub-item.

        Returns:
            str: The aggregated category; EXECUTION_ERROR for an empty list
                (an aggregate step with nothing to check could not run).
        """
        if not items:
            return cls.EXECUTION_ERROR
        failing = [category for category, success in items if not success]
        pool = failing if failing else [category for category, _ in items]
        counts: Dict[str, int] = {}
        for category in pool:
            counts[category] = counts.get(category, 0) + 1
        max_count = max(counts.values())
        tied = [c for c, n in counts.items() if n == max_count]
        for category in cls.AGGREGATE_PRECEDENCE:
            if category in tied:
                return category
        return tied[0]  # Unknown categories: arbitrary but deterministic


@dataclass
class EvaluationStep:
    name: str
    success: bool
    step_id: int
    details: Optional[str] = None
    score: int = 0
    max_score: int = 1
    execution_time: Optional[float] = None  # Time in seconds
    category: Optional[str] = None  # StepCategory value: mechanism that decided the outcome

@dataclass
class Checkpoint:
    total: int
    result: int
    steps: List[EvaluationStep] = field(default_factory=list)
    name: Optional[str] = None
    execution_time: Optional[float] = None  # Time in seconds
    
    def __post_init__(self):
        if not isinstance(self.total, int):
            raise TypeError(f"total must be an integer, got {type(self.total)}")
        if not isinstance(self.result, int):
            raise TypeError(f"result must be an integer, got {type(self.result)}")
        if self.total < 0:
            raise ValueError(f"total cannot be negative, got {self.total}")
        if self.result < 0:
            raise ValueError(f"result cannot be negative, got {self.result}")
        if self.result > self.total:
            raise ValueError(f"result ({self.result}) cannot be greater than total ({self.total})")
    
    def add_step(self, name: str, success: bool, step_id: int, details: Optional[str] = None, score: int = None, max_score: int = 1, execution_time: Optional[float] = None, category: Optional[str] = None):
        """Add an evaluation step to this checkpoint.

        Args:
            name (str): Human-readable step name.
            success (bool): Whether the step passed.
            step_id (int): Unique step identifier within the evaluator.
            details (str, optional): Explanation of the outcome.
            score (int, optional): Points earned; defaults to max_score if success else 0.
            max_score (int): Maximum points for this step.
            execution_time (float, optional): Time in seconds.
            category (str, optional): StepCategory value naming the mechanism
                that decided the outcome (see StepCategory).
        """
        if score is None:
            score = max_score if success else 0
        self.result += score
        step = EvaluationStep(name=name, success=success, step_id=step_id, details=details, score=score, max_score=max_score, execution_time=execution_time, category=category)
        self.steps.append(step)
        return step
    
    def get_step_summary(self) -> Dict[str, Any]:
        """Get a summary of all steps in this checkpoint."""
        return {
            "total_steps": len(self.steps),
            "successful_steps": sum(1 for step in self.steps if step.success),
            "failed_steps": sum(1 for step in self.steps if not step.success),
            "steps": [
                {
                    "step_id": step.step_id,
                    "name": step.name,
                    "success": step.success,
                    "details": step.details,
                    "score": step.score,
                    "max_score": step.max_score,
                    "execution_time": step.execution_time,
                    "category": step.category
                }
                for step in self.steps
            ]
        }

@dataclass
class Result:
    checkpoints: List[Checkpoint]
    scoring_strategy: Optional[Callable[[List[Checkpoint]], dict]] = None
    total_execution_time: Optional[float] = None  # Time in seconds
    
    def __post_init__(self):
        if self.scoring_strategy is None:
            # Default scoring strategy: simple sum
            self.scoring_strategy = lambda checkpoints: {
                "total": sum(cp.total for cp in checkpoints),
                "result": sum(cp.result for cp in checkpoints)
            }
    
    @property
    def final_score(self) -> dict:
        return self.scoring_strategy(self.checkpoints)
    
    @classmethod
    def from_dict(cls, data: dict, scoring_strategy: Optional[Callable] = None) -> 'Result':
        """Create a Result instance from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"Input must be a dict, got {type(data)}")
        
        if "checkpoints" not in data:
            raise KeyError("Input must contain 'checkpoints' field")
        
        checkpoints = [
            Checkpoint(**checkpoint_data)
            for checkpoint_data in data["checkpoints"]
        ]
        
        return cls(checkpoints=checkpoints, scoring_strategy=scoring_strategy)
    
    def to_dict(self) -> dict:
        """Convert the Result instance to a dictionary."""
        return {
            "checkpoints": [
                {
                    "total": cp.total, 
                    "result": cp.result,
                    "name": cp.name,
                    "execution_time": cp.execution_time,
                    "step_summary": cp.get_step_summary()
                }
                for cp in self.checkpoints
            ],
            "final_score": self.final_score,
            "total_execution_time": self.total_execution_time
        }
    
    def get_category_summary(self) -> Dict[str, Dict[str, int]]:
        """Aggregate step outcomes by category across all checkpoints.

        Steps without a category are grouped under "uncategorized".

        Returns:
            dict: Mapping of category -> {"total": int, "passed": int, "failed": int}.
        """
        summary: Dict[str, Dict[str, int]] = {}
        for cp in self.checkpoints:
            for step in cp.steps:
                category = step.category or "uncategorized"
                entry = summary.setdefault(category, {"total": 0, "passed": 0, "failed": 0})
                entry["total"] += 1
                entry["passed" if step.success else "failed"] += 1
        return summary

    def get_detailed_report(self) -> Dict[str, Any]:
        """Get a detailed report of all evaluation steps."""
        return {
            "final_score": self.final_score,
            "checkpoints": [
                {
                    "name": cp.name or f"Checkpoint {i+1}",
                    "score": f"{cp.result}/{cp.total}",
                    "execution_time": cp.execution_time,
                    "steps": cp.get_step_summary()["steps"]
                }
                for i, cp in enumerate(self.checkpoints)
            ]
        }