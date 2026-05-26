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


@dataclass
class EvaluationStep:
    name: str
    success: bool
    step_id: int
    details: Optional[str] = None
    score: int = 0
    max_score: int = 1
    execution_time: Optional[float] = None  # Time in seconds

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
    
    def add_step(self, name: str, success: bool, step_id: int, details: Optional[str] = None, score: int = None, max_score: int = 1, execution_time: Optional[float] = None):
        """Add an evaluation step to this checkpoint."""
        if score is None:
            score = max_score if success else 0
        self.result += score
        step = EvaluationStep(name=name, success=success, step_id=step_id, details=details, score=score, max_score=max_score, execution_time=execution_time)
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
                    "execution_time": step.execution_time
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