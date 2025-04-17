from dataclasses import dataclass
from typing import List, Callable, Optional

@dataclass
class Checkpoint:
    total: int
    result: int
    
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

@dataclass
class Result:
    checkpoints: List[Checkpoint]
    scoring_strategy: Optional[Callable[[List[Checkpoint]], dict]] = None
    
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
                {"total": cp.total, "result": cp.result}
                for cp in self.checkpoints
            ],
            "final_score": self.final_score
        }