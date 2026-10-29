

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field 
from enum import Enum 
import yaml 

class ParamType(Enum):
    CATEGORICAL = "categorical"
    FLOAT = "float"
    INT = "int"
    LOG_UNIFORM = "log_uniform"


@dataclass
class ParamSpec: 
    param_type: ParamType
    values: Optional[List[Any]] = None
    low: Optional[float] = None 
    high: Optional[float] = None 
    step: Optional[float] = None 
    log: bool = False 

    def __post_init__(self):
        if self.param_type == ParamType.CATEGORICAL and self.values is None:
            raise ValueError("Categorical param requires 'values'")

        if self.param_type in (ParamType.FLOAT, ParamType.INT, ParamType.LOG_UNIFORM):
            if self.low is None or self.high is None: 
                raise ValueError(f"{self.param_type.value} param requires 'low' and 'high'")


class SearchSpace: 
    def __init__(self, params: Dict[str, ParamSpec]):
        self.params = params


    @classmethod
    def from_yaml(cls, path: str) -> "SearchSpace":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict) -> "SearchSpace":
        params = {}
        for name, spec, in data.items():
            param_type = ParamType(spec["type"])
            params[name] = ParamSpec(
                param_type=param_type,
                values=spec.get("values"),
                low=spec.get("low"),
                high=spec.get("high"),
                step=spec.get("step"),
                log=spec.get("log", False),
            )
        return cls(params)

    def to_dict(self) -> dict: 
        return {
            name: {
                "type": spec.param_type.value,
                "values": spec.values,
                "low": spec.low,
                "high": spec.high,
                "log": spec.log
            }
            for name, spec, in self.params.items()
        }
    def grid_configs(self) -> List[Dict[str, Any]]:
        import itertools 

        param_names = list(self.params.keys())
        param_values = []

        for name, spec in self.params.items():
            if spec.param_type == ParamType.CATEGORICAL:
                param_values.append(spec.values)
            elif spec.param_type == ParamType.INT:
                if spec.step is None: 
                    spec.step = 1
                values = list(range(int(spec.low), int(spec.high) +1 , int(spec.step)))
                param_values.append(values)
            elif spec.param_type == ParamType.FLOAT:
                if spec.step is None:
                    raise ValueError(f"Float param '{name}' requires 'step' for grid search")
                if spec.log:
                    import numpy as np
                    values = np.logspace(
                        np.log10(spec.low), np.log10(spec.high),
                        num=int((np.log10(spec.high) - np.log10(spec.low)) / spec.step) + 1
                    ).tolist()
                else: 
                    import numpy as np 
                    values = np.arange(spec.low, spec.high + spec.step /2 , spec.step).tolist()
                param_values.append(values)
            elif spec.param_type == ParamType.LOG_UNIFORM:
                import numpy as np 
                if spec.step is None:
                    raise ValueError(f"Log-uniform param '{name}' requires 'step' for grid search")
                values = np.logspace(
                    np.log10(spec.low), np.log10(spec.high),
                    num=int((np.log10(spec.high) - np.log10(spec.low)) / spec.step) + 1
                ).tolist()
                param_values.append(values)
        
        configs = []
        for combo in itertools.product(*param_values):
            configs.append(dict(zip(param_names, combo)))
        return configs

    def sample_config(self, rng: Any) -> Dict[str, Any]:
        import numpy as np 


        config = {}
        for name, spec in self.params.items():
            if spec.param_type == ParamType.CATEGORICAL:
                config[name] = rng.choice(spec.values)
            elif spec.param_type == ParamType.INT:
                config[name] = int(rng.integers(spec.low, spec.high + 1))
            elif spec.param_type == ParamType.FLOAT:
                if spec.log: 
                    config[name] = 10 ** rng.uniform(np.log10(spec.low), np.log10(spec.high))
                else: 
                    config[name] = rng.uniform(spec.low, spec.high)
            elif spec.param_type == ParamType.LOG_UNIFORM:
                config[name] = 10 ** rng.uniform(np.log10(spec.low), np.log10(spec.high))
            
        return config 

    
            