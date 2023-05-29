import statistics

import pandas as pd


class Evaluator:
    def __init__(self) -> None:
        self.table = pd.DataFrame()

    def update_evaluations(
        self, model_name: str, eval_dict: dict[str, float], epoch_times: list[float]
    ) -> None:
        modified_eval_dict: dict[str, list] = {"Model Name": [model_name]}
        modified_eval_dict.update({k: [v] for k, v in eval_dict.items()})
        modified_eval_dict.update({"mean_epoch_time": [statistics.mean(epoch_times)]})
        new_eval = pd.DataFrame(modified_eval_dict)
        self.table = pd.concat([self.table, new_eval])
        self.table.reset_index(drop=True, inplace=True)
