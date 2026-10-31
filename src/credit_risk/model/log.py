"""A simple experiment log: one CSV row per trial, appended as the run goes."""
import csv
import datetime as dt
import json
from pathlib import Path


class ExperimentLog:
    FIELDS = ["run_id", "timestamp", "family", "input_set", "params", "class_weight", "monotone", "fold1_auc", "fold2_auc", "mean_auc", "mean_brier", "note"]

    def __init__(self, path, run_id):
        self.path, self.run_id = Path(path), run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, self.FIELDS).writeheader()

    def add(self, family, input_set, params, class_weight, monotone, aucs, briers, note=""):
        row = {"run_id": self.run_id, "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "family": family, "input_set": input_set,
               "params": json.dumps(params, sort_keys=True), "class_weight": class_weight, "monotone": monotone, "fold1_auc": round(aucs[0], 5),
               "fold2_auc": round(aucs[1], 5), "mean_auc": round(sum(aucs) / len(aucs), 5), "mean_brier": round(sum(briers) / len(briers), 5), "note": note}
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self.FIELDS).writerow(row)
        return row
