import json
import os


class Leaderboard:

    def __init__(self, path="signal_leaderboard.json"):
        self.path = path

        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump([], f)

    def update(self, entry: dict):

        with open(self.path, "r") as f:
            data = json.load(f)

        data.append(entry)

        data = sorted(
            data,
            key=lambda x: x["mean_auc"],
            reverse=True,
        )

        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)