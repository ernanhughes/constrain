import os
from constrain.utils.json_utils import dumps_safe

class DashboardExporter:

    @staticmethod
    def export_json(results: dict, run_id: str, folder="dashboard"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{run_id}.json")

        with open(path, "w") as f:
            f.write(dumps_safe(results, indent=2))

        return path

    @staticmethod
    def export_html(results: dict, run_id: str, folder="dashboard"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{run_id}.html")

        html = f"""
        <html>
        <head>
            <title>Signal Report {run_id}</title>
        </head>
        <body>
            <h1>Signal Report</h1>
            <p><b>AUC:</b> {results.get("model_results", {}).get("auc", "N/A")}</p>
            <p><b>Mean Energy:</b> {results.get("diagnostics", {}).get("mean_energy", "N/A")}</p>
            <p><b>Energy Slope:</b> {results.get("energy_slopes", "N/A")}</p>
            <p><b>IRD:</b> {results.get("intervention_recovery_delta", "N/A")}</p>
            <h2>Top Features</h2>
            <ul>
        """

        for item in results["model_results"]["feature_importance"][:10]:
            html += f"<li>{item['feature']}: {item['importance']:.4f}</li>"

        html += """
            </ul>
        </body>
        </html>
        """

        with open(path, "w") as f:
            f.write(html)

        return path
