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

        # ✅ Safe access with defaults
        model_results = results.get("model_results", {})
        diagnostics = results.get("diagnostics", {})

        auc = model_results.get("auc_mean", "N/A")
        mean_energy = diagnostics.get("mean_energy", "N/A")
        energy_slope = diagnostics.get("mean_energy_slope", "N/A")
        ird = diagnostics.get("intervention_recovery_delta", "N/A")
        feature_importance = model_results.get("feature_importance", [])

        html = f"""
        <html>
        <head>
            <title>Signal Report {run_id}</title>
        </head>
        <body>
            <h1>Signal Report: {run_id}</h1>
            <p><b>Status:</b> {"✅ Complete" if not results.get("skipped") else f"⚠️ Skipped: {results.get('reason')}"} </p>
            <p><b>AUC:</b> {auc if auc != "N/A" else "N/A"}</p>
            <p><b>Mean Energy:</b> {mean_energy if mean_energy != "N/A" else "N/A"}</p>
            <p><b>Energy Slope:</b> {energy_slope if energy_slope != "N/A" else "N/A"}</p>
            <p><b>Intervention Recovery Delta:</b> {ird if ird != "N/A" else "N/A"}</p>
        """

        # ✅ Safe feature list rendering
        if feature_importance:
            html += "<h2>Top Features</h2><ul>"
            for item in feature_importance[:10]:
                feature = item.get("feature", "unknown")
                importance = item.get("importance", 0)
                html += f"<li>{feature}: {importance:.4f}</li>"
            html += "</ul>"
        else:
            html += "<p><i>No feature importance data available (run may have been skipped or had insufficient samples).</i></p>"

        html += """
            </body>
            </html>
        """

        with open(path, "w") as f:
            f.write(html)

        return path
