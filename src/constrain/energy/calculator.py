# src/certum/runner.py

import hashlib
import json
import logging
from pathlib import Path

from tqdm import tqdm

from .calibration import AdaptiveCalibrator
from .gate import VerifiabilityGate
from .geometry.claim_evidence import ClaimEvidenceGeometry
from .utils.id_utils import compute_ids
from .utils.math_utils import accept_margin_ratio
from .protocols.embedder import Embedder
from .protocols.gate import Gate
from .protocols.calibrator import Calibrator
from .protocols.geometry import GeometryComputer
from .embedding.backends.sqlite_backend import SQLiteEmbeddingBackend
from .embedding.hf_embedder import HFEmbedder

from constrain.config import get_config

logger = logging.getLogger(__name__)


class EnergyCalculator:

    
    def caclculate (
        self,
        claim: str,
        evidence_texts: list[str],
        policies: str,

    ):


        embedding_model = get_config().embedding_model
        embedding_db = get_config().embedding_db
        embedder = self.build_embedder(model=embedding_model, embedding_db=embedding_db)

        samples = [{
            "claim": claim,
            "evidence": evidence_texts,
        }]

        self._ensure_vectors(samples, embedder)

        energy_computer = self.build_energy_computer()
        gate = self.build_gate(embedder, energy_computer)

        # -------------------------------------------------
        # 3. Calibration
        # -------------------------------------------------

        calibrator = AdaptiveCalibrator(gate, embedder=embedder)

        claim_vec_cache = {}

        policy_names = [p.strip() for p in policies.split(",") if p.strip()]
        logger.debug(f"Building policies: {policy_names}")

    # -----------------------------------------------------
    # Utilities
    # -----------------------------------------------------

    def _ensure_vectors(self, samples, embedder):
        """
        Efficient batched embedding with progress.
        """

        logger.debug("Preparing embeddings...")

        # ---------------------------
        # 1️⃣ Claims
        # ---------------------------

        claims_to_embed = [
            s["claim"]
            for s in samples
            if "claim_vec" not in s
        ]

        if claims_to_embed:
            logger.debug(f"Embedding {len(claims_to_embed)} claims...")
            claim_vecs = embedder.embed(claims_to_embed)

            idx = 0
            for s in samples:
                if "claim_vec" not in s:
                    s["claim_vec"] = claim_vecs[idx]
                    idx += 1

        # ---------------------------
        # 2️⃣ Evidence
        # ---------------------------

        for s in tqdm(samples, desc="Embedding evidence", unit="sample"):
            if "evidence_vecs" not in s or s["evidence_vecs"] is None:
                s["evidence_vecs"] = embedder.embed(s["evidence"])

        logger.debug("All embeddings prepared.")

    def _write_policy_rows(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


    def _stable_sample_id(self, sample: dict) -> str:
        """
        Stable ID for joining across tables/files.
        Prefer dataset-provided 'id'; otherwise hash claim+evidence.
        """
        sid = sample.get("id", None)
        if sid is not None:
            return str(sid)

        claim = sample.get("claim", "") or ""
        evidence = sample.get("evidence", []) or []
        blob = claim + "\n" + "\n".join(map(str, evidence))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


    def _evaluate_policy_suite(
        self,
        *,
        gate,
        sample: dict,
        policies_list: list,
        run_id: str,
        split: str,
    ) -> list[dict]:
        """
        Returns JSON-serializable rows (one per policy) for policy sweep outputs.
        Does NOT change your existing main outputs.
        """
        # Compute axes once (vector-aware) — requires gate.evaluate to accept claim_vec/evidence_vecs,
        # OR you can call gate.compute_axes if you added it.
        base, axes, embedding_info = gate.compute_axes(
            sample["claim"],
            sample["evidence"],
            claim_vec=sample.get("claim_vec"),
            evidence_vecs=sample.get("evidence_vecs"),
        ) 

        pair_id, claim_id, evidence_id = compute_ids(sample["claim"], sample["evidence"])

        rows: list[dict] = []
        for policy in policies_list:
            tau = getattr(policy, "tau_accept", None)
            if tau is None:
                eff = 0.0  # or None, but your Policy.decide expects a float
            else:
                eff = accept_margin_ratio(energy=float(axes.get("energy")), tau=float(tau))

            verdict = policy.decide(axes, float(eff))
            g = base.geometry
            rows.append({
                "run_id": run_id,
                "split": split,

                "sample_id": sample.get("id"),  
                "id": pair_id,
                "pair_id": pair_id,
                "claim_id": claim_id,
                "evidence_id": evidence_id,
                "row_id": self._row_id(sample.get("id"), policy.name, split),


                "policy_name": policy.name,
                "policy_key": getattr(policy, "key", None),  # harmless if absent
                "verdict": verdict.value,
                "effectiveness": float(eff),

                # a few extra geometry fields (high-value)
                "effective_rank": int(getattr(g, "effective_rank", 0)),
                "used_count": int(getattr(g, "used_count", 0)),
                "sigma1_ratio": float(getattr(g, "sigma1_ratio", 0.0)),
                "sigma2_ratio": float(getattr(g, "sigma2_ratio", 0.0)),
                "entropy_rank": float(getattr(g, "entropy_rank", 0.0)),
                "sim_top1": float(getattr(g, "sim_top1", 0.0)),
                "sim_top2": float(getattr(g, "sim_top2", 0.0)),

                "embedding_backend": embedding_info.get("embedding_backend"),
                "claim_dim": embedding_info.get("claim_dim"),
                "evidence_count": embedding_info.get("evidence_count"),

                "tau_accept": policy.tau_accept,

                # keep full nested structure for later deep dives
                "energy": base.energy,
                "energy_result": base.to_dict(),
            })
        return rows

    def _row_id(self, sample_id: str, policy_name: str, split: str) -> str:
        blob = f"{split}|{sample_id}|{policy_name}"
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


    def build_embedder(self, model: str, embedding_db: Path) -> Embedder:
        backend = SQLiteEmbeddingBackend(str(embedding_db))
        return HFEmbedder(model_name=model, backend=backend)

    # -------------------------------------------------
    # Energy
    # -------------------------------------------------

    def build_energy_computer(self) -> GeometryComputer:
        return ClaimEvidenceGeometry(top_k=12, rank_r=8)

    # -------------------------------------------------
    # Gate
    # -------------------------------------------------

    def build_gate(self, embedder: Embedder, energy_computer: GeometryComputer) -> Gate:
        return VerifiabilityGate(embedder, energy_computer)


    def build_calibrator(self, gate: Gate, embedder: Embedder) -> Calibrator:
        return AdaptiveCalibrator(gate, embedder)
   
