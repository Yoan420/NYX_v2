"""Persist actual forecast/report milestones for the NYX application."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4


STEPS = (
    ("verify", "Vérifier les modèles enregistrés"),
    ("sources", "Actualiser les données"),
    ("cpu_FR", "Prévisions France"),
    ("cpu_DE", "Prévisions Allemagne"),
    ("cpu_BE", "Prévisions Belgique"),
    ("cpu_NL", "Prévisions Pays-Bas"),
    ("experts", "Réajuster HHB/Test2, prix et probabilités"),
    ("comparators", "Préparer les comparaisons EPEX / Storm"),
    ("cwe_1", "Rapport CWE · J+1"),
    ("cwe_2", "Rapport CWE · J+2"),
    ("detail_BE", "Rapport détaillé Belgique"),
    ("detail_DE", "Rapport détaillé Allemagne"),
    ("detail_FR", "Rapport détaillé France"),
    ("detail_NL", "Rapport détaillé Pays-Bas"),
    ("publish", "Vérifier et publier les rapports"),
)


class ReportProgress:
    def __init__(self, directory, *, stream=None):
        self.path = Path(directory) / "nyx_progress.json"
        self.stream = stream if stream is not None else sys.stdout
        self.steps = [{"id": key, "label": label, "status": "pending"} for key, label in STEPS]
        self.active_id = None
        self.status = "running"
        self.first_day = self.last_day = None
        self.phase_detail = None

    def _save(self, event=None):
        completed = sum(step["status"] in ("complete", "skipped") for step in self.steps)
        current = next((s for s in self.steps if s["id"] == self.active_id), None)
        document = {"protocol": "nyx_report_progress_v1", "status": self.status,
            "completed": completed, "total": len(self.steps), "percent": round(100 * completed / len(self.steps)),
            "phase": self.phase_detail or (current["label"] if current else "Rapports disponibles" if self.status == "succeeded" else "Préparation"),
            "steps": self.steps, "first_delivery_day": self.first_day, "last_delivery_day": self.last_day,
            "updated_at_utc": datetime.now(timezone.utc).isoformat()}
        temporary = self.path.with_name(self.path.name + "." + uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(document, stream, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            for attempt in range(4):
                try:
                    os.replace(temporary, self.path)
                    break
                except PermissionError:
                    # A Windows reader can briefly hold the previous snapshot.
                    # The cumulative log remains available if it stays locked.
                    if attempt < 3:
                        time.sleep(.02 * (attempt + 1))
        finally:
            temporary.unlink(missing_ok=True)
        if event:
            self.stream.write("[NYX progress] " + json.dumps({**event,
                "completed": completed, "total": len(self.steps), "percent": document["percent"],
                "steps": self.steps, "run_status": self.status,
                "phase": document["phase"],
                "updated_at_utc": document["updated_at_utc"]}, ensure_ascii=False) + "\n")
            self.stream.flush()

    def set(self, key, status):
        if status not in ("pending", "running", "complete", "failed", "skipped"):
            raise ValueError("Unknown progress status")
        step = next((step for step in self.steps if step["id"] == key), None)
        if step is None:
            raise ValueError("Unknown forecast progress step")
        if step["status"] == status:
            return
        if step["status"] in ("complete", "skipped"):
            return
        step["status"] = status
        self.phase_detail = None
        if status in ("running", "failed"):
            self.active_id = key
        elif self.active_id == key:
            self.active_id = None
        self._save({"id": key, "status": status, "label": step["label"]})

    def detail(self, message, **details):
        if self.active_id is None:
            raise ValueError("An active milestone is required for detailed progress")
        self.phase_detail = str(message)
        self._save({"id": self.active_id, "status": "running",
                    "label": self.phase_detail, "details": details})

    def horizon(self, first_day, last_day):
        self.first_day, self.last_day = first_day, last_day
        self._save()

    def cpu_event(self, line):
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            return
        if not isinstance(value, dict) or value.get("stage") != "cpu_daily_baseline":
            return
        zone, phase = value.get("zone"), value.get("phase", "")
        if zone not in ("FR", "DE", "BE", "NL"):
            return
        self.set("sources", "complete")
        if phase == "two_day_complete":
            self.set("cpu_" + zone, "complete")
            if all(step["status"] == "complete" for step in self.steps if step["id"].startswith("cpu_")):
                self.set("experts", "running")
        elif phase.startswith(("bridge_", "two_day_")):
            self.set("cpu_" + zone, "running")

    def fail(self):
        self.status = "failed"
        if self.active_id:
            self.set(self.active_id, "failed")
        else:
            self._save()

    def finish(self):
        if any(step["status"] not in ("complete", "skipped") for step in self.steps):
            raise ValueError("Forecast progress cannot finish before its milestones")
        self.status, self.active_id = "succeeded", None
        self._save()

    @contextmanager
    def capture_cpu_events(self):
        with redirect_stdout(_CpuProgressStream(sys.stdout, self)):
            yield


class _CpuProgressStream:
    def __init__(self, original, tracker):
        self.original, self.tracker, self.pending = original, tracker, ""

    def write(self, text):
        result = self.original.write(text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line.startswith("{"):
                self.tracker.cpu_event(line)
        if len(self.pending) > 100_000:
            self.pending = ""
        return result

    def flush(self):
        return self.original.flush()

    def __getattr__(self, key):
        return getattr(self.original, key)
