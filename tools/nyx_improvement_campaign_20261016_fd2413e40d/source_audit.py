"""Read-only audit of the original EPEX donor and saved explanatory inputs."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from replay import ZONES, atomic_json, coverage, hourly, require, safe_bound, seed_record, sha


def describe(frame):
    result = {"rows": len(frame), "columns": len(frame.columns),
              "first_utc": str(frame.index.min()), "last_utc": str(frame.index.max()), "fields": {}}
    for name in frame:
        column = frame[name]
        numeric = pd.to_numeric(column, errors="coerce")
        finite = np.isfinite(numeric)
        item = {"nonmissing_rows": int(column.notna().sum()), "finite_numeric_rows": int(finite.sum()),
                "missing_rows": int(column.isna().sum()), "last_nonmissing_index": str(column.last_valid_index())}
        if finite.any():
            valid = numeric.loc[finite]
            recent = valid.iloc[-min(168, len(valid)):]
            item.update(minimum=float(valid.min()), median=float(valid.median()), maximum=float(valid.max()),
                        last_value=float(valid.iloc[-1]), recent_168_rows_mean=float(recent.mean()))
        result["fields"][str(name)] = item
    return result


def audit(seed, output):
    output = Path(output)
    require(not output.exists() or not any(output.iterdir()), "Choisir un dossier d audit vide")
    output.mkdir(parents=True, exist_ok=True)
    inputs = {str(seed["run_path"]): sha(seed["run_path"]), str(seed["receipt_path"]): sha(seed["receipt_path"])}
    requested = ["source_artifacts/saturn/covariates.parquet", "source_artifacts/saturn/origins.parquet",
                 "source_artifacts/fuel/market_fuel_features.parquet"]
    requested += [f"source_artifacts/auction_prices/{z}.parquet" for z in ZONES]
    requested += [key for key in seed["hashes"] if key.startswith("source_artifacts/thermal_capacity/") and key.endswith(".parquet")]
    frames, absent = {}, []
    for relative in dict.fromkeys(requested):
        if relative not in seed["hashes"]:
            absent.append(relative)
            continue
        path = safe_bound(seed["bundle"], relative, seed["hashes"])
        frame = pd.read_parquet(path)
        if isinstance(frame, pd.Series):
            frame = frame.to_frame()
        frames[relative] = describe(frame)
        inputs[str(path)] = sha(path)
        print("[NYX audit] " + relative + " : " + str(len(frame)) + " lignes", flush=True)
    for name, expected in inputs.items():
        require(sha(name) == expected, "Source modifiee pendant l audit : " + name)
    document = {"status": "SAVED_INPUTS_AUDITED", "seed_run": str(seed["run_path"].parent),
                "coverage": coverage(seed), "sources": frames, "absent_optional_inputs": absent,
                "source_sha256": inputs, "production_modified": False,
                "strict_08_provider_vintages_certified": False, "live_source_freshness_verified": False,
                "notes": ["Les statistiques de niveau et de disponibilite concernent les fichiers du donneur fige, pas les donnees live du jour.",
                          "Le prix cible est EPEX selon le recu du donneur ; les unitees et publications doivent etre confrontees aux specifications sources.",
                          "Aucun decalage ou biais n est corrige automatiquement. Aucune collecte Saturn ne se produit.",
                          "Un historique final annuel requiert des origines et des labels hors echantillon distincts des 835 etats initiaux."]}
    atomic_json(output / "audit.json", document)
    lines = []
    for name, frame in frames.items():
        rows = [{"variable": field, **values} for field, values in frame["fields"].items()]
        lines.append('<h2>' + html.escape(name) + '</h2><p>' + html.escape(str(frame["first_utc"]) + ' à ' + str(frame["last_utc"])) + '</p>' + pd.DataFrame(rows).to_html(index=False))
    body = '<h1>NYX — audit des entrées sauvegardées</h1><p>Lecture seule. Le donneur et les unités déclarées restent ceux du run source. Cet audit ne certifie pas les publications historiques à 08 h et ne contacte aucun fournisseur.</p>'
    body += '<h2>Origines rejouables</h2><pre>' + html.escape(json.dumps(document["coverage"], ensure_ascii=False, indent=2)) + '</pre>' + ''.join(lines)
    (output / "report.html").write_text('<!doctype html><html lang="fr"><meta charset="utf-8"><title>NYX — audit</title><style>body{font:14px/1.4 system-ui;margin:30px}table{border-collapse:collapse;display:block;overflow-x:auto}td,th{border:1px solid #ddd;padding:6px}pre{white-space:pre-wrap}</style>' + body + '</html>', encoding="utf-8")
    return document


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annual-repo", type=Path, required=True)
    p.add_argument("--seed-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    audit(seed_record(a.annual_repo, a.seed_run), a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
