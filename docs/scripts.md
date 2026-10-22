# Organisation des scripts NYX

Le point d'entrée Linux/AWS est `python -m nyx_runtime`. Les scripts existants
restent à leur emplacement : les recettes scellées, les reçus de qualification
et certains manifestes calculent leurs empreintes ou les appellent par chemin.
Un déplacement général casserait ces contrats. Le découpage ci-dessous permet
de choisir une chaîne sans importer le laboratoire dans un job quotidien.

| Usage | Entrée à utiliser | Code partagé / configuration |
|---|---|---|
| Refit quotidien CPU à J+1 et J+2, FR/DE/BE/NL | `python -m nyx_runtime` | `nyx_runtime/`, `production/nyx_production/`, `production/runtime/annual_extension/`, `Dockerfile`, `deploy/aws/` |
| Transfert des seeds, poids et états | `python -m nyx_runtime export-input` | `nyx_runtime/snapshot.py`, manifeste `snapshot.json` ; export sans réécriture des reçus |
| Prévisions historiques day-ahead autonome et Kalman | `run_multicountry_forecast.py --mode Both` | `chronos2_hourly/`, `chronos2_modular/`, `chronos2_hourly_live_zones.yaml`, `config/kalman_operational.yaml` |
| Production day-ahead Kalman nucléaire CWE | `run_nuclear_kalman.py` ; lanceur Windows `NuclearKalman.ps1` | `run_nuclear_forecast.py`, `chronos2_hourly/nuclear_forecast.py`, `config/nuclear_forecast.yaml` |
| Les quatre variantes day-ahead | `run_complete_forecast.py` | Orchestre autonome, Kalman, autonome nucléaire et Kalman nucléaire |
| Chaîne annuelle CPU | `run_nyx_annual_pipeline.py` ; consommateur `run_nyx_annual_cpu_live.py` | `chronos2_hourly/nyx_annual_*`, `config/nyx_annual_*` ; qualification et bundles requis |
| Collecte et préparation des entrées | `materialize_saturn_*.py`, `sync_saturn_hourly_candidates.py`, `materialize_jao_core_flowbased.py`, `materialize_noaa_gfs_weather.py`, `materialize_open_meteo_weather.py` | `chronos2_modular/saturn.py`, `data/cache/`, `data/pit/` |
| Captures et sources annuelles | `run_nyx_annual_daily_capture.py`, `run_nyx_annual_*_source.py` | Reçus de provenance ; voir les guides annuels dans `docs/` |
| Rapports sans recalcul de prévision | `run_model_storm_report.py`, `generate_hourly_html_report.py`, `run_kpi_report.py`, `run_nyx_annual_cpu_reports.py` | `kpi_report/`, `runs/reports/`, `runs/exports/` |
| Évaluation et laboratoire | `evaluate_*.py`, `run_*_experiment.py`, `run_*_comparison.py`, `run_chronos2_exogenous_*.py`, `run_auxiliary_lab.py` | `auxiliary_lab/`, `chronos2_exogenous/`, configurations d'expériences et archives locales |
| Interfaces et installation Windows | `Forecast.ps1`, `NYX.pyw`, `Start-ExperimentConsole.ps1`, `Install-*.ps1`, `Setup-NYXAnnualCPU.ps1` | `app_multizone.py`, `experiment_console/`, `requirements_app.txt`, `requirements_console.txt` |
| Outils ponctuels | `scripts/` | Voir [le catalogue des outils](../scripts/README.md) |

Les noms `run_nyx_*` désignent des familles différentes ; ils ne sont pas
interchangeables. `production/SOURCE_MANIFEST.json` identifie la source de la
recette quotidienne retenue ; son adaptation cloud n'en change pas la
qualification. Les scripts `run_nyx_test2_fast.py` et les utilitaires de reprise
du 24 septembre 2026 restent des expériences à date figée.

## Où mettre les prochains changements

- `nyx_runtime/` : options d'exécution, origine, horizons, validation des
  entrées, publication et stockage cloud.
- `deploy/aws/` : infrastructure et commandes de construction/déploiement.
- Paquets métiers existants : modèles, sources et transformations propres
  à une famille ; leurs nouvelles versions doivent suivre leur qualification.
- `scripts/` : utilitaires ponctuels, sans entrée de production concurrente.
- `tests/` : validation de la causalité, des jours DST et des publications.
- `docs/` : contrats, exploitation et méthode ; le README oriente vers ces guides.

## Code, données et sorties

L'image contient le code et les dépendances. Les poids du modèle, caches et
vintages PIT sont chargés séparément depuis un snapshot d'entrée. Les résultats
sont écrits dans un autre répertoire puis publiés dans S3.

Le dépôt suit historiquement une partie de `data/`, mais cela ne constitue pas
un flux de données quotidien. Les archives `runs/`, les modèles et certaines
captures sont ignorés par Git. Un clone seul ne reproduit donc pas une
installation équipée. Le runtime AWS exige les entrées correspondant à son
origine de prévision ; il n'utilise pas un vieux résultat comme prévision nouvelle.

Guide d'exécution : [prévisions J+1/J+2 sur AWS](aws_forecasts.md).
