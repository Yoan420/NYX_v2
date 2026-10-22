# Outils ponctuels

L'entrée pour les jobs Linux/AWS est `python -m nyx_runtime` ; l'infrastructure
se trouve dans `deploy/aws/`. Le répertoire `scripts/` contient les utilitaires
suivants et ne lance pas automatiquement une chaîne de production.

| Script | Usage |
|---|---|
| `launch_nyx_test2_fast.ps1` | Reprise Windows de l'expérience Test2 figée au 24 septembre 2026 ; chemins du poste d'étude |
| `start_nyx_test2_live.ps1` | Lanceur Windows Test2 avec identité du plan figé et surveillance des processus |
| `probe_nyx_fast_cpu.py` | Mesure CPU sur un snapshot DE de l'expérience du 24 septembre 2026 |
| `verify_nyx_hourly_curves.cjs` | Vérification Playwright d'un rapport Test2 local de septembre 2026 ; chemin Node/Edge du poste d'étude |

Les scripts racine sont classés par usage dans le
[catalogue du dépôt](../docs/scripts.md). Les commandes d'exploitation cloud
sont décrites dans le [guide AWS](../docs/aws_forecasts.md).
