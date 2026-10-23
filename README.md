# NYX — prévisions quotidiennes J+1 et J+2

Une commande calcule sur **CPU** les prix horaires d’électricité pour **FR, DE,
BE et NL**, sur les deux prochains jours, puis produit les CSV, Parquet et
rapports HTML. L’Espagne fournit certaines covariables, sans être un pays de sortie.
Ce README décrit le parcours quotidien et le rôle de chaque fichier conservé.

**J est le jour d’émission à Paris.** J+1 et J+2 partagent la coupure **J à
08:00 `Europe/Paris`**, même si le calcul commence plus tard. Les deux journées
civiles conservent leurs heures physiques, y compris les deux occurrences de
02:00 en automne. Le CSV commun contient normalement **192 lignes**, ou 188/196
avec un changement d’heure. Exemple : après le 23 octobre 2026 à 08:00 Paris,
les livraisons sont les 24 et 25 octobre, origine `2026-10-23T06:00:00Z`, **196 lignes**.
Avant 08:00, la commande par défaut refuse une origine encore future.

## Commandes quotidiennes

Après l’installation et la préparation du snapshot ci-dessous, depuis la racine
du dépôt, utiliser le même Python et la même configuration :

```powershell
# Vers 07:00 Paris : captures publiques pour la livraison de demain
.\.venv\Scripts\python.exe -m nyx_runtime capture --config config.local.json

# Après 08:00 Paris : contrôle local, puis les DEUX horizons et les QUATRE pays
.\.venv\Scripts\python.exe -m nyx_runtime preflight --config config.local.json
.\.venv\Scripts\python.exe -m nyx_runtime forecast --config config.local.json
```

`preflight` doit annoncer `ready: true`. Il ne contacte aucun fournisseur et ne
charge aucun estimateur, mais peut restaurer les copies locales des états.
Il contrôle la continuité CPU ; le cache de refit est vérifié pendant le calcul.
`forecast` refait les contrôles, collecte, réentraîne et publie. Consommer un
résultat seulement si `status.json.state == "COMPLETE"`, via `latest.json`.

La fenêtre de capture est **01:15–08:00 Paris, 08:00 exclu**. La commande réunit
JAO, hydro et échanges ; les scripts internes ne sont pas à lancer un par un.
Sans capture JAO admissible, le calcul quotidien peut continuer avec des variables
JAO `NaN` et `extra_jao__available = 0`. Une capture existante valide reste utilisée ;
une capture tardive est exclue, une capture corrompue bloque. Cela peut changer
les prévisions ; l’impact sur leur précision n’est pas établi ici. Cette tolérance
ne remplace pas les reçus et historiques JAO obligatoires du seed initial.
Une erreur de collecte donne un code non nul sans effacer les autres captures.

Pour planifier, prévoir une tâche à 07:00 et une après 08:00, par exemple 08:15,
selon **Europe/Paris**, avec les chemins absolus de Python/configuration, le dépôt
comme dossier de démarrage et les accès du compte opérationnel. Un seul écrivain
est autorisé par copie du runtime.

## Arbre d’exécution

```text
python -m nyx_runtime
├── export-input → snapshot.py                         PRÉPARATION INITIALE
│   └── Copie vérifiée : seed + poids + états + captures existants
├── capture → capture.py → run_nyx_annual_daily_capture.py
│   └── JAO + hydro + échanges, sans modèle, avant 08:00
├── plan → planning.py                                 VÉRIFICATION DES DATES
├── preflight → settings/planning → pipeline/preflight
│   └── Snapshot, code, versions, seed, poids et continuité des états
├── forecast → pipeline.py → forecast.py
│   └── annual_extension/nyx_daily_refit.py
│       ├── nyx_saved_horizon_inference.py
│       │   ├── nyx_daily_state_cache.py               Historique compatible
│       │   ├── nyx_saved_horizon_sources.py           Sources à la coupure commune
│       │   ├── nyx_daily_two_day_baseline.py          Chronos + résiduel + Kalman
│       │   └── Variables futures + inférence intermédiaire du seed, écartée
│       ├── nyx_daily_refit_history.py                 Historique et matrices
│       ├── nyx_daily_refit_reference.py               12 HGB + 2 paires Test2/prior90
│       └── nyx_daily_refit_experts.py                 Prix et probabilités négatives
│   ├── Validation des grilles, valeurs, horizons et reçus des modèles
│   ├── nyx_saved_horizon_reporting_sources.py          Comparateurs facultatifs
│   ├── forecast.py                                   CSV natifs + CSV commun
│   ├── detail_reports.py                             4 détails pays + CSV/Parquet
│   ├── reporting.py                                  3 vues CWE
│   └── Audit complet → status COMPLETE → latest.json
└── reports → reporting.py                             RENDU SEUL
    └── 3 vues CWE depuis predictions.csv → latest_reports.json
```

Les variables de source et les interactions alimentent le socle ; les matrices
des experts utilisent ensuite **ses quantiles**. Chronos-2 reste gelé sur CPU en
`torch.float32`. Les correcteurs et références sont ajustés selon la recette,
puis les trois experts prix et quatre recettes de probabilité négative sont
réentraînés à chaque nouvelle prévision. Les **sept tentatives** finales ne
signifient pas toujours sept nouveaux modèles : les recettes négatives conservent
leurs règles de calibration et de repli. Un échec du refit ne publie pas
l’inférence intermédiaire du seed. Les états peuvent être enregistrés pendant
le calcul ; `latest.json` n’avance qu’après réussite de tous les rapports.

## Installation et configuration

Utiliser **Python 3.11** et les versions exactes de `requirements.txt`. Les états
sont liés au code, aux configurations et aux versions scientifiques ; ne pas
modifier les pins ou reçus pour contourner un contrôle.

Sous Windows, installer le dépôt dans un chemin court, par exemple
`C:\NYX\code`, avec `C:\NYX\snapshot` pour les entrées et `C:\NYX\results`
pour les sorties. Les chemins imbriqués des états peuvent dépasser la limite
Windows de 260 caractères dans un dossier de travail trop profond. Exécuter
les commandes suivantes depuis le dossier du code.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install C:\paquets-internes\tshistory_lite-0.5-py3-none-any.whl
Copy-Item config.example.json config.local.json
$env:SATURN_AUTHOR = 'VOTRE_IDENTIFIANT_PROFESSIONNEL'
```

Le chemin du wheel est un exemple. Réutiliser le client privé approuvé
**`tshistory_lite==0.5`** déjà disponible dans l’équipe ; vérifier qu’il est installé
dans **le Python choisi pour NYX**, avec
`.\.venv\Scripts\python.exe -m pip show tshistory_lite`. S’il manque dans cet
environnement, y installer le wheel approuvé. Son installation ne donne pas
l’accès au réseau Saturn : le poste et le compte doivent disposer de l’accès
d’entreprise. Configurer l’authentification dans le mécanisme prévu par ce client,
hors Git. `SATURN_AUTHOR` est l’identifiant d’attribution,
pas un mot de passe ; il doit être explicite. Garder les YAML scientifiques
inchangés. Retirer toute variable héritée `NYX_ANNUAL_OPERATIONAL_CONTEXT`.

Sous Linux, utiliser `python3.11 -m venv .venv`, puis `.venv/bin/python` ; un
transfert de seed ne prouve pas sa compatibilité avec un autre OS. Aucun GPU ou
service cloud n’est requis. Pour un proxy JAO, fournir les certificats approuvés
avec `capture --ca-bundle C:\certificats\entreprise.pem`, sans désactiver TLS.

Exemple de `config.local.json` :

```json
{
  "input_root": "C:/NYX/snapshot",
  "output_root": "C:/NYX/results",
  "collect_reporting_sources": true
}
```

Les chemins du JSON sont absolus ou relatifs au fichier. Les options CLI
`--input-root`/`--output-root` priment et sont relatives au dossier courant.
Sans configuration : `snapshot/` et `output/` à la racine du dépôt. Les deux
racines doivent être distinctes, non imbriquées et hors de `production/`.
`collect_reporting_sources: false` désactive uniquement les comparaisons des
rapports ; la prévision nécessite toujours Saturn. Les clés inconnues sont refusées.

## Préparation initiale : seed, poids et états

Le **seed** est le résultat annuel complet qui fournit les modèles initiaux,
matrices, historiques et preuves de provenance. Le seed du 13 octobre et ses
états sont récupérables depuis le commit indiqué ci-dessous, ou depuis
l’installation opérationnelle. Les seuls prix, poids Chronos ou états quotidiens
ne suffisent pas. Le snapshot privé attendu contient :

```text
snapshot.json                  Inventaire SHA-256, protocole nyx_aws_input_snapshot_v1
seed_bundle/                   Matrices, historiques, audits et sources scellés
seed_output/receipt.json        REPORTS_COMPLETE ; future_labels_used = false
seed_output/models/             3 experts prix + 4 modèles négatifs du seed
huggingface/hub/models--amazon--chronos-2/snapshots/<revision>/
                               Configuration et poids locaux liés au seed
state/runs/nyx_daily_cpu_states/ Historiques CPU compatibles
state/runs/nyx_daily_refit_states/ Historiques de refit compatibles
state/data/pit/nyx_annual_*/     Captures publiques disponibles
```

Révision imposée d’`amazon/chronos-2` :
`29ec3766d36d6f73f0696f85560a422f50e8498c` ; aucun téléchargement implicite au calcul.
Neuf groupes de reçus sont requis : `saturn`, `auction_prices`, `fuel`,
`thermal_capacity`, `jao_initial`, `public_hydro`, `lagged_exchange`,
`nyx_quantiles`, `scarcity_confirmed_pair`.

### Récupérer le seed et les états publiés sur main

Le commit **`f8b9cae815f8fbcb3d5e2702fa9b9c51defeb28f`** contient le bundle,
les sept modèles et les états, sans changement du code scientifique. La branche
nettoyée conserve cet historique mais exclut ces données de son arbre de code.
Les commandes suivantes exigent un **clone Git avec cet historique**, pas un
simple ZIP du code. Depuis un clone normal, `git fetch origin` récupère les objets
manquants. Avec un ZIP du code, obtenir séparément l’archive d’artefacts créée
ci-dessous ou un snapshot exporté par un collègue.

Depuis le dépôt, avec `C:\NYX` existant, choisir une archive et un dossier de
destination nouveaux. Cette sélection extrait le seed, son reçu, ses modèles
et les deux arbres d’états ; elle laisse de côté caches, journaux et anciens HTML.

```powershell
$seedId = '2026-10-13_20261012T073755Z_5322a197f771'
git archive --format=zip --output=C:/NYX/nyx-assets-f8b9cae8.zip f8b9cae815f8fbcb3d5e2702fa9b9c51defeb28f "runs/nyx_annual_anytime/$seedId/bundle" "runs/nyx_annual_anytime_reports/$seedId/receipt.json" "runs/nyx_annual_anytime_reports/$seedId/models" runs/nyx_daily_cpu_states runs/nyx_daily_refit_states
Expand-Archive -LiteralPath C:\NYX\nyx-assets-f8b9cae8.zip -DestinationPath C:\NYX\assets
```

Préserver aussi les captures publiques disponibles sur le poste opérationnel,
sous `data/pit/nyx_annual_hydro_captures`, `data/pit/nyx_annual_exchange_captures`
et `data/pit/nyx_annual_jao_initial_live` du `runtime-root` choisi. Ces trois arbres
ne sont pas dans l’archive ci-dessus. Préférer ce runtime opérationnel complet ;
si `C:\NYX\assets` sert de runtime, y copier les archives d’origine disponibles
aux mêmes chemins. Une capture nouvelle ne recrée pas la disponibilité d’hier.
Changer de snapshot remplace les copies locales : les arbres absents du nouvel
export y sont supprimés.

Sur le poste équipé, arrêter les calculs concurrents puis exporter **le même
seed** et ses états vers une destination nouvelle. Exemple avec les artefacts
extraits et le cache Hugging Face habituel ; adapter `$hfCache` si nécessaire :

```powershell
$seedId = '2026-10-13_20261012T073755Z_5322a197f771'
$hfCache = Join-Path $env:USERPROFILE '.cache\huggingface\hub'
.\.venv\Scripts\python.exe -m nyx_runtime export-input --seed-bundle "C:\NYX\assets\runs\nyx_annual_anytime\$seedId\bundle" --seed-output "C:\NYX\assets\runs\nyx_annual_anytime_reports\$seedId" --runtime-root C:\NYX\assets --hf-hub-cache $hfCache --destination C:\NYX\snapshot --restore-git-newlines
```

Le cache doit déjà contenir la révision Chronos imposée. Pour ce seed extrait de
Git, `--restore-git-newlines` est nécessaire : Git a normalisé les fins de ligne
de **329 fichiers JSON**. L’option restaure uniquement les copies de destination
dont les octets obtenus correspondent **exactement au SHA-256 du reçu original** ;
elle enregistre les empreintes avant/après et refuse tout autre écart. Ces copies
incluent les reçus sources `auction_prices.json` et `thermal_capacity.json` ; leur
contenu JSON reste identique. Tous les fichiers du dossier source restent intacts,
ainsi que le reçu `seed_output/receipt.json` et les reçus des états copiés.
Depuis un export opérationnel aux empreintes déjà correctes, cette option n’est
pas nécessaire.

L’export vérifie aussi les copies et refuse les états modifiés pendant la copie.
Le nom historique `nyx_aws_input_snapshot_v1` n’impose pas AWS.
`preflight` vérifie ensuite la continuité CPU ; `forecast` vérifie les états de
refit contre les contrats du seed et du runtime. La continuation est limitée à
**sept jours manquants** ; au-delà, récupérer un état compatible plus récent.
La fabrication et la qualification d’un nouveau seed restent une préparation
séparée, dont le code est disponible dans l’historique Git.

Le seed conservé déclare une **préparation historique** : certaines captures
JAO, hydro et échanges ont été récupérées après sa coupure du 12 octobre.
Ses reçus enregistrent cette limite et ne certifient pas leur disponibilité
avant cette coupure. Les conserver tels quels ; cela n’autorise ni l’antidatage
des nouvelles captures ni une qualification rétrospective de cette prévision.

## Arborescence commentée du dépôt

Utiliser uniquement les sous-commandes `python -m nyx_runtime`. Les fichiers
internes ci-dessous fournissent des fonctions, constantes ou classes à cette
chaîne ; ils ne sont pas autant de programmes à lancer quotidiennement. Les
noms historiques (`candidate`, `ablation`, `dashboard`…) restent pour les imports,
modèles sérialisés et empreintes contractuelles, sans réactiver ces expériences.
Chaque fichier conservé est indiqué avec son rôle.

<details>
<summary>Afficher tous les fichiers et leur rôle dans le pipeline</summary>

```text
NYX_v2/
├── nyx_runtime/  # Interface des commandes quotidiennes
│   ├── templates/
│   │   └── report.html  # Gabarit HTML des vues CWE
│   ├── __init__.py  # Espace de noms ; compatibilité des imports et modèles
│   ├── __main__.py  # Entrée python -m nyx_runtime
│   ├── capture.py  # Capture publique avant 08:00 ou vérification des archives
│   ├── cli.py  # Sous-commandes et arguments de l’opérateur
│   ├── detail_reports.py  # Quatre détails pays, CSV/Parquet, index et audit final
│   ├── engine.py  # Contrats communs, empreintes, imports dynamiques, masquage des secrets
│   ├── forecast.py  # Refit deux jours, validation et export des prévisions
│   ├── pipeline.py  # Coordination, statuts et publication du dernier succès
│   ├── planning.py  # Origine commune, dates J+1/J+2 et heures DST
│   ├── preflight.py  # Versions, seed, poids et continuité des états, sans fournisseur
│   ├── reporting.py  # Trois vues CWE depuis le CSV commun
│   ├── runtime.py  # Manifestes, verrou, copies sûres des états, écritures atomiques
│   ├── settings.py  # Chemins et options de config.local.json
│   └── snapshot.py  # Export, contrôle des empreintes et restauration explicite des fins de ligne Git
├── production/  # Implémentation conservée et manifestes d’intégrité
│   ├── reporting/  # Rendus détaillés d’origine
│   │   ├── nyx_prod_templates/  # Gabarits et calculs des détails pays
│   │   │   ├── __init__.py  # Espace de noms ; compatibilité des imports et modèles
│   │   │   ├── common.py  # Types des données de reporting
│   │   │   ├── forecast_explanation.py  # Explications descriptives des variables/prévisions
│   │   │   ├── hourly_comparison.py  # Graphiques et tableaux horaires comparatifs
│   │   │   ├── metrics.py  # Mesures historiques affichées dans les rapports
│   │   │   └── report.py  # HTML détaillé, graphiques et tableaux pays
│   │   └── report_detail_adapter.py  # Sépare prévision finale et statistiques du socle
│   ├── runtime/  # Noyau scientifique appelé par la CLI
│   │   ├── annual_extension/  # Dix modules exposés dynamiquement sous chronos2_hourly
│   │   │   ├── nyx_daily_refit.py  # Coordonne le refit quotidien et scelle ses résultats
│   │   │   ├── nyx_daily_refit_experts.py  # Trois experts prix et quatre recettes de probabilité négative
│   │   │   ├── nyx_daily_refit_history.py  # Historique d’entraînement et état de refit
│   │   │   ├── nyx_daily_refit_reference.py  # 12 HGB, deux paires Test2 et sélection prior90
│   │   │   ├── nyx_daily_state_cache.py  # Historiques CPU liés au seed, code, versions et configuration
│   │   │   ├── nyx_daily_two_day_baseline.py  # Chronos + résiduel + Kalman ; continuation bornée
│   │   │   ├── nyx_saved_horizon_inference.py  # Socle, variables futures et inférence intermédiaire du seed
│   │   │   ├── nyx_saved_horizon_reporting_sources.py  # Comparateurs EPEX/Storm réservés aux rapports
│   │   │   ├── nyx_saved_horizon_resume.py  # Validation/reprise interne d’artefacts scellés
│   │   │   └── nyx_saved_horizon_sources.py  # Sources J+1/J+2 à la même coupure
│   │   ├── chronos2_hourly/  # Variables, socle, références et experts
│   │   │   ├── models/  # Classes nécessaires aux dépendances/états
│   │   │   │   ├── __init__.py  # Espace de noms ; compatibilité des imports et modèles
│   │   │   │   ├── base.py  # Interface commune des estimateurs
│   │   │   │   ├── blended_residual_corrector.py  # Combinaison déterministe des correcteurs ajustés
│   │   │   │   ├── calibration.py  # Calibration causale par médiane glissante
│   │   │   │   ├── catboost_hourly.py  # Classe CatBoost requise par les dépendances/états
│   │   │   │   ├── ensemble.py  # Utilitaires d’ensemble requis par les dépendances/états
│   │   │   │   ├── lear.py  # Classe LEAR requise par les dépendances/états
│   │   │   │   └── residual_corrector.py  # Correction résiduelle des quantiles
│   │   │   ├── __init__.py  # Espace de noms ; compatibilité des imports et modèles
│   │   │   ├── chronos_adapter.py  # Adaptation des sorties Chronos aux heures de livraison
│   │   │   ├── features.py  # Primitives de variables causales alignées en UTC
│   │   │   ├── historical_price_replay.py  # Primitives du rejeu et checkpoints historiques
│   │   │   ├── hourly_contract.py  # Contrat horaire, fuseaux et jours de 23/24/25 heures
│   │   │   ├── jao_flowbased.py  # Client JAO, normalisation et variables flow-based
│   │   │   ├── kalman_covariates.py  # Contrats et construction des covariables Kalman
│   │   │   ├── kalman_residual.py  # Correction résiduelle Kalman causale
│   │   │   ├── nuclear_cwe_forecast.py  # Protocoles nucléaires CWE réutilisés par le socle
│   │   │   ├── nuclear_forecast.py  # Primitives nucléaires réutilisées par le socle
│   │   │   ├── nyx_annual_cpu_baseline.py  # Chargement Chronos CPU gelé et historique du socle
│   │   │   ├── nyx_annual_cpu_bundle_builder.py  # Construction et scellement des matrices de variables
│   │   │   ├── nyx_annual_cpu_live.py  # Validation du bundle et composition/exécution des modèles
│   │   │   ├── nyx_annual_epex_prices.py  # Prix EPEX seuls ; aucune observation imputée
│   │   │   ├── nyx_annual_equal_ensemble.py  # Ensemble de référence à poids égaux
│   │   │   ├── nyx_annual_feature_projection.py  # Projection dans l’ordre contractuel des variables
│   │   │   ├── nyx_annual_jao_history.py  # Historique JAO de préparation du seed
│   │   │   ├── nyx_annual_jao_source.py  # Preuves et vérification des captures JAO prospectives
│   │   │   ├── nyx_annual_live_preflight.py  # Validation des sources, matrices et historiques
│   │   │   ├── nyx_annual_nyx_quantiles_gate.py  # Contrat des quantiles et de l’historique du socle
│   │   │   ├── nyx_annual_operational_context.py  # Contexte temporaire de coupure du calcul scientifique
│   │   │   ├── nyx_annual_prior90_cpu.py  # Sélection fondée sur les 90 jours antérieurs
│   │   │   ├── nyx_annual_public_history.py  # Historiques publics de préparation du seed
│   │   │   ├── nyx_annual_reference_pair.py  # Paires confirmées et signaux de référence
│   │   │   ├── nyx_annual_saturn_archive.py  # Récupération historique NL encadrée
│   │   │   ├── nyx_annual_saturn_source.py  # Profils/vintages Saturn et leurs reçus
│   │   │   ├── nyx_annual_test2_cpu.py  # Primitives d’ajustement Test2 sur CPU
│   │   │   ├── nyx_annual_wind_source.py  # Règles auditées de récupération éolienne
│   │   │   ├── nyx_forecast_profile_features.py  # Formes des profils de livraison
│   │   │   ├── nyx_fr_hydro_lagged_features.py  # Hydro retardée et disponibilité
│   │   │   ├── nyx_historical_hgb_reference.py  # Primitives d’ajustement des références HGB
│   │   │   ├── nyx_lagged_exchange_features.py  # Échanges retardés et disponibilité
│   │   │   ├── nyx_live_baseline.py  # Configuration et fabrique du correcteur résiduel
│   │   │   ├── nyx_live_hybrid.py  # Règles et variables des paires NYX/Test2
│   │   │   ├── nyx_local_extra_features.py  # Variables des prévisions et combustibles admissibles
│   │   │   ├── nyx_local_io.py  # Publication immuable et écritures atomiques
│   │   │   ├── nyx_local_price_features.py  # Retards de prix et disponibilité des observations
│   │   │   ├── nyx_local_squared_price_model.py  # Expert prix CatBoost individuel
│   │   │   ├── nyx_negative_probability_cpu.py  # Probabilités négatives, calibration et replis prévus
│   │   │   ├── nyx_pooled_calendar.py  # Calendriers nationaux des modèles mutualisés
│   │   │   ├── nyx_pooled_cpu_price_model.py  # Experts prix CatBoost mutualisés
│   │   │   ├── nyx_thermal_capacity_features.py  # Capacités thermiques et disponibilité
│   │   │   ├── process_lock.py  # Verrous de processus du noyau
│   │   │   ├── reporting_observations.py  # Observations EPEX pour les rapports
│   │   │   ├── solar_cwe_sources.py  # Catalogue des sources solaires CWE
│   │   │   ├── solar_wind_forecast.py  # Recette du socle solaire/éolien et réglages Kalman
│   │   │   ├── solar_wind_interaction_features.py  # Interactions physiques des profils
│   │   │   ├── solar_wind_scarcity_ablation.py  # Constantes de variables réutilisées par Test2
│   │   │   ├── solar_wind_scarcity_hybrid.py  # Sélection des quantiles selon le régime
│   │   │   ├── solar_wind_scarcity_regime.py  # Distributions des régimes de rareté
│   │   │   ├── solar_wind_sources.py  # Sources de production solaire/éolienne
│   │   │   └── storm_dashboard.py  # Comparateur Storm ; aucun tableau de bord lancé
│   │   ├── chronos2_modular/  # Acquisition partagée et adaptation Chronos
│   │   │   ├── __init__.py  # Espace de noms ; compatibilité des imports et modèles
│   │   │   ├── common.py  # Types pays/séries, YAML et reproductibilité
│   │   │   ├── data.py  # Lecture des séries et vintages
│   │   │   ├── exogenous_extensions.py  # Enrichissement causal des variables externes
│   │   │   ├── forecasting.py  # Chargement et adaptation des entrées/sorties Chronos
│   │   │   └── saturn.py  # Client Saturn et normalisation des séries
│   │   ├── config/  # Configurations scientifiques contractuelles
│   │   │   ├── kalman_operational.yaml  # Paramètres et covariables Kalman
│   │   │   ├── nyx_annual_cpu_ordered_features.json  # Ordre exact des variables par recette
│   │   │   └── nyx_historical_hgb_feature_schemas.json  # Ordre des 292/334 variables des 12 références HGB quotidiennes
│   │   ├── experiment_console/  # Seul le masquage des secrets est conservé
│   │   │   ├── __init__.py  # Espace de noms ; compatibilité des imports et modèles
│   │   │   └── security.py  # Masquage des secrets ; ancienne console retirée
│   │   ├── marginal_cost_expert/  # Seul le catalogue thermique est réutilisé
│   │   │   ├── __init__.py  # Espace de noms ; compatibilité des imports et modèles
│   │   │   └── sources.py  # Catalogue Pmax réutilisé ; expert de recherche non exécuté
│   │   ├── chronos2_hourly_be_residual_candidate_v1.yaml  # Configuration pays de la recette CPU conservée
│   │   ├── chronos2_hourly_de_residual_candidate_v1.yaml  # Configuration pays de la recette CPU conservée
│   │   ├── chronos2_hourly_fr_residual_v1.yaml  # Configuration pays de la recette CPU conservée
│   │   ├── chronos2_hourly_nl_residual_candidate_v1.yaml  # Configuration pays de la recette CPU conservée
│   │   ├── materialize_jao_core_flowbased.py  # Matérialisation des domaines JAO flow-based
│   │   ├── materialize_saturn_kalman_fuel.py  # Combustibles et coûts marginaux admissibles à la coupure
│   │   ├── materialize_saturn_kalman_weather.py  # Contrats et acquisition des covariables météo/Kalman
│   │   ├── run_nyx_annual_auction_prices_source.py  # Configuration et source de prix EPEX des quatre pays
│   │   ├── run_nyx_annual_daily_capture.py  # Orchestre JAO, hydro et échanges ; vérifie les captures
│   │   ├── run_nyx_annual_exchange_source.py  # Capture, vérification et variables d’échanges
│   │   ├── run_nyx_annual_hydro_source.py  # Capture, vérification et variables hydro publiques
│   │   ├── run_nyx_annual_jao_source.py  # Fonctions de collecte et d’assemblage JAO
│   │   └── run_nyx_annual_thermal_source.py  # Treize séries Pmax et disponibilité thermique
│   ├── REPORTING_MANIFEST.json  # Provenance/empreintes des rendus historiques
│   └── SOURCE_MANIFEST.json  # Provenance/empreintes des sources et configurations
├── tests/  # Contrats, causalité, états, DST et sorties
│   ├── __init__.py  # Espace de noms ; compatibilité des imports et modèles
│   ├── conftest.py  # Prépare les imports de test
│   ├── report_helpers.py  # Fixtures des rapports
│   ├── test_capture.py  # Fenêtre Paris, archives et échecs indépendants des collectes
│   ├── test_detail_reports.py  # Détails pays, probabilités et parité des exports CSV/Parquet
│   ├── test_hourly_contract.py  # Jours de 23/24/25 heures et agrégation quart-horaire
│   ├── test_hourly_features.py  # Retards et statistiques causales ; aucun prix futur utilisé
│   ├── test_kalman_covariates.py  # Groupes de covariables et couverture historique Kalman
│   ├── test_kalman_residual.py  # Correction Kalman causale, sélection et quantiles ordonnés
│   ├── test_nyx_annual_cpu_baseline.py  # Poids Chronos, identité du socle et prévision sans label futur
│   ├── test_nyx_annual_live_preflight.py  # Schémas, reçus et cohérence des indicateurs de disponibilité
│   ├── test_nyx_annual_nyx_quantiles_gate.py  # Origines exactes, quantiles et provenance du socle CPU
│   ├── test_nyx_annual_profile_feature_audit.py  # Traçabilité des profils sans modification des variables
│   ├── test_operational_configuration.py  # Auteur explicite et rejet des surcharges de coupure
│   ├── test_production_live.py  # Origine J+1/J+2, refit complet et publication validée
│   ├── test_production_reports.py  # Parité CSV/HTML, grille complète et reprise des vues CWE
│   ├── test_run_nyx_annual_auction_prices_source.py  # Prix EPEX complets, coupure et reçus immuables
│   ├── test_runtime_cli.py  # Commandes, dates et dernier succès après tous les rapports
│   ├── test_runtime_contract.py  # Manifestes et refus des imports d’une autre installation
│   ├── test_runtime_settings.py  # Chemins portables, types et priorité des arguments
│   ├── test_runtime_snapshot.py  # Export exact, inventaire et refus des entrées redirigées
│   ├── test_runtime_state_safety.py  # Copies d’états sûres, jonctions et verrous
│   ├── test_target_dst_regression.py  # Normalisation UTC et transitions DST sur plusieurs années
│   └── test_versions.py  # Python 3.11, pins exacts et client Saturn privé
├── .gitattributes  # Fins de ligne stables pour les empreintes
├── .gitignore  # Exclusion des secrets, données, états et sorties
├── config.example.json  # Exemple des chemins privés et options de reporting
├── LICENSE  # Licence MIT
├── pytest.ini  # Découverte des tests
├── README.md  # Guide unique : exploitation, pipeline et fichiers
├── requirements-dev.txt  # Runtime et pytest
└── requirements.txt  # Versions exactes du runtime CPU

```

</details>

Les poids, données, secrets, caches et sorties ne sont pas suivis dans cet
arbre. Les états évolutifs sont écrits sous `production/runtime/runs/`, les
captures sous `production/runtime/data/`, tous ignorés par Git. Sauvegarder ces
dossiers avec les résultats. Un même snapshot conserve ces états récents ; un
changement de snapshot réinitialise leurs copies. Ne pas modifier un snapshot
en place ni déplacer un run dont les reçus contiennent des chemins absolus.

## Sources et règles de disponibilité

| Source | Règle conservée |
| --- | --- |
| EPEX via Saturn | Seule référence des labels et retards de prix ; aucun remplacement ENTSO-E ni observation imputée. Une donnée requise absente bloque. |
| Profils Saturn, météo, nucléaire, combustibles | Vintages admissibles à la coupure commune et récupération historique décrite dans les reçus. Les observations devenues disponibles sont relues, jamais remplacées par d’anciennes prévisions. |
| Thermique | Capacités Pmax et indicateurs de disponibilité selon la recette. |
| JAO initial | Capture immuable publiée/reçue avant la coupure ; sinon variables absentes et disponibilité à zéro. Une archive présente doit être valide. |
| Hydro public et échanges | Captures locales avec leurs délais causaux ; seules les absences prévues restent NaN avec indicateurs. |
| Sources publiques J+2 | Souvent non publiées à J 08:00 ; aucune collecte tardive ne reconstitue une disponibilité passée. |
| EPEX/Storm des rapports | Comparateurs facultatifs collectés après l’inférence, séparés de l’entraînement et des prévisions. |

Les futurs labels restent absents. En particulier, les retards J+2 n’utilisent
pas les prix réalisés de J+1 encore inconnus à l’origine commune.

## Sorties et lecture du résultat

```text
<output_root>/
├── latest.json                         Dernier forecast entièrement réussi
├── latest_reports.json                 Dernier rendu seul, s’il a été demandé
├── captures/<capture_id>.json           Diagnostic de capture/vérification
└── <run_id>/
    ├── status.json                     COMPLETE requis avant consommation
    ├── predictions.csv                 Les deux jours et les quatre pays
    ├── native_forecasts/{FR,DE,BE,NL}.csv
    ├── live_result.json                Chemins, compteurs, empreintes ; CSV_COMPLETE
    ├── live_audit.json                 Audits modèle/sources/intervalles
    ├── forecast_extension/             Sources, matrices, modèles et reçus du refit
    └── reports/
        ├── CWE_Model_Storm_<J+1>.html
        ├── CWE_Model_Storm_<J+2>.html
        ├── CWE_Model_Storm_J1_J2_<J+1>.html
        ├── forecast_<pays>_<J+1>_nyx_annual_cpu.html          × 4
        ├── forecast_<pays>_<J+1>_nyx_annual_interval.csv     × 4
        ├── forecast_<pays>_<J+1>_nyx_annual_interval.parquet × 4
        ├── report_audit.json + render_audit.json
        └── index.html                 Liens vers les sept rapports
```

`<pays>` vaut `fr`, `de`, `be`, `nl` ; chaque détail couvre les deux jours.
Les graphiques historiques décrivent **le socle CPU**, pas une mesure hors
échantillon des experts finaux fraîchement ajustés.

| Champs de predictions.csv | Sens |
| --- | --- |
| `delivery_day`, `zone` | Jour civil et pays. |
| `delivery_start_utc`, `delivery_start_local` | Début de l’heure physique, avec décalage explicite. |
| `forecast_origin_utc`, `horizon_day` | Coupure commune, horizon 1 ou 2. |
| `prediction_eur_mwh` | Prix final EUR/MWh. |
| `lower_bound_eur_mwh`, `upper_bound_eur_mwh` | `min(q10 du socle, prix final)` / `max(q90 du socle, prix final)`. |
| `interval_method`, `nominal_interval_level`, `interval_calibrated` | `baseline_envelope_uncalibrated`, 0.8, false : enveloppe indicative, **non calibrée pour l’expert final**. |
| `storm_eur_mwh`, `observed_eur_mwh` | Comparaisons facultatives ; vide = indisponible. |
| `source_html_sha256`, `source_generated_at_utc`, `source_html_name` | Champs de provenance conservés, vides pour une prévision native. |

Les CSV natifs conservent notamment `price_eur_mwh`, `p_negative`,
`p_negative_raw`, `reference_eur_mwh`, `nyx__q10/q50/q90`. Les sorties intervalle
ajoutent `interval_lower_eur_mwh`, `interval_upper_eur_mwh` et leur méthode.
Les probabilités négatives ne figurent pas dans le CSV commun.

Commandes auxiliaires :

```powershell
# Plan seulement, sans fournisseur ni modèle
.\.venv\Scripts\python.exe -m nyx_runtime plan

# Exemple historique : --first-day désigne J+1, avec sa coupure J à 08:00 Paris
.\.venv\Scripts\python.exe -m nyx_runtime plan --first-day 2026-10-23 --as-of 2026-10-22T08:00:00+02:00

# Vérifier une capture archivée, sans contacter les fournisseurs
.\.venv\Scripts\python.exe -m nyx_runtime capture --config config.local.json --verify-only --delivery-day 2026-10-23

# Recréer seulement les trois vues CWE, sans modèle ni réseau
.\.venv\Scripts\python.exe -m nyx_runtime reports --csv C:\NYX\results\RUN_ID\predictions.csv --output-root C:\NYX\results
```

Les options `--first-day`/`--as-of` existent aussi pour `preflight` et `forecast` ;
un calcul historique exige les vintages et états adaptés. `reports` ne recrée
pas les quatre détails pays, qui nécessitent aussi les historiques/covariables.
Il publie `latest_reports.json` et ne modifie pas `latest.json`.

## Dépannage

| Symptôme | Action |
| --- | --- |
| Snapshot ou seed incomplet | Vérifier `input_root` et récupérer le bundle, ses reçus, modèles et poids liés ; les états seuls ne suffisent pas. |
| Chemin introuvable/trop long sous Windows | Utiliser des racines courtes, par exemple `C:\NYX\code`, `C:\NYX\snapshot`, `C:\NYX\results`, avant l’installation et le premier calcul. |
| Empreinte différente | Récupérer les octets d’origine, sans réécrire le manifeste ou les reçus. |
| Module/version incompatible | Installer les pins et le client privé dans le Python du planificateur. |
| Auteur absent / Saturn inaccessible | Définir `SATURN_AUTHOR` et vérifier le réseau/authentification d’entreprise. `preflight` ne teste pas le serveur. |
| Coupure non atteinte | Attendre 08:00 Paris ; respecter le décalage saisonnier avec `--as-of`. |
| Capture manquante/tardive | Ne pas antidater. Consulter les absences déclarées ; le calcul peut accepter celles prévues par la recette. Une archive corrompue reste bloquante. |
| État incompatible / plus de sept jours à reconstruire | Obtenir un export récent compatible avec le seed, le code et les bibliothèques. |
| `nyx_historical_hgb_feature_schemas.json` absent | Mettre à jour `cleanup/daily-j1-j2` avec `git pull --ff-only`, puis relancer `preflight`. Ce schéma de référence est livré avec le code et contrôlé avant les entraînements. |
| EPEX ou profil requis absent | Lire le reçu et faire corriger la source ; aucun remplissage avec des observations futures. |
| Verrou actif | Attendre. Après arrêt brutal, confirmer l’arrêt du processus avant de retirer `production/runtime/runs/.nyx_cloud.lock` (nom historique). |
| Modèle ou rendu en échec | Lire `status.json` et les reçus ; relancer crée un nouveau run. Le dernier succès reste accessible. |
| Comparateurs absents | Lire `live_audit.json` ; `--skip-reporting-sources` désactive leur collecte sans modifier le modèle. |
| Tests : accès refusé à `pytest-of-...` | Ajouter `--basetemp` avec un dossier temporaire court, nouveau et réservé à cet essai. Pytest efface ce dossier ; ne jamais désigner un répertoire contenant des données. |

Codes retour : **0** succès, **2** blocage/échec traité, **130** interruption
clavier. Un CSV isolé ou `live_result.json` ne prouve pas la réussite complète.

## Provenance et validation

Les 96 fichiers scientifiques/configurations proviennent de `main`
`b0fa97f4d690c52a73d5716c16b152b0124c3882` : **84 sont identiques octet pour octet**
aux objets Git ; **12 ont uniquement leurs fins de ligne LF restaurées en CRLF**
pour retrouver les empreintes exactes enregistrées dans les états opérationnels.
Leur structure Python est identique. `.gitattributes` conserve ces octets dans
les checkouts et archives ; `SOURCE_MANIFEST.json` garde les empreintes Git
d’origine et la provenance de la restauration. Les sept modules de rendu
historiques restent inchangés. Les imports dynamiques et les empreintes des
états expliquent les noms/fichiers conservés. Les notebooks,
campagnes, anciennes interfaces et guides redondants ont été retirés de l’arbre
courant ; l’historique Git n’est pas réécrit.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

Vérifications effectuées le 23 octobre 2026 : **220 tests réussis, quatre ignorés**
dans l’environnement Python 3.11 dédié, après restauration du schéma HGB et ajout
des contrôles de présence et d’intégrité avant entraînement.
Les quatre tests de vrais liens symboliques sont empêchés par les privilèges
Windows. Un dossier temporaire neuf a résolu l’accès refusé au répertoire pytest
partagé, sans changer ses permissions. Les 89 modules scientifiques s’importent ;
les ressources contrôlées et le schéma HGB se résolvent. Les **25 paquets publics ont leurs versions
exactes**, avec `pip check` réussi ; les anciens écarts certifi/tzdata sont résolus.
L’archive ZIP conserve les 96 empreintes scientifiques/configurations, y compris
les 12 fichiers CRLF, ainsi que les sept modules de rendu historiques.

Le rendu d’une prévision réelle existante de 192 lignes conserve exactement prix,
bornes, comparaisons et dates entre HTML d’origine, CSV et nouvelles vues CWE.
Les détails pays, états, causalité, captures et transitions DST ont aussi des
contrôles locaux ; ces tests ne prouvent pas l’accès aux fournisseurs.

Pour les états ajoutés par `8f7a076`, **29 reçus et 364 empreintes d’artefacts**
ont été vérifiés. Le code, les configurations et les versions correspondent aux
**16 états CPU et 12 des 13 états de refit** ; le plus ancien état de refit du
15 octobre utilise une autre version et reste exclu. Les contrats ont ensuite
été reconstruits depuis le seed publié dans `f8b9cae8` : les chargeurs natifs des
états récents passent pour l’origine du 23 octobre et la livraison du 24 octobre,
avec l’historique CPU jusqu’au 23 octobre et sans utiliser ses labels en cache.

Le commit `f8b9cae8` fournit désormais **les 15 556 entrées scellées**, le reçu
complet et **les sept modèles du seed**. **15 227 entrées** correspondent
directement aux empreintes originales ; les **329 JSON restants** retrouvent
exactement leur empreinte par restauration LF vers CRLF, avec un contenu JSON
identique. Les 19 fichiers de transformation et le schéma correspondent aussi
au manifeste de matérialisation. Les poids Chronos épinglés du cache local
correspondent aux empreintes du seed.

L’export réel avec `--restore-git-newlines` a réussi : **15 960 fichiers**, dont
les 329 restaurations tracées, avec le reçu principal et les états inchangés.
Le contrôle natif du seed valide ses 15 556 entrées, ses sept modèles et ses
neuf groupes de sources. Les **30 contrôles du bundle historique passent dans
son contexte de préparation d’origine**, sans changer ses reçus. Les trois
déclarations de sources publiques restent refusées en mode historique strict,
conformément à la limite de disponibilité expliquée plus haut.
Les sept modèles CatBoost se chargent avec les schémas de variables et nombres
d’arbres attendus, sans modification de leurs fichiers ni calcul de prévision.
Les commandes réelles `preflight` et `forecast` ne signalent plus de blocage du
seed ; elles retournent le code 2 pour les deux prérequis locaux absents :
`SATURN_AUTHOR` et `tshistory_lite==0.5`. Les contrôles des poids et de la continuité,
conditionnés par le client dans la CLI, ont été exécutés séparément avec les
fonctions natives et les artefacts réels, sans client de substitution.

Sur le poste de vérification, le client Saturn privé, `SATURN_AUTHOR` et l’accès
au réseau d’entreprise restent indisponibles. Les collègues déjà équipés doivent
utiliser leur client approuvé dans l’environnement Python choisi pour ce dépôt.

La recette conserve **`qualified_for_production: false`**. Aucun nouveau calcul
complet avec le seed privé et Saturn n’a été démontré sur ce poste ; la comparaison
numérique avant/après à entrées, états et bibliothèques identiques reste à faire.

L’essai du 23 octobre sur le poste professionnel a terminé le socle J+1/J+2
pour FR, DE, BE et NL, puis a révélé l’omission du schéma HGB avant le refit.
`nyx_historical_hgb_feature_schemas.json` est maintenant restauré depuis `main`,
à l’identique : ses colonnes correspondent aux huit matrices de référence du seed.
Le manifeste d’intégrité et `preflight` le contrôlent avant les entraînements ;
les tests chargent les douze schémas réellement utilisés. Une nouvelle exécution
complète sur le poste professionnel reste nécessaire après cette correction.
