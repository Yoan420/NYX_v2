# NYX — prévisions quotidiennes J+1 et J+2

NYX calcule sur **CPU** les prix horaires de l'électricité pour **FR, DE, BE et
NL** et produit des CSV, Parquet et rapports HTML consultables hors ligne.
Une commande `forecast` calcule les deux jours et les quatre pays. L'Espagne
n'est pas un pays de sortie ; ses covariables nécessaires sont conservées.
Les anciennes applications et campagnes expérimentales sont retirées.

La recette conserve **`qualified_for_production: false`** : le nettoyage du
projet n'est pas une nouvelle qualification scientifique. Avant utilisation
sur un nouveau poste, obtenir un snapshot privé compatible, l'accès Saturn et
vérifier un premier calcul complet. Les tests sans fournisseurs ne prouvent pas
qu'une prévision réelle a été produite.

## Dates et coupure

**J est le jour d'émission à Paris.** J+1 et J+2 sont les deux jours civils
suivants. Ils utilisent **la même coupure J à 08:00 `Europe/Paris`**, même si le
calcul commence plus tard. Le 22 octobre 2026 après 08:00 Paris, `forecast`
vise les 23 et 24 octobre, avec origine `2026-10-22T06:00:00Z`. J+2 ne reçoit
pas une nouvelle coupure à J+1.

Les heures sont stockées en UTC et étiquetées en heure locale avec leur
décalage. Les journées de 23 ou 25 heures conservent toutes leurs heures
physiques, y compris les deux occurrences de 02:00 en automne. Le CSV commun
contient normalement 192 lignes ; 188 ou 196 si la période inclut un changement
d'heure. Les observations futures restent absentes : les retards de prix J+2
ne sont pas complétés avec des observations J+1 encore inconnues.

## Installation et accès

Utiliser **Python 3.11**, un environnement dédié et les versions épinglées dans
`requirements.txt`. Les états et modèles sont liés aux versions scientifiques.
Ne pas remplacer les pins par « les dernières versions ». Les poids Chronos
restent en `torch.float32` sur CPU ; aucun GPU ni service cloud n'est requis.

Depuis la racine du dépôt, dans PowerShell :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install C:\paquets-internes\tshistory_lite-0.5-py3-none-any.whl
Copy-Item config.example.json config.local.json
```

Le chemin du wheel est un exemple : utiliser le client privé approuvé
**`tshistory_lite==0.5`** fourni par l'équipe. Il n'est pas distribué ici. Si un
pin manque sur votre miroir, obtenir le paquet de l'environnement opérationnel
auprès de l'équipe ; ne pas modifier les versions ou reçus pour passer les
contrôles. Sous Linux, utiliser `python3.11 -m venv .venv`, puis
`.venv/bin/python`. Le transfert d'un seed ne prouve pas sa compatibilité avec
un nouvel OS.

Le compte d'exécution doit disposer du réseau et de l'authentification Saturn
de l'entreprise. L'installation du client seul ne fournit pas ces accès. Les
secrets restent dans le mécanisme d'authentification approuvé du client privé,
jamais dans le dépôt. Définir aussi l'identifiant d'auteur de l'opérateur :

```powershell
$env:SATURN_AUTHOR = 'VOTRE_IDENTIFIANT_PROFESSIONNEL'
```

Ce champ n'est pas un mot de passe. Le précontrôle exige une valeur explicite
pour éviter de reprendre l'identifiant historique des YAML scellés. Ne pas
modifier ces YAML : ils participent à l'identité des états. Les scripts de
calcul résolvent leurs ressources depuis le dépôt, sans chemin de poste codé
en dur. Pour un proxy JAO, `capture --ca-bundle C:\certificats\entreprise.pem`
utilise les certificats approuvés ; ne pas désactiver TLS.

### Configuration locale

Exemple de `config.local.json` :

```json
{
  "input_root": "C:/NYX-private/snapshot",
  "output_root": "C:/NYX-results",
  "collect_reporting_sources": true
}
```

Les chemins JSON sont absolus ou relatifs à ce fichier. Les arguments
`--input-root`/`--output-root` priment et sont relatifs au dossier courant.
Sans configuration, les valeurs sont `snapshot/` et `output/` à la racine du
dépôt. Entrées et sorties doivent être distinctes, non imbriquées et hors de
`production/`. Les clés inconnues sont refusées.

`collect_reporting_sources: false` désactive seulement les comparaisons des
rapports ; le calcul nécessite toujours Saturn. Ne pas hériter de
`NYX_ANNUAL_OPERATIONAL_CONTEXT` d'un ancien lanceur « anytime » : la chaîne
quotidienne refuse cette surcharge de la coupure.

## Préparation initiale du seed et des modèles

Obtenir **le seed annuel complet et ses états compatibles** depuis
l'installation opérationnelle. Un dossier de prix ou les poids Chronos seuls
ne suffisent pas. Les données, modèles entraînés, captures et résultats privés
sont exclus de Git. Le snapshot attendu est :

```text
snapshot.json                 Inventaire SHA-256 de tous les fichiers
seed_bundle/                  Matrices, baseline_history, baseline_audits,
                              source_artifacts et source_receipts scellés
seed_output/receipt.json       Reçu annuel REPORTS_COMPLETE, sans labels futurs
seed_output/models/           3 experts prix + 4 modèles de probabilité négative
huggingface/hub/models--amazon--chronos-2/snapshots/<revision>/
                              Poids et configuration Chronos liés au seed
state/runs/nyx_daily_cpu_states/       Historiques quotidiens compatibles
state/runs/nyx_daily_refit_states/     États du réentraînement
state/data/pit/nyx_annual_*/           Captures JAO, hydro et échanges
```

La révision imposée d'`amazon/chronos-2` est
`29ec3766d36d6f73f0696f85560a422f50e8498c`. Le calcul utilise uniquement son cache
local vérifié. Les neuf groupes de reçus requis sont `saturn`, `auction_prices`,
`fuel`, `thermal_capacity`, `jao_initial`, `public_hydro`, `lagged_exchange`,
`nyx_quantiles` et `scarcity_confirmed_pair`.

Pour exporter depuis le poste équipé, arrêter ses calculs concurrents puis
remplacer les chemins d'exemple par ceux du **même seed** :

```powershell
.\.venv\Scripts\python.exe -m nyx_runtime export-input --seed-bundle C:\NYX-source\runs\nyx_annual_anytime\SEED\bundle --seed-output C:\NYX-source\runs\nyx_annual_anytime_reports\SEED --runtime-root C:\NYX-source --hf-hub-cache C:\cache-modeles\huggingface\hub --destination C:\NYX-private\snapshot
```

La destination doit être nouvelle et distincte des sources. L'export vérifie
les empreintes, refuse les états en cours de modification et ne réécrit aucun
reçu. Les exports existants au format `nyx_aws_input_snapshot_v1` restent
acceptés ; le nom historique du protocole n'impose pas AWS.

Les états évitent de rejouer toute la préparation annuelle. La reconstruction
est limitée à **sept jours manquants**. Sans état récent compatible, récupérer
un nouvel export auprès de l'installation opérationnelle. Ne pas modifier les
dates ou empreintes des reçus. La fabrication initiale d'un nouveau seed et sa
qualification ne sont pas des opérations quotidiennes ; leur code reste dans
l'historique Git. Cette version fournit le contrat d'entrée et l'export vérifié.

## Commandes quotidiennes

Utiliser le même environnement, le même dépôt et la même configuration :

1. **Vers 07:00 Paris**, capturer les sources publiques :

   ```powershell
   .\.venv\Scripts\python.exe -m nyx_runtime capture --config config.local.json
   ```

   Fenêtre autorisée : **01:15–08:00 Paris, 08:00 exclu**. La commande vise
   demain. Une source en échec n'efface pas les captures réussies des autres,
   mais produit un code non nul. Une capture tardive ne recrée pas une
   information qui aurait été disponible avant la coupure.

2. **Après 08:00 Paris**, contrôler puis calculer les deux horizons :

   ```powershell
   .\.venv\Scripts\python.exe -m nyx_runtime preflight --config config.local.json
   .\.venv\Scripts\python.exe -m nyx_runtime forecast --config config.local.json
   ```

   `preflight` doit annoncer `ready: true`, code 0. Il ne contacte aucun
   fournisseur et ne charge aucun estimateur ; il peut initialiser les copies
   locales des états. `forecast` refait le contrôle puis collecte et calcule.
   Le succès exige les quatre pays, les deux jours et tous les rapports.

Pour automatiser, prévoir deux tâches locales à 07:00 et après 08:00 (par
exemple 08:15), selon **Europe/Paris**, pas à heure UTC fixe. Fournir au
planificateur les chemins absolus de Python et du JSON, le dépôt comme dossier
de démarrage, l'environnement et les accès du compte opérationnel. Le verrou
autorise un seul écrivain par copie du runtime.

Afficher le plan sans modèles ni fournisseurs :

```powershell
.\.venv\Scripts\python.exe -m nyx_runtime plan
.\.venv\Scripts\python.exe -m nyx_runtime plan --first-day 2026-10-22 --as-of 2026-10-21T08:00:00+02:00
```

`--first-day` désigne J+1, jamais J. `--as-of` doit être exactement 08:00 Paris
du jour précédent, avec décalage explicite. Ces options existent aussi pour
`preflight` et `forecast`. Un calcul historique exige ses vintages et états
adéquats ; changer la date ne fabrique pas ces entrées.

Vérifier une capture archivée sans contacter les fournisseurs :

```powershell
.\.venv\Scripts\python.exe -m nyx_runtime capture --config config.local.json --verify-only --delivery-day 2026-10-23
```

## Pipeline complet

1. **Entrée** : `cli` résout la configuration ; `planning` fige les dates,
   heures physiques et coupure. Le verrou, le snapshot et les manifestes
   contrôlent l'identité des fichiers, du code et des états.
2. **Données** : Saturn fournit EPEX, profils de consommation/production,
   météo, nucléaire, combustibles et thermique selon les règles de vintages.
   JAO, hydro public et échanges proviennent des captures horodatées locales.
   La commande `capture` ne charge aucun modèle.
3. **Variables** : alignement UTC, calendrier, retards de prix, profils,
   hydro, capacité thermique, échanges et interactions. L'ordre des variables
   est fixé dans `config/nyx_annual_cpu_ordered_features.json`. Les règles de
   récupération des profils/cibles historiques restent dans les reçus.
4. **Socle** : poids Chronos-2 gelés, correcteur résiduel, interactions et
   Kalman de la recette. Prolongement borné de l'historique avec états vérifiés.
   Les observations devenues disponibles sont relues depuis EPEX, jamais
   reprises comme labels depuis une ancienne prévision.
5. **Refit quotidien obligatoire** : 12 références HGB, 2 paires Test2,
   prior90, trois experts prix CatBoost et quatre recettes de probabilité de
   prix négatif avec calibration historique. L'inférence intermédiaire du seed
   est écartée ; un échec du refit ne permet pas de publier ce résultat ancien.
6. **Prévisions** : validation des grilles complètes, origine commune,
   valeurs finies, probabilités dans [0,1] et quantiles ordonnés. Les sorties
   natives sont conservées sans transformation.
7. **Sorties** : CSV commun, comparaisons facultatives séparées, puis rapports.
   L'enveloppe affichée vaut `min(q10, prix final)` / `max(q90, prix final)` ;
   son niveau nominal 80 % **n'est pas calibré pour l'expert final**.
   `latest.json` avance seulement après succès de toutes les étapes.

### Données et disponibilités

| Source | Règle opérationnelle |
| --- | --- |
| EPEX via Saturn | Seule référence des prix d'entraînement et retards. Aucun remplacement ENTSO-E ni remplissage des observations requises. Une absence bloquante arrête le calcul. |
| Profils Saturn, météo, nucléaire, combustibles, thermique | Réseau privé et vintages admissibles à la coupure. Politiques historiques de récupération conservées dans les reçus. |
| JAO initial | Capture immuable publiée et reçue avant la coupure. |
| Hydro public et échanges retardés | Captures locales avec délais causaux. Seules les absences autorisées par la recette restent NaN avec leurs indicateurs. |
| Sources publiques J+2 | Parfois non publiées à J 08:00. Pas de substitution rétrospective ; conserver les indicateurs d'absence de la recette. |
| Storm/Observed des rapports | Comparaisons facultatives et séparées du modèle. Une panne les laisse indisponibles et auditables. Elles ne servent ni à entraîner ni à prévoir. |

## Sorties

Chaque prévision crée un dossier unique sous `output_root` :

```text
<run_id>/
  status.json                 COMPLETE, BLOCKED, FAILED, INTERRUPTED ou RUNNING
  predictions.csv             4 pays × les heures physiques des deux jours
  native_forecasts/{FR,DE,BE,NL}.csv
  live_result.json             Chemins, compteurs et empreintes
  live_audit.json              Audits modèle/sources et limites des intervalles
  forecast_extension/          Reçus, matrices et artefacts du refit
  reports/
    CWE_Model_Storm_<J+1>.html
    CWE_Model_Storm_<J+2>.html
    CWE_Model_Storm_J1_J2_<J+1>.html
    forecast_<pays>_<J+1>_nyx_annual_cpu.html
    forecast_<pays>_<J+1>_nyx_annual_interval.csv
    forecast_<pays>_<J+1>_nyx_annual_interval.parquet
    report_audit.json
    index.html
latest.json                   Dernier forecast entièrement réussi
captures/<capture_id>.json     Résultat de chaque capture/vérification publique
```

Les trois fichiers `forecast_<pays>...` existent pour `fr`, `de`, `be`, `nl` et
couvrent les deux jours. Les détails distinguent la prévision finale des
statistiques historiques du socle CPU, qui ne démontrent pas la performance
hors échantillon des experts finaux.

| Champs du CSV commun | Sens |
| --- | --- |
| `delivery_day`, `zone` | Jour civil et pays. |
| `delivery_start_utc`, `delivery_start_local` | Début de l'heure physique avec décalage. |
| `forecast_origin_utc`, `horizon_day` | Coupure commune, horizon 1 ou 2. |
| `prediction_eur_mwh` | Prix final EUR/MWh, sans arrondi de calcul. |
| `lower_bound_eur_mwh`, `upper_bound_eur_mwh` | Enveloppe indicative autour du prix final. |
| `interval_method`, `nominal_interval_level`, `interval_calibrated` | `baseline_envelope_uncalibrated`, 0.8, false. |
| `storm_eur_mwh`, `observed_eur_mwh` | Comparaisons facultatives ; vide = indisponible. |
| `source_html_sha256`, `source_generated_at_utc`, `source_html_name` | Provenance conservée dans le contrat ; vide pour une prévision native. |

Les CSV natifs contiennent notamment `price_eur_mwh`, `p_negative`,
`p_negative_raw`, `reference_eur_mwh`, `nyx__q10/q50/q90`. Les sorties intervalle
ajoutent `interval_lower_eur_mwh`, `interval_upper_eur_mwh` et leur méthode.

Recréer **les trois vues CWE** depuis le CSV commun, sans modèle ni réseau :

```powershell
.\.venv\Scripts\python.exe -m nyx_runtime reports --csv C:\NYX-results\RUN_ID\predictions.csv --output-root C:\NYX-results
```

Cette commande publie `latest_reports.json`, jamais `latest.json`, et ne
prétend pas calculer de nouvelles prévisions. Le CSV commun ne contient pas
les historiques/covariables nécessaires aux quatre détails pays.

Les états évolutifs restent sous `production/runtime/runs/`, les captures sous
`production/runtime/data/`, tous ignorés par Git. Sauvegarder ces dossiers avec
les résultats. Le même snapshot conserve les états récents ; changer de snapshot
réinitialise ces copies depuis le nouvel export. Ne jamais modifier le snapshot
en place. Les états ne remplacent pas les reçus des sorties.

## Dépannage

| Symptôme | Action |
| --- | --- |
| Snapshot/dossier absent | Vérifier `input_root`, `snapshot.json` et obtenir/exporter le seed complet. |
| Empreinte différente / source modifiée | Récupérer le fichier original. Ne pas réécrire le manifeste ou le reçu pour passer le contrôle. |
| Module/version absent ou incompatible | Installer les pins dans le bon Python et le client privé approuvé ; vérifier l'exécutable du planificateur. |
| Auteur absent / accès Saturn refusé | Définir `SATURN_AUTHOR`, configurer authentification et réseau. Le précontrôle local ne teste pas le serveur. |
| Coupure non atteinte / mauvais UTC | Exécuter après 08:00 Paris ; laisser les dates par défaut ou fournir le décalage saisonnier correct. |
| Capture hors fenêtre | Reprendre le lendemain avant 08:00. Ne pas antidater. Vérifier si les absences autorisées suffisent ou si le calcul bloque. |
| État incompatible / plus de 7 jours à reconstruire | Obtenir un export récent compatible avec recette et bibliothèques. Ne pas changer les dates de l'état. |
| EPEX/profils manquants à la coupure | Lire les reçus et faire corriger l'amont. Aucun remplissage avec des observations futures. |
| Verrou actif | Attendre le calcul. Après arrêt brutal, vérifier que le processus indiqué est terminé avant de retirer `production/runtime/runs/.nyx_cloud.lock` (nom historique). |
| Échec de modèle/rendu | Lire `status.json` et les reçus ; le dernier succès reste accessible. Relancer crée un nouveau run et réutilise seulement les états vérifiés. |
| Storm/Observed absents | Lire `live_audit.json`. `--skip-reporting-sources` désactive explicitement leur collecte, sans changer le modèle. |

Codes retour : **0** succès, **2** blocage/échec traité, **130** interruption
clavier. Un CSV isolé n'est pas une preuve de succès. Consommer le dossier de
`latest.json` et vérifier `status.json.state == "COMPLETE"`.

## Arborescence

```text
README.md                     Guide unique
config.example.json           Chemins et option de reporting
requirements.txt              Versions du runtime CPU
requirements-dev.txt          Runtime + pytest
nyx_runtime/                  Entrée → données → modèle → prévisions → sorties
  templates/report.html       Présentation CWE
production/
  SOURCE_MANIFEST.json        Provenance et empreintes du noyau
  REPORTING_MANIFEST.json     Provenance et empreintes des rendus historiques
  runtime/                   Sources scientifiques et configurations
  reporting/                 Rendus détaillés de l'équipe
tests/                       Contrats, causalité, DST, états et sorties
LICENSE                       Licence MIT
.gitignore, .gitattributes    Exclusion des données privées, fins de ligne stables
pytest.ini                    Découverte des tests
```

Les noms historiques du noyau sont conservés pour les modèles sérialisés,
imports dynamiques et empreintes des états. Certains fichiers contiennent aussi
des fonctions historiques : ils restent entiers lorsque leur empreinte est
contractuelle. Ils ne constituent pas des lanceurs alternatifs. Utiliser les
sous-commandes `python -m nyx_runtime`.

## Provenance et validation

Sélection depuis `main` au commit
`b0fa97f4d690c52a73d5716c16b152b0124c3882`, confrontée au paquet partagé le
22 octobre, au lanceur deux jours récupéré localement et dans les échanges
des 14–15 octobre, et aux corrections présentes dans `main`. Les sources
privées servent de preuves et ne sont pas distribuées ici.

L'ancien `NuclearKalman.ps1` et la chaîne annuelle « anytime » ne décrivent pas
le refit J+1/J+2 retenu. La première réorganisation omettait les captures
publiques et les quatre détails pays : ils sont restaurés. Les correctifs EPEX
et des audits scellés sont déjà implémentés. Les propositions AWS et expériences
de modèles sont exclues de l'exécution quotidienne. Le statut non qualifié et
les limites des intervalles sont conservés, sans transformer une suggestion en
correctif validé.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

Les tests portent sur imports/ressources, manifestes, horizons, coupure, DST,
causalité, captures, refit, états, rapports et dernier succès. Les tests simulés
ne remplacent pas l'accès réel aux fournisseurs.

Contrôles exécutés le 22 octobre 2026 :

| Contrôle | Résultat |
| --- | --- |
| Suite complète, Python 3.11.15 et bibliothèques scientifiques CPU de l'installation existante | **209 réussis, 4 ignorés**. Les quatre fixtures de liens symboliques sont refusées par les privilèges Windows ; ce comportement n'est pas compté comme vérifié. |
| Sources scientifiques/configurations comparées à `main` | **95 fichiers identiques octet pour octet**, plus 7 modules de rendu historiques préservés. |
| Imports et ressources | **89 modules scientifiques importés**, 21 ressources actives résolues ; aucun modèle chargé, aucun fournisseur contacté. |
| Rapport réel J+1/J+2 existant | **192 lignes** comparées entre HTML original, CSV et trois nouveaux HTML : prix, bornes, comparaisons, dates et horizons exactement conservés. Aucune nouvelle prévision calculée. |
| Détails pays sur données synthétiques | Valeurs tracées, probabilités, intervalles, heures DST, CSV/Parquet et reçus vérifiés pour les quatre pays. |
| Commandes `preflight` et `forecast` | Blocage attendu, code 2 : snapshot de production absent. Le client privé `tshistory_lite` manque également. |

Le contrôle strict relève aussi `certifi 2026.7.22` au lieu de `2026.10.19` et
`tzdata 2026.4` au lieu de `2026.5` sur cet environnement existant. Ces deux
écarts restent à corriger dans l'environnement d'exploitation ; les tests
réussis ne les transforment pas en versions approuvées.

**Limite de livraison :** aucun calcul complet avec seed privé de production et
Saturn n'a été démontré sur ce poste. Une comparaison numérique complète des
prévisions avant/après reste à effectuer avec les mêmes entrées, états et
bibliothèques. Les sources scientifiques sont conservées sans changement de
calcul. Les données privées et anciens résultats ont été retirés de l'arbre
courant ; l'historique Git existant n'a pas été réécrit.

## Référence des modules

Chaque fichier opérationnel/scientifique est répertorié ci-dessous, par dossier.
Les `__init__.py` maintiennent les espaces de noms historiques. Dans `tests/`,
chaque `test_*.py` contrôle le composant de son nom ; `conftest.py` prépare les
imports et `report_helpers.py` fournit les fixtures de reporting.

<!-- MODULE_CATALOG -->

### Interface quotidienne : `nyx_runtime/`

| Fichier | Rôle |
| --- | --- |
| `__init__.py`, `__main__.py`, `cli.py` | Paquet, entrée `python -m` et sous-commandes documentées. |
| `settings.py`, `planning.py` | Configuration portable ; dates, coupure et grilles horaires physiques. |
| `runtime.py`, `snapshot.py` | Manifestes de code ; export, contrôle et installation du snapshot privé. |
| `preflight.py` | Contrôles locaux des dépendances, ressources et contrats avant calcul. |
| `capture.py` | Commande de capture/vérification publique, verrou et compte rendu. |
| `pipeline.py`, `engine.py` | Coordination du run, statuts, verrou et publication du dernier succès ; adaptation des entrées/sorties communes. |
| `forecast.py` | Appel du refit deux jours, validation des prévisions et écriture des résultats/audits. |
| `reporting.py`, `templates/report.html` | Trois rapports CWE depuis le CSV commun, avec graphiques embarqués. |
| `detail_reports.py` | Quatre détails pays, enveloppe historique conservée, exports CSV/Parquet, index et vérification du jeu complet de rapports. |

### Collecteurs et configurations : `production/runtime/`

Ces scripts sont appelés par le runtime ou ses modules scientifiques. Les
commandes directes de leurs anciens lanceurs ne sont pas nécessaires au
collaborateur qui utilise les commandes quotidiennes ci-dessus.

| Fichier | Rôle |
| --- | --- |
| `run_nyx_annual_daily_capture.py` | Orchestre la capture précoce de JAO, de l'hydro et des échanges pour demain ; vérifie aussi les archives. |
| `run_nyx_annual_jao_source.py`, `materialize_jao_core_flowbased.py` | Capture JAO et assemblage de son historique causal. |
| `run_nyx_annual_hydro_source.py`, `run_nyx_annual_exchange_source.py` | Captures publiques horodatées et assemblage des variables retardées. |
| `run_nyx_annual_auction_prices_source.py` | Prix EPEX réalisés des quatre pays, identité de source et publication contrôlée. |
| `run_nyx_annual_thermal_source.py` | Treize séries de capacité thermique Pmax et leurs indicateurs de disponibilité. |
| `materialize_saturn_kalman_fuel.py`, `materialize_saturn_kalman_weather.py` | Acquisition causale des combustibles, charges résiduelles et variables météo du socle Kalman. |
| `chronos2_hourly_fr_residual_v1.yaml`, `chronos2_hourly_de_residual_candidate_v1.yaml`, `chronos2_hourly_be_residual_candidate_v1.yaml`, `chronos2_hourly_nl_residual_candidate_v1.yaml` | Quatre configurations pays utilisées par la recette ; le suffixe historique `candidate` n'indique pas une expérience à lancer. |
| `config/kalman_operational.yaml` | Paramètres et covariables de la correction Kalman. |
| `config/nyx_annual_cpu_ordered_features.json` | Ordre contractuel des variables de chaque recette annuelle. |

### Refit deux jours : `production/runtime/annual_extension/`

Le runtime expose ces dix fichiers sous l'espace de noms historique
`chronos2_hourly` pour conserver les imports et l'identité des états.

| Fichier | Rôle |
| --- | --- |
| `nyx_daily_refit.py` | Orchestre le refit, ses reprises et les reçus du calcul final. |
| `nyx_daily_refit_history.py`, `nyx_daily_state_cache.py` | Prépare l'historique disponible à la coupure ; charge/publie les états immuables avec reçus. |
| `nyx_daily_two_day_baseline.py` | Prolonge le seed avec les prévisions Chronos, résiduelles et Kalman sur deux jours. |
| `nyx_daily_refit_reference.py` | Réentraîne HGB/Test2 et applique la sélection prior90 strictement antérieure. |
| `nyx_daily_refit_experts.py` | Réentraîne les trois experts prix et quatre recettes de probabilité négative, puis calibre. |
| `nyx_saved_horizon_sources.py` | Réunit les sources des deux jours avec une origine commune. |
| `nyx_saved_horizon_inference.py` | Assemble les variables et contrats des deux jours ; l'inférence du seed sert à la préparation, le résultat publié vient du refit. |
| `nyx_saved_horizon_reporting_sources.py` | Comparateurs EPEX/Storm réservés aux rapports, isolés des entrées des modèles. |
| `nyx_saved_horizon_resume.py` | Vérification et reprise d'un calcul scellé par les fonctions historiques ; aucun contournement du refit quotidien n'est exposé par la CLI. |

### Contrats et modèles : `production/runtime/chronos2_hourly/`

Les modules aux noms `nuclear`, `solar_wind`, `ablation` ou `live` ci-dessous
fournissent encore des fonctions, constantes ou classes importées par la
recette. Leurs noms et docstrings historiques ne définissent pas de nouvelles
étapes quotidiennes ni des variantes à choisir.

| Fichier | Rôle conservé |
| --- | --- |
| `__init__.py`, `hourly_contract.py`, `features.py` | Espace de noms, cible horaire/DST et variables causales alignées en UTC. |
| `chronos_adapter.py`, `historical_price_replay.py` | Adaptation des sorties Chronos aux heures de livraison et primitives de rejeu historique avec checkpoints. |
| `kalman_covariates.py`, `kalman_residual.py` | Contrats des covariables et correction résiduelle Kalman causale. |
| `nuclear_forecast.py`, `nuclear_cwe_forecast.py` | Fonctions et protocoles nucléaires réutilisés par la chaîne résiduelle/solaire-éolienne. |
| `solar_cwe_sources.py`, `solar_wind_sources.py` | Descriptions et acquisition des séries de production solaire/éolienne. |
| `solar_wind_forecast.py`, `nyx_live_baseline.py` | Recette du socle solaire/éolien, configuration Kalman et fabrique du correcteur résiduel. |
| `solar_wind_interaction_features.py`, `nyx_live_hybrid.py` | Interactions physiques et règles des paires NYX/Test2. |
| `solar_wind_scarcity_regime.py`, `solar_wind_scarcity_hybrid.py`, `solar_wind_scarcity_ablation.py` | Distributions de rareté, sélection des quantiles et constantes des variables utilisées par Test2. |
| `nyx_annual_cpu_baseline.py` | Chargement CPU du Chronos gelé, prédictions du socle et validation de son historique. |
| `nyx_annual_cpu_bundle_builder.py`, `nyx_annual_feature_projection.py` | Construction, projection dans l'ordre contractuel et scellement des matrices. |
| `nyx_annual_cpu_live.py` | Chargement/validation du bundle, composition du prix et fonctions d'exécution des modèles. |
| `nyx_annual_live_preflight.py`, `nyx_annual_nyx_quantiles_gate.py` | Contrôles structurels des variables, sources, historiques et quantiles du socle. |
| `nyx_annual_operational_context.py` | Contexte de coupure explicite et temporaire pendant le calcul. |
| `nyx_annual_epex_prices.py`, `nyx_annual_saturn_source.py` | Source de prix EPEX unique et captures Saturn admissibles à la coupure, avec reçus. |
| `nyx_annual_saturn_archive.py`, `nyx_annual_wind_source.py` | Récupérations historiques NL strictement encadrées et substitutions éoliennes auditées. |
| `jao_flowbased.py`, `nyx_annual_jao_source.py` | Client JAO, normalisation, variables flow-based et preuves des captures prospectives. |
| `nyx_annual_jao_history.py`, `nyx_annual_public_history.py` | Validation/assemblage des historiques de préparation JAO et publics, distincts des preuves de capture quotidienne. |
| `nyx_historical_hgb_reference.py`, `nyx_annual_test2_cpu.py` | Ajustement des références HGB et Test2 appelé par le refit quotidien. |
| `nyx_annual_equal_ensemble.py`, `nyx_annual_reference_pair.py`, `nyx_annual_prior90_cpu.py` | Ensemble de référence, paires confirmées et politique de rareté fondée sur les jours antérieurs. |
| `nyx_local_price_features.py`, `nyx_local_extra_features.py` | Retards de prix et variables complémentaires issues des prévisions/combustibles admissibles. |
| `nyx_forecast_profile_features.py`, `nyx_pooled_calendar.py` | Formes des profils de livraison et calendriers nationaux des modèles mutualisés. |
| `nyx_fr_hydro_lagged_features.py`, `nyx_lagged_exchange_features.py`, `nyx_thermal_capacity_features.py` | Variables hydro/échanges retardés et capacités thermiques avec indicateurs d'absence. |
| `nyx_local_squared_price_model.py`, `nyx_pooled_cpu_price_model.py` | Experts prix CatBoost individuels et mutualisés. |
| `nyx_negative_probability_cpu.py` | Ajustement CPU des recettes de probabilité de prix négatif et fonctions de calibration/évaluation utilisées. |
| `nyx_local_io.py`, `process_lock.py` | Écritures atomiques, publication immuable et verrous de processus. |
| `reporting_observations.py`, `storm_dashboard.py` | Références EPEX et comparateur Storm pour les rapports ; aucun tableau de bord interactif n'est lancé. |

Sous-dossier `production/runtime/chronos2_hourly/models/` :

| Fichier | Rôle conservé |
| --- | --- |
| `__init__.py`, `base.py` | Classes exportées et contrats communs des estimateurs. |
| `residual_corrector.py`, `blended_residual_corrector.py` | Correction des quantiles et combinaison déterministe de correcteurs ajustés. |
| `calibration.py` | Calibration causale par médiane glissante. |
| `catboost_hourly.py`, `lear.py`, `ensemble.py` | Classes et utilitaires historiques de modèles horaires, chargés par les dépendances/états ; aucune campagne d'entraînement alternative dans la CLI. |

### Acquisition partagée : `production/runtime/chronos2_modular/`

| Fichier | Rôle |
| --- | --- |
| `__init__.py`, `common.py` | Types des séries/pays, lecture YAML, reproductibilité et réglages du modèle. |
| `data.py`, `exogenous_extensions.py` | Lecture des séries/vintages, régularisation et enrichissement causal des variables. |
| `forecasting.py` | Chargement Chronos et adaptation de ses entrées/sorties. |
| `saturn.py` | Client Saturn, normalisation des séries et archives de vintages. |

Deux petits espaces de noms historiques restent nécessaires :

| Dossier et fichiers | Rôle conservé |
| --- | --- |
| `production/runtime/experiment_console/__init__.py`, `production/runtime/experiment_console/security.py` | Fonctions de masquage des secrets dans les métadonnées et journaux. La console a été retirée. |
| `production/runtime/marginal_cost_expert/__init__.py`, `production/runtime/marginal_cost_expert/sources.py` | Catalogue Pmax, coupures et conversion des séries utilisés par le collecteur thermique. L'expert de recherche n'est pas exécuté. |

### Détails pays : `production/reporting/`

Ces sept fichiers conservent les rendus de l'équipe. Le manifeste séparé
`production/REPORTING_MANIFEST.json` contrôle leurs empreintes ;
`production/SOURCE_MANIFEST.json` couvre le noyau scientifique et ses
configurations.

| Fichier | Rôle |
| --- | --- |
| `report_detail_adapter.py` | Adapte les résultats annuels/deux jours aux détails pays et distingue les statistiques du socle de la prévision finale. |
| `nyx_prod_templates/__init__.py`, `nyx_prod_templates/common.py` | Espace de noms et types de données partagés par les rendus. |
| `nyx_prod_templates/report.py` | Assemblage HTML, graphiques, tableaux et styles. |
| `nyx_prod_templates/metrics.py` | Mesures historiques et couverture disponibles dans les rapports. |
| `nyx_prod_templates/hourly_comparison.py` | Comparaisons horaires et tableaux associés. |
| `nyx_prod_templates/forecast_explanation.py` | Présentation descriptive des variables et de la prévision. |
