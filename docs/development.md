# Développer et relire NYX

Commencer par [l’architecture](architecture.md), puis suivre une commande depuis
`nyx_runtime/cli.py`. La séparation des modules permet de modifier le lancement,
les rapports ou le déploiement sans parcourir les anciennes campagnes.

## Installer l’environnement de développement

Utiliser Python 3.11. Depuis la racine du dépôt, sous Linux :

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

Sous Windows, créer l’environnement avec `py -3.11 -m venv .venv`, puis remplacer
`.venv/bin/python` par `.venv\Scripts\python.exe`. L’utilisation directe de cet
exécutable évite de dépendre d’un script d’activation du terminal.

`requirements.txt` définit les dépendances du calcul et des rapports.
`requirements_aws.txt` ajoute l’accès AWS ; `requirements-dev.txt` ajoute les
outils de test. Les versions du calcul correspondent à l’environnement CPU
retenu. Le client Saturn privé provient de la distribution interne approuvée.

## Configurer une exécution

Copier `config.example.json` vers `config.local.json`, puis renseigner les
emplacements du snapshot et des résultats. Le fichier contient uniquement les
options opérationnelles suivantes :

| Clé | Sens |
| --- | --- |
| `input_root` | Snapshot exporté, contenant `snapshot.json`. |
| `output_root` | Racine des nouveaux lancements. |
| `input_s3_uri` | Préfixe S3 facultatif du snapshot. |
| `output_s3_uri` | Préfixe S3 facultatif des résultats et états publiés. |
| `collect_reporting_sources` | Collecter les comparaisons propres aux rapports. |

Les chemins du JSON sont relatifs à ce fichier. Les chemins passés en argument
sont relatifs au répertoire courant. Les arguments explicites priment sur les
variables `NYX_INPUT_S3_URI` et `NYX_OUTPUT_S3_URI`, puis sur le JSON et les
valeurs par défaut. Les clés inconnues sont rejetées.

```bash
python -m nyx_runtime --help
python -m nyx_runtime plan --first-day 2026-03-28 --as-of 2026-03-27T07:00:00Z
python -m nyx_runtime preflight --config config.local.json
python -m nyx_runtime forecast --config config.local.json
```

`preflight` et `forecast` s’utilisent après la coupure de 08 h à Paris pour les
livraisons du lendemain. Le précontrôle doit annoncer `ready: true` avant le
calcul. `reports` reconstruit les HTML à partir d’un CSV existant ; `export-input`
prépare un snapshot depuis l’installation contenant réellement les données,
modèles et états. Les cinq commandes partagent le même point d’entrée.

## Choisir le bon périmètre de modification

| Changement | Fichiers à examiner | Contrôle principal |
| --- | --- | --- |
| Un chemin ou une option de lancement | `config.local.json`, `settings.py`, `cli.py` | Résolution des chemins, priorité des options et messages d’erreur. |
| La présentation d’un rapport | `templates/report.html`, `reporting.py` | Valeurs conservées, jours physiques complets et rendu des trois HTML. |
| Le cycle d’exécution | `pipeline.py`, `runtime.py` | Verrous, restauration des états et publication du dernier succès. |
| Les transferts ou ressources AWS | `storage.py`, `deploy/aws/`, `Dockerfile` | Préfixes distincts, empreintes, droits et image Linux. |
| La recette, les variables ou les modèles | `production/runtime/`, manifeste, dépendances scientifiques | Nouvelle version scientifique, jeux de comparaison et nouveaux artefacts compatibles. |

Une option de chemin ou d’affichage ne doit pas modifier le sens des prévisions.
Les quatre pays, les horizons J+1/J+2 et la coupure commune sont des contrats de
la recette. Leur évolution demande les mêmes contrôles qu’un changement de
modèle.

## Faire évoluer la recette scientifique

1. Définir la modification et la comparaison attendue sur des données datées,
   avec une coupure explicite et des labels strictement historiques.
2. Créer une version de recette avec ses sources et leur provenance. Documenter
   les changements de variables, de paramètres et de dépendances.
3. Produire les modèles, le seed et les états compatibles avec cette version,
   puis établir les reçus à partir des artefacts réellement produits.
4. Vérifier les contrats, comparer les résultats et exécuter le précontrôle dans
   l’environnement cible avant le premier lancement complet.

Les empreintes du noyau et les versions des bibliothèques scientifiques font
partie des contrats des états. Mettre à jour un manifeste ne convertit pas un
ancien état. Les reçus existants restent des preuves des calculs qui les ont
produits ; ils ne doivent pas être réécrits pour faire accepter un changement.

## Relire et tester une modification

Lancer `python -m pytest` dans l’environnement du projet. Selon le changement,
examiner en priorité les tests de planification, de précontrôle, de calcul, de
rapports, de snapshots ou de stockage. Vérifier notamment la coupure partagée,
les jours de 23/25 heures, le refus des labels futurs et la conservation du
dernier résultat réussi après un échec.

Les modèles et fournisseurs simulés permettent de tester les erreurs et la
publication. Ils ne remplacent pas l’essai réel avec les entrées privées, le
SDK Saturn et le réseau de la plateforme cible. Indiquer dans chaque revue les
contrôles effectués et les validations encore nécessaires.

Les sources historiques restent consultables sur la branche précédente, par
exemple :

```bash
git show refactor/aws-daily-forecasts:NYXAnnualCPU.ps1
```
