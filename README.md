# NYX — prévisions J+1 et J+2

Ce dépôt contient la chaîne CPU retenue pour prévoir les prix de l’électricité
en France, Allemagne, Belgique et aux Pays-Bas, puis produire les CSV et les
rapports HTML journaliers et combiné J+1/J+2.

Les deux horizons partagent une coupure d’information à **08 h, heure de Paris**.
Les journées de 23 ou 25 heures conservent toutes leurs heures physiques.

## Organisation

```text
nyx_runtime/                 Commandes et étapes de l’application
  cli.py                     Point d’entrée unique
  settings.py                Chemins et stockage
  planning.py                Dates et coupure commune
  preflight.py               Vérification des entrées
  forecast.py                Calcul et validation des prévisions
  reporting.py               Données des rapports
  templates/report.html      Présentation HTML/CSS/JavaScript
  pipeline.py                Enchaînement des étapes et publication
  runtime.py                 Intégrité du noyau, verrous et états
  engine.py                  Contrats et chargement du noyau
  snapshot.py                Export vérifié des entrées
  storage.py                 Transferts S3 et marqueurs de succès
production/
  runtime/                   Noyau scientifique sélectionné
  SOURCE_MANIFEST.json       Provenance et empreintes de ses 92 fichiers
tests/                       Tests d’exécution et de calcul
deploy/aws/                  Ressources AWS Batch et déploiement
docs/                        Architecture, développement, migration, AWS
config.example.json          Exemple de configuration locale
```

Les sources scientifiques sont conservées avec leurs noms et octets d’origine :
les modèles sérialisés et les états vérifient ces identités. Les anciennes
copies à la racine, les variantes expérimentales et les données embarquées ont
été retirées de cette branche. Leur historique reste disponible dans Git.

## Un seul lanceur

Depuis la racine du dépôt, dans un environnement Python 3.11 préparé selon le
[guide de développement](docs/development.md) :

```bash
python -m nyx_runtime --help
python -m nyx_runtime plan
python -m nyx_runtime preflight --config config.local.json
python -m nyx_runtime forecast --config config.local.json
python -m nyx_runtime reports --csv output/UN_LANCEMENT/predictions.csv --output-root output
```

Les commandes de prévision pour demain s’utilisent après 08 h à Paris.
La commande `export-input` prépare les entrées depuis l’installation d’origine :
`python -m nyx_runtime export-input --help` décrit ses cinq chemins obligatoires.

Copier `config.example.json` vers `config.local.json`, puis renseigner les chemins
du snapshot et des résultats. Les modèles, poids Chronos, données et états
restent hors du code. Un snapshot déjà exporté au format
`nyx_aws_input_snapshot_v1` peut être réutilisé.

Chaque calcul réussi produit `predictions.csv`, deux rapports
`CWE_Model_Storm_<date>.html` et le rapport combiné
`CWE_Model_Storm_J1_J2_<première_date>.html`.
Le pointeur `output/latest.json` désigne le dernier lancement publié avec succès.

## Modifier et relire

- [Architecture et rôle des modules](docs/architecture.md)
- [Installation, configuration et modifications](docs/development.md)
- [Passage depuis l’ancien dépôt](docs/migration.md)
- [Exécution Linux et déploiement AWS](docs/aws_forecasts.md)

```bash
python -m pytest
```

Les tests couvrent les contrats horaires, la causalité, les prévisions, les
rapports, les états et la publication. Les sources scientifiques sont vérifiées
contre leur manifeste.

La recette conserve `qualified_for_production: false` et une enveloppe
`baseline_envelope_uncalibrated`. Un essai réel avec les données privées,
l’accès Saturn et l’environnement Linux cible reste nécessaire avant exploitation.
