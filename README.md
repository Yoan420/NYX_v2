# NYX — démarrage rapide

Prévisions horaires sur **CPU**, pour **FR, DE, BE et NL**. Une commande produit
**J+1 et J+2** : demain et après-demain, avec une origine commune à
**08:00 Europe/Paris aujourd’hui**. Les changements d’heure sont pris en compte.

**Les données initiales sont incluses dans `snapshot/`** : seed, sept modèles,
poids Chronos et états de reprise. Aucun snapshot à demander à un collègue.
Les données du jour sont ensuite récupérées via les sources d’entreprise.

## 1. Installer une seule fois

Prérequis : **Python 3.11 (64 bits)**, **Git avec Git LFS** et votre accès
d’entreprise Saturn. Cloner le dépôt dans un chemin court : `C:\NYX\code`.
Placer le paquet Saturn approuvé dans
`C:\NYX\tshistory_lite-0.5-py3-none-any.whl` ; ce client privé est fourni par l’équipe.

Dans **PowerShell**, exécuter dans l’ordre et s’arrêter en cas d’erreur :

```powershell
Set-Location C:\NYX\code
git lfs install --local
git lfs pull
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt C:\NYX\tshistory_lite-0.5-py3-none-any.whl
.\.venv\Scripts\python.exe -m pip check
```

Utiliser **Git clone**, pas le bouton Download ZIP : Git LFS récupère les gros
fichiers. Le snapshot complet occupe environ **1,6 Go**. `pip check` doit afficher
`No broken requirements found`. Saturn doit être installé dans **cette `.venv`**,
même s’il existe déjà dans un autre environnement Python.

## 2. Définir son identifiant et vérifier

Saisir votre identifiant professionnel Saturn, pas un mot de passe :

```powershell
$env:SATURN_AUTHOR = Read-Host 'Votre identifiant professionnel'
[Environment]::SetEnvironmentVariable('SATURN_AUTHOR', $env:SATURN_AUTHOR, 'User')
```

Il est enregistré pour votre compte Windows. Garder cette fenêtre pour le premier
essai ; ouvrir les suivantes depuis le menu Démarrer. L’authentification reste
gérée par le client Saturn de l’entreprise.

Après **08:00 Paris**, depuis `C:\NYX\code` :

```powershell
.\.venv\Scripts\python.exe -m nyx_runtime preflight
```

Attendre `ready: true`. Le contrôle vérifie les fichiers et versions locaux ;
l’accès aux sources est vérifié pendant la prévision. Les chemins par défaut sont
`snapshot/` pour les entrées et `output/` pour les résultats. Pour les modifier,
copier `config.example.json` vers `config.local.json`, adapter les chemins et
ajouter `--config config.local.json` aux commandes.

## 3. Lancer la prévision chaque jour

Depuis `C:\NYX\code`, **après 08:00 Paris** :

```powershell
.\.venv\Scripts\python.exe -m nyx_runtime forecast
```

Cette commande calcule **les deux jours, les quatre pays et les rapports**.
Elle effectue les contrôles et les ajustements quotidiens automatiquement.
Elle peut démarrer plus tard dans la journée : l’origine reste 08:00 Paris.
Aucune commande JAO séparée n’est requise pour ce démarrage ; l’absence de
capture JAO admissible est tolérée, mais peut modifier les prévisions.
Lancer **une seule tâche à la fois** par copie du dépôt.

## 4. Trouver les résultats

Une réussite termine avec **`state: COMPLETE`**,
**`new_predictions_computed: true`** et un **code retour 0**.

```text
output/
├── latest.json                 Chemins du dernier calcul entièrement réussi
└── <run_id>/
    ├── status.json             État et diagnostic de cette exécution
    ├── predictions.csv         J+1/J+2 pour les quatre pays
    ├── native_forecasts/       CSV par pays et probabilités de prix négatifs
    └── reports/index.html      Accès aux rapports HTML
```

Le CSV commun indique notamment le pays (`zone`), la livraison (`delivery_day`,
`delivery_start_utc`), l’origine (`forecast_origin_utc`), l’horizon (`horizon_day`)
et le prix (`prediction_eur_mwh`, en EUR/MWh). Les bornes sont indicatives,
non calibrées pour le modèle final. En cas d’échec, `latest.json` conserve le
succès précédent : vérifier sa date.

## Organisation et dépannage

```text
snapshot/               Données, modèles et états initiaux vérifiés
nyx_runtime/            Commandes, contrôles, coordination et exports
production/runtime/     Sources, variables, modèles et ajustements quotidiens
production/reporting/   Rapports détaillés par pays
tests/                  Vérifications des contrats et sorties
licenses/chronos-2/     Licence et attribution des poids Chronos
requirements.txt        Versions exactes des dépendances
config.example.json     Configuration facultative des chemins
```

Parcours : **sources → variables → Chronos et correcteurs → ajustements
quotidiens → prévisions → CSV et rapports**.

- **Blocage** : lire `blockers` ou `status.json`. Avant 08:00 Paris, attendre.
- **Fichier LFS absent ou empreinte incorrecte** : lancer `git lfs pull`.
  Si l’erreur persiste, récupérer les fichiers d’origine sans modifier les reçus.
- **Version incorrecte** : réinstaller les versions de `requirements.txt` dans `.venv`.
- **Saturn inaccessible** : vérifier `SATURN_AUTHOR`, le réseau et l’authentification.
- **État trop ancien/incompatible** : récupérer un snapshot compatible plus récent.
  Celui livré ici contient les états à la coupure du **22 octobre 2026** ;
  la continuation accepte au maximum **sept jours manquants**.

Le snapshot initial reste immuable. Les états et captures évolutifs sont conservés
dans `production/runtime/runs/` et `production/runtime/data/` : les sauvegarder.
Ces fichiers de travail, les secrets et les nouvelles sorties restent hors Git.
Le statut de qualification du modèle reste `qualified_for_production: false`.
