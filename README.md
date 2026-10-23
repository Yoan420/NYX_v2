# NYX — démarrage rapide

Prévisions horaires des prix de l’électricité sur **CPU**, pour **FR, DE, BE et NL**.
Une commande produit **J+1 et J+2** : demain et après-demain, avec une origine commune
à **08:00 Europe/Paris aujourd’hui**. Les changements d’heure sont pris en compte.

## 1. Préparer le poste

- Installer **Python 3.11 (64 bits)** et placer le code dans `C:\NYX\code`.
- Obtenir auprès d’un collègue un **snapshot récent et complet** : seed, modèles,
  poids Chronos, états et captures. Le copier dans `C:\NYX\snapshot` ;
  `C:\NYX\snapshot\snapshot.json` doit exister. Le code seul ne suffit pas.
- Placer le paquet Saturn approuvé dans `C:\NYX\tshistory_lite-0.5-py3-none-any.whl`.
- Disposer de son propre accès réseau et de l’authentification d’entreprise Saturn.

Dans **PowerShell**, exécuter les commandes dans l’ordre et s’arrêter en cas d’erreur :

```powershell
Set-Location C:\NYX\code
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt C:\NYX\tshistory_lite-0.5-py3-none-any.whl
.\.venv\Scripts\python.exe -m pip check
```

`pip check` doit afficher `No broken requirements found`. Installer Saturn dans
**cette `.venv`**, même s’il est déjà disponible dans un autre environnement Python.

## 2. Configurer une seule fois

Toujours depuis `C:\NYX\code`, créer la configuration et saisir son identifiant
professionnel Saturn (ce n’est pas un mot de passe) :

```powershell
@'
{
  "input_root": "C:/NYX/snapshot",
  "output_root": "C:/NYX/results",
  "collect_reporting_sources": true
}
'@ | Set-Content -Encoding UTF8 config.local.json

$env:SATURN_AUTHOR = Read-Host 'Votre identifiant professionnel'
[Environment]::SetEnvironmentVariable('SATURN_AUTHOR', $env:SATURN_AUTHOR, 'User')
```

L’identifiant est enregistré pour votre compte Windows. Garder cette fenêtre pour
le premier essai ; ouvrir les suivantes depuis le menu Démarrer. Les identifiants
d’authentification Saturn restent gérés par le client d’entreprise, hors Git.

Après **08:00 Paris**, vérifier l’installation :

```powershell
.\.venv\Scripts\python.exe -m nyx_runtime preflight --config config.local.json
```

Attendre `ready: true`, puis lancer la prévision ci-dessous. `preflight` vérifie les
fichiers et versions locaux ; l’accès aux sources est vérifié pendant `forecast`.

## 3. Exécuter chaque jour

Depuis `C:\NYX\code`, avec la même configuration :

```powershell
# Après 08:00 Paris : calculer J+1 ET J+2, les quatre pays et les rapports
.\.venv\Scripts\python.exe -m nyx_runtime forecast --config config.local.json
```

La prévision peut démarrer plus tard dans la journée : son origine reste
**08:00 Paris**. Aucune commande JAO séparée n’est requise pour ce démarrage ;
les captures existantes sont utilisées et l’absence de JAO admissible est tolérée,
mais peut modifier les prévisions.

`forecast` refait les contrôles et les ajustements quotidiens automatiquement.
Aucun entraînement initial ni téléchargement de modèle n’est nécessaire avec
un snapshot complet. Lancer **une seule tâche à la fois** par copie du dépôt.

## 4. Lire les résultats

Une exécution réussie termine avec **`state: COMPLETE`**,
**`new_predictions_computed: true`** et un **code retour 0**.

```text
C:\NYX\results\
├── latest.json                 Chemins du dernier calcul entièrement réussi
└── <run_id>\
    ├── status.json             État et diagnostic de cette exécution
    ├── predictions.csv         J+1/J+2 pour les quatre pays
    ├── native_forecasts\       CSV par pays, avec probabilités de prix négatifs
    └── reports\index.html      Accès aux rapports HTML
```

Le CSV commun contient notamment `zone`, `delivery_day`, `delivery_start_utc`,
`forecast_origin_utc`, `horizon_day` et `prediction_eur_mwh` (EUR/MWh).
Les bornes sont indicatives, non calibrées pour le modèle final.
En cas d’échec, `latest.json` reste sur le succès précédent : vérifier sa date.

## Organisation et dépannage

```text
nyx_runtime/            Commandes, contrôles, coordination et exports
production/runtime/     Sources, variables, modèles et ajustements quotidiens
production/reporting/   Rapports détaillés par pays
tests/                  Vérifications des dates, états, modèles et sorties
requirements.txt        Versions exactes des dépendances
config.example.json     Exemple de configuration
```

Parcours : **sources Saturn/EPEX et captures → variables → Chronos et correcteurs
→ ajustements quotidiens → prévisions → CSV et rapports**.

- **Blocage** : lire `blockers` ou `status.json`. Avant 08:00 Paris, attendre.
- **Version ou empreinte incorrecte** : réinstaller les versions imposées ou
  récupérer les fichiers d’origine ; ne pas modifier les reçus.
- **Saturn inaccessible** : vérifier `SATURN_AUTHOR`, le réseau et l’authentification.
- **État trop ancien ou incompatible** : obtenir un snapshot compatible plus récent
  (la continuation accepte au maximum sept jours manquants).

Conserver le snapshot ainsi que `production/runtime/runs/` et
`production/runtime/data/` : ces dossiers locaux portent les états et captures
mis à jour. Ne pas modifier le snapshot en place. Données, secrets et sorties
restent hors Git. Le statut de qualification du modèle reste
`qualified_for_production: false`.

<details>
<summary>Préparer un snapshot pour un collègue — sur le poste qui fonctionne</summary>

Après un calcul réussi, sans tâche NYX en cours, depuis `C:\NYX\code`.
Cet exemple suppose la même organisation des dossiers que ci-dessus ; adapter
les chemins si nécessaire. La destination doit être nouvelle.

```powershell
.\.venv\Scripts\python.exe -m nyx_runtime export-input `
  --seed-bundle C:\NYX\snapshot\seed_bundle `
  --seed-output C:\NYX\snapshot\seed_output `
  --runtime-root C:\NYX\code\production\runtime `
  --hf-hub-cache C:\NYX\snapshot\huggingface\hub `
  --destination C:\NYX\snapshot-transfert
```

Après un code retour 0, transférer **tout** `snapshot-transfert` vers
`C:\NYX\snapshot` sur le nouveau poste, ainsi que le paquet Saturn approuvé.
L’export inclut les états et captures récents du poste qui fonctionne.

</details>
