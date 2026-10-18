# Rapports quotidiens NYX annuel CPU

Cette entrée produit les rapports HTML habituels pour FR, DE, NL et BE avec la
mention **modèle non encore qualifié**. Elle utilise les modèles annuels CPU,
leurs contrôles de sources et leurs caches. Elle ne change pas l'activation de
production : le reçu garde `qualified_for_production: false` et
`forecast_published: false`, et indique `reports_published: true` seulement quand
les quatre rapports sont publiés.

## Lancer les rapports pour demain

Depuis la racine du dépôt, après l'installation de `.venv-annual` et du client
Saturn décrite dans [la procédure CPU](nyx_annual_production_cpu.md) :

```powershell
.\NYXAnnualReports.ps1
```

La livraison par défaut est **demain dans le fuseau Europe/Paris**, calculée par
Python. Le wrapper lance d'abord `NYXAnnualCPU.ps1 -Action prepare` pour cette
date, puis génère les rapports à partir du lot préparé. Il s'arrête si l'une
des deux étapes échoue. Le premier calcul peut être long ; une reprise conserve
les caches compatibles déjà calculés. Les étapes préparées sont revérifiées.

Les captures JAO, hydro et échanges de la livraison doivent déjà avoir été
effectuées **la veille avant 08:00 Paris**. La préparation et les rapports
s'exécutent après cette coupure. Un historique d'entraînement public téléchargé
plus tard peut servir à ces rapports non qualifiés ; son horodatage réel et son
statut tardif restent visibles. Cela ne remplace jamais une capture manquante
de la livraison à prévoir.

Pour la livraison du **3 octobre 2026**, les captures doivent donc avoir été
effectuées le **2 octobre avant 08:00 Paris**. La commande suivante ne peut pas
les reconstituer après coup :

```powershell
.\NYXAnnualReports.ps1 -DeliveryDay '2026-10-03'
```

Si ces captures sont absentes, la préparation reste bloquée. Il faut capturer
une prochaine livraison pendant sa fenêtre autorisée ; ne pas changer l'heure
des captures ni copier celles d'une autre date.

## Ouvrir les résultats et retrouver une erreur

Le point d'entrée des résultats est :

```text
runs/nyx_annual_cpu_reports/YYYY-MM-DD/index.html
```

Chaque pays possède ses HTML, CSV et Parquet dans `zones/FR`, `zones/DE`,
`zones/NL` ou `zones/BE`. Exemple pour la France :

```text
runs/nyx_annual_cpu_reports/2026-10-03/zones/FR/forecast_fr_2026-10-03_nyx_annual_cpu.html
```

Le `receipt.json` à la racine conserve le statut non qualifié, les contrôles
des sources et les empreintes des fichiers. Une sortie existante est immuable :
le wrapper la refuse avant de relancer la préparation. Une tentative de rapport
interrompue conserve ses diagnostics sous le dossier `.YYYY-MM-DD.attempts`,
sans publier un jeu de rapports incomplet.

Le journal de chaque lancement se trouve dans :

```text
runs/logs/nyx_annual_cpu_reports/YYYY-MM-DD/reports_<horodatage>_<identifiant>.log
```

Ce journal décrit l'enchaînement des étapes. Le fichier voisin
`reports_<horodatage>_<identifiant>.report.log` conserve les messages détaillés
du calcul des rapports, y compris les erreurs précédant la création d'une
tentative. Son chemin est affiché au lancement de cette deuxième étape.

Les journaux et diagnostics détaillés de la préparation restent dans
`runs/logs/nyx_annual_cpu/YYYY-MM-DD/`. En cas d'erreur, conserver ces journaux
et le diagnostic indiqué à l'écran ; relancer la même date après correction.
Le fichier `runs/nyx_annual_cpu_live/YYYY-MM-DD.pipeline.json` décrit la
préparation (`PREPARED`), pas la publication des rapports : consulter le reçu
dans `runs/nyx_annual_cpu_reports/YYYY-MM-DD/` pour leur résultat.

## Programmer les prochains jours

```powershell
.\Install-NYXAnnualTasks.ps1 -EnableDailyReports
Get-ScheduledTask -TaskName 'NYX CWE PIT', 'NYX CWE CPU rapports' | Get-ScheduledTaskInfo
```

Cette option installe les captures à **07:00, 07:15, 07:30, 07:45 et 07:55 Paris**,
puis une tâche séparée **NYX CWE CPU rapports** à **08:05 Paris**. Celle-ci lance
`NYXAnnualReports.ps1` dans une fenêtre masquée pour la livraison du lendemain.
Le poste doit être allumé, éveillé, connecté au réseau, avec la session ouverte.
Une installation après 08:05 ne lance pas rétroactivement le calcul du jour ;
utiliser la commande manuelle si les captures nécessaires existent déjà.

Sans option, `Install-NYXAnnualTasks.ps1` installe toujours uniquement les
captures. `-EnableDailyReports` et `-EnableQualifiedForecast` sont mutuellement
exclusifs. Les tâches installées précédemment ne sont pas supprimées ; vérifier
leur état avant de changer de mode. Le lancement existant
`NYXAnnualCPU.ps1 -Action forecast` conserve ses exigences de qualification.
