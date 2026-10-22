# Prévisions NYX J+1 / J+2 sur AWS

`nyx_runtime/` est l'entrée Linux/AWS de la recette quotidienne CPU retenue pour
FR, DE, BE et NL. Le calcul réutilise `produce_refit_horizon` : collecte des
sources, prolongement des historiques, réentraînement quotidien et prévision
des deux jours civils depuis une coupure commune à **08 h Europe/Paris**.
Les journées de 23 ou 25 heures conservent tous leurs produits horaires UTC.
Les rapports HTML sont reconstruits à partir du CSV publié par ce calcul.

Cette recette est celle du paquet de production préexistant, et conserve son
statut `qualified_for_production: false`. Son enveloppe affichée reste
`baseline_envelope_uncalibrated` : elle n'est pas un intervalle calibré du prix
final. Le portage AWS ne lui attribue pas une qualification supplémentaire.

## Organisation et provenance

| Emplacement | Rôle |
|---|---|
| `nyx_runtime/` | Commandes, chemins Linux, transport S3 et orchestration du job |
| `production/nyx_production/` | Adaptateur de calcul et restitution du paquet retenu |
| `production/runtime/chronos2_hourly/`, `chronos2_modular/` | Dépendances scientifiques sélectionnées |
| `production/runtime/annual_extension/` | Refit quotidien, historique, état CPU et calcul sur deux jours |
| `production/SOURCE_MANIFEST.json` | Origine, inventaire et empreintes des sources conservées |
| `Dockerfile`, `requirements_aws.txt` | Environnement Python 3.11 / PyTorch CPU |
| `deploy/aws/` | Infrastructure AWS Batch, S3, ECR et planification optionnelle |

Le paquet source reprend le bundle de production fourni le 22 octobre 2026.
Le manifeste conserve la révision Git source et les SHA-256. Les rapports
privés, les CSV de prévisions, les seeds et les états d'apprentissage ne font
pas partie du code publié. Les scripts historiques restent accessibles à
leurs chemins d'origine ; voir le [catalogue du dépôt](scripts.md).

## Préparer le snapshot depuis le poste équipé

Un clone seul ne contient pas les entrées de calcul. La commande `export-input`
construit un snapshot privé depuis l'installation ayant produit les rapports.
Elle copie les reçus sans les modifier, vérifie les empreintes et produit le
manifeste `snapshot.json`, exigé à chaque exécution.

Sur le poste d'origine, depuis le dépôt mis à jour, adapter les cinq chemins
ci-dessous au couple seed et au runtime réellement utilisés :

```powershell
python -m nyx_runtime export-input `
  --seed-bundle C:\NYX-donnees\seed\bundle `
  --seed-output C:\NYX-donnees\seed\reports `
  --runtime-root C:\NYX\production\runtime `
  --hf-hub-cache C:\Users\UTILISATEUR\.cache\huggingface\hub `
  --destination C:\NYX-transfert\snapshot
```

Le répertoire de destination doit être nouveau et distinct des sources.
Arrêter les calculs qui utilisent ces états pendant l'export : un verrou ou
une source modifiée provoque un refus. Le résultat contient uniquement les
seeds, le snapshot Chronos requis et les répertoires d'état autorisés :

```text
snapshot/
  snapshot.json          # inventaire exact des fichiers et SHA-256
  seed_bundle/           # bundle annuel complet et ses reçus de sources
  seed_output/           # receipt.json, modèles et artefacts liés au bundle
  huggingface/hub/       # modèle Chronos exact ; liens de fichiers matérialisés
  state/
    runs/nyx_daily_cpu_states/
    runs/nyx_daily_refit_states/
    data/pit/nyx_annual_hydro_captures/
    data/pit/nyx_annual_exchange_captures/
    data/pit/nyx_annual_jao_initial_live/
```

Copier le couple `seed_bundle` / `seed_output` correspondant au **même** rapport
annuel complet. Le précontrôle vérifie les entrées et modèles contre leurs
empreintes ; renommer des fichiers ou reconstruire un reçu ne rend pas un
autre couple compatible. Le seed doit être antérieur ou égal à la première
livraison demandée et attester l'absence de labels futurs.

Le modèle requis est `amazon/chronos-2`, révision
`29ec3766d36d6f73f0696f85560a422f50e8498c`, CPU `torch.float32`. Conserver son
snapshot Hugging Face avec les fichiers réellement référencés. L'exporteur
matérialise ses liens de fichiers ; le transfert n'a pas besoin d'emporter
les tokens ou tout le cache personnel. Les empreintes doivent correspondre
à celles des quatre baselines
du seed. Le précontrôle cherche uniquement les poids locaux ; il ne télécharge
pas silencieusement une autre version.

Les états quotidiens servent à poursuivre les historiques entre le seed et la
nouvelle livraison. L'exporteur conserve leurs reçus et leurs artefacts dans
l'arborescence `state/`. La recette borne la reconstruction intermédiaire à **7 jours**.
Un seed ancien sans état récent peut donc être bloqué même si les poids
Chronos sont présents. Ne pas supprimer ce contrôle pour forcer un lancement.
L'identité d'un état comprend aussi les versions scientifiques et les
empreintes des configurations, qui peuvent contenir des chemins absolus.
Un état Windows peut donc rester incompatible avec Linux même lorsque ses
fichiers sont copiés à l'identique et les bibliothèques alignées. L'exporteur
ne réécrit pas ces contrats. Si aucun état compatible n'est retenu, prévoir un
seed suffisamment récent pour rester dans la borne de 7 jours, ou reconstruire
les états avec la recette sur la plateforme cible. Le précontrôle tranche ce
point avant tout calcul.

Préparer ces seuls répertoires et les vérifier avant transfert. Les fichiers
de connexion, tokens, profils AWS et configurations personnelles restent hors
du snapshot et hors de l'image Docker. Les accès aux fournisseurs passent
par le mécanisme d'identité et le réseau de l'environnement d'exécution.

## Vérifier et exécuter en local

Afficher le plan ne nécessite ni modèle ni données :

```bash
python -m nyx_runtime plan \
  --as-of 2026-10-22T08:00:00+02:00 \
  --horizons 1 2 --zones FR DE BE NL
```

Cette origine correspond aux livraisons du 23 et du 24 octobre. Les deux
jours sont prévus à partir de l'information disponible le 22 octobre à 08 h,
sans avancer la coupure au lendemain pour J+2. Adapter l'origine à la journée
réelle ; une coupure future est refusée.

Le précontrôle décrit les entrées manquantes sans lancer de modèle ni
contacter les fournisseurs :

```bash
python -m nyx_runtime preflight \
  --as-of 2026-10-22T08:00:00+02:00 \
  --input-root /work/input --output-root /work/output
```

Avec `NYX_INPUT_S3_URI` ou `--input-s3-uri`, cette commande commence par
télécharger le snapshot ; `--input-root` doit alors être vide. Si une sortie
S3 est configurée, elle restaure aussi le dernier état complet publié dans le
runtime local sous verrou. Le contrôle scientifique qui suit lit uniquement
les entrées locales. Pour une vérification entièrement hors ligne, utiliser
un snapshot déjà présent et ne pas configurer les URI S3.

Lancer ensuite le calcul explicite :

```bash
python -m nyx_runtime forecast \
  --as-of 2026-10-22T08:00:00+02:00 \
  --horizons 1 2 --zones FR DE BE NL \
  --input-root /work/input --output-root /work/output
```

Le précontrôle local ne prouve pas que Saturn ou les autres fournisseurs sont
joignables depuis AWS. La collecte vérifie les vintages et les sources avant
les fits ; une entrée obligatoire absente bloque le calcul. Les absences
autorisées par la recette conservent ses indicateurs explicites. Storm et les
observations de comparaison restent des données de restitution et ne
deviennent pas des entrées d'entraînement.

Les sorties de chaque lancement ont leur répertoire propre : statut JSON,
`predictions.csv`, prévisions natives par pays, reçus de sources et de refit,
audits et rapports. Un rapport restauré depuis un CSV existant ne constitue
pas un nouveau calcul. La commande suivante reconstruit les rapports HTML
autonomes, sans recalculer ni collecter de prévisions :

```bash
python -m nyx_runtime reports \
  --csv /chemin/du/lancement/predictions.csv \
  --output-root /work/reports
```

## Construire l'image et installer AWS Batch

Le déploiement exige un compte AWS configuré, AWS CLI, Docker avec BuildKit et
des sous-réseaux du VPC autorisant les sorties nécessaires. Le template crée
le dépôt ECR, un bucket S3 privé, les rôles IAM, les logs CloudWatch et un
environnement AWS Batch EC2 dont la capacité minimale est nulle.

La tâche utilise le client privé `tshistory_lite`. Si le client n'est pas
disponible dans l'image de base, fournir son fichier de dépendances approuvé
au build comme secret BuildKit. Un wheel autorisé peut également être fourni
par `NYX_SATURN_WHEEL` :

```bash
export NYX_PRIVATE_REQUIREMENTS=/chemin/prive/requirements_private.txt
# Ou : export NYX_SATURN_WHEEL=/chemin/prive/tshistory_lite-0.5-py3-none-any.whl
AWS_REGION=eu-west-1 deploy/aws/deploy.sh \
  --stack-name nyx-forecasts \
  --vpc-id vpc-XXXXXXXX \
  --subnet-ids subnet-XXXXXXXX,subnet-YYYYYYYY
```

Le script construit et pousse l'image, puis crée ou met à jour la stack. Il
ne configure pas la connexion au réseau interne de Saturn. Le VPC doit
disposer du routage et de l'authentification autorisés vers les services
d'entreprise, ainsi que des sorties AWS et des sources publiques requises.
Le client privé seul ne fournit pas cet accès réseau.

La planification est désactivée par défaut. Après validation d'un job manuel,
`--schedule-state ENABLED` l'active à 08 h 15 dans le fuseau `Europe/Paris`.
L'heure locale reste stable lors des changements d'heure. Le calcul peut être
long et coûteux : dimensionner CPU, mémoire et capacité maximale Batch à
partir d'une exécution mesurée.

## Charger les entrées et soumettre un job

Récupérer les noms des ressources dans les sorties CloudFormation :

```bash
aws cloudformation describe-stacks --stack-name nyx-forecasts \
  --query 'Stacks[0].Outputs' --output table
```

Envoyer le snapshot privé dans le préfixe d'entrée du bucket créé :

```bash
aws s3 sync /chemin/du/snapshot/ s3://NOM_DU_BUCKET/snapshots/current/
```

Ce préfixe doit contenir exactement l'inventaire de `snapshot.json` : utiliser
un préfixe vide lors du premier transfert. Pour remplacer un snapshot,
préférer un nouveau préfixe et adapter `InputPrefix` dans la stack ; une
synchronisation qui laisse les fichiers d'un ancien export est refusée.

Le job reçoit `NYX_INPUT_S3_URI` et `NYX_OUTPUT_S3_URI` depuis sa définition.
La première variable désigne le snapshot à restaurer ; la seconde désigne
le préfixe de publication. Utiliser le nom de queue et la définition de job
retournés par la stack :

```bash
aws batch submit-job \
  --job-name nyx-j1-j2-manuel \
  --job-queue NOM_DE_LA_QUEUE \
  --job-definition NOM_DE_LA_DEFINITION
```

Suivre l'état avec `aws batch describe-jobs --jobs IDENTIFIANT_DU_JOB` et les
logs CloudWatch indiqués dans la réponse. Un échec ou un état `BLOCKED` ne
publie pas une prévision complète. Vérifier les statuts et reçus du lancement
avant de consommer un CSV ou un rapport dans S3.

Les publications distantes utilisent `results/runs/<identifiant>/` et un
marqueur `_SUCCESS.json` écrit après les artefacts. Les échecs portent un
marqueur `_FAILED.json`. Les états de continuation complets sont versionnés
sous `results/state/versions/` ; `results/state/latest.json` ne pointe vers une
nouvelle version qu'après sa publication. Un état complet peut être conservé
même si le transfert ultérieur du rapport échoue ; il ne vaut pas publication
d'une prévision. Le lancement suivant restaure cet état en priorité pour
poursuivre les historiques. Les consommateurs de prévisions exigent toujours
le marqueur `_SUCCESS.json` du lancement concerné.

Un verrou S3 sérialise les prévisions qui partagent ce préfixe de sortie.
Si un job s'arrête brutalement, vérifier dans Batch qu'il est terminé avant
de retirer un éventuel verrou abandonné `results/locks/forecast.json`.

La session de préparation du dépôt n'a pas déployé de ressource AWS ni produit
de nouvelles prévisions avec les données privées. L'exécution complète exige
le snapshot validé, le client privé et les accès aux sources.
