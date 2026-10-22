# Exécuter NYX sous Linux et préparer AWS

Le programme utilise le même lanceur localement et dans un conteneur :
`python -m nyx_runtime`. L’image Docker appelle ce lanceur directement.

AWS n’est pas requis pour relire le code ou lancer les tests.
Le déploiement est prévu après validation du programme et obtention des accès.

## Répartition des composants

| Composant | Rôle |
| --- | --- |
| Docker / ECR | Environnement Python 3.11 et bibliothèques CPU épinglées. |
| S3 | Snapshot d’entrée, résultats et états de reprise versionnés. |
| AWS Batch sur EC2 | Précontrôle puis calcul J+1/J+2. |
| CloudWatch Logs | Journaux et diagnostics de chaque lancement. |
| EventBridge Scheduler | Lancement facultatif à 08 h 15 Europe/Paris. |

La planification est désactivée par défaut. La recette reste non qualifiée
pour la production et conserve ses enveloppes indicatives non calibrées.

## Préparer le snapshot

Depuis l’installation d’origine contenant réellement les données et états :

```bash
python -m nyx_runtime export-input \
  --seed-bundle /installation/runs/nyx_annual_anytime/SEED/bundle \
  --seed-output /installation/runs/nyx_annual_anytime_reports/SEED \
  --runtime-root /installation \
  --hf-hub-cache /cache/huggingface/hub \
  --destination /transfert/snapshot-aws
```

Arrêter les calculs qui écrivent ces états pendant la copie. La destination
doit être nouvelle et distincte des sources. L’export vérifie les empreintes,
matérialise les liens de fichiers Hugging Face autorisés et copie les reçus
sans les réécrire. Le modèle Chronos-2 requis est la révision
`29ec3766d36d6f73f0696f85560a422f50e8498c`.

Le snapshot contient `snapshot.json`, `seed_bundle/`, `seed_output/`,
`huggingface/hub/` et `state/`. Un snapshot déjà exporté reste utilisable.

## Construire l’image sur une machine autorisée

Le poste de construction doit disposer d’un moteur Docker Linux fonctionnel.
La construction peut aussi être intégrée à une CI autorisée ; le script livré
`deploy.sh` utilise Docker sur la machine qui l’exécute. CloudShell peut servir
aux commandes AWS, tandis qu’un build CodeBuild nécessite sa propre configuration.

Le client Saturn privé approuvé se transmet comme secret BuildKit :

```bash
export NYX_SATURN_WHEEL=/chemin/tshistory_lite-0.5-py3-none-any.whl
docker build --platform linux/amd64 \
  --secret "id=saturn_wheel,src=$NYX_SATURN_WHEEL" -t nyx:local .
```

L’installation de ce client ne fournit pas à elle seule l’authentification
ni l’accès réseau à Saturn. Si le réseau impose une autorité de certification,
utiliser son certificat approuvé. Le secret `proxy_ca` accepte un bundle
de certificats pour les installations pip.

## Contrôler les entrées Linux

Après la coupure de 08 h Paris :

```bash
export NYX_SNAPSHOT=/chemin/absolu/snapshot-aws
docker run --rm \
  --mount "type=bind,src=$NYX_SNAPSHOT,dst=/work/input,readonly" \
  nyx:local preflight --input-root /work/input --output-root /work/output
```

Le résultat attendu est `ready: true` avec un code retour 0. Ce contrôle
vérifie les fichiers, dépendances et états, sans appeler les fournisseurs ni
entraîner les modèles. Le premier calcul réel vérifie ensuite les sources.

Les contrats des états comprennent les versions exactes, les sources et les
identités de configuration. Une copie Windows peut être incompatible sous
Linux. La reconstruction intermédiaire est limitée à sept jours ; un blocage
exige des artefacts compatibles ou une nouvelle préparation scientifique.

## Déployer les ressources

L’équipe AWS doit fournir le compte, le rôle autorisé, le VPC et les sous-réseaux
capables de joindre Saturn. Le raccord réseau, les certificats et les secrets
des fournisseurs sont propres à l’environnement d’exécution.

Depuis une machine avec AWS CLI et Docker, les commandes suivantes créent des
ressources facturables et construisent/publient l’image :

```bash
export AWS_PROFILE=nyx
export AWS_REGION=eu-west-1
aws sts get-caller-identity

bash deploy/aws/deploy.sh \
  --stack-name nyx-forecasts \
  --vpc-id vpc-IDENTIFIANT \
  --subnet-ids subnet-IDENTIFIANT1,subnet-IDENTIFIANT2 \
  --schedule-state DISABLED
```

Les valeurs de compte, profil et réseau sont à fournir par l’équipe AWS.
La sortie du script donne le bucket, la queue Batch, la définition du job
et le groupe de logs.

Charger le snapshot dans le préfixe d’entrée vide configuré par la stack,
par défaut `snapshots/current/`. L’arborescence doit correspondre exactement
à `snapshot.json`. Pour une nouvelle version, utiliser un nouveau préfixe :
une synchronisation qui laisse des fichiers obsolètes sera refusée.

## Vérifier puis lancer

Soumettre d’abord un job Batch avec la commande :

```json
["preflight", "--input-root", "/work/input", "--output-root", "/work/output"]
```

Le job hérite des URI S3 de la définition fournie par la stack.
Exiger le statut Batch `SUCCEEDED` et `ready: true` dans ses journaux.

Lancer ensuite le job avec sa commande par défaut `forecast`. Il produit
demain et après-demain depuis la même coupure du jour à 08 h Paris.
Les résultats publiés se trouvent sous `results/runs/<run_id>/`.
Le marqueur `_SUCCESS.json` doit exister avant de consommer les CSV et HTML.
Les états sont conservés sous `results/state/versions/` avec un pointeur
`results/state/latest.json`.

Après validation d’un premier résultat complet, la planification peut être
activée par une mise à jour CloudFormation avec `ScheduleState=ENABLED`.
Sa désactivation arrête les futurs lancements planifiés. Un verrou abandonné
ne doit être retiré qu’après vérification de la fin effective du job concerné.
