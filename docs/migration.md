# Passer au dépôt structuré

La branche `refactor/readable-j1-j2` remplace l’ensemble des lanceurs historiques
par `python -m nyx_runtime`, pour la chaîne CPU FR/DE/BE/NL J+1/J+2 retenue.

## Ce qui change

- Les anciens lanceurs Windows et scripts de campagnes sortent de la branche.
- Les anciennes copies des packages Python à la racine sont retirées : une
  seule copie du noyau scientifique demeure dans `production/runtime/`.
- Les données, caches, modèles et rapports générés sortent du code suivi.
- Les deux anciens lanceurs Python sont réunis dans `nyx_runtime/cli.py`.
- La configuration opérationnelle est décrite dans `config.example.json`.
- Les dépendances sont regroupées dans `requirements.txt`,
  `requirements_aws.txt` et `requirements-dev.txt`.
- La présentation des rapports se modifie dans un fichier HTML distinct.

L’ancien état est conservé par le commit
`f215e66fe93ba0d1ebe6c7c0ddc3900042da82a7`, sur la branche locale
`refactor/aws-daily-forecasts`. L’historique n’a pas été réécrit.

## Installer la nouvelle version

Extraire le nouveau ZIP complet dans **un nouveau dossier**, par exemple
`NYX_v2-structure`. Un nouveau dossier permet de conserver l’installation
d’origine et d’obtenir effectivement le nouvel arbre épuré.

Avec un bundle Git et un clone existant, l’alternative est un worktree séparé :

```bash
git fetch /chemin/NYX_V2_STRUCTURE.bundle refactor/readable-j1-j2:refactor/readable-j1-j2
git worktree add ../NYX_v2-structure refactor/readable-j1-j2
```

Le bundle incrémental nécessite la base
`b0fa97f4d690c52a73d5716c16b152b0124c3882` dans le clone existant.

Depuis le nouveau dossier, préparer Python selon le
[guide de développement](development.md), copier `config.example.json` vers
`config.local.json` et renseigner le chemin du snapshot déjà exporté.
Le format d’entrée et les reçus existants sont conservés.

## Retrouver une ancienne campagne

L’historique Git conserve les fichiers retirés. Pour lire un ancien lanceur :

```bash
git show f215e66fe93ba0d1ebe6c7c0ddc3900042da82a7:NYXAnnualCPU.ps1
```

Pour retravailler une ancienne famille, créer un worktree de cet ancien commit.
Les campagnes historiques ont leurs propres dépendances et validations.
