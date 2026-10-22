# Noyau scientifique retenu

`runtime/` contient les **86 fichiers Python et 6 configurations** nécessaires
à la recette CPU J+1/J+2 ou à ses contrats d’empreintes.

Le lancement se fait depuis la racine du projet :

```bash
python -m nyx_runtime forecast --config config.local.json
```

L’orchestration, les contrôles d’entrée et la restitution se trouvent dans
`nyx_runtime/`. Ce dossier ne contient aucun deuxième lanceur d’application.

`SOURCE_MANIFEST.json` décrit la provenance, la raison de conservation et
l’empreinte de chaque source. Douze fichiers qui ne sont pas sur le chemin
d’appel actif restent requis par les empreintes ou la reprise des états.
Les namespaces historiques des objets sérialisés restent conservés.

Pour relire la recette, commencer par `runtime/annual_extension/nyx_daily_refit.py`,
puis les modules `nyx_daily_refit_experts`, `nyx_daily_refit_reference` et
`nyx_daily_refit_history`. Les modules de sources réalisent les collectes datées.

Une modification scientifique demande de nouveaux artefacts compatibles :
voir [le guide de développement](../docs/development.md).
