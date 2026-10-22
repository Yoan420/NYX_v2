# Runtime retenu pour les rapports NYX

`runtime/` contient les 86 scripts et six configurations sélectionnés dans
`SOURCE_MANIFEST.json`, recopiés sans modification depuis la révision indiquée.
L'extension quotidienne est exposée sous ses namespaces Python historiques :
les objets sérialisés et les empreintes scientifiques gardent leur identité.

`nyx_production/` reprend l'adaptateur de calcul et de restitution du dossier
de production préexistant du 22 octobre 2026. Le rendu peut également embarquer
le JavaScript de la distribution Plotly installée dans l'image ; aucune donnée
des rapports privés ne figure ici.

Le lanceur Linux est `python -m nyx_runtime`. Voir
[le guide AWS](../docs/aws_forecasts.md) pour les seeds, les états, les poids
et les dépendances nécessaires. La recette conserve son statut non qualifié
et ses enveloppes indicatives non calibrées.
