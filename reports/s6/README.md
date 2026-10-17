# S6: métricas base por nivel taxonómico

Generado por `scripts/train/taxonomic_metrics.py` a partir de las predicciones guardadas; no edita a mano.
Dataset `v002` · huella `144323e165ecfbe86d29a3484e56fd91b8e8cb1dce1eef5f1d1355fea7bedb90`.

Regla principal: el taxón predicho es el del género top-1 (la decisión real del modelo, juzgada a un
nivel superior). Por observación se promedian primero las probabilidades de sus imágenes (`mean_prob`).
IC95 = bootstrap percentil sobre observaciones. La regla alternativa (sumar probabilidades por taxón)
queda en `taxonomic_summary.csv` (`marginal_*`) solo como comparación.

## Taxonomía usada (FungiTastic, sin ediciones)

| Género | Familia | Orden | Clase | Filo | Especies | Obs. val | Obs. test |
|---|---|---|---|---|---:|---:|---:|
| Amanita | Amanitaceae | Agaricales | Agaricomycetes | Basidiomycota | 19 | 904 | 1315 |
| Auricularia | Auriculariaceae | Auriculariales | Agaricomycetes | Basidiomycota | 2 | 288 | 319 |
| Ganoderma | Polyporaceae | Polyporales | Agaricomycetes | Basidiomycota | 5 | 335 | 346 |
| Geastrum | Geastraceae | Geastrales | Agaricomycetes | Basidiomycota | 9 | 79 | 182 |
| Hericium | Hericiaceae | Russulales | Agaricomycetes | Basidiomycota | 3 | 66 | 51 |
| Lycoperdon | Lycoperdaceae | Agaricales | Agaricomycetes | Basidiomycota | 10 | 408 | 407 |
| Morchella | Morchellaceae | Pezizales | Pezizomycetes | Ascomycota | 3 | 11 | 20 |
| Pleurotus | Pleurotaceae | Agaricales | Agaricomycetes | Basidiomycota | 3 | 251 | 319 |
| Trametes | Polyporaceae | Polyporales | Agaricomycetes | Basidiomycota | 7 | 756 | 791 |
| Xylaria | Xylariaceae | Xylariales | Sordariomycetes | Ascomycota | 5 | 354 | 289 |

## Macro-F1 por observación (métrica principal)

| Modelo | Partición | Nivel | Taxones | n | Macro-F1 | IC95 | Exactitud | Exactitud balanceada |
|---|---|---|---:|---:|---:|---|---:|---:|
| ma_resnet18_v002 | val | género | 10 | 3452 | 0.9545 | 0.9454–0.9621 | 0.9594 | 0.9557 |
| ma_resnet18_v002 | val | familia | 9 | 3452 | 0.9588 | 0.9492–0.9667 | 0.9664 | 0.9611 |
| ma_resnet18_v002 | val | orden | 7 | 3452 | 0.9623 | 0.9512–0.9719 | 0.9742 | 0.9600 |
| ma_resnet18_v002 | val | clase | 3 | 3452 | 0.9870 | 0.9822–0.9917 | 0.9928 | 0.9898 |
| ma_resnet18_v002 | val | filo | 2 | 3452 | 0.9810 | 0.9742–0.9879 | 0.9928 | 0.9851 |
| ma_resnet18_v002 | test | género | 10 | 4039 | 0.9463 | 0.9329–0.9567 | 0.9579 | 0.9400 |
| ma_resnet18_v002 | test | familia | 9 | 4039 | 0.9495 | 0.9346–0.9608 | 0.9639 | 0.9434 |
| ma_resnet18_v002 | test | orden | 7 | 4039 | 0.9521 | 0.9338–0.9655 | 0.9735 | 0.9387 |
| ma_resnet18_v002 | test | clase | 3 | 4039 | 0.9611 | 0.9249–0.9820 | 0.9908 | 0.9518 |
| ma_resnet18_v002 | test | filo | 2 | 4039 | 0.9681 | 0.9570–0.9772 | 0.9908 | 0.9757 |
| s5_mobilenet_v3_large_v002_a | val | género | 10 | 3452 | 0.9706 | 0.9563–0.9809 | 0.9797 | 0.9665 |
| s5_mobilenet_v3_large_v002_a | val | familia | 9 | 3452 | 0.9721 | 0.9567–0.9831 | 0.9835 | 0.9668 |
| s5_mobilenet_v3_large_v002_a | val | orden | 7 | 3452 | 0.9705 | 0.9513–0.9838 | 0.9864 | 0.9623 |
| s5_mobilenet_v3_large_v002_a | val | clase | 3 | 3452 | 0.9788 | 0.9372–0.9970 | 0.9968 | 0.9644 |
| s5_mobilenet_v3_large_v002_a | val | filo | 2 | 3452 | 0.9916 | 0.9861–0.9962 | 0.9968 | 0.9910 |

## Macro-F1 por imagen

| Modelo | Partición | Nivel | Taxones | n | Macro-F1 | IC95 | Exactitud | Exactitud balanceada |
|---|---|---|---:|---:|---:|---|---:|---:|
| ma_resnet18_v002 | val | género | 10 | 6290 | 0.8997 | — | 0.9207 | 0.8946 |
| ma_resnet18_v002 | val | familia | 9 | 6290 | 0.9041 | — | 0.9342 | 0.8996 |
| ma_resnet18_v002 | val | orden | 7 | 6290 | 0.9065 | — | 0.9482 | 0.8934 |
| ma_resnet18_v002 | val | clase | 3 | 6290 | 0.9226 | — | 0.9862 | 0.9148 |
| ma_resnet18_v002 | val | filo | 2 | 6290 | 0.9585 | — | 0.9863 | 0.9659 |
| ma_resnet18_v002 | test | género | 10 | 7159 | 0.8875 | — | 0.9109 | 0.8865 |
| ma_resnet18_v002 | test | familia | 9 | 7159 | 0.8912 | — | 0.9205 | 0.8904 |
| ma_resnet18_v002 | test | orden | 7 | 7159 | 0.8984 | — | 0.9412 | 0.8848 |
| ma_resnet18_v002 | test | clase | 3 | 7159 | 0.9050 | — | 0.9838 | 0.9024 |
| ma_resnet18_v002 | test | filo | 2 | 7159 | 0.9379 | — | 0.9838 | 0.9395 |
| s5_mobilenet_v3_large_v002_a | val | género | 10 | 6290 | 0.9254 | — | 0.9469 | 0.9222 |
| s5_mobilenet_v3_large_v002_a | val | familia | 9 | 6290 | 0.9299 | — | 0.9591 | 0.9260 |
| s5_mobilenet_v3_large_v002_a | val | orden | 7 | 6290 | 0.9267 | — | 0.9657 | 0.9211 |
| s5_mobilenet_v3_large_v002_a | val | clase | 3 | 6290 | 0.9346 | — | 0.9919 | 0.9225 |
| s5_mobilenet_v3_large_v002_a | val | filo | 2 | 6290 | 0.9750 | — | 0.9919 | 0.9762 |

## Clase y filo por observación: F1 por taxón (resto en `taxonomic_per_class.csv`)

| Modelo | Partición | Nivel | Taxón | Géneros | Soporte | Precisión | Recall | F1 |
|---|---|---|---|---|---:|---:|---:|---:|
| ma_resnet18_v002 | val | clase | Agaricomycetes | Amanita, Auricularia, Ganoderma, Geastrum, Hericium, Lycoperdon, Pleurotus, Trametes | 3087 | 0.9971 | 0.9948 | 0.9959 |
| ma_resnet18_v002 | val | clase | Pezizomycetes | Morchella | 11 | 1.0000 | 1.0000 | 1.0000 |
| ma_resnet18_v002 | val | clase | Sordariomycetes | Xylaria | 354 | 0.9557 | 0.9746 | 0.9650 |
| ma_resnet18_v002 | val | filo | Ascomycota | Morchella, Xylaria | 365 | 0.9570 | 0.9753 | 0.9661 |
| ma_resnet18_v002 | val | filo | Basidiomycota | Amanita, Auricularia, Ganoderma, Geastrum, Hericium, Lycoperdon, Pleurotus, Trametes | 3087 | 0.9971 | 0.9948 | 0.9959 |
| ma_resnet18_v002 | test | clase | Agaricomycetes | Amanita, Auricularia, Ganoderma, Geastrum, Hericium, Lycoperdon, Pleurotus, Trametes | 3730 | 0.9965 | 0.9936 | 0.9950 |
| ma_resnet18_v002 | test | clase | Pezizomycetes | Morchella | 20 | 1.0000 | 0.9000 | 0.9474 |
| ma_resnet18_v002 | test | clase | Sordariomycetes | Xylaria | 289 | 0.9205 | 0.9619 | 0.9408 |
| ma_resnet18_v002 | test | filo | Ascomycota | Morchella, Xylaria | 309 | 0.9250 | 0.9579 | 0.9412 |
| ma_resnet18_v002 | test | filo | Basidiomycota | Amanita, Auricularia, Ganoderma, Geastrum, Hericium, Lycoperdon, Pleurotus, Trametes | 3730 | 0.9965 | 0.9936 | 0.9950 |
| s5_mobilenet_v3_large_v002_a | val | clase | Agaricomycetes | Amanita, Auricularia, Ganoderma, Geastrum, Hericium, Lycoperdon, Pleurotus, Trametes | 3087 | 0.9981 | 0.9984 | 0.9982 |
| s5_mobilenet_v3_large_v002_a | val | clase | Pezizomycetes | Morchella | 11 | 1.0000 | 0.9091 | 0.9524 |
| s5_mobilenet_v3_large_v002_a | val | clase | Sordariomycetes | Xylaria | 354 | 0.9859 | 0.9859 | 0.9859 |
| s5_mobilenet_v3_large_v002_a | val | filo | Ascomycota | Morchella, Xylaria | 365 | 0.9863 | 0.9836 | 0.9849 |
| s5_mobilenet_v3_large_v002_a | val | filo | Basidiomycota | Amanita, Auricularia, Ganoderma, Geastrum, Hericium, Lycoperdon, Pleurotus, Trametes | 3087 | 0.9981 | 0.9984 | 0.9982 |

## Especies con recall de género < 0,90 (≥ 5 observaciones)

| Modelo | Partición | Especie | Obs. | Recall género | Error más frecuente |
|---|---|---|---:|---:|---|
| ma_resnet18_v002 | test | *Auricularia mesenterica* | 13 | 0.615 | Trametes (4) |
| ma_resnet18_v002 | test | *Trametes cinnabarina* | 19 | 0.684 | Lycoperdon (3) |
| ma_resnet18_v002 | test | *Hericium cirrhatum* | 15 | 0.733 | Trametes (2) |
| ma_resnet18_v002 | test | *Amanita franchetii* | 11 | 0.818 | Lycoperdon (2) |
| ma_resnet18_v002 | test | *Lycoperdon mammiforme* | 6 | 0.833 | Amanita (1) |
| ma_resnet18_v002 | test | *Lycoperdon molle* | 6 | 0.833 | Pleurotus (1) |
| ma_resnet18_v002 | test | *Morchella esculenta* | 13 | 0.846 | Trametes (1) |
| ma_resnet18_v002 | test | *Geastrum melanocephalum* | 7 | 0.857 | Lycoperdon (1) |
| ma_resnet18_v002 | test | *Ganoderma lucidum* | 71 | 0.887 | Amanita (4) |
| ma_resnet18_v002 | test | *Ganoderma applanatum* | 159 | 0.893 | Trametes (10) |
| ma_resnet18_v002 | test | *Pleurotus dryinus* | 19 | 0.895 | Trametes (1) |
| ma_resnet18_v002 | val | *Auricularia mesenterica* | 7 | 0.429 | Xylaria (2) |
| ma_resnet18_v002 | val | *Hericium cirrhatum* | 17 | 0.765 | Trametes (3) |
| ma_resnet18_v002 | val | *Trametes cinnabarina* | 26 | 0.769 | Lycoperdon (3) |
| ma_resnet18_v002 | val | *Ganoderma resinaceum* | 14 | 0.857 | Pleurotus (1) |
| ma_resnet18_v002 | val | *Ganoderma applanatum* | 177 | 0.881 | Trametes (13) |
| s5_mobilenet_v3_large_v002_a | val | *Auricularia mesenterica* | 7 | 0.571 | Xylaria (1) |
| s5_mobilenet_v3_large_v002_a | val | *Geastrum striatum* | 7 | 0.857 | Amanita (1) |
| s5_mobilenet_v3_large_v002_a | val | *Hericium cirrhatum* | 17 | 0.882 | Trametes (2) |

## Evaluaciones incluidas

| Modelo | Partición | Origen | Observaciones | Imágenes | Reproduce métricas guardadas |
|---|---|---|---:|---:|---|
| ma_resnet18_v002 | val | development run | 3452 | 6290 | sí |
| ma_resnet18_v002 | test | development run | 4039 | 7159 | sí |
| s5_mobilenet_v3_large_v002_a | val | development run | 3452 | 6290 | sí |
