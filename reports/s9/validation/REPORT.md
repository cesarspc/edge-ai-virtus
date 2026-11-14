# S9: revisión de errores de validación

Las corridas marcadas legacy_environment usan hábitat, sustrato, elevación y fecha: no estiman el aporte de GPS ni de clima. Las corridas corregidas declaran geo o geo_climate. No se leyó test.

| run | feature_set | mode | observations | errors | macro_f1 | accuracy | multi_image_observations | discordant_observations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20261030T184438858210Z_s8_visual | legacy_environment | visual | 3452 | 69 | 0.9771 | 0.9800 | 1712 | 223 |
| 20261030T191206787347Z_s8_metadata | legacy_environment | metadata | 3452 | 1976 | 0.1890 | 0.4276 | 1712 | 0 |
| 20261030T193246267806Z_s8_multimodal | legacy_environment | multimodal | 3452 | 69 | 0.9771 | 0.9800 | 1712 | 223 |

La cola prioriza errores confiados, errores de clases con poco train y vistas discordantes. Una discrepancia entre vistas no demuestra error de etiqueta; la adjudicación humana sigue PENDING. La inspección visual preliminar de seis casos se documenta aparte en ../visual_review/REVIEW.md; la cola completa no está adjudicada.

## 20261030T184438858210Z_s8_visual

| genus | train_observations | support | recall | f1 |
| --- | --- | --- | --- | --- |
| Amanita | 3752 | 904 | 0.9856 | 0.9884 |
| Auricularia | 1443 | 288 | 0.9583 | 0.9735 |
| Ganoderma | 1832 | 335 | 0.9761 | 0.9575 |
| Geastrum | 779 | 79 | 0.9873 | 0.9811 |
| Hericium | 494 | 66 | 0.9091 | 0.9375 |
| Lycoperdon | 2166 | 408 | 0.9755 | 0.9767 |
| Morchella | 256 | 11 | 1.0000 | 1.0000 |
| Pleurotus | 1465 | 251 | 0.9880 | 0.9861 |
| Trametes | 3654 | 756 | 0.9828 | 0.9815 |
| Xylaria | 1958 | 354 | 0.9915 | 0.9887 |

Confusiones más frecuentes (por observación):

| true_genus | pred_genus | errors | fraction_of_true_class |
| --- | --- | --- | --- |
| Trametes | Ganoderma | 10 | 0.0132 |
| Lycoperdon | Amanita | 7 | 0.0172 |
| Amanita | Lycoperdon | 5 | 0.0055 |
| Amanita | Ganoderma | 4 | 0.0044 |
| Hericium | Trametes | 4 | 0.0606 |
| Ganoderma | Trametes | 3 | 0.0090 |
| Amanita | Trametes | 3 | 0.0033 |
| Auricularia | Ganoderma | 3 | 0.0104 |

## 20261030T191206787347Z_s8_metadata

| genus | train_observations | support | recall | f1 |
| --- | --- | --- | --- | --- |
| Amanita | 3752 | 904 | 0.9768 | 0.6972 |
| Auricularia | 1443 | 288 | 0.1181 | 0.1650 |
| Ganoderma | 1832 | 335 | 0.0269 | 0.0437 |
| Geastrum | 779 | 79 | 0.0886 | 0.1400 |
| Hericium | 494 | 66 | 0.0000 | 0.0000 |
| Lycoperdon | 2166 | 408 | 0.0490 | 0.0891 |
| Morchella | 256 | 11 | 0.0000 | 0.0000 |
| Pleurotus | 1465 | 251 | 0.0398 | 0.0673 |
| Trametes | 3654 | 756 | 0.5635 | 0.4931 |
| Xylaria | 1958 | 354 | 0.2458 | 0.1942 |

Confusiones más frecuentes (por observación):

| true_genus | pred_genus | errors | fraction_of_true_class |
| --- | --- | --- | --- |
| Lycoperdon | Amanita | 360 | 0.8824 |
| Trametes | Xylaria | 199 | 0.2632 |
| Xylaria | Trametes | 181 | 0.5113 |
| Auricularia | Trametes | 144 | 0.5000 |
| Ganoderma | Trametes | 111 | 0.3313 |
| Ganoderma | Amanita | 98 | 0.2925 |
| Pleurotus | Trametes | 96 | 0.3825 |
| Ganoderma | Xylaria | 88 | 0.2627 |

## 20261030T193246267806Z_s8_multimodal

| genus | train_observations | support | recall | f1 |
| --- | --- | --- | --- | --- |
| Amanita | 3752 | 904 | 0.9856 | 0.9884 |
| Auricularia | 1443 | 288 | 0.9583 | 0.9735 |
| Ganoderma | 1832 | 335 | 0.9761 | 0.9575 |
| Geastrum | 779 | 79 | 0.9873 | 0.9811 |
| Hericium | 494 | 66 | 0.9091 | 0.9375 |
| Lycoperdon | 2166 | 408 | 0.9755 | 0.9767 |
| Morchella | 256 | 11 | 1.0000 | 1.0000 |
| Pleurotus | 1465 | 251 | 0.9880 | 0.9861 |
| Trametes | 3654 | 756 | 0.9828 | 0.9815 |
| Xylaria | 1958 | 354 | 0.9915 | 0.9887 |

Confusiones más frecuentes (por observación):

| true_genus | pred_genus | errors | fraction_of_true_class |
| --- | --- | --- | --- |
| Trametes | Ganoderma | 10 | 0.0132 |
| Lycoperdon | Amanita | 7 | 0.0172 |
| Amanita | Lycoperdon | 5 | 0.0055 |
| Amanita | Ganoderma | 4 | 0.0044 |
| Hericium | Trametes | 4 | 0.0606 |
| Ganoderma | Trametes | 3 | 0.0090 |
| Amanita | Trametes | 3 | 0.0033 |
| Auricularia | Ganoderma | 3 | 0.0104 |
