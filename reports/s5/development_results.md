# S5: corridas completas de desarrollo

Selección: macro-F1 por observación de VAL.

| Ensayo | LR | Pesos | F1 obs. val | F1 imagen val | Corrida |
|---|---:|---|---:|---:|---|
| c | 0.0003 | inverse_sqrt | 0.9717 | 0.9188 | runs\20261024T031650621510Z_s5_mobilenet_v3_large_v002_c |
| a | 0.0003 | none | 0.9706 | 0.9254 | runs\20261012T161041257797Z_s5_mobilenet_v3_large_v002_a |
| b | 0.0001 | none | 0.9698 | 0.9152 | runs\20261024T022849507889Z_s5_mobilenet_v3_large_v002_b |
| d | 0.0001 | inverse_sqrt | 0.9617 | 0.9134 | runs\20261024T132723373718Z_s5_mobilenet_v3_large_v002_d |

## Corridas excluidas de la comparación

- runs\20261023T005149730394Z_s5_mobilenet_v3_large_v002_b: D:\Dev\Virtus\edge-ai-virtus\runs\20261023T005149730394Z_s5_mobilenet_v3_large_v002_b\predictions_val.csv: 6272/6290 rows contain non-finite probabilities
- runs\20261023T020544245443Z_s5_mobilenet_v3_large_v002_c: D:\Dev\Virtus\edge-ai-virtus\runs\20261023T020544245443Z_s5_mobilenet_v3_large_v002_c\predictions_val.csv: 6272/6290 rows contain non-finite probabilities
- runs\20261024T015337199324Z_s5_mobilenet_v3_large_v002_d: D:\Dev\Virtus\edge-ai-virtus\runs\20261024T015337199324Z_s5_mobilenet_v3_large_v002_d\predictions_val.csv: 6272/6290 rows contain non-finite probabilities
