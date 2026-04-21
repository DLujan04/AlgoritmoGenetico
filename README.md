# Algoritmo Genetico — Problema de la Mochila (Fallout)

Implementacion de un algoritmo genetico para resolver el problema de la mochila, ambientado en el universo de Fallout. Selecciona los items de mayor valor sin superar la capacidad maxima de carga.

## Archivo

| Archivo | Descripcion |
|---|---|
| `ga_fallout_fusion.py` | Algoritmo genetico completo + pruebas integradas |

## Requisitos

- Python 3.10+
- Sin dependencias externas

## Ejecucion

```bash
# Ejecutar el algoritmo
python ga_fallout_fusion.py

# Ejecutar las pruebas (8 secciones, ~25 casos)
python ga_fallout_fusion.py --test
```

## Parametros

| Parametro | Valor | Descripcion |
|---|---|---|
| `POP_SIZE` | 50 | Tamano de la poblacion |
| `PC` | 0.70 | Probabilidad de crossover |
| `PM` | 0.20 | Probabilidad de mutacion |
| `PR` | 0.10 | Probabilidad de reproduccion (elitismo) |
| `MAX_GEN` | 150 | Maximo de generaciones |
| `TORNEO_K` | 3 | Participantes por torneo |
| `CONV_PATIENCE` | 30 | Generaciones sin mejora para detener |
| `PESO_MAXIMO` | 30 | Capacidad maxima (kg) |

## Operadores geneticos

- **Seleccion:** Torneo de tamano k=3
- **Crossover:** Single-point con punto aleatorio
- **Mutacion:** Bit-flip por gen (PM_GEN = 0.15)
- **Reemplazo:** Generacional completo con elitismo (PR=10%)
