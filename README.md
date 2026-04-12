# Algoritmo Genético — Problema de la Mochila (Fallout)

Implementación de un algoritmo genético fusionado para resolver el problema de la mochila, ambientado en el universo de Fallout. El objetivo es seleccionar los ítems de mayor valor sin superar la capacidad máxima de carga.

## Archivos

| Archivo | Descripción |
|---|---|
| `ga_fallout_fusion.py` | Implementación principal del algoritmo genético |
| `tests_ga.py` | Suite de pruebas: unitarias, integración y casos borde |
| `ga_fusion.png` | Gráfica generada al ejecutar el algoritmo (se crea automáticamente) |

## Requisitos

- Python 3.10 o superior
- matplotlib

### Instalación de dependencias

```bash
pip install matplotlib
```

## Ejecución

### Ejecutar el algoritmo genético

```bash
python ga_fallout_fusion.py
```

Imprime el progreso por generación, el inventario final seleccionado y guarda la gráfica `ga_fusion.png` en la misma carpeta.

### Ejecutar las pruebas

```bash
python tests_ga.py
```

Corre 23 pruebas divididas en 8 secciones y muestra un resumen `PASS/FAIL` al final.

## Parámetros del algoritmo

| Parámetro | Valor | Descripción |
|---|---|---|
| `POP_SIZE` | 50 | Tamaño de la población |
| `PC` | 0.70 | Probabilidad de crossover |
| `PM` | 0.20 | Probabilidad de mutación |
| `PR` | 0.10 | Probabilidad de reproducción (elitismo) |
| `MAX_GEN` | 150 | Máximo de generaciones |
| `TORNEO_K` | 3 | Participantes por torneo de selección |
| `CONV_PATIENCE` | 30 | Generaciones sin mejora para detener |
| `PESO_MAXIMO` | 30 | Capacidad máxima de la mochila (kg) |

## Operadores genéticos

- **Selección:** Torneo de tamaño k=3
- **Crossover:** Punto de corte aleatorio (single-point)
- **Mutación:** Bit-flip por gen con probabilidad `PM_GEN = 0.15`
- **Reemplazo:** Generacional completo
