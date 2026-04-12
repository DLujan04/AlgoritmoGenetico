# Pruebas del algoritmo genético: unitarias, integración y casos borde

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ga_fallout_fusion import (
    ITEMS, N_GENES, PESO_MAXIMO, PC, PM, PR,
    calcular_fitness, peso_y_valor, make_population,
    seleccion_torneo, crossover, mutacion, ejecutar_ag,
)

# Utilidad de reporte de resultados
resultados = []

def prueba(nombre: str, condicion: bool, detalle: str = "") -> None:
    estado = "PASS" if condicion else "FAIL"
    resultados.append((nombre, estado, detalle))
    marca = "✓" if condicion else "✗"
    print(f"  [{estado}] {marca} {nombre}")
    if detalle:
        print(f"          {detalle}")


def seccion(titulo: str) -> None:
    print(f"\n{'-'*60}")
    print(f"  {titulo}")
    print(f"{'-'*60}")


# Sección 1: calcular_fitness
seccion("1. FITNESS — calcular_fitness()")

crom_vacio = [0] * N_GENES
prueba("Cromosoma vacío → fitness 0",
       calcular_fitness(crom_vacio) == 0.0,
       f"fitness={calcular_fitness(crom_vacio)}")

crom_stimpak = [1, 0, 0, 0, 0, 0, 0, 0]
prueba("Solo Stimpak → fitness 10",
       calcular_fitness(crom_stimpak) == 10.0,
       f"fitness={calcular_fitness(crom_stimpak)}")

crom_servo = [0, 0, 0, 1, 0, 0, 0, 0]
prueba("Solo Servoarmadura (25kg/30max) → fitness 100",
       calcular_fitness(crom_servo) == 100.0,
       f"fitness={calcular_fitness(crom_servo)}")

crom_todo = [1] * N_GENES
peso_total, valor_total = peso_y_valor(crom_todo)
fit_todo = calcular_fitness(crom_todo)
exceso = max(0, peso_total - PESO_MAXIMO)
penalizado = max(0.0, valor_total - exceso * 8)
prueba("Todos los items (exceso de peso) → penalización aplicada",
       fit_todo == penalizado,
       f"peso={peso_total}, valor={valor_total}, fit={fit_todo:.1f}, esperado={penalizado:.1f}")

# Servoarmadura(25) + Caja de Municion(4) + Stimpak(1) = 30kg
crom_limite = [1, 0, 0, 1, 0, 0, 0, 1]
p_lim, v_lim = peso_y_valor(crom_limite)
prueba("Peso exactamente en el límite (30kg) → sin penalización",
       calcular_fitness(crom_limite) == v_lim and p_lim == PESO_MAXIMO,
       f"peso={p_lim}, valor={v_lim}, fitness={calcular_fitness(crom_limite)}")


# Sección 2: make_population
seccion("2. POBLACIÓN — make_population()")

random.seed(0)
pop = make_population(50)

prueba("Tamaño de población correcto",
       len(pop) == 50,
       f"len={len(pop)}")

prueba("Cada cromosoma tiene N_GENES genes",
       all(len(ind) == N_GENES for ind in pop),
       f"N_GENES={N_GENES}")

prueba("Solo bits 0 y 1 en cromosomas",
       all(g in (0, 1) for ind in pop for g in ind),
       "todos los genes son binarios")

prueba("Población tiene variedad (no todos iguales)",
       len(set(tuple(ind) for ind in pop)) > 1,
       f"individuos únicos={len(set(tuple(ind) for ind in pop))}")


# Sección 3: seleccion_torneo
seccion("3. SELECCIÓN — seleccion_torneo()")

random.seed(1)
pop_test = make_population(20)
seleccionado = seleccion_torneo(pop_test, k=3)

prueba("Seleccionado es lista de longitud N_GENES",
       isinstance(seleccionado, list) and len(seleccionado) == N_GENES,
       f"tipo={type(seleccionado)}, len={len(seleccionado)}")

prueba("Seleccionado solo contiene bits válidos",
       all(g in (0, 1) for g in seleccionado),
       "todos los genes son 0 o 1")

# Valida que el torneo favorece individuos con mayor fitness vs selección aleatoria
pop_mitad = [[1,1,1,0,0,0,0,0]] * 25 + [[0,0,0,0,0,0,0,0]] * 25
fit_torneo = [calcular_fitness(seleccion_torneo(pop_mitad, k=3))
              for _ in range(200)]
fit_azar   = [calcular_fitness(random.choice(pop_mitad))
              for _ in range(200)]
prom_torneo = sum(fit_torneo) / 200
prom_azar   = sum(fit_azar)   / 200
prueba("Torneo favorece individuos con mejor fitness (prom > selección aleatoria)",
       prom_torneo > prom_azar,
       f"prom torneo={prom_torneo:.1f} vs prom azar={prom_azar:.1f}")


# Sección 4: crossover
seccion("4. CROSSOVER — crossover()")

random.seed(2)
p1 = [1, 1, 1, 1, 1, 1, 1, 1]
p2 = [0, 0, 0, 0, 0, 0, 0, 0]

puntos_usados = set()
for _ in range(200):
    h1, h2 = crossover(p1, p2)
    for i in range(1, N_GENES):
        if h1[i] != h1[i-1]:
            puntos_usados.add(i)
            break

prueba("Hijos tienen longitud N_GENES",
       len(h1) == N_GENES and len(h2) == N_GENES,
       f"h1={h1}, h2={h2}")

prueba("h1 y h2 son complementarios (single-point con p1 vs p2 all-ones/all-zeros)",
       h1[i] != h2[i] if (h1 != p1 and h1 != p2) else True,
       "los hijos intercambian segmentos correctamente")

prueba("Punto de corte es aleatorio (varios puntos observados en 200 cruzas)",
       len(puntos_usados) > 3,
       f"puntos distintos usados: {sorted(puntos_usados)}")

prueba("Padres no se modifican (inmutabilidad)",
       p1 == [1]*N_GENES and p2 == [0]*N_GENES,
       "padres originales intactos")


# Sección 5: mutacion
seccion("5. MUTACION — mutacion()")

random.seed(3)
crom_original = [1, 0, 1, 0, 1, 0, 1, 0]

mutaciones_detectadas = 0
for _ in range(500):
    mutado = mutacion(crom_original[:])
    if mutado != crom_original:
        mutaciones_detectadas += 1

prueba("La mutación cambia genes en >50% de las ejecuciones (PM_GEN=0.15)",
       mutaciones_detectadas > 250,
       f"mutaciones en 500 intentos: {mutaciones_detectadas}")

prueba("El cromosoma mutado tiene la misma longitud",
       len(mutacion(crom_original)) == N_GENES,
       f"N_GENES={N_GENES}")

prueba("La mutación solo produce bits válidos",
       all(g in (0, 1) for g in mutacion(crom_original)),
       "todos bits 0 o 1")

prueba("El original no se modifica (inmutabilidad)",
       crom_original == [1, 0, 1, 0, 1, 0, 1, 0],
       f"original={crom_original}")


# Sección 6: restricción de probabilidades
seccion("6. RESTRICCION — PC + PM + PR = 1")

suma = PC + PM + PR
prueba("PC + PM + PR == 1.0",
       abs(suma - 1.0) < 1e-9,
       f"PC={PC}, PM={PM}, PR={PR}, suma={suma:.4f}")

prueba("PC es la operación más frecuente",
       PC > PM and PC > PR,
       f"PC={PC} > PM={PM} y PR={PR}")


# Sección 7: integración del AG completo
seccion("7. INTEGRACION — ejecutar_ag()")

print("  Ejecutando AG (puede tardar unos segundos)...")
resultado = ejecutar_ag(verbose=False)

mejor   = resultado["mejor"]
bh      = resultado["mejor_historia"]
ph      = resultado["prom_historia"]
dh      = resultado["div_historia"]
gens    = resultado["generaciones"]

prueba("El AG termina y devuelve resultado",
       resultado is not None and "mejor" in resultado,
       "resultado contiene clave 'mejor'")

prueba("El cromosoma final tiene N_GENES genes",
       len(mejor) == N_GENES,
       f"len={len(mejor)}")

prueba("El cromosoma es binario",
       all(g in (0, 1) for g in mejor),
       f"cromosoma={mejor}")

p_final, v_final = peso_y_valor(mejor)
prueba("La solución final respeta el peso máximo",
       p_final <= PESO_MAXIMO,
       f"peso={p_final} <= {PESO_MAXIMO}")

# El AG no garantiza elitismo absoluto (PR=10%), por lo que se verifica
# que el fitness final sea >= al fitness inicial
prueba("El fitness final es mejor o igual que el inicial",
       bh[-1] >= bh[0],
       f"fitness inicial={bh[0]:.1f}, fitness final={bh[-1]:.1f}")

prueba("El fitness promedio siempre <= mejor fitness",
       all(ph[i] <= bh[i] + 1e-9 for i in range(len(bh))),
       "promedio <= mejor en todas las generaciones")

prueba("La diversidad inicia alta (>10 individuos únicos en gen 0)",
       dh[0] > 10,
       f"diversidad gen 0 = {dh[0]}")

prueba("El AG converge antes del máximo de generaciones",
       gens < 150,
       f"convergió en generación {gens}")

fit_final = calcular_fitness(mejor)
prueba("La solución tiene un fitness razonable (>80% del óptimo conocido)",
       fit_final >= 100,
       f"fitness final = {fit_final:.1f}")


# Sección 8: casos borde
seccion("8. CASOS BORDE")

crom_servo = [0, 0, 0, 1, 0, 0, 0, 0]
prueba("Servoarmadura sola (25kg): válida y fitness correcto",
       calcular_fitness(crom_servo) == 100.0 and peso_y_valor(crom_servo)[0] == 25,
       f"peso=25, fitness={calcular_fitness(crom_servo)}")

crom_srv_rifle = [0, 0, 1, 1, 0, 0, 0, 0]
fit_sr = calcular_fitness(crom_srv_rifle)
prueba("Servoarmadura+Rifle(35kg) excede → fitness penalizado",
       fit_sr < 150,
       f"fitness={fit_sr:.1f} (valor bruto sería 150)")

pop_1 = [[1, 0, 0, 0, 0, 0, 0, 0]]
try:
    sel = seleccion_torneo(pop_1, k=1)
    prueba("Torneo con k=1 no lanza excepción",
           len(sel) == N_GENES,
           f"seleccionado={sel}")
except Exception as e:
    prueba("Torneo con k=1 no lanza excepción", False, str(e))

p_igual = [1, 0, 1, 0, 1, 0, 1, 0]
h1_ig, h2_ig = crossover(p_igual[:], p_igual[:])
prueba("Crossover con padres idénticos → hijos idénticos",
       h1_ig == p_igual and h2_ig == p_igual,
       f"h1={h1_ig}")


# Resumen final de pruebas
total  = len(resultados)
passed = sum(1 for _, e, _ in resultados if e == "PASS")
failed = total - passed

print(f"\n{'='*60}")
print(f"  RESUMEN: {passed}/{total} pruebas pasaron")
if failed:
    print(f"  FALLIDAS ({failed}):")
    for nombre, estado, det in resultados:
        if estado == "FAIL":
            print(f"    ✗ {nombre}")
            if det:
                print(f"      {det}")
print(f"{'='*60}")

sys.exit(0 if failed == 0 else 1)
