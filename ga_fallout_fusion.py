import sys, random

# Items: (nombre, peso, valor)
ITEMS = [
    ("Stimpak",          1,  10),
    ("Nuka-Cola",        3,  15),
    ("Rifle Laser",     10,  50),
    ("Servoarmadura",   25, 100),
    ("Med-X",            1,  12),
    ("Chatarra",         5,   5),
    ("Nucleo de Fusion",  8,  40),
    ("Caja de Municion",  4,  20),
]
PESO_MAXIMO = 30
N_GENES     = len(ITEMS)

# Parametros del AG
POP_SIZE      = 50
PC, PM, PR    = 0.70, 0.20, 0.10   # crossover, mutacion, reproduccion
assert abs(PC + PM + PR - 1.0) < 1e-9, "PC + PM + PR debe ser 1"
MAX_GEN       = 150
TORNEO_K      = 3
CONV_PATIENCE = 30
PM_GEN        = 0.15   # prob. de mutar cada gen
SEED          = 42


def calcular_fitness(c):
    peso  = sum(c[i] * ITEMS[i][1] for i in range(N_GENES))
    valor = sum(c[i] * ITEMS[i][2] for i in range(N_GENES))
    return max(0.0, valor - max(0, peso - PESO_MAXIMO) * 8)


def peso_y_valor(c):
    return (sum(c[i] * ITEMS[i][1] for i in range(N_GENES)),
            sum(c[i] * ITEMS[i][2] for i in range(N_GENES)))


def make_population(size):
    return [[random.randint(0, 1) for _ in range(N_GENES)] for _ in range(size)]


def seleccion_torneo(pop, k=TORNEO_K):
    return max(random.sample(pop, k), key=calcular_fitness)[:]


def crossover(p1, p2):
    pt = random.randint(1, N_GENES - 1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]


def mutacion(c):
    return [g ^ 1 if random.random() < PM_GEN else g for g in c]


def ejecutar_ag(verbose=True):
    random.seed(SEED)
    pop = make_population(POP_SIZE)
    mejor_h, prom_h, div_h = [], [], []

    for gen in range(MAX_GEN):
        fits = [calcular_fitness(ind) for ind in pop]
        mejor_h.append(max(fits))
        prom_h.append(sum(fits) / len(fits))
        div_h.append(len(set(tuple(ind) for ind in pop)))

        if verbose and gen % 25 == 0:
            print(f"  Gen {gen:3d} | Mejor: {mejor_h[-1]:.1f} | Prom: {prom_h[-1]:.1f} | Unicos: {div_h[-1]}/{POP_SIZE}")

        if gen >= CONV_PATIENCE and mejor_h[-1] == mejor_h[-CONV_PATIENCE]:
            if verbose: print(f"\n  Convergencia en generacion {gen}")
            break

        nueva = []
        while len(nueva) < POP_SIZE:
            r = random.random()
            if r < PR:
                nueva.append(pop[fits.index(max(fits))][:])
            elif r < PR + PC:
                h1, h2 = crossover(seleccion_torneo(pop), seleccion_torneo(pop))
                nueva += [mutacion(h1), mutacion(h2)]
            else:
                nueva.append(mutacion(seleccion_torneo(pop)))
        pop = nueva[:POP_SIZE]

    fits  = [calcular_fitness(ind) for ind in pop]
    mejor = pop[fits.index(max(fits))]
    return {"mejor": mejor, "mejor_historia": mejor_h, "prom_historia": prom_h,
            "div_historia": div_h, "generaciones": len(mejor_h)}


def mostrar_resultado(res):
    mejor = res["mejor"]
    peso, valor = peso_y_valor(mejor)
    print("\n" + "=" * 52)
    print("   RESULTADO FINAL - INVENTARIO DE FALLOUT")
    print("=" * 52)
    print(f"{'ITEM':<22} {'PESO':>5} {'VALOR':>6}  INC")
    print("-" * 44)
    for i, (n, w, v) in enumerate(ITEMS):
        print(f"{n:<22} {w:>5} {v:>6}  {'[SI]' if mejor[i] else '[NO]'}")
    print("-" * 44)
    print(f"{'TOTAL':<22} {peso:>5.0f} {valor:>6.0f}")
    print(f"\nCapacidad : {peso:.0f}/{PESO_MAXIMO} kg  |  Fitness: {calcular_fitness(mejor):.2f}")
    print(f"Cromosoma : [{''.join(str(g) for g in mejor)}]  |  Generaciones: {res['generaciones']}")
    print("=" * 52)


# ── PRUEBAS ────────────────────────────────────────────────
def correr_pruebas():
    ok = fail = 0

    def prueba(nombre, cond, detalle=""):
        nonlocal ok, fail
        if cond: ok += 1
        else:    fail += 1
        det = f" ({detalle})" if detalle else ""
        print(f"  [{'PASS' if cond else 'FAIL'}] {nombre}{det}")

    def sec(t): print(f"\n--- {t} ---")

    # 1. Fitness
    sec("1. calcular_fitness")
    prueba("Vacio -> 0",           calcular_fitness([0]*N_GENES) == 0.0)
    prueba("Solo Stimpak -> 10",   calcular_fitness([1,0,0,0,0,0,0,0]) == 10.0)
    prueba("Solo Servo -> 100",    calcular_fitness([0,0,0,1,0,0,0,0]) == 100.0)
    p, v = peso_y_valor([1]*N_GENES)
    prueba("Todos -> penalizacion", calcular_fitness([1]*N_GENES) == max(0.0, v - max(0, p-PESO_MAXIMO)*8))
    p_l, v_l = peso_y_valor([1,0,0,1,0,0,0,1])   # 30 kg exactos
    prueba("30 kg exactos -> sin penalizacion",
           calcular_fitness([1,0,0,1,0,0,0,1]) == v_l and p_l == PESO_MAXIMO)

    # 2. Poblacion
    sec("2. make_population")
    random.seed(0); pop = make_population(50)
    prueba("Tamano 50",   len(pop) == 50)
    prueba("N_GENES genes por individuo", all(len(ind) == N_GENES for ind in pop))
    prueba("Solo bits 0/1", all(g in (0,1) for ind in pop for g in ind))
    prueba("Hay variedad", len(set(tuple(i) for i in pop)) > 1)

    # 3. Seleccion por torneo
    sec("3. seleccion_torneo")
    random.seed(1); pop_t = make_population(20)
    sel = seleccion_torneo(pop_t, k=3)
    prueba("Devuelve lista de N_GENES", isinstance(sel, list) and len(sel) == N_GENES)
    prueba("Solo bits validos", all(g in (0,1) for g in sel))
    pop_m = [[1,1,1,0,0,0,0,0]]*25 + [[0,0,0,0,0,0,0,0]]*25
    p_t = sum(calcular_fitness(seleccion_torneo(pop_m, 3)) for _ in range(200)) / 200
    p_a = sum(calcular_fitness(random.choice(pop_m))       for _ in range(200)) / 200
    prueba("Torneo favorece mejor fitness", p_t > p_a, f"torneo={p_t:.1f} azar={p_a:.1f}")

    # 4. Crossover
    sec("4. crossover")
    random.seed(2); p1, p2 = [1]*N_GENES, [0]*N_GENES
    puntos = set()
    for _ in range(200):
        h1, h2 = crossover(p1, p2)
        for i in range(1, N_GENES):
            if h1[i] != h1[i-1]: puntos.add(i); break
    prueba("Hijos tienen N_GENES genes", len(h1) == N_GENES and len(h2) == N_GENES)
    prueba("Punto de corte aleatorio",   len(puntos) > 3, f"puntos={sorted(puntos)}")
    prueba("Padres no se modifican",     p1 == [1]*N_GENES and p2 == [0]*N_GENES)

    # 5. Mutacion
    sec("5. mutacion")
    random.seed(3); orig = [1,0,1,0,1,0,1,0]
    cambios = sum(1 for _ in range(500) if mutacion(orig[:]) != orig)
    prueba("Muta en >50% de casos",  cambios > 250, f"{cambios}/500")
    prueba("Longitud conservada",    len(mutacion(orig)) == N_GENES)
    prueba("Solo bits 0/1",         all(g in (0,1) for g in mutacion(orig)))
    prueba("Original no se altera", orig == [1,0,1,0,1,0,1,0])

    # 6. Probabilidades
    sec("6. PC + PM + PR = 1")
    prueba("Suma == 1.0", abs(PC+PM+PR-1.0) < 1e-9, f"{PC}+{PM}+{PR}={PC+PM+PR:.4f}")
    prueba("PC es el mayor",  PC > PM and PC > PR)

    # 7. Integracion
    sec("7. ejecutar_ag (integracion)")
    print("  Ejecutando AG...")
    res = ejecutar_ag(verbose=False)
    mej = res["mejor"]; bh = res["mejor_historia"]; ph = res["prom_historia"]
    dh = res["div_historia"]; gens = res["generaciones"]
    p_f, _ = peso_y_valor(mej)
    prueba("Devuelve resultado",         "mejor" in res)
    prueba("Cromosoma N_GENES genes",    len(mej) == N_GENES)
    prueba("Cromosoma binario",          all(g in (0,1) for g in mej))
    prueba("Peso <= maximo",             p_f <= PESO_MAXIMO, f"{p_f}<={PESO_MAXIMO}")
    prueba("Fitness mejora o mantiene",  bh[-1] >= bh[0], f"{bh[0]:.1f}->{bh[-1]:.1f}")
    prueba("Promedio <= mejor siempre",  all(ph[i] <= bh[i]+1e-9 for i in range(len(bh))))
    prueba("Diversidad inicial alta",    dh[0] > 10, f"dh[0]={dh[0]}")
    prueba("Converge antes del maximo",  gens < MAX_GEN, f"gen={gens}")
    prueba("Fitness final >= 100",       calcular_fitness(mej) >= 100, f"fit={calcular_fitness(mej):.1f}")

    # 8. Casos borde
    sec("8. Casos borde")
    prueba("Servo sola -> fit 100",         calcular_fitness([0,0,0,1,0,0,0,0]) == 100.0)
    prueba("Servo+Rifle (35kg) penalizado", calcular_fitness([0,0,1,1,0,0,0,0]) < 150)
    try:
        s = seleccion_torneo([[1,0,0,0,0,0,0,0]], k=1)
        prueba("Torneo k=1 no lanza error",   len(s) == N_GENES)
    except Exception as e:
        prueba("Torneo k=1 no lanza error", False, str(e))
    h1i, h2i = crossover([1,0]*4, [1,0]*4)
    prueba("Crossover padres iguales -> hijos iguales", h1i == [1,0]*4 and h2i == [1,0]*4)

    # Resumen
    total = ok + fail
    print(f"\n{'='*46}")
    print(f"  RESUMEN: {ok}/{total} pruebas pasaron", end="")
    print(f"  ({fail} fallidas)" if fail else "")
    print(f"{'='*46}")
    sys.exit(0 if fail == 0 else 1)


# ── MAIN ───────────────────────────────────────────────────
if __name__ == "__main__":
    if "--test" in sys.argv:
        correr_pruebas()
    else:
        print(f"AG Fallout Knapsack - POP={POP_SIZE} PC={PC} PM={PM} PR={PR}")
        mostrar_resultado(ejecutar_ag(verbose=True))
