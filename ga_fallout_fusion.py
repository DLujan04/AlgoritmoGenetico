
import os
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Definición de ítems: (nombre, peso, valor)
ITEMS = [
    ("Stimpak",          1,  10),
    ("Nuka-Cola",        3,  15),
    ("Rifle Laser",     10,  50),
    ("Servoarmadura",   25, 100),
    ("Med-X",            1,  12),
    ("Chatarra",         5,   5),
    ("Nucleo de Fusion", 8,  40),
    ("Caja de Municion", 4,  20),
]
PESO_MAXIMO = 30
N_GENES     = len(ITEMS)

# Parámetros del algoritmo genético
POP_SIZE = 50
PC       = 0.70   # probabilidad de crossover
PM       = 0.20   # probabilidad de mutación
PR       = 0.10   # probabilidad de reproducción
assert abs(PC + PM + PR - 1.0) < 1e-9, "PC + PM + PR debe ser igual a 1"

MAX_GEN         = 150
TORNEO_K        = 3     # participantes en torneo de selección
CONV_PATIENCE   = 30    # generaciones sin mejora para detener
SEED            = 42

# Función de aptitud con penalización por exceso de peso
def calcular_fitness(cromosoma: list[int]) -> float:
    peso_total  = sum(cromosoma[i] * ITEMS[i][1] for i in range(N_GENES))
    valor_total = sum(cromosoma[i] * ITEMS[i][2] for i in range(N_GENES))

    if peso_total > PESO_MAXIMO:
        exceso = peso_total - PESO_MAXIMO
        return max(0.0, valor_total - exceso * 8)
    return float(valor_total)

def peso_y_valor(cromosoma: list[int]) -> tuple[float, float]:
    """Retorna (peso, valor) sin penalización."""
    peso  = sum(cromosoma[i] * ITEMS[i][1] for i in range(N_GENES))
    valor = sum(cromosoma[i] * ITEMS[i][2] for i in range(N_GENES))
    return peso, valor

def make_population(size: int) -> list[list[int]]:
    return [[random.randint(0, 1) for _ in range(N_GENES)]
            for _ in range(size)]

# Selección por torneo: elige el mejor entre k candidatos aleatorios
def seleccion_torneo(poblacion: list, k: int = TORNEO_K) -> list[int]:
    participantes = random.sample(poblacion, k)
    return max(participantes, key=calcular_fitness)[:]

def crossover(padre1: list[int], padre2: list[int]) -> tuple[list[int], list[int]]:
    punto  = random.randint(1, N_GENES - 1)
    hijo1  = padre1[:punto] + padre2[punto:]
    hijo2  = padre2[:punto] + padre1[punto:]
    return hijo1, hijo2

PM_GEN = 0.15   # probabilidad de mutar cada gen individual

# Mutación bit-flip: invierte cada gen con probabilidad PM_GEN
def mutacion(cromosoma: list[int]) -> list[int]:
    return [gen ^ 1 if random.random() < PM_GEN else gen
            for gen in cromosoma]


def ejecutar_ag(verbose: bool = True) -> dict:
    random.seed(SEED)

    poblacion = make_population(POP_SIZE)
    if verbose:
        print(f"Poblacion inicial creada ({POP_SIZE} individuos, {N_GENES} genes)")

    mejor_historia  = []
    prom_historia   = []
    div_historia    = []

    for gen in range(MAX_GEN):

        fitnesses   = [calcular_fitness(ind) for ind in poblacion]
        mejor_fit   = max(fitnesses)
        prom_fit    = sum(fitnesses) / len(fitnesses)
        diversidad  = len(set(tuple(ind) for ind in poblacion))

        mejor_historia.append(mejor_fit)
        prom_historia.append(prom_fit)
        div_historia.append(diversidad)

        if verbose and gen % 25 == 0:
            print(f"  Gen {gen:3d} | Mejor: {mejor_fit:.1f} | "
                  f"Prom: {prom_fit:.1f} | Unicos: {diversidad}/{POP_SIZE}")

        # Criterio de convergencia: sin mejora en CONV_PATIENCE generaciones
        if gen >= CONV_PATIENCE:
            if mejor_historia[-1] == mejor_historia[-CONV_PATIENCE]:
                if verbose:
                    print(f"\n  Convergencia en generacion {gen}")
                break

        nueva_poblacion = []

        while len(nueva_poblacion) < POP_SIZE:
            r = random.random()

            if r < PR:
                # Reproducción: elitismo directo
                idx_mejor = fitnesses.index(max(fitnesses))
                nueva_poblacion.append(poblacion[idx_mejor][:])

            elif r < PR + PC:
                # Crossover seguido de mutación
                p1 = seleccion_torneo(poblacion)
                p2 = seleccion_torneo(poblacion)
                h1, h2 = crossover(p1, p2)
                h1 = mutacion(h1)
                h2 = mutacion(h2)
                nueva_poblacion.append(h1)
                if len(nueva_poblacion) < POP_SIZE:
                    nueva_poblacion.append(h2)

            else:
                # Mutación pura
                ind = seleccion_torneo(poblacion)
                nueva_poblacion.append(mutacion(ind))

        # Reemplazo generacional completo
        poblacion = nueva_poblacion[:POP_SIZE]

    # Evaluación final
    fitnesses  = [calcular_fitness(ind) for ind in poblacion]
    idx_mejor  = fitnesses.index(max(fitnesses))
    mejor_ind  = poblacion[idx_mejor]

    return {
        "mejor":         mejor_ind,
        "mejor_historia": mejor_historia,
        "prom_historia":  prom_historia,
        "div_historia":   div_historia,
        "generaciones":   len(mejor_historia),
    }

def mostrar_resultado(resultado: dict) -> None:
    mejor = resultado["mejor"]
    peso, valor = peso_y_valor(mejor)

    print("\n" + "=" * 58)
    print("   RESULTADO FINAL — INVENTARIO DE FALLOUT")
    print("=" * 58)
    print(f"\n{'ITEM':<22} {'PESO':>6} {'VALOR':>7}  INCLUIDO")
    print("-" * 50)
    for i, (nombre, peso_i, valor_i) in enumerate(ITEMS):
        marca = "  [SI]" if mejor[i] == 1 else "  [NO]"
        print(f"{nombre:<22} {peso_i:>6} {valor_i:>7}{marca}")
    print("-" * 50)
    print(f"{'TOTAL':<22} {peso:>6.1f} {valor:>7.0f}")
    print(f"\nCapacidad usada : {peso:.1f} / {PESO_MAXIMO} kg")
    print(f"Fitness final   : {calcular_fitness(mejor):.2f}")
    print(f"Generaciones    : {resultado['generaciones']}")
    cromosoma_str = "".join(str(g) for g in mejor)
    print(f"Cromosoma       : [{cromosoma_str}]")
    print("=" * 58)

# Generación de gráficas: evolución del fitness, diversidad e inventario final
def graficar(resultado: dict, path_out: str = None):
    if path_out is None:
        path_out = os.path.join(_SCRIPT_DIR, "ga_fusion.png")
    mejor   = resultado["mejor"]
    bh      = resultado["mejor_historia"]
    ph      = resultado["prom_historia"]
    dh      = resultado["div_historia"]
    gens    = range(len(bh))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Algoritmo Genetico Fusionado — Mochila de Fallout",
                 fontsize=14, fontweight="bold", color="#1a1a2e")
    fig.patch.set_facecolor("#f0f0f0")

    # Panel 1: Evolución del fitness
    axes[0].set_facecolor("#1a1a2e")
    axes[0].plot(gens, bh, color="#00ff41", linewidth=2, label="Mejor fitness")
    axes[0].plot(gens, ph, color="#4ecdc4", linewidth=1.5,
                 linestyle="--", label="Promedio")
    axes[0].fill_between(gens, ph, bh, alpha=0.15, color="#00ff41")
    axes[0].set_title("Evolucion del Fitness", color="white")
    axes[0].set_xlabel("Generacion", color="white")
    axes[0].set_ylabel("Fitness", color="white")
    axes[0].tick_params(colors="white")
    axes[0].legend(facecolor="#2d2d2d", labelcolor="white")
    axes[0].grid(True, alpha=0.2, color="white")
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#00ff41")

    # Panel 2: Diversidad de la población
    axes[1].set_facecolor("#1a1a2e")
    axes[1].plot(gens, dh, color="#ffd700", linewidth=2)
    axes[1].axhline(y=POP_SIZE, color="#ff6b6b", linestyle=":",
                    label=f"Max ({POP_SIZE})")
    axes[1].set_title("Diversidad de la Poblacion", color="white")
    axes[1].set_xlabel("Generacion", color="white")
    axes[1].set_ylabel("Individuos unicos", color="white")
    axes[1].tick_params(colors="white")
    axes[1].legend(facecolor="#2d2d2d", labelcolor="white")
    axes[1].grid(True, alpha=0.2, color="white")
    for spine in axes[1].spines.values():
        spine.set_edgecolor("#ffd700")

    # Panel 3: Inventario de la solución final
    axes[2].set_facecolor("#1a1a2e")
    nombres = [it[0] for it in ITEMS]
    valores = [it[2] for it in ITEMS]
    colors  = ["#00ff41" if mejor[i] == 1 else "#333333"
               for i in range(N_GENES)]

    bars = axes[2].barh(nombres, valores, color=colors,
                        edgecolor="#555555", height=0.6)
    for bar, it, sel in zip(bars, ITEMS, mejor):
        if sel == 1:
            axes[2].text(bar.get_width() + 0.5,
                         bar.get_y() + bar.get_height() / 2,
                         f"{it[1]}kg", va="center", fontsize=8,
                         color="#00ff41")

    axes[2].set_title("Inventario Final", color="white")
    axes[2].set_xlabel("Valor", color="white")
    axes[2].tick_params(colors="white")
    verde = mpatches.Patch(color="#00ff41", label="Seleccionado")
    gris  = mpatches.Patch(color="#333333", label="No incluido",
                           edgecolor="#555555")
    axes[2].legend(handles=[verde, gris], facecolor="#2d2d2d",
                   labelcolor="white", fontsize=8)
    axes[2].grid(True, axis="x", alpha=0.2, color="white")
    for spine in axes[2].spines.values():
        spine.set_edgecolor("#ffd700")

    plt.tight_layout()
    plt.savefig(path_out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Grafica guardada: {path_out}")
    return path_out

if __name__ == "__main__":
    print("=" * 58)
    print("  ALGORITMO GENETICO FUSIONADO — FALLOUT KNAPSACK")
    print(f"  POP={POP_SIZE} | PC={PC} | PM={PM} | PR={PR}")
    print(f"  PC+PM+PR = {PC+PM+PR:.2f} | Peso max: {PESO_MAXIMO}kg")
    print("=" * 58)

    resultado = ejecutar_ag(verbose=True)
    mostrar_resultado(resultado)
    graficar(resultado)
