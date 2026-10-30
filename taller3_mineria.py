# -*- coding: utf-8 -*-
"""Taller 3, Minería de datos: análisis completo de las 12 compras del caso.

Sigue el orden de la guía:
  Actividad 1. Preparación de los datos
  Actividad 2. Análisis exploratorio (las cinco preguntas)
  Actividad 3. Clasificación por rangos de valor y agrupamiento con K-means
  Fase 3. Gráfico de barras y gráfico de dispersión

Usa pandas, matplotlib y scikit-learn, que ya vienen instalados en Google Colab.

    python3 taller3_mineria.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Los datos tal como vienen en la guía, con el valor escrito como texto.
DATOS = [
    ("C001", 22, "Bogotá", "Tecnología", "$250.000", 2),
    ("C002", 35, "Medellín", "Ropa", "$120.000", 4),
    ("C003", 28, "Bogotá", "Tecnología", "$320.000", 3),
    ("C004", 42, "Cali", "Hogar", "$180.000", 2),
    ("C005", 31, "Medellín", "Ropa", "$150.000", 5),
    ("C006", 24, "Bogotá", "Hogar", "$90.000", 1),
    ("C007", 45, "Cali", "Tecnología", "$450.000", 3),
    ("C008", 29, "Bogotá", "Ropa", "$110.000", 4),
    ("C009", 38, "Medellín", "Hogar", "$210.000", 2),
    ("C010", 26, "Cali", "Tecnología", "$280.000", 3),
    ("C011", 33, "Bogotá", "Ropa", "$160.000", 5),
    ("C012", 41, "Medellín", "Tecnología", "$390.000", 2),
]
COLUMNAS = ["id", "edad", "ciudad", "producto", "valor_texto", "compras_mes"]

SALIDA = Path("figuras")
TINTA, TINTA_SUAVE, REJILLA, AZUL = "#1a1a19", "#555550", "#e4e3df", "#2a78d6"
# Un color y una forma por grupo, porque el color solo no basta para todos.
GRUPOS = {
    1: ("#2a78d6", "o", "Grupo 1: ropa frecuente"),
    2: ("#eb6834", "s", "Grupo 2: jóvenes, valor medio"),
    3: ("#1baf7a", "^", "Grupo 3: mayores, valor alto"),
}


def pesos(x):
    """250000 -> '250.000', como se escribe en Colombia."""
    return f"{x:,.0f}".replace(",", ".")


def titulo(texto):
    print("\n" + "=" * 70 + "\n" + texto + "\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Actividad 1. Preparación de los datos
# ---------------------------------------------------------------------------
def preparar():
    titulo("ACTIVIDAD 1. Preparación de los datos")
    df = pd.DataFrame(DATOS, columns=COLUMNAS)

    print("Cualitativas: id, ciudad, producto")
    print("Cuantitativas: edad, valor de compra, compras mensuales")
    print("Datos faltantes en toda la tabla:", int(df.isna().sum().sum()))
    print("Identificadores repetidos:", int(df["id"].duplicated().sum()))

    # Error de formato: el valor viene como texto, "$250.000", y así no se
    # puede sumar ni promediar. Se le quita el signo y el punto de miles.
    print("Valor antes:", df["valor_texto"].dtype, "->", df.loc[0, "valor_texto"])
    df["valor"] = (df["valor_texto"].str.replace("$", "", regex=False)
                   .str.replace(".", "", regex=False).astype(int))
    print("Valor después:", df["valor"].dtype, "->", df.loc[0, "valor"])

    # Inconsistencia de significado: cada fila es una compra, pero la edad y las
    # compras mensuales son del cliente, y no hay un identificador de cliente.
    print("Supuesto: la frecuencia mensual es la del cliente que hizo esa compra.")
    return df


# ---------------------------------------------------------------------------
# Actividad 2. Análisis exploratorio
# ---------------------------------------------------------------------------
def explorar(df):
    titulo("ACTIVIDAD 2. Análisis exploratorio")

    print("\n1. Producto con más compras")
    por_producto = df["producto"].value_counts()
    print(por_producto.to_string())
    total = df["valor"].sum()
    tec = df.loc[df["producto"] == "Tecnología", "valor"].sum()
    print(f"Tecnología suma {pesos(tec)} de {pesos(total)}, el {tec / total:.0%}")

    print("\n2. Valor promedio de las compras")
    print("Promedio:", pesos(df["valor"].mean()))
    print("Mediana:", pesos(df["valor"].median()))
    print("Total:", pesos(total))

    print("\n3. Ciudad con más compras")
    ciudades = df.groupby("ciudad")["valor"].agg(
        compras="count", promedio="mean", total="sum")
    print(ciudades.sort_values("compras", ascending=False).map(pesos).to_string())

    print("\n4. Grupo de edad con los valores más altos")
    df["grupo_edad"] = pd.cut(df["edad"], bins=[20, 30, 40, 50], right=False,
                              labels=["20 a 29", "30 a 39", "40 a 49"])
    edades = df.groupby("grupo_edad", observed=True)["valor"].agg(
        compras="count", promedio="mean")
    print(edades.map(pesos).to_string())
    print("Correlación edad contra valor:", round(df["edad"].corr(df["valor"]), 2))

    print("\n5. Relación entre la frecuencia mensual y el valor")
    print("Correlación:", round(df["compras_mes"].corr(df["valor"]), 2))
    resumen = df.groupby("producto").agg(
        compras=("id", "count"), valor_promedio=("valor", "mean"),
        compras_mes=("compras_mes", "mean"), total=("valor", "sum"))
    print(resumen.round(1).to_string())


# ---------------------------------------------------------------------------
# Actividad 3. Técnicas de minería de datos
# ---------------------------------------------------------------------------
def clasificar(df):
    titulo("ACTIVIDAD 3a. Clasificación por valor de compra")
    # Los cortes los pusimos nosotros: el grupo del medio queda cerca de la mediana.
    df["categoria"] = pd.cut(df["valor"], bins=[0, 150_000, 300_000, float("inf")],
                             right=False, labels=["Baja", "Media", "Alta"])
    tabla = df.groupby("categoria", observed=True)["valor"].agg(
        compras="count", promedio="mean")
    print(tabla.map(pesos).to_string())
    print(pd.crosstab(df["categoria"], df["producto"]).to_string())


def agrupar(df):
    titulo("ACTIVIDAD 3b. Agrupamiento con K-means")
    variables = ["edad", "valor", "compras_mes"]
    # Se estandariza porque el valor va en cientos de miles; si no, decide solo.
    x = StandardScaler().fit_transform(df[variables])

    # Para cada k se mira la silueta y también cuántas compras caen en cada grupo.
    for k in range(2, 6):
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(x)
        tamanos = sorted(pd.Series(km.labels_).value_counts().tolist())
        silueta = silhouette_score(x, km.labels_)
        print(f"k={k}  silueta={silueta:.3f}  compras por grupo={tamanos}")

    km = KMeans(n_clusters=3, n_init=20, random_state=42).fit(x)
    df["grupo_kmeans"] = km.labels_
    # Los grupos se numeran por valor promedio, de menor a mayor.
    orden = df.groupby("grupo_kmeans")["valor"].mean().sort_values().index
    df["grupo"] = df["grupo_kmeans"].map({g: i + 1 for i, g in enumerate(orden)})

    perfil = df.groupby("grupo").agg(
        compras=("id", "count"), edad=("edad", "mean"), valor=("valor", "mean"),
        compras_mes=("compras_mes", "mean"),
        productos=("producto", lambda s: ", ".join(
            f"{p} {n}" for p, n in s.value_counts().items())),
        ids=("id", ", ".join),
    )
    print(perfil.drop(columns="ids").round(2).to_string())
    for g, ids in perfil["ids"].items():
        print(f"Grupo {g}: {ids}")


# ---------------------------------------------------------------------------
# Fase 3. Visualización
# ---------------------------------------------------------------------------
def estilo(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_SUAVE)
    ax.tick_params(colors=TINTA_SUAVE, labelcolor=TINTA, labelsize=10)
    ax.grid(axis="y", color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def graficar(df):
    titulo("FASE 3. Gráficos")
    SALIDA.mkdir(exist_ok=True)

    # Gráfico de barras: cantidad de compras por categoría de producto.
    conteo = df["producto"].value_counts()
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    barras = ax.bar(conteo.index, conteo.values, color=AZUL, width=0.55)
    for b, n in zip(barras, conteo.values):
        ax.text(b.get_x() + b.get_width() / 2, n + 0.08, str(n),
                ha="center", va="bottom", color=TINTA, fontsize=11)
    ax.set_ylabel("Número de compras", color=TINTA, fontsize=10)
    ax.set_ylim(0, conteo.max() + 1)
    estilo(ax)
    fig.tight_layout()
    fig.savefig(SALIDA / "figura1_compras_por_producto.png", facecolor="white")
    plt.close(fig)

    # Gráfico de dispersión: valor contra frecuencia, con el grupo de K-means.
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    for g, (color, marca, nombre) in GRUPOS.items():
        s = df[df["grupo"] == g]
        ax.scatter(s["compras_mes"], s["valor"] / 1000, c=color, marker=marca, s=70,
                   edgecolors="white", linewidths=1.5, label=nombre, zorder=3)
    ax.set_xlabel("Compras mensuales", color=TINTA, fontsize=10)
    ax.set_ylabel("Valor de compra (miles de pesos)", color=TINTA, fontsize=10)
    ax.set_xticks(range(1, 6))
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0, 500)
    ax.grid(axis="x", color=REJILLA, linewidth=0.8)
    estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right",
              bbox_to_anchor=(1.02, 1.0), labelcolor=TINTA)
    fig.tight_layout()
    fig.savefig(SALIDA / "figura2_valor_vs_frecuencia.png", facecolor="white")
    plt.close(fig)
    print("Guardados en la carpeta figuras:")
    print("  figura1_compras_por_producto.png")
    print("  figura2_valor_vs_frecuencia.png")


if __name__ == "__main__":
    datos = preparar()
    explorar(datos)
    clasificar(datos)
    agrupar(datos)
    graficar(datos)
