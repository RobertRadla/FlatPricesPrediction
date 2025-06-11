import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df):
    # Histogram ceny za m/2
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Cena za m/2"], bins=50, kde=True)
    plt.title("Distribuce ceny za m²")
    plt.xlabel("Cena za m²")
    plt.ylabel("Počet bytů")
    plt.tight_layout()
    plt.show()

    # Scatter plot: Obytná plocha vs. Cena za m/2
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x="Obytná plocha", y="Cena za m/2", data=df)
    plt.title("Vztah mezi obytnou plochou a cenou za m²")
    plt.xlabel("Obytná plocha (m²)")
    plt.ylabel("Cena za m²")
    plt.tight_layout()
    plt.show()

    # Boxplot: Cena za m² podle počtu místností
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="Počet místností", y="Cena za m/2", data=df)
    plt.title("Cena za m² podle počtu místností")
    plt.xlabel("Počet místností")
    plt.ylabel("Cena za m²")
    plt.tight_layout()
    plt.show()

    # Boxplot: Cena za m² podle vybraných lokalit
    top_locations = df["Lokalita"].value_counts().head(10).index
    df_top_lokality = df[df["Lokalita"].isin(top_locations)]

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Lokalita", y="Cena za m/2", data=df_top_lokality)
    plt.title("Cena za m² podle TOP 10 lokalit")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Korelační heatmapa
    plt.figure(figsize=(8, 6))
    corr = df[["Obytná plocha", "Počet místností", "Podlaží", "Datum prodeje", "Cena za m/2"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Korelační matice")
    plt.tight_layout()
    plt.show()
