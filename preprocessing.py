import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from visualizations import visualize_data

def prepare_data():
    
    # === STEP 1: DATA COLLECTION ===
    # Load files
    # Concatenate dataframes
    # Quick data preview and shape check

    df1 = pd.read_csv('data/DATA PART I.txt', delimiter=";", decimal=".")
    df2 = pd.read_csv('data/PART II.txt', delimiter=";", decimal=".", header=None)
    df2.columns = df1.columns 

    df = pd.concat([df1, df2], ignore_index=True)

    print("Data preview:")
    print(df.head())
    print(f"\nRows: {len(df)}")
    print(f"Columns: {df.shape[1]}")




    # === STEP 2: DATA EXPLORATION + STEP 3: DATA PREPROCESSING ===
    # dataset's columns display
    # data types display
    # basic descriptive statistics of the dataset
    # check for any missing (NaN) values and their removal
    # check for consistent row counts across columns
    # duplicate rows check and removal
    # analyzing the values of individual columns and adjustments


    print("Sloupce v datasetu:")
    print(df.columns)

    print("\nTypy hodnot ve sloupcích:")
    print(df.dtypes)

    print("\nPopisné statistiky:")
    print(df.describe())

    print("\nPočet chybějících hodnot v každém sloupci:")
    print(df.isnull().sum())
    df = df.dropna(subset=["Obytná plocha"])
    df = df.dropna(subset=["Počet místností"])
    df = df.dropna(subset=["Podlaží"])
    df = df.dropna(subset=["Datum prodeje"])
    print("\nKontrola, že proběhlo vymazání NaN řádků:")
    print(df.isnull().sum())

    print("\nPočet hodnot v jednotlivých sloupcích:")
    print("Index:             ", df["Index"].count())
    print("Lokalita:          ", df["Lokalita"].count())
    print("Obytná plocha:     ", df["Obytná plocha"].count())
    print("Počet místností:   ", df["Počet místností"].count())
    print("Podlaží:           ", df["Podlaží"].count())
    print("Datum prodeje:     ", df["Datum prodeje"].count())
    print("Místo/čas:         ", df["Místo/čas"].count())
    print("Cena za m/2:       ", df["Cena za m/2"].count())

    duplicates = df[df.duplicated(keep=False)]
    print(f"\nPočet úplně stejných řádků: {len(duplicates)}")
    print("Příklady duplicitních řádků:")
    print(duplicates.head())
    first_duplicate_row = duplicates.iloc[0] # find one completely identical row
    identical_rows = df[df.eq(first_duplicate_row).all(axis=1)]
    print("\nUkázka duplicitního řádku")
    print(identical_rows)
    df = df.drop_duplicates(keep="first").reset_index(drop=True) # keeps the first occurrence
    remaining_duplicates = df[df.duplicated(keep=False)]
    print(f"\nPočet úplně stejných řádků PO odstranění: {len(remaining_duplicates)}")

   

    # -- Analyzing the 'Index' column --
    # Listing unique values and number of rows – they should match  
    # Example of a selected duplicate index – to check if there are fully duplicated rows  
    # Removing the 'Index' column – it does not contain unique values 

    print("\n--- ANALÝZA SLOUPCE: Index ---")
    print("Unikátní hodnoty: ", df["Index"].nunique())
    print("Počet řádků:", len(df))

    print("\nUkázka duplicitního indexu:")
    print(df[df["Index"] == 132])

    df.drop(columns=["Index"], inplace=True)

    # -- Analyzing the 'Lokalita' column --
    # Check if there are any values that are not of type string (data type: object)

    print("\n--- ANALÝZA SLOUPCE: Lokalita ---")
    non_string_values = df[~df["Lokalita"].apply(lambda x: isinstance(x, str))]
    print(f"Počet ne-string hodnot: {len(non_string_values)}")

    # -- Analyzing the 'Obytná plocha' column --
    # Basic descriptive statistics
    # Check for outliers – maximum
    # Remove outliers above 300 m²

    print("\n--- ANALÝZA SLOUPCE: Obytná plocha ---")
    print(df["Obytná plocha"].describe())

    too_large = df[df["Obytná plocha"] > 300]
    print(f"\nPočet podezřele velkých bytů (> 300 m²): {len(too_large)}")
    print(too_large.head(10))  # Sample of several rows
    
    print(f"\nPočet řádků před odstraněním extrémních hodnot: {len(df)}")
    df = df[df["Obytná plocha"] <= 300]
    print(f"\nPočet řádků po odstranění extrémních hodnot: {len(df)}")


    # -- Analyzing the 'Počet místností' column --
    # Count unique values
    # Convert values to numeric type and extract only numbers
    # Example of the first few rows after conversion

    print("\n--- ANALÝZA SLOUPCE: Počet místností ---")
   
    print("Unikátní hodnoty:")
    print(df["Počet místností"].value_counts())
    
    df["Počet místností"] = df["Počet místností"].str.extract(r"(\d+)").astype(int)

    print("\nUkázka upraveného sloupce 'Počet místností':")
    print(df["Počet místností"].head())


    # -- Analyzing the 'Podlaží' column --
    # Count unique values
    # Replace "Přízemí" with 0
    # Replace "Podkroví" with 999999
    # Convert the rest of the column to integers
    # Showcase of the modified column

    print("\n--- ANALÝZA SLOUPCE: Podlaží ---")
    
    print("Unikátní hodnoty:")
    print(df["Podlaží"].value_counts())
    
    df["Podlaží"] = df["Podlaží"].replace("Přízemí", 0)
    
    df["Podlaží"] = df["Podlaží"].replace("Podkroví", 12)
    
    df["Podlaží"] = df["Podlaží"].astype(int)
    
    print("\nUkázka upraveného sloupce 'Podlaží':")
    print(df["Podlaží"].value_counts().sort_index())


    # -- Analyzing the 'Datum prodeje' column --
    # Convert the column to int (to avoid decimals)
    # Example of the modified column

    print("\n--- ANALÝZA SLOUPCE: Datum prodeje ---")
    
    df["Datum prodeje"] = df["Datum prodeje"].astype(int)
    
    print("Ukázka upraveného sloupce 'Datum prodeje':")
    print(df["Datum prodeje"].value_counts().sort_index())


    # -- Analyzing the 'Místo/čas' column --
    # Count unique values
    # Extract the part before the underscore
    # Rename the column to 'Místo'
    # Showcase of the modified column

    print("\n--- ANALÝZA SLOUPCE: Místo/čas ---")
    
    print("Unikátní hodnoty:")
    print(df["Místo/čas"].value_counts())
    
    df["Místo/čas"] = df["Místo/čas"].str.extract(r"^(.*?)_")
    
    df.rename(columns={"Místo/čas": "Místo"}, inplace=True)
    
    print("\nUkázka upraveného sloupce 'Místo':")
    print(df["Místo"].value_counts().head(10))


    # -- Analyzing the 'Cena za m/2' column --
    # Remove all spaces (including unicode) and replace comma with dot
    # Convert to float
    # Showcase of the modified column

    print("\n--- ANALÝZA SLOUPCE: Cena za m/2 ---")
    
    df["Cena za m/2"] = df["Cena za m/2"].str.replace(r"\s+", "", regex=True)  # removes all spaces including Unicode spaces
    df["Cena za m/2"] = df["Cena za m/2"].str.replace(",", ".", regex=False)
    df["Cena za m/2"] = df["Cena za m/2"].astype(float)
    
    print("\nPo úpravě:")
    print("Datový typ:", df["Cena za m/2"].dtype)
    print("Nová podoba čísel:")
    print(df["Cena za m/2"].head())





    # === VIZUALIZACE DAT ===

    visualize_data(df) 




    # === ONE-HOT ENCODING ===
    # encoding the 'Lokalita' and 'Místo' columns
    
    df = pd.get_dummies(df, columns=["Místo", "Lokalita"], drop_first=True)

    print("\nZakódované sloupce (Místo):")
    print(df.columns[df.columns.str.startswith("Místo_")])

    print("\nZakódované sloupce (Lokalita):")
    print(df.columns[df.columns.str.startswith("Lokalita_")])









    # === STEP 4: MODEL SELECTION ===
    # -- Splitting the dataset into training and test sets --
    # Definition of the target variable and the input features
    # Spliting data into training and test sets (80% training, 20% test)

    X = df.drop(columns=["Cena za m/2"])
    y = df["Cena za m/2"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -- Normalization of data using StandardScaler --
    # Creation of an instance of StandardScaler
    # Columns to be scaled
    # Using fit_transform on the training data
    # Application of the same scaler to the test data

    scaler = StandardScaler()
    num_cols = ["Obytná plocha", "Počet místností", "Podlaží"]
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return df, X_train, X_test, y_train, y_test
