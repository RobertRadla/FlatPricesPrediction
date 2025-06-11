
============ PREDIKCE CEN BYTŮ NA ČESKÉM TRHU ===============

Tento projekt řeší predikci ceny bytu (v Kč/m²) na základě vlastností nemovitosti jako jsou lokalita, 
obytná plocha, počet místností a další. Cílem je vytvořit spolehlivý model, který může být nasazen jako 
jednoduchá webová aplikace pro odhad cen.

---


## ------ 1. main.py --------
Soubor main.py slouží jako hlavní orchestrátor celého projektu. Spouští ve správném pořadí všechny klíčové
části pipeline od načtení dat až po uložení modelu.



## ------ 2. preprocessing.py --------

V první části dochází k: 
- načtení dat
- spojení dat do jednoho dataframe 
- ukázce podoby dat a výpisu základních charakteristik
- DŮLEŽITÉ POZNATKY: 
- datafame má 1 496 243 záznamů
- sloupce datafame: Index | Lokalita | Obytná plocha | Počet místností | Podlaží | Datum prodeje | Místo/čas | Cena za m/2

V druhé části dochází k: 
- zjištění datových typů sloupců -> datové typy v několika případech nesouhlasí s tím, jaká data uchovávají (int/str např.),
- výpisu základních statistik o dataframe -> obsahuje málo dat vzhledem k problému s datovými typy,
- vyhledání NaN hodnot -> vzhledek k nízkému počtu oproti dataframe a k tomu, že by např. průměry a jiné úpravy 
    mohly zavádět výslednou cenu (například v případě, když by byt o velikosti 3000m2 měl průměrnou cenu za m2), dochází k
    vymazání řádků s NaN hodnotami,
- kontrole, že je datset souměrný (tzn. že má v každém sloupci stejný počet řádků) -> kontrola úspěšná
- kontrole existence stejných řádků -> existuje 30 248 totožných řádků -> zanechání prvního výskytu a smazání ostatních výskytů stejného řádku
- kontrole sloupce INDEX: kontrola zda-li obsahuje unikátní hotnoty -> neobsahuje (obsahuje duplicitní hodnoty, přičemž celé 
    řádky duplicitní nejsou = sloupec index je zavádějící) -> smazání sloupce index a ponechání vygenerovaného indexu pandas
- kontrole sloupce LOKALITA: má datový typ objekt, dojde tedy ke kontrole, jestli obsahuje pouze stringy -> obsahuje, vše OK
- kontrole sloupce OBYTNÁ PLOCHA: prveden describe() pro přehled -> zajímají nás extrémy (min = 20m2 = ok, max = 15 044m2 = není ok ) ->
    -> bytů nad 300m2 (náhodně vybrané číslo) je 1 459 -> dochází k jejich vymazání
- kontrole sloupce POČET MÍSTNOSTÍ: má formát string (např. 2 + KK) -> dle výpisu unikátních hodnot obsahuje hodnoty od 1 + KK do 
    6 + KK -> odstraníme " + KK" a převedeme vše předtím na číslo (int),
- kontrole sloupce PODLAŽÍ: je datatype object (obsahuje stringy nebo mix), patra mají podobu čísel kromě "přízemí" a "podkroví" ->
    -> "přízemí" je převedeno na 0 a "podkroví" na 12 (nejvyšší patro v datasetu je 11) -> celý sloupec je poté převeden na číslený datový typ,
- kontrole sloupce DATUM PRODEJE: má desetinná čísla, jelikož je datový typ float -> převedení na int -> kontrola správnosti převodu
- kontrole sloupce MÍSTO/ČAS: data mají podobu nazevKraje_datum (např. Středočeský kraj_2020) -> datum již je obsažen ve sloupci 
    "Datum prodeje", takže je extrahován a ponechán pouze název kraje -> sloupec je přejmenován na "Místo" -> kontrola proběhla OK,
- kontrole sloupce CENA ZA M/2: data před úpravou mají datový typ object a obsahují různé mezery a desetinné čárky např. 43518 ,7781638779 ->
    -> dochází k úpravě dat (odstraněníé mezer a přeměna "," za ".") -> změna datového typu na float -> kontrola OK
- vizualizaci dat -> vizualizace je obsažena v visualizations.py 

                # ------------- visualizations.py --------------

                - Histogram ceny za m²: zobrazuje rozložení cílové proměnné -> je vidět, jestli je rozložení normální, zkreslené, má extrémy.
                - Scatter plot - Obytná plocha vs. Cena za m²: zkoumá vztah mezi velikostí bytu a cenou za m², umožňuje vidět trendy, shluky, 
                    možné outliery (např. drahé miniaturní byty nebo levné velké).
                - Boxplot - Cena za m² podle počtu místností: pomáhá vidět mediany, rozsahy a extrémní hodnoty pro každou kategorii.
                - Boxplot - Cena za m² podle TOP 10 lokalit: toto pomáhá vizuálně potvrdit, že některé lokality jsou systematicky dražší.
                - Korelační heatmapa - ukazuje míru lineární závislosti mezi numerickými sloupci -> může odhalit, které vstupní proměnné 
                    mají silný vliv na cenu nebo jsou mezi sebou korelované.

- ONE-HOT ENCODING: byl použit encoding pro sloupce "Lokalita" a "Místo", protože obsahují stringy -> kontrola OK

Ve třetí části dochází k:
- rozdělení datasetu na trénovací a testovací sady -> proměnná X obsahuje "vstupy" a proměnna y obsahuje "výstupy" (Cena za m2)-> dochází k 
    rozdělení "vstupů" a "výstupů" na dvě sady v poměru 8:2 (80% tréninková sada, 20% testovací sada),
- normalizaci dat -> převedeny jsou data ze sloupců "Obytná plocha", "Počet místností" a "Podlaží" ("Datum prodeje" normalizováno není, protože 
    tento údaj není čistě matematicky lineární) -> nejprve je použit scaler na trénovací data a poté je stejný scaler použit na testovací data ->
    -> data mají váhově vyvážený rozsah



## ------ 3. training.py --------
- v této části jsou vybrány 3 modely, podle jejichž výsledků bude vybrán ten nejpřesnější pro další práci,
- tyto tři modely jsou: lineární regrese, rozhodovací stromy a random forest,
- nejprve jsou modely inicializovány -> random_state=42 pro reprodukovatelnost výsledků a n_estimators=100 u Random Forest pro vytvoření
     100 rozhodovacích stromů (jako kompromis mezi přesností a výkonem),
- následně trénovány na trénovacích datech,
- uložení natrénovaných modelů do .pkl souborů.



## ------ 4. evaluation.py --------
- V této části dochází k použití natrénovaných modelů na testovacích datech a následnému porovnání výsledků v tabulce dle MAE, MSE a R²,
- funkce evaluate_model obsahuje obsahuje logiku pro výpočet metrik MAE, MSE, R² a jejich výpis do konzole spolu se jménem modelu,
- funkce evaluate_all_models provede funkci evaluate_model pro každý natrénovaný model s použitím testovacích dat a výsledky vypíše do konzole
    v podobě tabulky,
- dle výsledků se jeví jako nejlepší model Random Forest a proto je vybrán pro další práci –> má nejnižší MAE i MSE + má nejvyšší R².



## ------ 5. tuning.py --------
- V této části probíhá ladění hyperparametrů u modelu Random Forest, jehož cílem je najít co nejlepší možnou kombinaci nastavení modelu,
- nejprve je vytvořen slovník param_grid (parametrický grid), jakožto seznam možných hodnot pro různé parametry modelu,
- poté je vytvořen objekt grid_search -> systematicky vyzkouší všechny kombinace parametrů z výše uvedeného gridu -> pro každý model použije 
    3násobnou cross-validaci -> optimalizuje podle metriky R² -> využívá všechna dostupná CPU jádra pro urychlení výpočtů,
- vypsána nejlepší kombinace hyperparametrů (grid_search.best_params_),
- vypsáno nejlepší průměrné R² skóre z cross-validace,
- nejlepší model (best_estimator_) je dále použit k predikci na testovacích datech a je zhodnocen pomocí metrik MAE, MSE a R²,
- výsledný model je uložen jako .pkl soubor pod názvem final_random_forest_model.pkl, aby mohl být použit např. ve webové aplikaci,
- jsou také uloženy scaler.pkl – obsahuje normalizovaný tvar trénovacích dat (pro správné škálování vstupních hodnot ve webové aplikaci), 
    misto_columns.pkl a lokalita_columns.pkl – obsahují názvy všech nově vytvořených sloupců při one-hot encodingu sloupců "Místo" a "Lokalita",
    což je nezbytné při převodu nových vstupních dat (např. z formuláře ve webové aplikaci) do stejné struktury, na které byl model natrénován.


## ------ 6. test_utils.py --------
- Slouží jako pomocný skript pro unit testy -> umožňuje rychle načíst a připravit malou podmnožinu dat, aby testy byly rychlé a nezatěžovaly 
    systém,
- import funkce prepare_data() ze souboru preprocessing.py,
- definice vlastní funkce prepare_data_limited, která funguje jako zjednodušená verze prepare_data(), ale omezuje množství dat na 5 000 řádků,
- spuštění plné přípravy dat – načtení, čištění, rozdělení na trénovací a testovací sady,
- oříznutí trénovacích a testovacích dat jen na prvních n_rows řádků (5000),
- oříznutí dataframe na 2x n_rows řádků (aby X_train + X_test = df).


## ------ 6. test_utils.py --------
- Slouží jako unit test skript pro ověření funkčnosti jednotlivých částí projektu,
- funkce test_data_preparation načte omezený počet řádků datasetu přes prepare_data_limited() (pro rychlé testování) a ověří, že ->
    -> data nejsou prázdná -> rozdělení na trénovací a testovací množinu proběhlo správně (součet se rovná původnímu počtu řádků) -> 
    -> ve vstupních datech nejsou žádné NaN hodnoty,
- funkce test_model_training natrénuje modely (Linear Regression, Decision Tree, Random Forest) a ověří, že -> každý model byl vytvořen ->
    -> každý model umí predikovat ( že .predict() metoda existuje),
- funkce test_model_evaluation_output posoudí Random Forest na testovacích datech a ověří, že -> výstupem jsou metriky MAE, MSE, R² -> 
    -> R² je typu float.


## ------ 6. web_app.py --------
- Tento soubor slouží jako frontend aplikace, ve které může uživatel zadat parametry bytu a aplikace na základě natrénovaného Random Forest 
    modelu odhadne cenu bytu za metr čtvereční,
- načtení modelu a pomocných dat,
- zadání titles,
- vytvoří se formulář pro zadání parametrů,
- připraví se vstupy pro model -> inputy se uloží do input_data a z toho se vytvoří dataframe do X -> místo a Lokalita jsou převedeny na 
    one-hot encoded sloupce -> pokud v aktuálním vstupu některé sloupce chybí, doplní se nulami -> sloupce se seřadí přesně tak, jak to 
    očekává model (podle uloženého scaleru),
- číselné hodnoty (Obytná plocha, Počet místností, Podlaží) jsou transformovány pomocí původního StandardScaleru, aby odpovídaly formátu, 
    na kterém byl model natrénován,
- pokud uživatel klikne na tlačítko „Spočítej odhad ceny za m²“, vstup se předá do modelu a zobrazí se výsledek (odhadovaná cena za 1 m² v Kč),
- streamlit run web_app.py. 