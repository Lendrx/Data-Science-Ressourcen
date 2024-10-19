# Detaillierte 3-Monats-Roadmap zum Erlernen von Python für Data Science

## Monat 1: Grundlagen und Einführung

### Woche 1-2: Python-Basics

#### Lernziele:
- Python-Syntax verstehen
- Mit Variablen und Datentypen arbeiten
- Grundlegende Operationen durchführen

#### Beispiele:

1. Variablen und Datentypen:
```python
# Variablen verschiedener Datentypen
name = "Anna"  # String
alter = 25     # Integer
groesse = 1.75 # Float
ist_student = True  # Boolean

print(f"{name} ist {alter} Jahre alt und {groesse}m groß.")
print(f"Student: {ist_student}")
```

2. Grundlegende Operationen:
```python
# Arithmetische Operationen
a = 10
b = 3

print(f"Addition: {a + b}")
print(f"Subtraktion: {a - b}")
print(f"Multiplikation: {a * b}")
print(f"Division: {a / b}")
print(f"Ganzzahlige Division: {a // b}")
print(f"Modulo: {a % b}")
print(f"Potenz: {a ** b}")
```

#### Projekt: Einfacher Taschenrechner
```python
def taschenrechner():
    zahl1 = float(input("Geben Sie die erste Zahl ein: "))
    zahl2 = float(input("Geben Sie die zweite Zahl ein: "))
    operation = input("Wählen Sie die Operation (+, -, *, /): ")

    if operation == "+":
        ergebnis = zahl1 + zahl2
    elif operation == "-":
        ergebnis = zahl1 - zahl2
    elif operation == "*":
        ergebnis = zahl1 * zahl2
    elif operation == "/":
        ergebnis = zahl1 / zahl2 if zahl2 != 0 else "Error: Division durch Null"
    else:
        ergebnis = "Ungültige Operation"

    print(f"Ergebnis: {ergebnis}")

taschenrechner()
```

### Woche 3-4: Kontrollstrukturen und Funktionen

#### Lernziele:
- if-else Anweisungen verwenden
- for und while Schleifen beherrschen
- Funktionen definieren und aufrufen

#### Beispiele:

1. if-else Anweisung:
```python
alter = int(input("Wie alt sind Sie? "))

if alter < 18:
    print("Sie sind minderjährig.")
elif 18 <= alter < 65:
    print("Sie sind erwachsen.")
else:
    print("Sie sind im Rentenalter.")
```

2. for Schleife:
```python
# Quadratzahlen von 1 bis 10 berechnen
for i in range(1, 11):
    print(f"{i} zum Quadrat ist {i**2}")
```

3. while Schleife:
```python
# Zähle rückwärts von 10
countdown = 10
while countdown > 0:
    print(countdown)
    countdown -= 1
print("Start!")
```

4. Funktion definieren:
```python
def begruessung(name, sprache="Deutsch"):
    if sprache.lower() == "deutsch":
        return f"Hallo, {name}!"
    elif sprache.lower() == "englisch":
        return f"Hello, {name}!"
    else:
        return f"Sprache nicht unterstützt. Hallo, {name}!"

print(begruessung("Maria"))
print(begruessung("John", "Englisch"))
```

#### Projekt: Zahlenratespiel
```python
import random

def zahlenraten():
    ziel = random.randint(1, 100)
    versuche = 0

    print("Ich denke an eine Zahl zwischen 1 und 100.")

    while True:
        versuch = int(input("Dein Tipp: "))
        versuche += 1

        if versuch < ziel:
            print("Zu niedrig!")
        elif versuch > ziel:
            print("Zu hoch!")
        else:
            print(f"Richtig! Du hast die Zahl in {versuche} Versuchen erraten.")
            break

zahlenraten()
```

## Monat 2: Datenstrukturen und Bibliotheken

### Woche 5-6: Listen, Dictionaries und Dateien

#### Lernziele:
- Mit Listen und Dictionaries arbeiten
- Datei-Ein- und Ausgabe beherrschen

#### Beispiele:

1. Listen:
```python
# Liste erstellen und manipulieren
fruechte = ["Apfel", "Banane", "Kirsche"]
print(fruechte)

fruechte.append("Orange")
print(fruechte)

print(f"Das zweite Element ist: {fruechte[1]}")

# List Comprehension
quadratzahlen = [x**2 for x in range(1, 6)]
print(quadratzahlen)
```

2. Dictionaries:
```python
# Dictionary erstellen und verwenden
person = {
    "name": "Max Mustermann",
    "alter": 30,
    "beruf": "Ingenieur"
}

print(person["name"])

person["wohnort"] = "Berlin"
print(person)

for key, value in person.items():
    print(f"{key}: {value}")
```

3. Datei-Ein- und Ausgabe:
```python
# In Datei schreiben
with open("beispiel.txt", "w") as file:
    file.write("Dies ist ein Beispieltext.\n")
    file.write("Python ist eine großartige Sprache!")

# Aus Datei lesen
with open("beispiel.txt", "r") as file:
    inhalt = file.read()
    print(inhalt)
```

#### Projekt: Einfaches Adressbuch
```python
def adressbuch():
    kontakte = {}

    while True:
        aktion = input("(A)nzeigen, (H)inzufügen, (L)öschen oder (B)eenden? ").lower()

        if aktion == "a":
            for name, nummer in kontakte.items():
                print(f"{name}: {nummer}")
        elif aktion == "h":
            name = input("Name: ")
            nummer = input("Telefonnummer: ")
            kontakte[name] = nummer
        elif aktion == "l":
            name = input("Zu löschender Name: ")
            if name in kontakte:
                del kontakte[name]
            else:
                print("Kontakt nicht gefunden.")
        elif aktion == "b":
            break
        else:
            print("Ungültige Eingabe.")

adressbuch()
```

### Woche 7-8: Einführung in NumPy und Pandas

#### Lernziele:
- NumPy-Arrays erstellen und manipulieren
- Pandas DataFrames verstehen und nutzen

#### Beispiele:

1. NumPy:
```python
import numpy as np

# Array erstellen
arr = np.array([1, 2, 3, 4, 5])
print(arr)

# Mathematische Operationen
print(f"Mittelwert: {np.mean(arr)}")
print(f"Standardabweichung: {np.std(arr)}")

# Mehrdimensionales Array
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix)
```

2. Pandas:
```python
import pandas as pd

# DataFrame erstellen
data = {
    'Name': ['Anna', 'Bob', 'Charlie'],
    'Alter': [25, 30, 35],
    'Stadt': ['Berlin', 'Hamburg', 'München']
}
df = pd.DataFrame(data)
print(df)

# Daten filtern
print(df[df['Alter'] > 28])

# Gruppieren und Aggregieren
print(df.groupby('Stadt').mean())
```

#### Projekt: Titanic-Datensatz Analyse
```python
import pandas as pd
import matplotlib.pyplot as plt

# Daten laden (angenommen, Sie haben die CSV-Datei)
df = pd.read_csv('titanic.csv')

# Überlebensrate nach Geschlecht
survival_gender = df.groupby('Sex')['Survived'].mean()
survival_gender.plot(kind='bar')
plt.title('Überlebensrate nach Geschlecht')
plt.show()

# Altersverteilung
df['Age'].hist(bins=20)
plt.title('Altersverteilung der Passagiere')
plt.xlabel('Alter')
plt.ylabel('Anzahl')
plt.show()

# Überlebensrate nach Passagierklasse
survival_class = df.groupby('Pclass')['Survived'].mean()
survival_class.plot(kind='bar')
plt.title('Überlebensrate nach Passagierklasse')
plt.show()
```

## Monat 3: Data Science Anwendungen

### Woche 9-10: Datenvisualisierung mit Matplotlib

#### Lernziele:
- Verschiedene Plot-Typen erstellen
- Plots anpassen und verschönern

#### Beispiele:

1. Linienplot:
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('Sinuskurve')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.show()
```

2. Streudiagramm:
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.random.randn(100)
y = 2*x + np.random.randn(100)

plt.scatter(x, y)
plt.title('Streudiagramm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

#### Projekt: COVID-19 Dashboard
```python
import pandas as pd
import matplotlib.pyplot as plt

# Daten laden (angenommen, Sie haben die CSV-Datei)
df = pd.read_csv('covid_data.csv')

# Zeitreihe der Fälle
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['total_cases'])
plt.title('COVID-19 Fälle im Zeitverlauf')
plt.xlabel('Datum')
plt.ylabel('Gesamtfälle')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Balkendiagramm der Top 10 Länder
top_10 = df.groupby('country')['total_cases'].max().sort_values(ascending=False).head(10)
top_10.plot(kind='bar')
plt.title('Top 10 Länder nach COVID-19 Fällen')
plt.xlabel('Land')
plt.ylabel('Gesamtfälle')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Woche 11-12: Einführung in Machine Learning mit scikit-learn

#### Lernziele:
- Grundlegende ML-Konzepte verstehen
- Einfache Modelle trainieren und evaluieren

#### Beispiele:

1. Lineare Regression:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Daten generieren
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell trainieren
model = LinearRegression()
model.fit(X_train, y_train)

# Vorhersagen machen
y_pred = model.predict(X_test)

# Modell evaluieren
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

2. K-Nearest Neighbors:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Daten laden
iris = load_iris()
X, y = iris.data, iris.target

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modell trainieren
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Vorhersagen machen
y_pred = knn.predict(X_test)

# Genauigkeit berechnen
accuracy = accuracy_score(y_test, y_pred)
print(f"Genauigkeit: {accuracy:.2f}")
```

#### Projekt: Hauspreisvorhersage
```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Daten laden
boston = load_boston()
X, y = boston.data, boston.target

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell trainieren
model = LinearRegression()
model.fit(X_train, y_train)

# Vorhersagen machen
y_pred = model.predict(X_test)

# Modell evaluieren
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Feature-Wichtigkeit
for feature, importance in zip(boston.feature_names, model.coef_):
    print(f"{feature}: {importance}")
```
