
# 1. Bibliotheken importieren
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 2. CSV-Datei einlesen
file_path = "mushrooms.csv"
df = pd.read_csv(file_path, delimiter=',')

# 3. Erste Zeilen ausgeben
print("\nErste 5 Zeilen des DataFrames:")
print(df.head())

# 4. Leere Zellen zählen
print("\nAnzahl der leeren Zellen:")
print(df.isnull().sum().sum())

# 5. Zielspalte trennen und OHE-kodieren
col_name = 'class'
col = pd.get_dummies(df[col_name], dtype=float)  # Zielspalte kodieren
df = df.drop([col_name], axis=1)  # Zielspalte entfernen

# 6. Alle restlichen Spalten mit LabelEncoder kodieren
le = LabelEncoder()
df = df.apply(le.fit_transform)

# 7. Korrelationen berechnen
correlations = df.corr()
correlation_sum = correlations.abs().sum()
print("\nSpalten mit schwächster Korrelation:")
weakest_cols = correlation_sum.nsmallest(5)
print(weakest_cols)

# 8. Spalten mit sehr geringer Korrelation entfernen (optional)
threshold = 10  # Schwellenwert für die Korrelation (Summe)
low_correlation_cols = correlation_sum[correlation_sum < threshold].index
df = df.drop(columns=low_correlation_cols)

# 9. Daten skalieren
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 10. Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(df_scaled, col, test_size=0.2, random_state=42)

# 11. Modell erstellen
model = tf.keras.Sequential([
    tf.keras.Input(shape=(df_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='sigmoid'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(2, activation='softmax')  # 2 Output-Klassen (essbar/giftig)
])

# 12. Modell kompilieren
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 13. Modell trainieren
model.fit(X_train, y_train, epochs=50)

# 14. Modell testen
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# === Aufgabe 9: Tipp für Hyperparameter-Tuning ===
# Du kannst folgende Parameter anpassen, um die Genauigkeit zu verbessern:
# - Anzahl der Neuronen pro Layer (z. B. 32, 64, 128)
# - Anzahl der Layer (z. B. mehr oder weniger Dense-Schichten)
# - Aktivierungsfunktionen ('relu', 'tanh', 'sigmoid' statt nur 'sigmoid')
# - Optimizer (z. B. 'adam', 'sgd', 'rmsprop')
# - Anzahl der Epochen (z. B. 100 statt 50)
# - Batch-Größe (standardmäßig 32, kann z. B. auf 64 erhöht werden)
