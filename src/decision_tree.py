# 1. Preparació del dataset
# Importem el fitxer 'dades_aion.csv'
import pandas as pd

data = pd.read_csv("dades_aion.csv", sep=';')

# Avaluem la mida del dataset
num_files, num_columnes = data.shape
print(f"Nombre de files: {num_files}")
print(f"Nombre de columnes: {num_columnes}")

# Llistem les columnes amb valors nuls i la quantitat de valors nuls per columna
valors_nuls = data.isnull().sum()
columnes_amb_nuls = valors_nuls[valors_nuls > 0]
print("\nColumnes amb valors nuls:")
print(columnes_amb_nuls)

# Emplenem amb 0 només les columnes amb NaNs que són features d'activitat en grup
cols_a_omplir = ['total_party_time', 'guild_join_count', 'average_party_time']
data[cols_a_omplir] = data[cols_a_omplir].fillna(0)

# Eliminem les columnes no predictives
data = data.drop(columns=['Unnamed: 0', 'actor_account'])

# Separem les variables predictives (X) i la variable objectiu (y)
X = data.drop(columns=['class'])
y = data['class']

# Desem el dataset net en un fitxer
data.to_csv("dades_aion_preparat.csv", index=False)
print("\nFitxer 'dades_aion_preparat.csv' desat correctament.")

# 2. Construcció d’un arbre de decisió inicial
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Seleccionem només les files on la variable objectiu és coneguda
X_model = X[y.notna()]
y_model = y[y.notna()]

# Separem les dades en entrenament i test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2)

print("\n==========================")
print("MODEL 1: Arbre sense límit de profunditat")
print("==========================")

# Creem i entrenem l'arbre de decisió amb "entropy"
arbre = DecisionTreeClassifier(criterion="entropy")
arbre.fit(X_train, y_train)

# Prediccions
train_pred = arbre.predict(X_train)
test_pred = arbre.predict(X_test)

# Precisió
acc_train = accuracy_score(y_train, train_pred)
acc_test = accuracy_score(y_test, test_pred)
print(f"[Arbre Inicial] Precisió en entrenament: {acc_train:.3f}")
print(f"[Arbre Inicial] Precisió en test: {acc_test:.3f}")

# Matriu de confusió i càlcul de FP/FN
cm = confusion_matrix(y_test, test_pred)
tn, fp, fn, tp = cm.ravel()
print("\n[Arbre Inicial] Matriu de confusió:")
print(cm)

taxa_fp = fp / (fp + tn)
taxa_fn = fn / (fn + tp)
print(f"[Arbre Inicial] Taxa de falsos positius (FP): {taxa_fp:.3f}")
print(f"[Arbre Inicial] Taxa de falsos negatius (FN): {taxa_fn:.3f}")

print("\n→ Aquest model aconsegueix una precisió molt alta i detecta bé els jugadors amb estrès, però probablement està sobreajustat, ja que la precisió en entrenament és del 100%. També cal vigilar amb la taxa de falsos negatius si l'objectiu és protegir la salut dels usuaris.")

# 3. Modificació de l’arbre de decisió
print("\n==========================")
print("MODEL 2: Arbre AMB límit de profunditat (max_depth=5)")
print("==========================")

# Creem un nou arbre amb entropia i profunditat màxima 5
arbre_limitat = DecisionTreeClassifier(criterion="entropy", max_depth=5)
arbre_limitat.fit(X_train, y_train)

# Prediccions
train_pred_lim = arbre_limitat.predict(X_train)
test_pred_lim = arbre_limitat.predict(X_test)

# Precisió
acc_train_lim = accuracy_score(y_train, train_pred_lim)
acc_test_lim = accuracy_score(y_test, test_pred_lim)
print(f"[Arbre Limitat] Precisió en entrenament: {acc_train_lim:.3f}")
print(f"[Arbre Limitat] Precisió en test: {acc_test_lim:.3f}")

# Matriu de confusió i càlcul de FP/FN
cm_lim = confusion_matrix(y_test, test_pred_lim)
tn_l, fp_l, fn_l, tp_l = cm_lim.ravel()
print("\n[Arbre Limitat] Matriu de confusió:")
print(cm_lim)

taxa_fp_lim = fp_l / (fp_l + tn_l)
taxa_fn_lim = fn_l / (fn_l + tp_l)
print(f"[Arbre Limitat] Taxa de falsos positius (FP): {taxa_fp_lim:.3f}")
print(f"[Arbre Limitat] Taxa de falsos negatius (FN): {taxa_fn_lim:.3f}")

print("\n→ Aquest segon model és més senzill i generalitza millor, però com a contrapartida detecta menys casos de jugadors realment estressats (FN elevats). Això pot ser un problema si el nostre objectiu principal és la detecció precoç i preventiva.")

# 4. Comparació amb l’algorisme Dummy
from sklearn.dummy import DummyClassifier

print("\n==========================")
print("MODEL 3: Classificador Dummy (estratègia 'most_frequent')")
print("==========================")

# Entrenem el classificador dummy
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

# Prediccions
train_pred_dummy = dummy.predict(X_train)
test_pred_dummy = dummy.predict(X_test)

# Precisió
acc_train_dummy = accuracy_score(y_train, train_pred_dummy)
acc_test_dummy = accuracy_score(y_test, test_pred_dummy)
print(f"[Dummy] Precisió entrenament: {acc_train_dummy:.3f}")
print(f"[Dummy] Precisió test: {acc_test_dummy:.3f}")

# Matriu de confusió i càlcul de FP/FN
cm_dummy = confusion_matrix(y_test, test_pred_dummy)
tn_d, fp_d, fn_d, tp_d = cm_dummy.ravel()
print("\n[Dummy] Matriu de confusió:")
print(cm_dummy)

taxa_fp_dummy = fp_d / (fp_d + tn_d)
taxa_fn_dummy = fn_d / (fn_d + tp_d)
print(f"[Dummy] Taxa de falsos positius (FP): {taxa_fp_dummy:.3f}")
print(f"[Dummy] Taxa de falsos negatius (FN): {taxa_fn_dummy:.3f}")

print("\n→ El model Dummy sempre escull la classe majoritària (probablement 0 = sense estrès). "
      "Té una precisió aparentment alta, però ignora completament la classe minoritària. "
      "Com a conseqüència, té una taxa de falsos negatius del 100%, cosa que el fa inútil per detectar jugadors estressats.")
