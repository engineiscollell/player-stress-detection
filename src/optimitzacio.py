# 5. Optimització del model de decisió

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Carreguem i preparem les dades
data = pd.read_csv("dades_aion.csv", sep=';')

cols_a_omplir = ['total_party_time', 'guild_join_count', 'average_party_time']
data[cols_a_omplir] = data[cols_a_omplir].fillna(0)
data = data.drop(columns=['Unnamed: 0', 'actor_account'])
X = data.drop(columns=['class'])
y = data['class']

X_model = X[y.notna()]
y_model = y[y.notna()]
X_pred = X[y.isna()]  # Usuaris que no s’han avaluat

X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2)

# 2. Definim les configuracions
configuracions = [
    {"label": "Base", "params": {}},
    {"label": "max_depth=5", "params": {"max_depth": 5}},
    {"label": "max_depth=6", "params": {"max_depth": 6}},
    {"label": "min_samples_leaf=10", "params": {"min_samples_leaf": 10}},
    {"label": "class_weight=balanced", "params": {"class_weight": "balanced"}},
    {"label": "depth=6, leaf=10", "params": {"max_depth": 6, "min_samples_leaf": 10}},
    {"label": "depth=6, leaf=10, balanced", "params": {"max_depth": 6, "min_samples_leaf": 10, "class_weight": "balanced"}},
]

# 3. Avaluem cada configuració
resultats_opt = []

print("\n--- COMPARACIÓ DE CONFIGURACIONS ---")
for config in configuracions:
    arbre = DecisionTreeClassifier(criterion="entropy", **config["params"])
    arbre.fit(X_train, y_train)
    test_pred = arbre.predict(X_test)
    train_pred = arbre.predict(X_train)

    acc_train = accuracy_score(y_train, train_pred)
    acc_test = accuracy_score(y_test, test_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
    taxa_fp = fp / (fp + tn)
    taxa_fn = fn / (fn + tp)

    resultats_opt.append({
        "Model": config["label"],
        "Train Acc": acc_train,
        "Test Acc": acc_test,
        "FP Rate": taxa_fp,
        "FN Rate": taxa_fn
    })

    print(f"\nModel: {config['label']}")
    print(f"- Precisió entrenament: {acc_train:.3f}")
    print(f"- Precisió test: {acc_test:.3f}")
    print(f"- Taxa de falsos positius (FP): {taxa_fp:.3f}")
    print(f"- Taxa de falsos negatius (FN): {taxa_fn:.3f}")

    if taxa_fn < 0.25:
        print("✅ Baixa taxa de falsos negatius → millor detecció de jugadors amb estrès.")
    elif taxa_fn > 0.5:
        print("⚠️ Molts falsos negatius → el model ignora casos d’estrès.")
    if acc_train - acc_test > 0.05:
        print("⚠️ Possible sobreajustament: diferència gran entre entrenament i test.")

# 4. Gràfica comparativa
labels = [r['Model'] for r in resultats_opt]
accs = [r['Test Acc'] for r in resultats_opt]
fps = [r['FP Rate'] for r in resultats_opt]
fns = [r['FN Rate'] for r in resultats_opt]

plt.figure(figsize=(10, 6))
plt.plot(labels, accs, marker='o', label='Test Accuracy')
plt.plot(labels, fps, marker='o', label='FP Rate')
plt.plot(labels, fns, marker='o', label='FN Rate')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.title("Comparació de rendiment entre configuracions")
plt.xlabel("Configuració de l'arbre")
plt.ylabel("Valor (0-1)")
plt.legend()
plt.tight_layout()
plt.show()
