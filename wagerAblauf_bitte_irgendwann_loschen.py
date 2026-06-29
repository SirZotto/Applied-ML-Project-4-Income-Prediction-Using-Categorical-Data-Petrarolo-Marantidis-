import numpy as np

# Aus eurer courselib:
from courselib.splits import k_fold_split
from courselib.normalization import standardize
from courselib.glm import LogisticRegression
from courselib.optimizers import GDOptimizer
from courselib.metrics import binary_accuracy, confusion_matrix

# Aus eurer neuen, autarken Datei:
from encoding import Target_Encoder_Fit, Target_Encoder_Transform, drop_rows

# 1. Daten vorbereiten (X und Y als numpy arrays generieren)
# ... (Hier ladet ihr das Adult-Dataset und splittet X und Y)

# 2. Cross Validation Setup (z.B. K=5)
X_folds, Y_folds = k_fold_split(X, Y, k=5)

for i in range(5):
    # Validierungs-Fold ist i, der Rest ist Training
    X_val, Y_val = X_folds[i], Y_folds[i]
    X_train = np.concatenate([X_folds[j] for j in range(5) if j != i])
    Y_train = np.concatenate([Y_folds[j] for j in range(5) if j != i])
    
    # --- LEAKAGE CONTROL: Target Encoding NUR auf Training fitten ---
    # (Wir nehmen an, Spalte 3 und 5 sind die kategorialen Features)
    maps = Target_Encoder_Fit(X_train, Y_train, columns_to_be_encoded=[3, 5], alpha=10)
    
    # Transformation auf Train UND Val anwenden
    X_train_encoded = Target_Encoder_Transform(X_train, columns_to_be_encoded=[3, 5], encoding_maps=maps)
    X_val_encoded = Target_Encoder_Transform(X_val, columns_to_be_encoded=[3, 5], encoding_maps=maps)
    
    # --- SKALIERUNG ---
    X_train_scaled = standardize(X_train_encoded)
    X_val_scaled = standardize(X_val_encoded)
    
    # --- MODELL TRAINING ---
    # Gewichte w (Länge = Anzahl Features) und b (Skalar) initialisieren
    w_init = np.zeros(X_train_scaled.shape[1])
    b_init = 0.0
    
    opt = GDOptimizer(learning_rate=0.01)
    model = LogisticRegression(w_init, b_init, opt, penalty="lasso", lam=0.1)
    
    # Das Modell nutzt die iterativen Epochen aus eurer base.py
    model.fit(X_train_scaled, Y_train, num_epochs=50, batch_size=128)
    
    # --- EVALUIERUNG ---
    predictions = model(X_val_scaled)
    acc = binary_accuracy(predictions, Y_val)
    print(f"Fold {i+1} Accuracy: {acc}%")