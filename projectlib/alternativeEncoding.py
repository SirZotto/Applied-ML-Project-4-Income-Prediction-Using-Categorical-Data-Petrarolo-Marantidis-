import pandas as pd
import numpy as np

def drop_rows(df, drop_these_data_rows):
    """
    Löscht alle Zeilen, die unerwünschte Werte (z.B. ['?']) in irgendeiner Spalte enthalten.
    Arbeitet rein vektoriell auf Basis von Pandas.
    """
    _df = df.copy()
    row_amount_before = len(_df)
    
    # Erstellt eine Maske über das gesamte DataFrame
    mask = _df.isin(drop_these_data_rows).any(axis=1)
    _df = _df[~mask].reset_index(drop=True)
    
    amount_of_rows_deleted = row_amount_before - len(_df)
    print(f"{amount_of_rows_deleted} out of {row_amount_before} rows were deleted.")
    return _df


def One_Hot_Encoder(df, columns_to_be_encoded, drop_first=True):
    """
    Erstellt ein One-Hot-Encoding für die angegebenen Spalten.
    
    Mathematische Definition: OneHot_k(x) = 1{x = c_k}
    
    Um die Dummy-Variable-Falle (Multikollinearität) bei der unregularisierten 
    logistischen Regression zu vermeiden, wird standardmäßig die erste Kategorie gedroppt.
    """
    _df = df.copy()
    
    for col in columns_to_be_encoded:
        # Einzigartige Kategorien in der Spalte bestimmen
        categories = _df[col].unique()
        
        # Um perfekte Multikollinearität zu verhindern (wichtig für lineare Modelle)
        if drop_first and len(categories) > 1:
            categories = categories[1:]
            
        for cat in categories:
            new_column_name = f"{col}_{cat}"
            # Mathematische Indikatorfunktion umsetzen
            _df[new_column_name] = (_df[col] == cat).astype(int)
            
        # Alte Textspalte entfernen
        _df = _df.drop(columns=[col])
        
    return _df


def Ordinal_Encoder(df, column_to_be_encoded, categories_order, starting_number=1, step_size=1):
    """
    Transformiert kategoriale Spalten anhand einer vordefinierten, sinnvollen Ordnung.
    Besonders wichtig für Features wie 'education' im Adult-Datensatz.
    
    Mathematische Definition: Ordinal(x = c_k) = rank(c_k)
    """
    _df = df.copy()
    
    if isinstance(column_to_be_encoded, str):
        column_to_be_encoded = [column_to_be_encoded]
        categories_order = [categories_order]
        
    for i, col in enumerate(column_to_be_encoded):
        current_order = categories_order[i]
        replacement_dict = {}
        current_value = starting_number
        
        for item in current_order:
            if isinstance(item, list):
                # Wenn Kategorien den gleichen Rang erhalten sollen
                for sub_item in item:
                    replacement_dict[sub_item] = current_value
            else:
                replacement_dict[item] = current_value
            current_value += step_size
            
        # Zuweisung der Ränge
        old_column = _df[col].copy()
        _df[col] = old_column.map(replacement_dict)
        
        # Validierung, ob alle Kategorien in der Reihenfolge abgedeckt wurden
        if _df[col].isna().any():
            missing = list(old_column[_df[col].isna()].unique())
            raise ValueError(f"Categories {missing} from column '{col}' not found in categories_order.")
            
    return _df


def Frequency_Encoder(df, columns_to_be_encoded):
    """
    Ersetzt Kategorien durch ihre relative Häufigkeit im Datensatz.
    
    Mathematische Definition: Freq(x = c_k) = count(x = c_k) / N
    """
    _df = df.copy()
    
    for col in columns_to_be_encoded:
        # Berechne relative Häufigkeiten (normalize=True teilt direkt durch N)
        freq_map = _df[col].value_counts(normalize=True).to_dict()
        
        new_column_name = f"{col}_freq"
        _df[new_column_name] = _df[col].map(freq_map)
        _df = _df.drop(columns=[col])
        
    return _df


# --- STRATEGISCHE TRENNUNG FÜR TARGET ENCODING (Verhindert Target Leakage) ---

def Target_Encoder_Fit(X_train, y_train, columns_to_be_encoded, alpha=0):
    """
    BERECHNET die Target-Statistiken aus den Trainingsdaten.
    Kann absolut autark aufgerufen werden und erzeugt KEIN Leakage auf Validierungsdaten.
    
    Mathematische Basisdefinition: Target(x = c_k) = E[y | x = c_k]
    Optionale Erweiterung (Smoothing): SmoothTarget(c) = (n_c * y_c + alpha * y_global) / (n_c + alpha)
    """
    encoding_maps = {}
    global_mean = y_train.mean()
    
    for col in columns_to_be_encoded:
        # Gruppierung nach Kategorien, um Anzahl (count) und Mittelwert (mean) zu bestimmen
        stats = pd.DataFrame({'target': y_train}).groupby(X_train[col])['target'].agg(['count', 'mean'])
        
        if alpha > 0:
            # Geglättete Variante gegen Ausreißer bei geringer Probenanzahl (High-Cardinality-Problem)
            smoothed_values = (stats['count'] * stats['mean'] + alpha * global_mean) / (stats['count'] + alpha)
            encoding_maps[col] = smoothed_values.to_dict()
        else:
            # Reine bedingte Erwartungswerte
            encoding_maps[col] = stats['mean'].to_dict()
            
        # Globalen Mittelwert als Fallback für unbekannte Kategorien im Testset mitspeichern
        encoding_maps[col]['_global_fallback_'] = global_mean
        
    return encoding_maps


def Target_Encoder_Transform(X, columns_to_be_encoded, encoding_maps):
    """
    WENDET die zuvor exklusiv gelernten Trainings-Statistiken auf ein beliebiges 
    Feature-DataFrame X an (sei es das Trainingsset selbst oder das Validierungs-/Testset).
    """
    _X = X.copy()
    
    for col in columns_to_be_encoded:
        if col not in encoding_maps:
            continue
            
        fallback = encoding_maps[col]['_global_fallback_']
        new_column_name = f"{col}_target"
        
        # Mappe die Werte; Kategorien, die im Fit nicht existierten, erhalten den globalen Durchschnitt
        _X[new_column_name] = _X[col].map(encoding_maps[col]).fillna(fallback)
        _X = _X.drop(columns=[col])
        
    return _X