import numpy as np
#TP = true positives
#FP = false positives
#FN = false negatives

# precision formula is given by: TP / ( TP + FP) 
def binary_precision(y_pred,y_true, threshold=0.5, labels = [1,0]):
    """Precision function for binary classification models, it follows the formula TP / (TP + FP) where 
        TP = true positives
        FP = false positives
    """
    first_label = labels[0]
    second_label = labels[1]
    
    TP = np.sum((y_pred == first_label) & (y_true == first_label))
    FP = np.sum((y_pred == first_label) & (y_true == second_label))
    
    if TP + FP == 0:
        return 0.0
    
    return TP / (TP + FP)

# recall formula is given by: TP / ( TP + FP) 
def binary_recall(y_pred,y_true, threshold=0.5, labels = [1,0]):
    """Recall function for binary classification models it follows the formula TP / (TP + FN) where 
        TP = true positives
        FN = false negatives
    """
    first_label = labels[0]
    second_label = labels[1]
    
    TP = np.sum((y_pred == first_label) & (y_true == first_label))
    FN = np.sum((y_pred == second_label) & (y_true == first_label))
    
    if TP + FN == 0:
        return 0.0
    
    return TP / (TP + FN)

