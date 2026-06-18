import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 1. Overall Accuracy (same for all averaging modes)
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = float(correct_predictions / len(y_true)) if len(y_true) > 0 else 0.0
    
    # 2. Extract unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if average == "binary":
        # Binary mode: treat pos_label as the only positive class
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        
    elif average == "micro":
        # Micro mode: aggregate global TP, FP, FN across all classes
        global_tp = 0
        global_fp = 0
        global_fn = 0
        
        for c in classes:
            global_tp += np.sum((y_true == c) & (y_pred == c))
            global_fp += np.sum((y_true != c) & (y_pred == c))
            global_fn += np.sum((y_true == c) & (y_pred != c))
            
        precision = float(global_tp / (global_tp + global_fp)) if (global_tp + global_fp) > 0 else 0.0
        recall = float(global_tp / (global_tp + global_fn)) if (global_tp + global_fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        
    else:
        # Macro and Weighted modes: compute per-class metrics first
        per_class_precision = []
        per_class_recall = []
        per_class_f1 = []
        supports = []
        
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            support = np.sum(y_true == c)
            
            p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            
            per_class_precision.append(p)
            per_class_recall.append(r)
            per_class_f1.append(f)
            supports.append(support)
            
        per_class_precision = np.array(per_class_precision)
        per_class_recall = np.array(per_class_recall)
        per_class_f1 = np.array(per_class_f1)
        supports = np.array(supports)
        
        if average == "macro":
            precision = float(np.mean(per_class_precision))
            recall = float(np.mean(per_class_recall))
            f1 = float(np.mean(per_class_f1))
        elif average == "weighted":
            total_support = np.sum(supports)
            if total_support > 0:
                precision = float(np.sum(per_class_precision * supports) / total_support)
                recall = float(np.sum(per_class_recall * supports) / total_support)
                f1 = float(np.sum(per_class_f1 * supports) / total_support)
            else:
                precision, recall, f1 = 0.0, 0.0, 0.0
                
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}