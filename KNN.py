# KNN.py
from math import sqrt
from collections import Counter
import pandas as pd

class KNN:

    # Constructor to hold all of the neccasry data
    def __init__(self, data, k, train_ratio, val_ratio, test_ratio, distance):
        self.k = k
        self.data = self._prep_from_dataframe(data) 
        self.train_data, self.val_data, self.test_data = self._train_val_test_split(
            train_ratio, val_ratio, test_ratio
        )
        self.distance = distance
        
    # Data prep so we can work with list of tuples
    def _prep_from_dataframe(self, df):
        df = df.sample(frac=1, random_state = 43).reset_index(drop=True)             #Shuffle data
        species_to_id = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}
        df["label"] = df["species"].map(species_to_id)
        return [([r.sepal_length, r.sepal_width, r.petal_length, r.petal_width], (r.label))     #Return list of tuples 
                for r in df.itertuples(index=False)]

    ##Split data into training, validation, and testing
    def _train_val_test_split(self, train_ratio, val_ratio, test_ratio):
        d = self.data[:]
        n = len(d) 
        n_train = int(train_ratio*n) 
        n_val = int(val_ratio*n)
        train = d[:n_train]
        val = d[n_train:n_train+n_val]
        test = d[n_train+n_val:]
        return train, val, test

    # Return only feature vectors for distance calculations
    @staticmethod
    def _features(point): return point[0]                            

    #Euclidean
    @staticmethod
    def euclidean_dist(p1, p2):
        x1, x2 = KNN._features(p1), KNN._features(p2)
        s = 0.0
        for a, b in zip(x1, x2):
            d = a - b
            s += d * d
        return sqrt(s)
    
    #Manhattan
    @staticmethod
    def manhattan_dist(p1,p2):
        x1, x2 = KNN._features(p1), KNN._features(p2)
        s = 0.0
        for a, b in zip(x1, x2):
            d = abs(a-b)
            s+=d
        return(s)
    
    # Sorts which distance we want to use
    def get_dist(self,test_point, train_data):
        if (self.distance == 'euclidean'):
            return self.euclidean_dist(test_point,train_data)
        elif (self.distance == 'manhattan'):
            return self.manhattan_dist(test_point,train_data)
        else: 
            print("Invalid distance choice! Please choose \'euclidean\'" \
            "or \'manhattan\' \n Program will crash shortly ....")
            return

    # Get distance for all neihgbors but only return K-Nearest 
    def get_neighbors(self, test_point, number_neighbors):
        k = number_neighbors
        dists = [ (tp, self.get_dist(test_point, tp)) for tp in self.train_data ]    # distances from test point to all train points
        dists.sort(key=lambda t: t[1])
        return [dists[i][0] for i in range(k)]

    ## Classification for a single point
    def predict_classification(self, test_point, number_neighbors):
        neighbors = self.get_neighbors(test_point, number_neighbors)  
        labels = [row[-1] for row in neighbors]  
        return Counter(labels).most_common(1)[0][0]
    
    ## Classification for whole test set
    def predict_all_classification(self, test_data, number_neighbors):
        preds = [self.predict_classification(tp, number_neighbors) for tp in test_data]
        return self._results_df(test_data, preds)

    #Validation Step for chosing the best K
    def choose_best_k(self, k_grid):
        best_k, best_acc = None, -1.0
        for k in k_grid:
            correct = 0
            for tp in self.val_data:
                pred = self.predict_classification(tp, k)
                correct += int(pred == tp[1])
            acc = correct / len(self.val_data)
            if acc > best_acc:
                best_k, best_acc = k, acc
        return best_k, best_acc

    # Return a data frame that is nice and neaty
    @staticmethod
    def _results_df(test_data, preds):
        rows = [{"test point": tuple(x), "label": int(y), "prediction": int(p)}
                for (x, y), p in zip(test_data, preds)]
        return pd.DataFrame(rows, columns=["test point","label","prediction"])
    
    # Run evaluation on entire testing set after validation 
    def evaluate(self, k): 
        results = self.predict_all_classification(self.test_data, k)
        return results, self.multiclass_analytics(results)
    

    # calculating precision, recall, and F1
    @staticmethod
    def _f1_from_counts(tp, fp, fn):
        p_den, r_den = tp + fp, tp + fn
        prec = tp / p_den if p_den else 0.0
        rec  = tp / r_den if r_den else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
        return prec, rec, f1

    def multiclass_analytics(self, results_df):
        y_true = results_df["label"].tolist()
        y_pred = results_df["prediction"].tolist()
        n = len(y_true)
        labels = sorted(set(y_true) | set(y_pred))
        acc = sum(int(t == p) for t, p in zip(y_true, y_pred)) / n
        per = {}
        tp_sum = fp_sum = fn_sum = 0
        for c in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
            cnt = sum(1 for t in y_true if t == c)
            tp_sum += tp; fp_sum += fp; fn_sum += fn
            pr_c, rc_c, f1_c = self._f1_from_counts(tp, fp, fn)
            per[c] = {"precision": pr_c, "recall": rc_c, "f1": f1_c, "count": cnt}

        # overall precision/recall
        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
        recall    = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 0.0

        # F1 macro & weighted
        L = len(labels)
        total = sum(v["count"] for v in per.values())
        f1_macro    = sum(v["f1"] for v in per.values()) / L
        f1_weighted = sum(v["f1"] * v["count"] for v in per.values()) / total

        return {
            "accuracy": acc,
            "precision": precision, 
            "recall": recall, 
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "per_class": per,
    }


