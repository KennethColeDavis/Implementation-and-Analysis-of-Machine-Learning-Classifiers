import pandas as pd
class analytics:
    def __init__(self, results):
        self.results = results
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        rows_iter = ((int(r['label']), int(r['prediction'])) for _, r in results.iterrows())
        for true, pred in rows_iter:
            if true == 1 and pred == 1:
                self.tp += 1
            elif true == 0 and pred == 0:
                self.tn += 1
            elif true == 0 and pred == 1:
                self.fp += 1
            elif true == 1 and pred == 0:
                self.fn += 1

        # total points / examples
        self.total_points = self.tp + self.tn + self.fp + self.fn

    def confusion_matrix(self):
        data = [
            [self.tp, self.fp],   # Predicted Positive row
            [self.fn, self.tn],   # Predicted Negative row
        ]
        index = ["Predicted Positive (1)", "Predicted Negative (0)"]
        columns = ["Actually Positive (1)", "Actually Negative (0)"]
        return pd.DataFrame(data, index=index, columns=columns)

    def get_accuracy(self):
        if (self.total_points == 0): return 0.00
        else: 
            return(self.tp + self.tn) / self.total_points
        
    def get_precision(self):
        if ((self.tp +self.fp) == 0): return 0.00
        else:
            return (self.tp)/(self.tp + self.fp)
    
    def get_recall(self):
        if ((self.tp +self.fn) == 0): return 0.00
        else:
            return (self.tp)/(self.tp +self.fn)
    
    def get_f1_score(self):
        if ((self.get_recall() + self.get_precision()) == 0): return 0.00
        else:
            return (2 *(self.get_recall() * self.get_precision()))/(self.get_recall() + self.get_precision() )
        
    def separate_classes(self):

        class1 = {'TP': 0, 'FP': 0, 'FN': 0, 'count': 0}  # actual 1s
        class0 = {'TP': 0, 'FP': 0, 'FN': 0, 'count': 0}  # actual 0s


        for row in self.results.itertuples(index=False, name="Row"):
            actual_label = int(row.label)
            predicted_label = int(row.prediction)

            if actual_label == 1:
                class1['count'] += 1
                if predicted_label == 1:
                    class1['TP'] += 1
                else:
                    class1['FN'] += 1
                    class0['FP'] += 1
            else:  # actual_label == 0
                class0['count'] += 1
                if predicted_label == 0:
                    class0['TP'] += 1
                else:
                    class0['FN'] += 1
                    class1['FP'] += 1
        return class1, class0

    @staticmethod
    def _f1_from_counts(tp, fp, fn):
        p_den = tp + fp
        r_den = tp + fn
        precision = tp / p_den if p_den else 0.0
        recall    = tp / r_den if r_den else 0.0
        return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    def get_f1_macro(self):
        class1, class0 = self.separate_classes()
        f1_1 = self._f1_from_counts(class1['TP'], class1['FP'], class1['FN'])
        f1_0 = self._f1_from_counts(class0['TP'], class0['FP'], class0['FN'])
        return (f1_1 + f1_0) / 2.0

    def get_f1_weighted(self):
        class1, class0 = self.separate_classes()
        f1_1 = self._f1_from_counts(class1['TP'], class1['FP'], class1['FN'])
        f1_0 = self._f1_from_counts(class0['TP'], class0['FP'], class0['FN'])
        total = (class1['count'] + class0['count']) or 1
        return (f1_1 * class1['count'] + f1_0 * class0['count']) / total
