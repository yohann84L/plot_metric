import matplotlib.pyplot as plt
from numpy import newaxis, arange, argmin
from sklearn.metrics import confusion_matrix
from itertools import product

import seaborn as sns

sns.set_style('darkgrid')


class BinaryClassification:
    def __init__(self, y_true, y_pred, labels, threshold=0.5):
        '''Constructor of the class'''
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels
        self.threshold = threshold

    def plot_confusion_matrix(self, threshold=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Reds):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if threshold is None:
            t = self.threshold
        else:
            t = threshold

        # Define the confusion matrix
        y_pred_class = [1 if y_i > t else 0 for y_i in self.y_pred]
        cm = confusion_matrix(self.y_true, y_pred_class, labels=self.labels)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, newaxis]
            title = title + ' normalized'

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = arange(len(self.labels))
        plt.xticks(tick_marks, self.labels, rotation=45)
        plt.yticks(tick_marks, self.labels)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def plot_both_matrix(self, threshold=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Reds):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if threshold is None:
            t = self.threshold
        else:
            t = threshold

        # Define the confusion matrix
        y_pred_class = [1 if y_i > t else 0 for y_i in self.y_pred]
        cm = confusion_matrix(self.y_true, y_pred_class, labels=self.labels)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, newaxis]
            title = title + ' normalized'

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = arange(len(self.labels))
        plt.xticks(tick_marks, self.labels, rotation=45)
        plt.yticks(tick_marks, self.labels)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def plot_roc(self, threshold=None, linewidth=2, y_text_margin=0.05, x_text_margin=0.3):
        from sklearn.metrics import roc_curve, auc

        if threshold is None:
            t = self.threshold
        else:
            t = threshold

        # Compute ROC Curve
        fpr, tpr, thresh = roc_curve(self.y_true, self.y_pred)
        roc_auc = auc(fpr, tpr)

        # Compute the y & x axis to trace the threshold
        idx_thresh, idy_thresh = fpr[argmin(abs(thresh - t))], tpr[argmin(abs(thresh - t))]

        # Plot roc curve
        plt.plot(fpr, tpr, color='black',
                 lw=linewidth, label='ROC curve (area = %0.2f)' % roc_auc)

        # Plot reference line
        plt.plot([0, 1], [0, 1], color='red', lw=linewidth, linestyle='--')
        plt.axhline(y=idy_thresh, color='black', linestyle=':', lw=linewidth)
        plt.axvline(x=idx_thresh, color='black', linestyle=':', lw=linewidth)

        if idx_thresh > 0.5 and idy_thresh > 0.5:
            plt.text(x=idx_thresh - x_text_margin, y=idy_thresh - y_text_margin,
                     s='Threshold : {:.2f}'.format(t))
        elif idx_thresh <= 0.5 and idy_thresh <= 0.5:
            plt.text(x=idx_thresh + x_text_margin, y=idy_thresh + y_text_margin,
                     s='Threshold : {:.2f}'.format(t))
        elif idx_thresh <= 0.5 < idy_thresh:
            plt.text(x=idx_thresh + x_text_margin, y=idy_thresh - y_text_margin,
                     s='Threshold : {:.2f}'.format(t))
        elif idx_thresh > 0.5 >= idy_thresh:
            plt.text(x=idx_thresh - x_text_margin, y=idy_thresh + y_text_margin,
                     s='Threshold : {:.2f}'.format(t))

        plt.plot(idx_thresh, idy_thresh, 'ro')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

    def plot_class_distribution(self, threshold=None, alpha=.3, jitter=.3):
        from pandas import DataFrame

        if threshold is None:
            t = self.threshold
        else:
            t = threshold

        def _compute_thresh(row, _threshold):
            if (row['pred'] >= _threshold) & (row['class'] == 1):
                return "TP"
            elif (row['pred'] >= _threshold) & (row['class'] == 0):
                return 'FP'
            elif (row['pred'] < _threshold) & (row['class'] == 1):
                return 'FN'
            elif (row['pred'] < _threshold) & (row['class'] == 0):
                return 'TN'

        pred_df = DataFrame({'class': self.y_true,
                             'pred': self.y_pred})

        pred_df['type'] = pred_df['pred']
        pred_df['type'] = pred_df.apply(lambda x: _compute_thresh(x, t), axis=1)

        sns.set_palette(sns.color_palette("husl", 8))
        sns.violinplot(x='class', y='pred', data=pred_df, inner=None, color="white", cut=0)
        sns.stripplot(x='class', y='pred', hue='type', data=pred_df, jitter=jitter, alpha=alpha, size=4)
        plt.axhline(y=t, color='red')
        plt.title('Threshold at {:.2f}'.format(t))

    def print_report(self, threshold=.5):
        from sklearn.metrics import classification_report

        if threshold is None:
            t = self.threshold
        else:
            t = threshold

        y_pred_class = [1 if y_i > t else 0 for y_i in self.y_pred]

        print("                   ________________________")
        print("                  |  Classification Report |")
        print("                   ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")
        print(classification_report(self.y_true, y_pred_class, target_names=['0', '1']))
