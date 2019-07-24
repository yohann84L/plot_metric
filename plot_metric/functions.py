import matplotlib.pyplot as plt
from numpy import newaxis, arange, argmin, unique, concatenate, zeros_like, argmax, linspace
from scipy import interp
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_curve, average_precision_score
from itertools import product, cycle
import colorlover as cl
import random
from statistics import mean
import seaborn as sns
from pprint import pprint
import pandas as pd

sns.set_style('darkgrid')


class BinaryClassification:
    __param_precision_recall_curve = {'threshold': None,
                                    'plot_threshold': True,
                                    'beta': 1,
                                    'linewidth': 2,
                                    'fscore_iso': [0.2, 0.4, 0.6, 0.8],
                                    'iso_alpha': 0.7,
                                    'y_text_margin': 0.03,
                                    'x_text_margin': 0.2,
                                    'c_pr_curve': 'black',
                                    'c_mean_prec': 'red',
                                    'c_thresh': 'black',
                                    'c_f1_iso': 'grey',
                                    'c_thresh_point': 'red',
                                    'ls_pr_curve': '-',
                                    'ls_mean_prec': '--',
                                    'ls_thresh': ':',
                                    'ls_fscore_iso': ':',
                                    'marker_pr_curve': None}

    def __init__(self, y_true, y_pred, labels, threshold=0.5):
        '''Constructor of the class'''
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels
        self.threshold = threshold

    def get_function_parameters(self, function, as_df=False):
        """
        Function to get all available parameters for a given function.
`

        Parameters
        -------
        :param function: func
            Function parameter's wanted.
        :param as_df: boolean, default=False
            Set to True to return a dataframe with parameters instead of dictionnary.

        Returns
        -------
        :return: dict,
            Dictionnary containing parameters for the given function and their default value.
        """
        if function.__name__ is "plot_precision_recall_curve":
            if as_df:
                return pd.DataFrame.from_dict(self.__param_precision_recall_curve, orient='index')
            else:
                return self.__param_precision_recall_curve

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

    param = {

    }

    def plot_roc(self, threshold=None, plot_threshold=True, linewidth=2, y_text_margin=0.05, x_text_margin=0.3):
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
        # Plot threshold
        if plot_threshold:
            # Plot vertical and horizontal line
            plt.axhline(y=idy_thresh, color='black', linestyle=':', lw=linewidth)
            plt.axvline(x=idx_thresh, color='black', linestyle=':', lw=linewidth)

            # Plot text threshold
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

            # Plot redpoint of threshold on the ROC curve
            plt.plot(idx_thresh, idy_thresh, 'ro')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")


    def plot_precision_recall_curve(self, threshold=None, plot_threshold=True, beta=1,
                                    linewidth=2, fscore_iso=[0.2, 0.4, 0.6, 0.8], iso_alpha=0.7,
                                    y_text_margin=0.03, x_text_margin=0.2,
                                    c_pr_curve='black', c_mean_prec='red', c_thresh='black', c_f1_iso='grey', c_thresh_point='red',
                                    ls_pr_curve='-', ls_mean_prec='--', ls_thresh=':', ls_fscore_iso=':',
                                    marker_pr_curve=None):
        """
        Compute and plot the precision-recall curve.

        Note : this implementation is restricted to binary classification only.
        See MultiClassClassification for multi-classes implementation.

        F1-iso are curve where a given f1-score is constant.

        We also consider the use of F_beta-score, change the parameter beta to use an other f-score.
        "Two other commonly used F measures are the F_2 measure, which weighs recall higher than
        precision (by placing more emphasis on false negatives), and the F_0.5 measure, which weighs
        recall lower than precision (by attenuating the influence of false negatives). (Wiki)"

        Parameters
        ----------
        :param threshold: float, default=0.5
            Threshold to determnine the rate between positive and negative values of the classification.
        :param plot_threshold: boolean, default=True
            Plot or not precision and recall lines for the given threshold.
        :param beta: float, default=1,
            Set beta to another float to use a different f_beta score. See definition of f_beta-score
            for more information : https://en.wikipedia.org/wiki/F1_score
        :param linewidth: float, default=2
        :param fscore_iso: array, list, default=[0.2, 0.4, 0.6, 0.8]
            List of float f1-score. Set to None or empty list to remove plotting of iso.
        :param iso_alpha: float, default=0.7
            Transparency of iso-f1.
        :param y_text_margin: float, default=0.03
            Margin (y) of text threshold.
        :param x_text_margin: float, default=0.2
            Margin (x) of text threshold.
        :param c_pr_curve: string, default='black'
            Define the color of precision-recall curve.
        :param c_mean_prec: string, default='red'
            Define the color of mean precision line.
        :param c_thresh: string, default='black'
            Define the color of threshold lines.
        :param c_f1_iso: string, default='grey'
            Define the color of iso-f1 curve.
        :param c_thresh_point: string, default='red'
            Define the color of threshold point.
        :param ls_pr_curve: string, default='-'
            Define the linestyle of precision-recall curve.
        :param ls_mean_prec: string, default='--'
            Define the linestyle of mean precision line.
        :param ls_thresh: string, default=':'
            Define the linestyle of threshold lines.
        :param ls_fscore_iso: string, default=':'
            Define the linestyle of iso-f1 curve.
        :param marker_pr_curve: string, default=None
            Define the marker of precision-recall curve.
        """

        # Set f1-iso and threshold parameters
        if fscore_iso is None:
            fscore_iso = []
        if threshold is None:
            t = self.threshold
        else:
            t = threshold

        # List for legends
        lines, labels = [], []

        # Compute precision and recall
        prec, recall, thresh = precision_recall_curve(self.y_true, self.y_pred)
        # Compute area
        pr_auc = average_precision_score(self.y_true, self.y_pred)
        # Compute the y & x axis to trace the threshold
        idx_thresh, idy_thresh = recall[argmin(abs(thresh - t))], prec[argmin(abs(thresh - t))]

        # Plot PR curve
        l, = plt.plot(recall, prec, color=c_pr_curve, lw=linewidth, linestyle=ls_pr_curve, marker=marker_pr_curve)
        lines.append(l)
        labels.append('PR curve (area = {})'.format(round(pr_auc, 2)))

        # Plot mean precision
        l, = plt.plot([0, 1], [mean(prec), mean(prec)], color=c_mean_prec,
                      lw=linewidth, linestyle=ls_mean_prec)
        lines.append(l)
        labels.append('Mean precision = {}'.format(round(mean(prec), 2)))

        # Fscore-iso
        if len(fscore_iso) > 0: # Check to plot or not the fscore-iso
            for f_score in fscore_iso:
                x = linspace(0.005, 1, 100) # Set x range
                y = f_score * x / (beta**2 * x + x - beta**2 * f_score) # Compute fscore-iso using f-score formula
                l, = plt.plot(x[y >= 0], y[y >= 0], color=c_f1_iso,linestyle=ls_fscore_iso,
                                                    alpha=iso_alpha)
                plt.text(s='f{:s}={:0.1f}'.format(str(beta), f_score), x=0.9, y=y[-10] + 0.02, alpha=iso_alpha)
            lines.append(l)
            labels.append('iso-f{:s} curves'.format(str(beta)))

            # Set ylim to see entire iso and to avoid a max ylim really high
            plt.ylim([0.0, 1.05])

        # Plot threshold
        if plot_threshold:
            # Plot vertical and horizontal line
            plt.axhline(y=idy_thresh, color=c_thresh, linestyle=ls_thresh, lw=linewidth)
            plt.axvline(x=idx_thresh, color=c_thresh, linestyle=ls_thresh, lw=linewidth)

            # Plot text threshold
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

            # Plot redpoint of threshold on the ROC curve
            plt.plot(idx_thresh, idy_thresh, marker='o', color=c_thresh_point)

        # Axis and legends
        plt.xlim([0.0, 1.0])
        plt.legend(lines, labels)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if plot_threshold:
            plt.title('Precision and Recall Curve (Threshold = {})'.format(round(t, 2)))
        else:
            plt.title('Precision and Recall Curve')

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
        print(classification_report(self.y_true, y_pred_class, target_names=list(map(str, self.labels))))


class MultiClassClassification:
    def __init__(self, y_true, y_pred, labels, threshold=.5):
        '''Constructor of the class'''
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels
        self.threshold = threshold
        self.n_classes = len(labels)

    def __to_hex(self, scale):
        ''' converts scale of rgb or hsl strings to list of tuples with rgb integer values. ie,
            [ "rgb(255, 255, 255)", "rgb(255, 255, 255)", "rgb(255, 255, 255)" ] -->
            [ (255, 255, 255), (255, 255, 255), (255, 255, 255) ] '''
        numeric_scale = []
        for s in scale:
            s = s[s.find("(") + 1:s.find(")")].replace(' ', '').split(',')
            numeric_scale.append((float(s[0]), float(s[1]), float(s[2])))

        return ['#%02x%02x%02x' % tuple(map(int, s)) for s in numeric_scale]

    def plot_roc(self, threshold=None, linewidth=2, show_threshold=False):
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        if threshold is None:
            t = self.threshold
        else:
            t = threshold

        # Binarize the output
        y = label_binarize(self.y_true, classes=self.labels)

        # Compute ROC curve and ROC area for each class
        fpr, tpr = dict(), dict()
        roc_auc = dict()
        idx_thresh, idy_thresh = dict(), dict()
        for i in range(self.n_classes):
            fpr[i], tpr[i], thresh = roc_curve(y[:, i], self.y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute the y & x axis to trace the threshold
            idx_thresh[i], idy_thresh[i] = fpr[i][argmin(abs(thresh - t))], tpr[i][argmin(abs(thresh - t))]

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], thresh = roc_curve(y.ravel(), self.y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        idx_thresh["micro"] = fpr["micro"][argmin(abs(thresh - t))]
        idy_thresh["micro"] = tpr["micro"][argmin(abs(thresh - t))]

        # Aggregate all false positive rates
        all_fpr = unique(concatenate([fpr[i] for i in range(self.n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Average it and compute AUC
        mean_tpr /= self.n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=linewidth)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=linewidth)

        random.seed(124)
        colors = self.__to_hex(cl.scales['6']['qual']['Set1'])
        random.shuffle(colors)
        for i, color in zip(range(self.n_classes), cycle(colors)):
            plt.plot(fpr[i], tpr[i], color=color, lw=linewidth, alpha=.5,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        if show_threshold:
            plt.plot(idx_thresh.values(), idy_thresh.values(), 'ro')

        plt.plot([0, 1], [0, 1], 'k--', lw=linewidth)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for multi-class (One-Vs-All')
        plt.legend(loc="lower right")

    def plot_confusion_matrix(self, normalize=False, title='Confusion matrix', cmap=plt.cm.Reds):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        # Define the confusion matrix
        y_pred_class = argmax(self.y_pred, axis=1)
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

    def print_report(self):
        from sklearn.metrics import classification_report

        y_pred_class = argmax(self.y_pred, axis=1)

        print("                   ________________________")
        print("                  |  Classification Report |")
        print("                   ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")
        print(classification_report(self.y_true, y_pred_class, target_names=list(map(str, self.labels))))