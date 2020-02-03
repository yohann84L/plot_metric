from .binary_classification import BinaryClassification
import matplotlib.pyplot as plt
import pandas as pd

### Parameters definition ###
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

__param_confusion_matrix = {'threshold': None,
                            'normalize': False,
                            'title': 'Confusion matrix',
                            'cmap': plt.cm.Reds,
                            'colorbar': True,
                            'label_rotation': 45}

__param_roc_curve = {'threshold': None,
                     'plot_threshold': True,
                     'linewidth': 2,
                     'y_text_margin': 0.05,
                     'x_text_margin': 0.2,
                     'c_roc_curve': 'black',
                     'c_random_guess': 'red',
                     'c_thresh_lines': 'black',
                     'ls_roc_curve': '-',
                     'ls_thresh_lines': ':',
                     'ls_random_guess': '--',
                     'title': 'Receiver Operating Characteristic',
                     'loc_legend': 'lower right'}

__param_class_distribution = {'threshold': None,
                              'display_prediction': True,
                              'alpha': .5,
                              'jitter': .3,
                              'pal_colors': None,
                              'display_violin': True,
                              'c_violin': 'white',
                              'strip_marker_size': 4,
                              'strip_lw_edge': None,
                              'strip_c_edge': None,
                              'ls_thresh_line': ':',
                              'c_thresh_line': 'red',
                              'lw_thresh_line': 2,
                              'title': None}

__param_threshold = {'threshold': None,
                     'beta': 1,
                     'title': None,
                     'annotation': True,
                     'bbox_dict': None,
                     'bbox': True,
                     'arrow_dict': None,
                     'arrow': True,
                     'plot_fscore': True,
                     'plot_recall': True,
                     'plot_prec': True,
                     'plot_fscore_max': True,
                     'c_recall_line': 'green',
                     'lw_recall_line': 2,
                     'ls_recall_line': '-',
                     'label_recall': 'Recall',
                     'marker_recall': '',
                     'c_prec_line ': 'blue',
                     'lw_prec_line': 2,
                     'ls_prec_line': '-',
                     'label_prec': 'Precision',
                     'marker_prec': '',
                     'c_fscr_line ': 'red',
                     'lw_fscr_line': 2,
                     'ls_fscr_line': '-',
                     'label_fscr': None,
                     'marker_fscr': '',
                     'marker_fscore_max': 'o',
                     'c_fscore_max': 'red',
                     'markersize_fscore_max': 5,
                     'plot_threshold': True,
                     'c_thresh_line': 'black',
                     'lw_thresh_line': 2,
                     'ls_thresh_line': '--',
                     'plot_best_threshold': True,
                     'c_bestthresh_line': 'black',
                     'lw_bestthresh_line': 1,
                     'ls_bestthresh_line': ':'}


def get_function_parameters(function, as_df=False):
    """
    Function to get all available parameters for a given function.

    Parameters
    ----------
    function : func
        Function parameter's wanted.
    as_df : boolean, default=False
        Set to True to return a dataframe with parameters instead of dictionnary.

    Returns
    -------
    param_dict : dict
        Dictionnary containing parameters for the given function and their default value.
    """

    if function.__name__ is "plot_precision_recall_curve":
        param_dict = __param_precision_recall_curve
    elif function.__name__ is "plot_confusion_matrix":
        param_dict = __param_confusion_matrix
    elif function.__name__ is "plot_roc_curve":
        param_dict = __param_roc_curve
    elif function.__name__ is "plot_class_distribution":
        param_dict = __param_class_distribution
    elif function.__name__ is "plot_threshold":
        param_dict = __param_threshold
    else:
        print("Wrong function given, following functions are available : ")
        for func in filter(lambda x: callable(x), BinaryClassification.__dict__.values()):
            print(func.__name__)
        return [func() for func in filter(lambda x: callable(x), BinaryClassification.__dict__.values())]

    if as_df:
        return pd.DataFrame.from_dict(param_dict, orient='index')
    else:
        return param_dict
