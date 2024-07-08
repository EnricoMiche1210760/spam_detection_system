import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def print_table_from_classification_report(title, report):
    print('\033[94m' + title+ '\x1b[0m')
    print('\033[1m' + '\t\tprecision    recall  f1-score   support'+ '\033[0m')
    for line in report.split('\n')[2:-5]:
        print(line)
    print('\033[92m'+ "\t-------------------------------------------------"+'\x1b[0m')
    print('\n')

def print_confusion_matrix(y_test, y_pred, title:str = None):
    '''
    Function to print the confusion matrix
    Parameters:
        y_test: np.array, true labels
        y_pred: np.array, predicted labels
        title: string, plot title
    '''
    _, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", ax=ax)
    print(title)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Predicted severity")
    ax.set_ylabel("Severity")
    ax.set_xticklabels(labels=['Pred Ham', 'Pred Spam'])
    ax.set_yticklabels(labels=['Ham', 'Spam'])

    plt.subplots_adjust(hspace=0.3)
    plt.show();
