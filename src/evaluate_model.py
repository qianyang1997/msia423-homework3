from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import pandas as pd

import logging

logger3 = logging.getLogger(__name__)


def evaluation(y_test: pd.DataFrame, pred_df: pd.DataFrame, output_path: str) -> None:
    """evaluate model performance on test set.

    :param y_test: :obj: pandas dataframe of actual y's
    :param pred_df: pandas dataframe of predicted y's
    :param output_path: filepath of output file
    :return: None
    """
    ypred_proba_test = pred_df['ypred_proba'].values
    ypred_bin_test = pred_df['ypred_bin'].values

    auc = roc_auc_score(y_test, ypred_proba_test)
    confusion = confusion_matrix(y_test, ypred_bin_test)
    accuracy = accuracy_score(y_test, ypred_bin_test)

    confusion_df = pd.DataFrame(confusion,
                                index=['Actual negative', 'Actual positive'],
                                columns=['Predicted negative', 'Predicted positive'])

    with open(output_path, 'w') as file:
        file.write(f'Model evaluation:\n'
                   f'Accuracy: {accuracy}'
                   f'AUC: {auc}\n'
                   f'Confusion matrix: {confusion_df}')

    print('AUC on test: %0.3f' % auc)
    print('Accuracy on test: %0.3f' % accuracy)
    print()
    print(confusion_df)
