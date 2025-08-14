import logging

import pandas as pd
import yaml
from omegaconf import OmegaConf
from pycaret.regression import RegressionExperiment
from rich.logging import RichHandler

from utils.data_loader import change_column_data_type, load_scaler, threshold_time_features
from utils.exp import VO2Experiment

# logging with timestamps
logging.basicConfig(
    level=logging.INFO, format='%(message)s', datefmt='[%X]', handlers=[RichHandler(show_path=False, show_time=False)]
)
print = logging.info


def load_experiment(model_id, time_list):
    exp = VO2Experiment.load_exp(model_id, time_list, 0)
    caret = RegressionExperiment()
    model = caret.load_model(f'results/models/{model_id}_{time_list}_0', verbose=False)
    return caret, model, exp


if __name__ == '__main__':
    args = OmegaConf.create(dict(data='data/sample.csv'))
    data = pd.read_csv(args.data)
    print(f'Loaded data with shape: {data.shape}')
    with open('data/model_mapping.yml', 'r') as f:
        model_mapping = yaml.safe_load(f)

    group = int(input('Enter model group\n1. NG: Normal Group\n2. OG: Other Group\n3. NG+OG: Both\n: '))
    assert group in [1, 2, 3], f'Invalid group selection, expected 1, 2, or 3, got {group}'
    group = ['NG', 'OG', 'NG+OG'][group - 1]

    time_sel = int(
        input(
            'Enter time list\n\t1. Demographic\n\t2. Demographic + Rest\n\t3. Demographic + Rest + Submaximal\n\t4. Demographic + Rest + Submaximal + CI\n: '
        )
    )
    assert time_sel in [1, 2, 3, 4], f'Invalid time list selection, expected 1, 2, 3 or 4, got {time_sel}'
    time_list = [[0], [0, 1], [0, 1, 2], [0, 1, 2]][time_sel - 1]
    model_name = f'{group}_{time_list}{"_(-ci)" if time_sel == 3 else ""}'
    model_id = model_mapping.get(model_name)

    caret, model, exp = load_experiment(model_id, time_list)

    features = model._feature_names_in
    print(f'Loaded model {model_id} require features({len(features) - 1}): {features[:-1]}')

    data.rename(columns={'vo2pkg': 'target'}, inplace=True)  # Rename vo2max column to target
    X = change_column_data_type(data)
    X = threshold_time_features(X, time_list)
    scaler = load_scaler(scaler_id=f'{model_id}_0')
    X = scaler.tranform_df(X)

    prediction = caret.predict_model(model, data=X[features])
    prediction['pred'] = prediction['prediction_label']
    print(prediction[['target', 'pred']])
