import logging
from datetime import datetime
from pathlib import Path

import yaml
from aim import Repo
from omegaconf import OmegaConf
from rich.logging import RichHandler

from utils.exp import VO2Experiment

# Create log directory if it doesn't exist
log_dir = Path('log')
log_dir.mkdir(exist_ok=True)

# Create log file with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = log_dir / f'train_{timestamp}.log'

# Configure logging to both file and console
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = RichHandler(show_path=False)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

logging.basicConfig(
    level=logging.INFO,
    datefmt='[%X]',
    handlers=[console_handler, file_handler],
)

logger = logging.getLogger(__name__)
logger.info(f'Logging to {log_file}')

repo = Repo('.')

default_filters = [['majdysrh', '=', 0], ['myocisch', '=', 0]]
# Default filters for neither major dysrhythmia nor myocardial ischemia
model_mapping = {}


def run_exp(group, time_list, filters, fold, args, full_load, drop_ci=False):
    drop_columns = args.drop_columns.copy()
    if drop_ci and 'ci' not in drop_columns:
        drop_columns.append('ci')
    experiment = VO2Experiment(
        group=group,
        time_list=time_list,
        filters=filters,
        fold=fold,
        version=args.version,
        drop_columns=drop_columns,
        repo=repo,
        full_load=full_load,
    )
    experiment.fit_and_evaluate()
    experiment.close()
    model_mapping[f'{group}_{time_list}{"_(-ci)" if drop_ci else ""}'] = experiment.id


def main():
    """
    Arguments:
    - filters: List of filters to apply to the data. (ex. ['majdysrh', '=', 0])
    - version: Version of the experiment to run
    - drop_columns: List of columns to drop from the dataset. (ex. ['ci'])
    - gender_group: 'all', 'each' - whether to run the experiment for all or each gender separately.
    - full_load: Whether to load the entire dataset or hold out data for cross-validation.
    - time_list List of features to include (0: Demographic, 1: Rest, 2: Submaximal) (ex. [[0], [0, 1], [0, 1, 2]])
    """

    args = OmegaConf.create(
        dict(
            filters=[],
            version='default',
            drop_columns=[],
            gender_group='all',
            full_load=False,
            time_list=[[0], [0, 1], [0, 1, 2]],
        )
    )
    args.merge_with_cli()
    assert args.gender_group in ['all', 'each']
    full_load = args.full_load
    logger.info(args)
    groups = ['NG', 'OG', 'NG+OG']
    filter = default_filters + args.filters

    if args.gender_group == 'all':
        for group in groups:
            for time in args.time_list:  # 0: Demographic, 1: Rest, 2: Submaximal
                if full_load:
                    run_exp(group, time, filter, 0, args, full_load=full_load)
                    if time == [0, 1, 2]:  # If all time features are included, also run without CI
                        run_exp(group, time, filter, 0, args, full_load=full_load, drop_ci=True)
                else:
                    for fold in range(5):  # 5-fold cross validation
                        run_exp(group, time, filter, fold, args, full_load=full_load)
                        if time == [0, 1, 2]:  # If all time features are included, also run without CI
                            run_exp(group, time, filter, fold, args, full_load=full_load, drop_ci=True)
    else:
        for gender_filter in [['gender', '=', 0], ['gender', '=', 1]]:
            nf = filter + [gender_filter]
            for group in groups:
                for time in args.time_list:
                    if full_load:
                        run_exp(group, time, nf, 0, args, full_load=full_load)
                    else:
                        for fold in range(5):
                            run_exp(group, time, nf, fold, args, full_load=full_load)

    if full_load:
        with open('data/model_mapping.yml', 'w') as f:
            yaml.dump(model_mapping, f, default_flow_style=False)


if __name__ == '__main__':
    main()
