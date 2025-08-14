import logging
import pickle
from abc import abstractmethod
from copy import deepcopy
from typing import List, Tuple

import pandas as pd
import shap
from aim import Run
from pycaret.regression import RegressionExperiment

from . import root_path
from .data_loader import TARGET_FEATURE, load_vo2_df, threshold_time_features

print = logging.getLogger(__name__).info

N_JOBS = 10


class Experiment:
    def __init__(self, time_list, drop_columns, filters, fold, version, log, file_name, hold_out_folder, hash, repo):
        self.time_list = time_list
        self.drop_columns = drop_columns
        self.filters = filters
        self.fold = fold
        self.version = version
        self.log = log
        self.file_name = file_name
        self.hold_out_folder = hold_out_folder
        self._hash = hash
        self.repo = repo
        if log:
            self.logger = self.get_run()
            self.id = self.logger.hash[:6]

    def log_info(self, *objects):
        if self.log:
            self.logger.log_info(*objects)
        logging.info(*objects)

    @abstractmethod
    def get_run(self):
        pass

    @abstractmethod
    def set_data(self, full_load=False):
        pass

    @abstractmethod
    def load_data(self, train: bool, drop_id: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass

    def close(self):
        if self.log:
            self.logger.close()
            del self.logger

    def get_context(self):
        return dict(time=''.join([str(t) for t in self.time_list]))


class VO2Experiment(Experiment):
    def __init__(
        self,
        group,
        time_list=[0],
        drop_columns=[],
        filters=[],
        fold=0,
        version=None,
        log=True,
        file_name='data/data.csv',
        hold_out_folder='data/split',
        hash=None,
        repo=None,
        full_load=False,
    ):
        """
        group: NG, OG, NG+OG
        drop_columns: list of columns to drop
        filters: list of filters to apply(filter: [column_name, operator, value])
        fold: 0-4(5-fold cross validation)
        version: version of the experiment
        """
        self.group = group
        self.full_load = full_load
        super().__init__(
            time_list=time_list,
            drop_columns=drop_columns,
            filters=filters,
            fold=fold,
            version=version,
            log=log,
            file_name=file_name,
            hold_out_folder=hold_out_folder,
            hash=hash,
            repo=repo,
        )
        if full_load:
            self.log_info(f'Experiment: {self.group}, Time: {self.time_list}')
        else:
            self.log_info(f'Experiment: {self.group}, Time: {self.time_list}, Fold: {self.fold}')

    def get_run(self):
        args = dict(
            group=self.group,
            drop_columns=self.drop_columns,
            filters=self.filters,
            full_load=self.full_load,
        )
        args = deepcopy(args)
        query = f'run.hparams == {args} and run.experiment == "{self.version}"'
        result = list(self.repo.query_runs(query).iter_runs())
        print(f'query: {query}, result: {result}, length: {len(result)}')
        if len(result) > 0:
            run = Run(result[0].run.hash, repo=self.repo, log_system_params=True)
        else:
            run = Run(repo=self.repo, experiment=self.version)
            run.name = f'{self.group}'
            if ['gender', '=', 0] in self.filters:
                run.name += ' M'
            elif ['gender', '=', 1] in self.filters:
                run.name += ' F'
            run['hparams'] = args

        file = root_path / 'results/hparams' / f'{run.hash[:6]}.pkl'
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, 'wb') as f:
            pickle.dump(args, f)
        return run

    def get_pycaret_exp(self) -> RegressionExperiment:
        exp = RegressionExperiment()
        exp.setup(
            data=self.train_X,
            test_data=None if self.full_load else self.test_X,
            target='target',
            session_id=self.fold,
            preprocess=False,
            experiment_name=self.group,
            feature_selection=True,
            log_experiment=False,
            remove_multicollinearity=True,
            transform_target=True,
            n_jobs=N_JOBS,
            index=False,
            fold=5,
        )
        self.log_info(
            f'Features: {len(exp.X_train_transformed.columns)} {", ".join(exp.X_train_transformed.columns.tolist())}'
        )
        return exp

    def fit_and_evaluate(self):
        exp, best_model = self.fit()
        self.evaluate(exp, best_model)

    def fit(self):
        self.set_data()
        if len(self.train_X) < 150:
            self.log_info('Not enough data to train')
            return None, None
        exp: RegressionExperiment = self.get_pycaret_exp()
        # best_model = exp.compare_models(include=['lr', 'ridge', 'br', 'gbr', 'lightgbm', 'rf'], n_select=1)
        if self.group == 'NG':
            model = ['br']
        else:
            model = ['lightgbm']
        best_model = exp.compare_models(include=model, n_select=1)
        if self.log:
            if self.full_load:
                best_model = exp.finalize_model(best_model)
            model_path = root_path / 'results/models' / f'{self.logger.hash[:6]}_{self.time_list}_{self.fold}'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            exp.save_model(best_model, model_path)
        return exp, best_model

    def evaluate(self, exp: RegressionExperiment, best_model):
        exp.predict_model(best_model)
        result = exp.pull()

        if self.log:
            run = self.logger
            run.track(result['R2'].values[0], name='R2', context=self.get_context(), step=self.fold)
            run.track(result['RMSE'].values[0], name='RMSE', context=self.get_context(), step=self.fold)
            run.track(result['MAE'].values[0], name='MAE', context=self.get_context(), step=self.fold)
            run.track(result['MSE'].values[0], name='MSE', context=self.get_context(), step=self.fold)
            model_name = exp._get_model_name(best_model)
            self.log_info(f'Best model of {self.get_context()} {self.fold}: {model_name}')

            if self.fold == 0:
                self.log_shap(best_model, exp, model_name)

    def log_shap(self, best_model, exp: RegressionExperiment, model_name):
        data = exp.X_train_transformed
        if model_name == 'Light Gradient Boosting Machine':
            shap_values = best_model.predict(data, pred_contrib=True)
            shap_values = shap_values[:, :-1]
            shap_values = shap.Explanation(shap_values, data=data, feature_names=data.columns)
        else:
            shap_values = shap.Explainer(best_model, data)(data).values
            shap_values = shap.Explanation(shap_values, data=data, feature_names=data.columns)

        with open(
            root_path
            / 'results/shap'
            / f'{self.logger.hash[:6]}_({"".join(map(str, self.time_list))})_{self.fold}.pkl',
            'wb',
        ) as f:
            pickle.dump(shap_values, f)

    @staticmethod
    def load_exp(hash: str, time_list: List[int], fold: int):
        with open(root_path / 'results/hparams' / f'{hash[:6]}.pkl', 'rb') as f:
            args = pickle.load(f)
        return VO2Experiment(**args, time_list=time_list, fold=fold, log=False, hash=hash)

    def load_data(self, train: bool, drop_id: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        scaler_id = (
            f'{self.logger.hash[:6]}_{self.fold}'
            if self.log
            else f'{self._hash[:6]}_{self.fold}'
            if self._hash
            else None
        )
        scaled_df, unscaled_df = load_vo2_df(
            train,
            group=self.group,
            drop_columns=deepcopy(self.drop_columns),
            filters=self.filters,
            fold=self.fold,
            file_name=self.file_name,
            hold_out_folder=self.hold_out_folder,
            scaler_id=scaler_id,
            full_load=self.full_load,
        )
        scaled_df = threshold_time_features(scaled_df, self.time_list)
        scaled_df.drop([*TARGET_FEATURE], axis=1, inplace=True)
        assert 'target' in scaled_df.columns
        if drop_id:
            scaled_df.drop(['id'], axis=1, inplace=True)

        return scaled_df, unscaled_df

    def set_data(self):
        train_X, train_X_raw = self.load_data(train=True)
        test_X, test_X_raw = self.load_data(train=False)
        setattr(self, 'train_X', train_X)
        setattr(self, 'test_X', test_X)
        setattr(self, 'train_X_raw', train_X_raw)
        setattr(self, 'test_X_raw', test_X_raw)
        self.log_info(f'Training data shape: {train_X.shape}, Testing data shape: {test_X.shape}')
        if 'target' in train_X.select_dtypes(include=['category', 'object']).columns:
            self.log_info(f'TARGET description (categorical):\n{train_X["target"].value_counts()}')
        else:
            self.log_info(f'TARGET description:\n{train_X["target"].describe()}')
