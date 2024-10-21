import copy
import math
import os
import pickle
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

# PyTorch
import torch
from optuna.trial import TrialState
from sklearn.model_selection import KFold
from tqdm import tqdm

from tidy3d.plugins.design.surrogate_model_funcs import (
    train_network,
    validate_regression,
)

# Local
from tidy3d.plugins.design.surrogate_preprocess_funcs import (
    Dummy_Scaler,
    pytorch_load,
    save_scalers,
    scale_feature,
    scale_label,
    split_data,
)


def _build_directory(dir_path):
    dir_path = Path(dir_path)
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        print(f"Using existing directory at {dir_path.name}")

    return dir_path


class AI_Model:
    def __init__(self, output_dir, data_dir: list, seed=None) -> None:
        """Initialize the space building directories and finding the input models

        Current state:
        Only regression problems

        Ideas:
        Functions to take pd.DF inputs
        Investigate @property dec relevance here
        """
        self.output_dir = Path(output_dir)
        self.pickle_dir = _build_directory(self.output_dir / "pickle_data")
        self.model_dir = _build_directory(self.output_dir / "models")

        # Get data files
        # Separate as data will exist before building model
        self.data_dir = data_dir

        self.hdf5_files = {}
        for data in data_dir:
            file_list = [
                file_name
                for file_name in os.listdir(data)
                if ".hdf5" in file_name and file_name != "batch.hdf5"
            ]
            self.hdf5_files[data] = file_list

        # Setup device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device {self.device}")

        # Set seeds
        self.seed = seed
        np.random.seed(seed)

        # Torch cannot cope with None seed like numpy can - don't set seed so Torch is random
        if seed is not None:
            torch.cuda.empty_cache()
            torch.manual_seed(seed)

    def load_data_from_hdf5(
        self,
        fn_data,
        test_percentage,
        valid_percentage,
        batch_size,
        individual_feature_scaling: bool,
        scale_inputs=True,
        pickle_name=None,
        fn_data_kwargs=(),
    ):
        """Format the data used for an AI model form HDF5 files"""

        # Checks
        # Percentages don't equal > 1

        if pickle_name is not None:
            # Convenience function for if .pkl not in name
            if pickle_name[-4:] != ".pkl":
                pickle_name = pickle_name + ".pkl"

            try:
                os.chdir(self.pickle_dir)
                with open(pickle_name, "rb") as d:
                    raw_features, raw_labels = pickle.load(d)
                print("Loaded data from pickle")
            except FileNotFoundError:
                print("Calculating data from scratch")
                raw_features = []
                raw_labels = []
                for directory_path, file_list in self.hdf5_files.items():
                    os.chdir(directory_path)
                    features, labels = fn_data(hdf5_files=file_list, **fn_data_kwargs)

                    labels = labels.reshape(-1, 1)  # NEED TO CHANGE!
                    raw_features.append(features)
                    raw_labels.append(labels)

                # Assemble feature and label arrays into one
                raw_features = np.vstack(raw_features)
                raw_labels = np.vstack(raw_labels)

                print(f"Saving data to {pickle_name}")
                os.chdir(self.pickle_dir)
                with open(pickle_name, "wb") as file:
                    pickle.dump((raw_features, raw_labels), file)

        else:
            raw_features, raw_labels = fn_data(self.hdf5_files)

        self._prep_data(
            raw_features,
            raw_labels,
            test_percentage,
            valid_percentage,
            batch_size,
            individual_feature_scaling,
            scale_inputs,
        )

    def load_data_from_df(
        self,
        df: pd.DataFrame,
        label_names: list[str],
        feature_names: list[str],
        test_percentage: float,
        valid_percentage: float,
        batch_size: int,
        individual_feature_scaling: bool,
        scale_inputs: bool = True,
        label_as_array: bool = False,
    ):
        if label_as_array:
            # For single label being supplied where the label is an array
            raw_labels = df.loc[:, label_names].values
            raw_labels = np.vstack(raw_labels[:, 0])
        else:
            raw_labels = df.loc[:, label_names].values.reshape(-1, len(label_names))

        raw_features = df.loc[:, feature_names].values

        self._prep_data(
            raw_features,
            raw_labels,
            test_percentage,
            valid_percentage,
            batch_size,
            individual_feature_scaling,
            scale_inputs,
        )

    def _prep_data(
        self,
        raw_features,
        raw_labels,
        test_percentage,
        valid_percentage,
        batch_size,
        individual_feature_scaling,
        scale_inputs,
    ):
        # Randomise labels for sanity check
        # if random_labels:
        #     raw_labels = np.random.uniform(raw_labels.min(), raw_labels.max(), size=raw_labels.shape)

        # Shuffle and split data
        shuffle_index = np.random.permutation(len(raw_features))
        shuffle_features = raw_features[shuffle_index]
        shuffle_labels = raw_labels[shuffle_index]

        train_features, test_features, valid_features = split_data(
            shuffle_features, test_percentage, valid_percentage
        )
        train_labels, test_labels, valid_labels = split_data(
            shuffle_labels, test_percentage, valid_percentage
        )

        if scale_inputs:
            # Scale labels
            scaled_train_labels, label_scaler = scale_label(train_labels)
            scaled_test_labels = scale_label(test_labels, label_scaler)
            scaled_valid_labels = scale_label(valid_labels, label_scaler)

            # Scale features
            scaled_train_features, feature_scaler = scale_feature(
                train_features, individual_feature_scaling
            )
            scaled_test_features = scale_feature(
                test_features, individual_feature_scaling, feature_scaler=feature_scaler
            )
            scaled_valid_features = scale_feature(
                valid_features, individual_feature_scaling, feature_scaler=feature_scaler
            )
        else:
            print("Inputs are not being scaled - this is not recommended")

            scaled_train_labels = train_labels
            scaled_test_labels = test_labels
            scaled_valid_labels = valid_labels
            scaled_train_features = train_features
            scaled_test_features = test_features
            scaled_valid_features = valid_features
            label_scaler = Dummy_Scaler()
            feature_scaler = Dummy_Scaler()

        # Load data into PyTorch DataLoaders
        train_loaded = pytorch_load(scaled_train_features, scaled_train_labels, batch_size)
        test_loaded = pytorch_load(scaled_test_features, scaled_test_labels, batch_size)
        valid_loaded = pytorch_load(scaled_valid_features, scaled_valid_labels, batch_size)

        # Store key objects in the class
        self.label_scaler = label_scaler
        self.feature_scaler = feature_scaler
        self.train_loaded = train_loaded
        self.test_loaded = test_loaded
        self.valid_loaded = valid_loaded

        # Kept for analysis
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.valid_labels = valid_labels
        self.scaled_test_labels = scaled_test_labels
        self.scaled_valid_labels = scaled_valid_labels

        # Kept for k-fold training
        self.raw_features = raw_features
        self.raw_labels = raw_labels

    def train_model(
        self,
        model_name,
        network_model,
        optimizer,
        loss_function,
        epochs: int,
        plot_output: bool,
        verbose: bool = True,
    ):
        network_model.to(self.device)
        if model_name is not None:
            current_model_dir = _build_directory(self.model_dir / model_name)
            save_scalers(current_model_dir, self.label_scaler, self.feature_scaler)

            # Convenience function for if .pt not in model name
            if model_name[-3:] != ".pt":
                model_name = model_name + ".pt"

        else:
            print("Making temp model folder")
            current_model_dir = _build_directory(self.model_dir / "temp")
            model_name = "temp"

        trained_network = copy.deepcopy(network_model)

        with tqdm(total=epochs) as pbar:
            loss, _ = train_network(
                network_model,
                optimizer,
                self.train_loaded,
                self.test_loaded,
                loss_function,
                self.device,
                current_model_dir,
                model_name,
                epochs=epochs,
                pbar=pbar,
                show_plot=plot_output,
                verbose=verbose,
            )
            print(f" Best Test Loss: {loss}")

        trained_network.load_state_dict(
            torch.load(current_model_dir / model_name, weights_only=True)
        )

        return trained_network

    def k_fold_training(
        self,
        num_folds,
        network_model,
        optimizer,
        loss_function,
        epochs,
        individual_feature_scaling,
        batch_size,
    ):
        """
        Iterate through `num_folds` splits of the data and calculate the RMSE.
        Not using scikit learn for the cross-validation tools as need to handle scaling
        """
        network_model.to(self.device)
        k_fold_splitter = KFold(n_splits=num_folds, shuffle=True, random_state=self.seed)
        rmse_store = []
        for fold, (train_index, test_index) in enumerate(k_fold_splitter.split(self.raw_features)):
            print(f"Fold: {fold}")

            # Subsetting and scaling
            scaled_train_labels, label_scaler = scale_label(self.raw_labels[train_index])
            scaled_test_labels = scale_label(self.raw_labels[test_index], label_scaler)

            scaled_train_features, feature_scaler = scale_feature(
                self.raw_features[train_index], individual_feature_scaling
            )

            scaled_test_features = scale_feature(
                self.raw_features[test_index],
                individual_feature_scaling,
                feature_scaler=feature_scaler,
            )

            train_loaded = pytorch_load(scaled_train_features, scaled_train_labels, batch_size)
            test_loaded = pytorch_load(scaled_test_features, scaled_test_labels, batch_size)

            # Train network and evaluate
            with tqdm(total=epochs) as pbar:
                loss, best_state_dict = train_network(
                    network_model,
                    optimizer,
                    train_loaded,
                    test_loaded,
                    loss_function,
                    self.device,
                    output_dir=None,
                    model_name=f"Fold_{fold}",
                    epochs=epochs,
                    pbar=pbar,
                    show_plot=False,
                    verbose=False,
                )
                print(f" Best Test Loss: {loss}")

            trained_network = copy.deepcopy(network_model)
            trained_network.load_state_dict(best_state_dict)

            rmse, _, _ = self.validate_model(
                trained_network, data=test_loaded, labels=scaled_test_labels
            )
            rmse_store.append(rmse)

        print(rmse_store)
        mean_rmse = sum(rmse_store) / len(rmse_store)

        # Calculating mean at 95% confidence interval
        z = 1.96
        stderror = statistics.stdev(rmse_store) / math.sqrt(len(rmse_store))
        margin_error = z * stderror

        print(f"Mean RMSE: {mean_rmse:.3f} +/- {margin_error:.3f}")

        return [mean_rmse, margin_error]

    def optimize_network(
        self, trial_count, direction, network_dict, optimizer_dict, loss_function, epochs
    ):
        # Set sampler for Optuna if seed is provided - TPESampler appears to be the default
        sampler = optuna.samplers.TPESampler(seed=self.seed)

        def setup_trial(trial, input_dict):
            multi = input_dict.pop("multi") if "multi" in input_dict else None

            if "choices" in input_dict:
                suggest_fn = trial.suggest_categorical
            elif isinstance(input_dict["low"], int):
                suggest_fn = trial.suggest_int
            elif isinstance(input_dict["low"], float):
                suggest_fn = trial.suggest_float
            else:
                print("Unknown input dict")

            if multi is not None:
                output = []
                for i in range(multi):
                    new_input_dict = input_dict.copy()
                    new_input_dict["name"] = new_input_dict["name"] + str(i)
                    output.append(suggest_fn(**new_input_dict))

                # Add multi back into the input_dict as the same input_dict is used every trial
                input_dict["multi"] = multi
            else:
                output = suggest_fn(**input_dict)

            return output

        def objective(trial):
            # Network
            network_trials = {
                key: setup_trial(trial, input_dict)
                for key, input_dict in network_dict["optimize_kwargs"].items()
            }
            network_trials.update(network_dict["kwargs"])
            network_model = network_dict["network"](**network_trials)

            # Optimizer
            params = network_model.parameters()
            optimizer_trials = {
                key: setup_trial(trial, input_dict)
                for key, input_dict in optimizer_dict["optimize_kwargs"].items()
            }
            optimizer_trials.update(optimizer_dict["kwargs"])
            optimizer = optimizer_dict["optimizer"](params, **optimizer_trials)

            trained_model = self.train_model(
                None,
                network_model,
                optimizer,
                loss_function,
                epochs,
                plot_output=False,
                verbose=False,
            )
            rmse, _, _ = self.validate_model(trained_model, "test")

            return rmse

        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(objective, n_trials=trial_count, timeout=600)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Get multi keys
        network_multi = {
            key: []
            for key, input_dict in network_dict["optimize_kwargs"].items()
            if "multi" in input_dict
        }
        optimizer_multi = {
            key: []
            for key, input_dict in optimizer_dict["optimize_kwargs"].items()
            if "multi" in input_dict
        }

        # Split into optimizer and network params
        optimizer_params = {}
        network_params = {}
        for key, value in trial.params.items():
            if key in optimizer_dict["optimize_kwargs"]:
                optimizer_params[key] = value
            elif key in network_dict["optimize_kwargs"]:
                network_params[key] = value
            else:
                for multi_key in network_multi:
                    if multi_key in key:
                        network_multi[multi_key].append(value)
                        break

                for multi_key in optimizer_multi:
                    if multi_key in key:
                        optimizer_multi[key].append(value)
                        break

        # Restore multi keys
        network_params.update(network_multi)
        optimizer_params.update(optimizer_multi)

        optimizer_params.update(optimizer_dict["kwargs"])
        network_params.update(network_dict["kwargs"])

        best_network = network_dict["network"](**network_params)
        best_optimizer = optimizer_dict["optimizer"](
            params=best_network.parameters(), **optimizer_params
        )

        return best_network, best_optimizer

    def print_array_stats(self, arr, rmse=None, round_val=3):
        mean = round(arr.mean(), round_val)
        stdev = round(arr.std(), round_val)
        min = round(arr.min(), round_val)
        max = round(arr.max(), round_val)

        print(f"\nMean: {mean}\nStandard Dev: {stdev}\nMin: {min}\nMax: {max}")

        if rmse is not None:
            percent_error = round(rmse / mean * 100, 1)
            print(f"% RMSE: {percent_error}%")

    def validate_model(self, trained_network, data_source=None, data=None, labels=None):
        trained_network.to(self.device)

        if data is None and labels is None:
            if data_source == "test":
                data = self.test_loaded
                labels = self.scaled_test_labels

            elif data_source == "valid":
                data = self.valid_loaded
                labels = self.scaled_valid_labels

        rmse, mae, predictions = validate_regression(
            trained_network, data, labels, self.label_scaler, self.device
        )

        return rmse, mae, predictions

    def make_prediction(self, network, individual_feature_scaling, raw_features, output_size):
        """Make a prediction"""

        scaled_features = scale_feature(
            raw_features, individual_feature_scaling, self.feature_scaler
        )

        torch_tensor = torch.Tensor(scaled_features)
        torch_data = torch.utils.data.TensorDataset(torch_tensor)

        with torch.no_grad():
            network.eval()
            output = network(torch_data[0][0].to(self.device)).cpu().numpy()

            prediction = output.reshape(-1, output_size)
            prediction = self.label_scaler.inverse_transform(prediction)
            prediction = prediction.reshape(output_size)

            return prediction.tolist()

    def plot_label_distribution(
        self, label: np.array, bin_count=10, predictions=None, plot_error=False
    ):
        """Plot the distribution of the labels.

        Optionally add distribution of predictions and error associated with each group
        """

        bins = np.linspace(label.min(), label.max(), bin_count)

        # hist, bin_edges = np.histogram(label, bins='auto')

        plt.hist(label, bins, alpha=0.5, label="label")

        if predictions is not None:
            plt.hist(predictions, bins, alpha=0.5, label="predictions")

        plt.xlabel("Bins")
        plt.ylabel("Count")
        plt.title("Distribution of Labels")
        plt.legend()
        plt.show()

        if plot_error and predictions is not None:
            bin_idx = np.digitize(label, bins)
            error_percent = abs(predictions - label) / label * 100
            error_bins = {i: [] for i in range(1, len(bins) + 1)}
            for idx, bin in enumerate(bin_idx):
                error_bins[bin[0]].append(error_percent[idx])

            error_bin_average = []
            for error_bin in error_bins.values():
                error_bin_average.append(list(sum(error_bin) / len(error_bin))[0])

            bar_width = bins.max() / bin_count

            plt.bar(bins.tolist(), error_bin_average, width=bar_width)
            plt.xlabel("Bins")
            plt.ylabel("Percentage Error / %")
            plt.title("Percentage Error in Predictions")
            plt.show()
