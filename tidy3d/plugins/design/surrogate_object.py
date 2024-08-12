import copy
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna

# PyTorch
import torch
from optuna.trial import TrialState
from tqdm import tqdm

from tidy3d.plugins.design.surrogate_model_funcs import (
    train_network,
    validate_regression,
)

# Local
from tidy3d.plugins.design.surrogate_preprocess_funcs import (
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
    def __init__(self, output_dir, data_dir: list, rng_seed=None) -> None:
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
        self.rng_seed = rng_seed
        np.random.seed(rng_seed)

        # Torch cannot cope with None seed like numpy can - don't set seed so Torch is random
        if rng_seed is not None:
            torch.cuda.empty_cache()
            torch.manual_seed(rng_seed)

    def load_data_from_hdf5(
        self,
        fn_data,
        test_percentage,
        valid_percentage,
        batch_size,
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

                    labels = labels.reshape(-1, 1)
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

        self._prep_data(raw_features, raw_labels, test_percentage, valid_percentage, batch_size)

    def load_data_from_df(
        self,
        df,
        label_name,
        feature_names,
        test_percentage,
        valid_percentage,
        batch_size,
    ):
        raw_labels = df[label_name].values
        raw_labels = raw_labels.reshape(-1, 1)
        raw_features = df.loc[:, feature_names].values

        self._prep_data(raw_features, raw_labels, test_percentage, valid_percentage, batch_size)

    def _prep_data(self, raw_features, raw_labels, test_percentage, valid_percentage, batch_size):
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

        # Scale labels
        scaled_train_labels, label_scaler = scale_label(train_labels)
        scaled_test_labels = scale_label(test_labels, label_scaler)
        scaled_valid_labels = scale_label(valid_labels, label_scaler)

        # Scale features
        scaled_train_features, feature_scaler = scale_feature(train_features)
        scaled_test_features = scale_feature(test_features, feature_scaler)
        scaled_valid_features = scale_feature(valid_features, feature_scaler)

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

    def train_model(
        self, model_name, network_model, optimizer, loss_function, epochs: int, plot_output: bool
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
            loss = train_network(
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
            )
            print(f" Best Test Loss: {loss}")

        trained_network.load_state_dict(torch.load(current_model_dir / model_name))

        return trained_network

    def optimize_network(
        self, trial_count, direction, network_dict, optimizer_dict, loss_function, epochs
    ):
        # Set sampler for Optuna if seed is provided - TPESampler appears to be the default
        sampler = optuna.samplers.TPESampler(seed=self.rng_seed)

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
                None, network_model, optimizer, loss_function, epochs, plot_output=False
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

    def validate_model(self, trained_network, data_source):
        trained_network.to(self.device)

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
