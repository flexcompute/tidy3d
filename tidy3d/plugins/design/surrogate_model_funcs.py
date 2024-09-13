import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error


def train_network(
    net,
    optimizer,
    dataloader_train,
    dataloader_test,
    loss_function,
    device,
    output_dir,
    model_name,
    epochs,
    pbar=None,
    show_plot=False,
    verbose=True,
):
    test_loss_best = np.inf

    loss_store = []
    test_loss_store = []
    best_state_dict = None
    # Prepare loss history
    for idx_epoch in range(epochs):
        net.train()
        epoch_loss = []
        for x, y in dataloader_train:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            # Propagate input
            netout = net(x)

            # Compute loss
            loss = loss_function(netout, y)
            epoch_loss.append(loss.detach().cpu())

            # Backpropagate loss
            loss.backward()

            # Update weights
            optimizer.step()

        test_loss = compute_regression_stats(net, dataloader_test, loss_function, device)

        # Average loss over the epoch - test loss average already calculated
        ave_epoch_loss = sum(epoch_loss) / len(epoch_loss)

        # Store useful values
        loss_store.append(ave_epoch_loss)
        test_loss_store.append(test_loss)

        # Update best results
        if test_loss < test_loss_best:
            test_loss_best = test_loss
            best_state_dict = net.state_dict().copy()

            if verbose:
                print(f"Best State Updated. Epoch: {idx_epoch}")

        if pbar is not None:
            pbar.update()

    if show_plot:
        _, ax = plt.subplots()
        epoch_list = list(range(epochs))
        ax.plot(epoch_list, loss_store, color="blue", label="Train Loss")
        ax.plot(epoch_list, test_loss_store, color="orange", label="Test Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        plt.legend()

        plt.show()

    # Save state_dict
    torch.save(best_state_dict, output_dir / model_name)

    return test_loss_best


def compute_regression_stats(
    net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    net.eval()
    running_loss = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            netout = net(x)

            # Compute loss
            running_loss.append(loss_function(netout, y).detach().cpu())

    # Average running loss
    ave_running_loss = sum(running_loss) / len(running_loss)

    return ave_running_loss


def validate_regression(net, dataloader, labels, scaler, device):
    predictions = []
    with torch.no_grad():
        for x, _ in dataloader:
            net.eval()
            predictions.append(net(x.to(device)))

    np_predictions = []
    for val in predictions:
        np_predictions.append(val.cpu().numpy())
    prediction_array = np.concatenate(np_predictions, axis=0)

    scaled_prediction_array = scaler.inverse_transform(prediction_array)
    scaled_labels = scaler.inverse_transform(labels)

    # RMSE & MAE Calculation
    RMSE = round(calc_rmse(scaled_prediction_array, scaled_labels), 6)
    MAE = round(mean_absolute_error(scaled_labels, scaled_prediction_array), 6)
    print(f"RMSE: {RMSE}")
    print(f"MAE: {MAE}")

    return RMSE, MAE, scaled_prediction_array


def calc_rmse(pred_array, label_array):
    return np.sqrt(np.mean((pred_array - label_array) ** 2))


def print_label_stats(arr):
    mean = arr.mean()
    stdev = arr.std()
    min = arr.min()
    max = arr.max()

    print(f"Mean: {mean}\nStandard Dev: {stdev}\nMin: {min}\nMax: {max}")
