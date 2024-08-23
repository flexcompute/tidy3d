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
    epochs=10,
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
        epoch_loss = 0
        for x, y in dataloader_train:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            # Propagate input
            netout = net(x)

            # Compute loss
            loss = loss_function(netout, y)
            epoch_loss += loss

            # Backpropagate loss
            loss.backward()

            # Update weights
            optimizer.step()

        test_loss = compute_regression_stats(net, dataloader_test, loss_function, device)

        # Store useful values
        loss_store.append(epoch_loss.detach())
        test_loss_store.append(test_loss.detach())

        # Update best results
        if test_loss < test_loss_best:
            test_loss_best = test_loss
            best_state_dict = net.state_dict().copy()
            # torch.save(net.state_dict(), output_dir / model_name)

            if verbose:
                print(f"Best State Updated. Epoch: {idx_epoch}")

        if pbar is not None:
            pbar.update()

    if show_plot:
        cpu_loss_store = [loss_val.cpu() for loss_val in loss_store]
        cpu_test_loss_store = [loss_val.cpu() for loss_val in test_loss_store]
        fig, ax = plt.subplots()
        plt.xticks(np.array(range(epochs)))
        ax.plot(np.array(range(epochs)), cpu_loss_store, color="blue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss", color="blue")

        ax2 = ax.twinx()
        ax2.plot(np.array(range(epochs)), cpu_test_loss_store, color="orange")
        ax2.set_ylabel("Test Loss", color="orange")

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
    running_loss = 0
    loss_count = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            netout = net(x)

            # Compute loss
            running_loss += loss_function(netout, y)

        loss_count += running_loss

    return running_loss


def validate_regression(net, dataloader_val, y_valid, scaler, device):
    predictions = []
    with torch.no_grad():
        for x, _ in dataloader_val:
            net.eval()
            predictions.append(net(x.to(device)))

    predictionList = []
    for val in predictions:
        predictionList.append(val.cpu().numpy())
    predictionArray = np.concatenate(predictionList, axis=0)

    scaledPredictionArray = scaler.inverse_transform(predictionArray)
    scaledYValid = scaler.inverse_transform(y_valid)

    # RMSE & MAE Calculation
    RMSE = round(calc_rmse(scaledPredictionArray, scaledYValid), 6)
    MAE = round(mean_absolute_error(scaledYValid, scaledPredictionArray), 6)
    print(f"RMSE: {RMSE}")
    print(f"MAE: {MAE}")

    return RMSE, MAE, scaledPredictionArray


def calc_rmse(predArray, validArray):
    return np.sqrt(np.mean((predArray - validArray) ** 2))


def print_label_stats(arr):
    mean = arr.mean()
    stdev = arr.std()
    min = arr.min()
    max = arr.max()

    print(f"Mean: {mean}\nStandard Dev: {stdev}\nMin: {min}\nMax: {max}")
