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

    lossStore = []
    testLossStore = []
    # Prepare loss history
    for idx_epoch in range(epochs):
        epoch_loss = 0
        for x, y in dataloader_train:
            optimizer.zero_grad()

            # Propagate input
            net.train()
            netout = net(x.to(device))

            # Compute loss
            loss = loss_function(netout, y.to(device))
            epoch_loss += loss

            # Backpropagate loss
            loss.backward()

            # Update weights
            optimizer.step()

        test_loss = compute_regression_stats(net, dataloader_test, loss_function, device)

        # Print useful metrics
        # print(f'Train Loss: {epoch_loss:.5}')
        # print(f'Test Loss: {test_loss:.5}')

        # Store useful values
        lossStore.append(epoch_loss.cpu().detach())
        testLossStore.append(test_loss)

        # Update best results
        if test_loss < test_loss_best:
            test_loss_best = test_loss
            torch.save(net.state_dict(), output_dir / model_name)

            if verbose:
                print(f"Best State Updated. Epoch: {idx_epoch}")

        if pbar is not None:
            pbar.update()

    # Plot loss curves
    # print('Cutting Start')
    # plt.plot(lossStore[150:], label='Loss')
    # plt.plot(testLossStore[150:], label='Test_Loss')

    if show_plot:
        fig, ax = plt.subplots()
        plt.xticks(np.array(range(epochs)))
        ax.plot(np.array(range(epochs)), lossStore, color="blue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss", color="blue")

        ax2 = ax.twinx()
        ax2.plot(np.array(range(epochs)), testLossStore, color="orange")
        ax2.set_ylabel("Test Loss", color="orange")

        # plt.legend([ax, ax2], ['Training Loss', 'Test Loss'])
        plt.show()

    return test_loss_best


def compute_regression_stats(
    net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device = "cpu",
) -> torch.Tensor:
    net.eval()
    running_loss = 0
    lossCount = 0
    with torch.no_grad():
        for x, y in dataloader:
            netout = net(x.to(device)).cpu()

            # Compute loss
            running_loss += loss_function(netout, y)

        lossVal = running_loss
        lossCount += lossVal

    return lossVal.cpu()


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
