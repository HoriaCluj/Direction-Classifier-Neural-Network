# Direction Classifier Neural Network

<a href="https://colab.research.google.com/github/HoriaCluj/Direction-Classifier-Neural-Network/blob/main/Stocks_Binary_Classifier_Neural_Network.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

A PyTorch binary classifier that predicts whether the next trading day for the S&P 500 ETF (**SPY**) will be an **Up day** (close > open) or a **Down day** (close ≤ open), based on the previous 10 days of price and volume data.

The whole project lives in a single, step-by-step annotated notebook: [Stocks_Binary_Classifier_Neural_Network.ipynb](Stocks_Binary_Classifier_Neural_Network.ipynb).

## How it works

### 1. Data download

About 30 years of daily SPY data is downloaded with [`yfinance`](https://github.com/ranaroussi/yfinance):

```python
df = yf.download('SPY', period='30y')
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
```

Each row is one trading day, and the five columns (OHLCV) are the model's features. `df.values` turns the DataFrame into a NumPy array of shape `(days, 5)`.

### 2. Turning the history into training samples

The model doesn't look at a single day. It looks at a **10-day window** and predicts what happens on the day right after it. The `create_sequences(data, window=10)` function slides this window across the whole history, one day at a time:

```
Days:     d0  d1  d2  ...  d9 | d10   → sample 0 (label from d10)
              d1  d2  ...  d9  d10 | d11   → sample 1 (label from d11)
                  d2  ...  d9  d10  d11 | d12   → sample 2 ...
```

For each position `i`:

- **Input (`X_seq`)**: rows `i` to `i + 9`, i.e. 10 days × 5 features, shape `(10, 5)`.
- **Label**: taken from the 11th day, `data[i + 10]`. It is `1` if that day's Close is higher than its Open (an **Up day**), and `0` otherwise (a **Down day**). Open is column 0 and Close is column 3 of the array.

All the windows are stacked into:

- `X` with shape `(samples, 10, 5)`, 7,539 samples in the sample run
- `y` with shape `(samples,)`

Note that the label compares the 11th day's close with its **own open**. It describes the move during that day, not the gap from the previous close.

### 3. Normalizing the features

Raw prices (tens to hundreds of dollars) and volume (millions of shares) are on very different scales, which makes training harder. scikit-learn's `StandardScaler` rescales each feature to mean 0 and standard deviation 1:

```
x_standardized = (x - μ) / σ
```

`StandardScaler` only accepts 2D input of shape `(n_samples, n_features)`, so `X` is temporarily flattened, scaled, and then reshaped back:

```python
X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1]))  # (samples * 10, 5)
X_scaled = X_scaled.reshape(X.shape)                         # (samples, 10, 5)
```

This way each of the five features is standardized using statistics computed across every day of every window.

### 4. Train/test split and batching

- `train_test_split(X_scaled, y, test_size=0.2)` puts 80% of the samples into the training set (6,031) and 20% into the test set (1,508).
- The NumPy arrays are converted to `float32` PyTorch tensors.
- Inputs and labels are paired with `TensorDataset` and wrapped in `DataLoader`s with a batch size of 64. The training loader shuffles the samples every epoch, so the model sees the batches in a different order each time.

### 5. Model architecture

`BinaryClassification` is a small fully connected (feed-forward) network:

```
Input (10 days × 5 features = 50 values)
  → Linear(50, 64) → ReLU
  → Linear(64, 32) → ReLU
  → Linear(32, 1)  → Sigmoid → P(Up day)
```

- **Flattening.** `nn.Linear` expects one flat vector per sample, so `forward()` first reshapes each `(10, 5)` window into 50 values with `x.view(x.size(0), -1)`.
- **Hidden layer 1** turns the 50 inputs into 64 weighted combinations. ReLU (`max(0, x)`) then adds non-linearity, so the network can learn more than straight-line relationships.
- **Hidden layer 2** combines those 64 values into 32 new ones, again followed by ReLU. Having two hidden layers gives the model more flexibility to learn non-linear patterns.
- **Output layer** reduces the 32 values to a single number, and the sigmoid squashes it into a probability between 0 and 1. Values above 0.5 are read as an Up day, and values at or below 0.5 as a Down day.

Before training, the weights are random, so the model's outputs hover around 0.5.

### 6. Training

| Setting | Value |
|---|---|
| Loss function | Binary cross-entropy (`nn.BCELoss`), which fits because the sigmoid is already applied in `forward()` |
| Optimizer | Stochastic gradient descent (`torch.optim.SGD`), learning rate 0.1 |
| Batch size | 64 |
| Max epochs | 1000 |
| Early stopping | Stops after 20 epochs in a row without a lower test loss |

Each epoch has two phases:

1. **Training phase** (`model.train()`). For every batch in `train_loader`:
   1. Forward pass: compute the predicted probabilities.
   2. Compute the BCE loss against the true labels.
   3. Reset the old gradients with `optimizer.zero_grad()`.
   4. Backpropagation: `loss.backward()` computes how much each weight contributed to the error.
   5. `optimizer.step()` nudges each weight in the direction that lowers the loss.
2. **Testing phase** (`model.eval()` and `torch.no_grad()`). The model runs over `test_loader` without updating any weights, and the losses of all test batches are summed.

If the summed test loss is the lowest so far, the model's state is saved and the patience counter resets. Otherwise the counter goes up, and when it reaches 20, training stops. The saved best state is then loaded back into the model. Early stopping keeps the model from training long after it has stopped improving on unseen data, which helps limit overfitting.

### 7. Evaluation

On the test set:

- **Hit rate (accuracy)**: the share of test days where the predicted direction (probability > 0.5) matches the real one.
- **Confusion matrix**: how many Up and Down days were predicted correctly or incorrectly.
- **Classification report**: precision, recall and F1 score for each class.

### 8. Predicting the next trading day

Finally, the latest window in `X_scaled` is run through the trained model. It gets an extra batch dimension (`unsqueeze(0)`), and the output probability is rounded to 0 or 1. The notebook then prints whether the next day is expected to be an Up day or a Down day, together with the hit rate from the test set.

## Results

On a sample run (7,539 sequences, 1,508 of them in the test set):

| Metric | Value |
|---|---|
| Test accuracy (hit rate) | ~54.2% |
| Up-day F1 score | 0.632 |
| Down-day F1 score | 0.392 |

Results vary from run to run because the train/test split and the weight initialization are random.

For context, simply predicting "Up" every day would score about 52.8% on the same test set. The model is only slightly better than that baseline. Daily market direction is very noisy, and this project is meant to be a learning exercise, not a trading tool.

## Tech stack

- Python
- PyTorch: model, training loop, DataLoaders
- scikit-learn: `StandardScaler`, `train_test_split`, evaluation metrics
- pandas / NumPy: data handling
- yfinance: market data

## Possible improvements

- Use a chronological train/test split instead of a random one, and fit the scaler on the training data only, so no future information leaks into training.
- Add engineered features such as returns, moving averages, RSI or volatility, instead of raw prices.
- Try sequence models such as LSTMs, GRUs or 1D CNNs.
- Use walk-forward validation to get a more realistic estimate of performance.
