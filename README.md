# fraud_detection

Various fraud detection methods applied to the [Kaggle Credit Card Fraud
Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Nature of Problem
The given dataset is extremely unbalanced between genuine and fraudulent
transactions, with ~28,000 genuine transactions and ~450 fraudulent examples.
For security purposes, the features of the data are expressed as the result of
PCA on the original dataset, giving us 28 anonymous, context-free features.

## Methods
[Anomaly Detecting Autoencoder](./notebooks/anomaly_autoencoder.ipynb)
This method trains an autoencoder (MSE loss with stochastic gradient descent)
against the bulk of the genuine transactions in the dataset (leaving a few
thousand genuine transactions aside for testing and model validation). A
decision boundary is then chosen on the loss of evaluated examples, with high
reconstruction loss being presumed indicative of a fraudulent transaction,
which the autoencoder has not been trained on.

After ~200 epochs of training, the mean reconstruction error for the validation
holdout set is 0.484, while the mean reconstruction error for fraudulent
transactions is 26.74. Both of these sets produce "pathological" cases which
would be on the wrong side of any reasonable decision boundary, but given the
context-free nature of the features, and the fact that the classifier has never
been exposed to any fraudulent transactions in the training phase, the
distinction in reconstruction error is remarkable.
