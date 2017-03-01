"""
Time Series Modeling:
1. Get features
    1) Autocorrelation analysis for feature selection
    2) Weekly and daily circadian rhythmicity of physical activity
2. Construct AR model
3. Make prediction
"""
from sklearn import linear_model
import numpy as np


def get_ar_model(data, feature, target):
    X = data[feature]
    X['x_0'] = [1] * len(X)
    y = data[target]
    # Create linear regression object
    model = linear_model.LinearRegression()
    # Train the model
    model = model.fit(X, y)
    print('Coefficients: \n', model.coef_)
    return model, feature


def ar_predict(model, data, feature, target):
    X = data[feature]
    X['x_0'] = [1] * len(X)
    y = data[target]
    predicted = model.predict(X)
    data['predicted'] = predicted
    print("Mean squared error: %.2f"
          % np.mean((model.predict(X) - y) ** 2))
    print('Variance score: %.2f' % model.score(X, y))
    return predicted, model.score(X, y)
    # # Plot outputs
    # pyplot.scatter(range(len(X)), y, color='black')
    # pyplot.plot(range(len(X)), predicted, color='blue', linewidth=3)
    # pyplot.show()

# ar_model, X_col = get_ar_model(train, 'steps')
# ar_predict(ar_model, test, 'steps')
