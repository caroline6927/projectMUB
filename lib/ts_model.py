"""
Time Series Modeling:
1. Get features
    1) Autocorrelation analysis for feature selection
    2) Weekly and daily circadian rhythmicity of physical activity
2. Construct AR model
3. Make prediction
"""

from sklearn import linear_model


def get_ar_model(data, target):
    X_col = []
    for i in data.columns:
        if (target + 'feature') in i:
            X_col.append(i)
    X = data[X_col]  # TODO: add a vector of 1 here
    y = data[target]
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model
    regr = regr.fit(X, y)

    #     # The coefficients
    print(X_col)
    print('Coefficients: \n', regr.coef_)
    return regr, X_col


#     # The mean squared error
#     print("Mean squared error: %.2f"
#           % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
#     # Explained variance score: 1 is perfect prediction
#     print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

def ar_predict(model, data, target):
    X = data[X_col]
    y = data[target]
    print(len(X))
    print(len(y))
    predicted = model.predict(X)
    data['predicted'] = predicted
    print("Mean squared error: %.2f"
          % np.mean((model.predict(X) - y) ** 2))
    print('Variance score: %.2f' % model.score(X, y))

    # Plot outputs
    pyplot.scatter(range(len(X)), y, color='black')
    pyplot.plot(range(len(X)), predicted, color='blue', linewidth=3)
    pyplot.show()


ar_model, X_col = get_ar_model(train, 'steps')
ar_predict(ar_model, test, 'steps')