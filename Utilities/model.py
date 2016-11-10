from __future__ import division
from Utilities.performance_measure import mmre


def generate_model(training_data, testing_data, performance_measure=mmre):
    training_independent = training_data[0]
    training_dependent = training_data[-1]

    testing_independent = testing_data[0]
    testing_dependent = testing_data[-1]

    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor()
    # try:
    model.fit(training_independent, training_dependent)

    predicted = model.predict(testing_independent)
    return model, performance_measure(predicted, testing_dependent)