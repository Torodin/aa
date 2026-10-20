import numpy as numpy
import pandas as pd

import keras
import ml_edu.experiment
import ml_edu.results

import plotly.express as px

import matplotlib.pyplot as plt
import argparse

def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
    # Create and compile the regression model
    # Describe topography
    inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
    concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
    outputs = keras.layers.Dense(units=1)(concatenated_inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate), 
        loss="mean_squared_error", 
        metrics=metrics
    )

    return model

def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    label_name: str,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
    # Feed the model
    features = {name: dataset[name].values for name in settings.input_features}
    label = dataset[label_name].values
    history = model.fit(
        x=features, 
        y=label,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs
    )

    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )

def experiment_1(dataset: pd.DataFrame) -> ml_edu.experiment.Experiment:
    settings_1 = ml_edu.experiment.ExperimentSettings(
        learning_rate = 0.001,
        number_epochs = 20,
        batch_size = 50,
        input_features = ['TRIP_MILES']
    )

    metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

    model_1 = create_model(settings_1, metrics)
    experiment_model_1 = train_model('one_feature', model_1, dataset, 'FARE', settings_1)

    ml_edu.results.plot_experiment_metrics(experiment_model_1, ['rmse'])
    ml_edu.results.plot_model_predictions(experiment_model_1, dataset, 'FARE')

    return experiment_model_1

def experiment_2(dataset: pd.DataFrame) -> ml_edu.experiment.Experiment:
    settings_2 = ml_edu.experiment.ExperimentSettings(
        learning_rate = 0.001,
        number_epochs = 100,
        batch_size = 500,
        input_features = ['TRIP_MILES']
    )

    metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

    model_2 = create_model(settings_2, metrics)
    experiment_model_2 = train_model('one_feature_hyper', model_2, dataset, 'FARE', settings_2)

    ml_edu.results.plot_experiment_metrics(experiment_model_2, ['rmse'])
    ml_edu.results.plot_model_predictions(experiment_model_2, dataset, 'FARE')

    return experiment_model_2

    # With learning rate 1.0 and batch 50 the rmse take a bit longer to reduce
    # when learning rate is to high training would be harder

    # With learning rate 0.0001 and batch 50 the model dont reach the convergence
    # with more epochs surely but its not ideal

    # With learning rate 0.001 and batch 500 I see something between the two previous tests
    # The rmse decrease slowly and need more epochs. That said, the training runs much faster.

    # Repeating the last test until the model reach the convergence it needs around 60 epochs

def experiment_3(dataset: pd.DataFrame) -> ml_edu.experiment.Experiment:
    settings_3 = ml_edu.experiment.ExperimentSettings(
        learning_rate = 0.001,
        number_epochs = 20,
        batch_size = 50,
        input_features = ['TRIP_MILES', 'TRIP_MINUTES']
    )

    dataset['TRIP_MINUTES'] = dataset['TRIP_SECONDS']/60

    metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

    model_3 = create_model(settings_3, metrics)
    experiment_model_3 = train_model("two_features", model_3, dataset, "FARE", settings_3)

    ml_edu.results.plot_experiment_metrics(experiment_model_3, ['rmse'])
    ml_edu.results.plot_model_predictions(experiment_model_3, dataset, 'FARE')

    return experiment_model_3

    # The rmse is consistently lower than experiment 1
    # Changing the scale from seconds to minutes help the model training closing the gap between the features

EXPERIMENTS = {
    1: experiment_1,
    2: experiment_2,
    3: experiment_3
}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AA experiments"
    )

    parser.add_argument(
        "-e",
        "--experiments",
        type=int,
        nargs="+",
        choices=list(EXPERIMENTS.keys()),
        required=True,
        help="IDs of the experiments"
    )

    args = parser.parse_args()
    selected_experiments = list(dict.fromkeys(args.experiments))

    chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

    training_df = chicago_taxi_dataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

    print('Dataset read successfully')
    print(f'Total number of rows: {len(training_df.index)}')

    print(training_df.describe(include='all'))
    print(training_df.corr(numeric_only = True))

    fig = px.scatter_matrix(training_df, dimensions=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
    fig.show()

    experiments_results = []
    for experiment_id in selected_experiments:
        experiment_result = EXPERIMENTS[experiment_id](training_df)
        experiments_results.append(experiment_result)

    ml_edu.results.compare_experiment(experiments_results, ["rmse"], training_df, training_df["FARE"].values)

    plt.show()