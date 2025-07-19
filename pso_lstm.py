import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import random

class Particle:
    def __init__(self, bounds):
        self.position = [random.uniform(b[0], b[1]) for b in bounds]
        self.velocity = [random.uniform(-1, 1) for _ in bounds]
        self.best_position = list(self.position)
        self.best_score = float('inf')

    def update_velocity(self, global_best, inertia=0.5, cognitive=1.5, social=1.5):
        for i in range(len(self.velocity)):
            r1, r2 = random.random(), random.random()
            cognitive_component = cognitive * r1 * (self.best_position[i] - self.position[i])
            social_component = social * r2 * (global_best[i] - self.position[i])
            self.velocity[i] = inertia * self.velocity[i] + cognitive_component + social_component

    def update_position(self, bounds):
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]
            self.position[i] = max(bounds[i][0], min(bounds[i][1], self.position[i]))

def create_lstm_model(input_shape, units):
    model = Sequential()
    model.add(LSTM(units=int(units), input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(series, n_steps):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    return np.array(X), np.array(y)

def pso_optimize(data, n_particles=5, iterations=10, bounds=[(10, 100), (5, 30)]):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    best_global_position = None
    best_global_score = float('inf')
    particles = [Particle(bounds) for _ in range(n_particles)]

    for _ in range(iterations):
        for particle in particles:
            units, n_steps = int(particle.position[0]), int(particle.position[1])
            X, y = prepare_data(scaled_data, n_steps)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            model = create_lstm_model((X.shape[1], 1), units)
            model.fit(X, y, epochs=5, verbose=0)

            y_pred = model.predict(X).flatten()
            score = mean_squared_error(y, y_pred)

            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = list(particle.position)

            if score < best_global_score:
                best_global_score = score
                best_global_position = list(particle.position)

        for particle in particles:
            particle.update_velocity(best_global_position)
            particle.update_position(bounds)

    return best_global_position

def run_lstm_pso_forecast(series):
    bounds = [(20, 100), (5, 30)]
    best_params = pso_optimize(series.values, bounds=bounds)
    units, n_steps = int(best_params[0]), int(best_params[1])
    print(f"[LSTM-PSO] Best Params: units={units}, n_steps={n_steps}")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    X, y = prepare_data(scaled_data, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = create_lstm_model((X.shape[1], 1), units)
    model.fit(X, y, epochs=20, verbose=1)

    y_pred = model.predict(X).flatten()
    y_true = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True')
    plt.plot(y_pred_rescaled, label='LSTM-PSO Forecast')
    plt.legend()
    plt.title("LSTM-PSO Forecast")
    plt.tight_layout()
    plt.savefig("static/plots/lstm_pso_forecast.png")

    return {
        "MAPE": mean_absolute_percentage_error(y_true, y_pred_rescaled) * 100,
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred_rescaled)),
        "R2": r2_score(y_true, y_pred_rescaled)
    }
