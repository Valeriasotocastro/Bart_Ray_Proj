import numpy as np
import pymc3 as pm
import ray
from ray import tune
from models.bart_model import build_bart_model

# Datos de ejemplo
x = np.random.randn(100, 2)
y = np.random.randn(100)

def train_bart(config):
    # Modelo BART
    with pm.Model() as bart_model:
        # (Aquí incluir la definición del modelo BART)
        # Hiperparámetros
        num_trees = 50
        tree_depth = 3
        alpha = 0.95

        # Priors para los hiperparámetros
        sigma = pm.HalfCauchy('sigma', beta=10, testval=1.0)
        mu = pm.Normal('mu', mu=0, sigma=10, shape=num_trees)
        tau = pm.Gamma('tau', alpha=alpha, beta=alpha, shape=num_trees)
        p = pm.Beta('p', alpha=alpha, beta=alpha, shape=num_trees)

        # Número de iteraciones de MCMC
        num_samples = config.get("num_samples", 1000)

        # Muestreo MCMC
        trace = pm.sample(num_samples)

        # Guardar los resultados o métricas que quieras registrar, como el error de prueba, etc.

        # Devolver la métrica que se quiere maximizar o minimizar durante la búsqueda de hiperparámetros
        return -np.mean(trace['y_obs'])

if __name__ == "__main__":
    # Inicializar Ray (hacer esto antes de llamar a tune.run())
    ray.init(ignore_reinit_error=True)

    # Configuración de hiperparámetros a ajustar
    config = {
        "num_samples": tune.grid_search([1000, 2000, 3000]), # Puedes ajustar otros hiperparámetros también
        # Agrega otros hiperparámetros para ajustar aquí...
    }

    # Lanzar la búsqueda de hiperparámetros en paralelo
    result = tune.run(train_bart, config=config, num_samples=3, resources_per_trial={"cpu": 1})

    # Obtener los mejores hiperparámetros encontrados y sus resultados
    best_config = result.get_best_config(metric="neg_mean")  # Metric to maximize/minimize
    best_result = result.get_best_trial(metric="neg_mean", mode="max")  # Metric to maximize

    print("Mejores hiperparámetros:", best_config)
    print("Mejor resultado:", best_result)


#Unirlos
def train_bart(config):
    # Datos de ejemplo
    x = np.random.randn(100, 2)
    y = np.random.randn(100)

    # Construir el modelo BART con los hiperparámetros dados
    model = build_bart_model(x, y)

    # Número de iteraciones de MCMC
    num_samples = config.get("num_samples", 1000)

    # Muestreo MCMC
    with model:
        trace = pm.sample(num_samples)

    # Calcular el error medio cuadrático negativo
    mse = np.mean((trace['y_obs'] - y) ** 2)

    # Devolver la métrica a optimizar (menos es mejor)
    return -mse

if __name__ == "__main__":
    # Inicializar Ray (debes hacer esto antes de llamar a tune.run())
    ray.init(ignore_reinit_error=True)

    # Configuración de hiperparámetros a ajustar
    config = {
        "num_samples": tune.grid_search([1000, 2000, 3000]), # Puedes ajustar otros hiperparámetros también
        # Agrega otros hiperparámetros para ajustar aquí...
    }

    # Lanzar la búsqueda de hiperparámetros en paralelo
    result = tune.run(train_bart, config=config, num_samples=3, resources_per_trial={"cpu": 1})

    # Obtener los mejores hiperparámetros encontrados y sus resultados
    best_config = result.get_best_config(metric="neg_mean")  # Metric to maximize/minimize
    best_result = result.get_best_trial(metric="neg_mean", mode="max")  # Metric to maximize

    print("Mejores hiperparámetros:", best_config)
    print("Mejor resultado:", best_result)