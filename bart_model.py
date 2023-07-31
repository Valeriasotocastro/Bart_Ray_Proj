import pandas as pd
import numpy as np
import pymc3 as pm

# Leer el archivo CSV con pandas y cargar datos

data = pd.read_csv('Datos_Bart_C.csv', delimiter=';')

print(data)

#Definir Variables

# Modelo BART
with pm.Model() as bart_model:
    # Hiperparámetros
    num_trees = 50
    tree_depth = 3
    alpha = 0.95

    # Priors para los hiperparámetros
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.0)
    mu = pm.Normal('mu', mu=0, sigma=10, shape=num_trees)
    tau = pm.Gamma('tau', alpha=alpha, beta=alpha, shape=num_trees)
    p = pm.Beta('p', alpha=alpha, beta=alpha, shape=num_trees)

    # Árboles de regresión
    trees = []
    for i in range(num_trees):
        tree = pm.glm.forest.BARTRegressionTree('tree_{}'.format(i), np.column_stack((X1, X2, X3)), y, mu[i], tau[i], p[i], alpha=tree_depth)
        trees.append(tree)

    # Modelo BART
    bart = pm.Deterministic('bart', sum(trees))
    
    # Verosimilitud
    y_obs = pm.Normal('y_obs', mu=bart, sigma=sigma, observed=y)

# Ajustar el modelo utilizando MCMC
with bart_model:
    trace = pm.sample(2000, tune=1000)

# Resumen de los resultados
print(pm.summary(trace))
