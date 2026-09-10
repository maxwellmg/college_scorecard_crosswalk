"""
Minimal LightGBM smoke test — no miceforest, no College Scorecard data.

Why: every crash we've caught so far happens on the FIRST variable trained
in a given kernel.mice() call, regardless of which variable that is or
what its data looks like (confirmed across 10 different columns with no
common profile — continuous, binary, low- and high-cardinality alike). That
pattern points away from "some column is bad" and toward something about
the environment/first LightGBM call itself. This script isolates that
question completely: if THIS crashes too, on 100 rows of made-up numbers
that have nothing to do with the real dataset, the problem is the LightGBM
install in this environment, not this project's data or code.

Run this in a FRESH kernel restart (not appended to the notebook that's
already crashed once — see the module docstring in the parent conversation
for why a process that already hit a native access violation may not be
trustworthy for anything run afterward).
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

print("lightgbm version:", lgb.__version__)

rng = np.random.default_rng(0)
n = 200
X = pd.DataFrame({
    "a": rng.normal(size=n),
    "b": rng.integers(0, 5, size=n),
})
y = rng.normal(size=n)

print("Building Dataset...")
train_set = lgb.Dataset(X, label=y)

print("Training...")
booster = lgb.train(
    params={"objective": "regression", "verbosity": -1, "num_threads": 1},
    train_set=train_set,
    num_boost_round=5,
)

print("SUCCESS — LightGBM trained a model with no crash.")
print(booster.predict(X)[:5])
