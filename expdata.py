import pandas as pd
import numpy as np

# Levy Russian data
levy_exp1a_verb = pd.DataFrame({
    "mean": 600 + 800/276 * np.array([28, 211, 36, 154]),
    "rc_type": ["src", "src", "orc", "orc"],
    "locality": ["local", "non-local", "local", "non-local"],
    "data type": np.repeat("actual data", 4)
})

levy_exp2a_verb = pd.DataFrame({
    "mean": np.concatenate((
        600 + 500/260  * np.array([-10 + 12, 23 + 53, 40 + 59, ]),
        600 + 1000/351 * np.array([23 + 33, 37 + 56]))),
    "n intervener": [0, 1, 2, 1, 2],
    "intervener type": ["-", "arg", "arg", "adj", "adj"],
})