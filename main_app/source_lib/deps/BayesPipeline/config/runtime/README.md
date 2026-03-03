# Bayes Runtime Config

`runtime.config.json` is a sample development configuration.

Production systems are expected to resolve the model path externally and pass it
to `BayesRuntimeManager(runtime_config_path, model_config_path)`.

`runtime.config.json` intentionally excludes `model_config_path`.
