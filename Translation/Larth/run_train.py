"""
Run the training experiments from CLI,
"""
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import time
import logging
import argparse

import jax

# Ensure GPU
jax.config.update("jax_platform_name", "gpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Larth training script", description="IthacaLike training script"
    )
    parser.add_argument(
        "--model-config", "-m", help="Config file for the model (JSON or YAML)"
    )
    parser.add_argument(
        "--train-config",
        "-t",
        help="Config file with the training settings (JSON or YAML)",
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Load limited data for a test run"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Load limited data for a test run"
    )

    args = parser.parse_args()

    # Slow import here: you don't have to wait for the "--help" command
    from train import train_and_evaluate
    from train_utils import TrainConfig, parse_config
    from larth import LarthTranslationConfig

    model_config = LarthTranslationConfig(**parse_config(args.model_config))

    if args.debug:
        tmp = parse_config(args.train_config)
        tmp["debug"] = True
        train_config = TrainConfig(**tmp)
    else:
        train_config = TrainConfig(**parse_config(args.train_config))

    print("Model config:")
    print(model_config)

    print("Train config:")
    print(train_config)

    logging.basicConfig(level=logging.INFO)
    if args.verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.WARN)

    t = time.time()
    print("Backend:", jax.default_backend())
    print("Devices:", jax.local_device_count())
    print(f"Start at {time.ctime()}..")

    # with jax.profiler.trace("jax-trace", create_perfetto_link=True):
    train_and_evaluate(model_config, train_config)

    print(f"End at {time.ctime()}..")
    tot = time.time() - t
    print(f"Time: {tot}")
    print(f"Time per epoch: {tot / train_config.epochs}")
