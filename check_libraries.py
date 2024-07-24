try:
    import jax
    print(f"JAX version: {jax.__version__}")
except ImportError:
    print("JAX is not installed.")

try:
    import flax
    print(f"Flax version: {flax.__version__}")
except ImportError:
    print("Flax is not installed.")

try:
    import distrax
    print("Distrax is installed.")
except ImportError:
    print("Distrax is not installed.")

try:
    import hydra
    print("Hydra is installed.")
except ImportError:
    print("Hydra is not installed.")

try:
    import omegaconf
    print("OmegaConf is installed.")
except ImportError:
    print("OmegaConf is not installed.")

try:
    import wandb
    print("Weights & Biases (wandb) is installed.")
except ImportError:
    print("Weights & Biases (wandb) is not installed.")

try:
    import jaxmarl
    print("JAXMARL is installed.")
except ImportError:
    print("JAXMARL is not installed.")