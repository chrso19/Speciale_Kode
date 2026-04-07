import os
import joblib
import wandb


RF_MODEL_CACHE_ENV = "WANDB_RF_MODEL_DIR"
DEFAULT_RF_MODEL_CACHE_DIR = r"C:\Users\chris\Documents\Python\Personlig\Artifacts"


RF_ARTIFACTS = {
    "OffshoreWindPower_DK1": "Energinet_speciale/Shallow_learners/rf_OffshoreWindPower_DK1:latest",
    "OffshoreWindPower_DK2": "Energinet_speciale/Shallow_learners/rf_OffshoreWindPower_DK2:latest",
    "OnshoreWindPower_DK1": "Energinet_speciale/Shallow_learners/rf_OnshoreWindPower_DK1:latest",
    "OnshoreWindPower_DK2": "Energinet_speciale/Shallow_learners/rf_OnshoreWindPower_DK2:latest",
    "SolarPower_DK1": "Energinet_speciale/Shallow_learners/rf_SolarPower_DK1:latest",
    "SolarPower_DK2": "Energinet_speciale/Shallow_learners/rf_SolarPower_DK2:latest",
    "TotalProduction_DK1": "Energinet_speciale/Shallow_learners/rf_TotalProduction_DK1:latest",
    "TotalProduction_DK2": "Energinet_speciale/Shallow_learners/rf_TotalProduction_DK2:latest",
}

_model_cache = None


def _resolve_cache_root(user: str, cache_root: str | None) -> str:
    """Resolve the model cache directory, preferring explicit and absolute paths."""
    if cache_root:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(cache_root)))

    env_cache_root = os.getenv(RF_MODEL_CACHE_ENV)
    if env_cache_root:
        resolved_env = os.path.abspath(os.path.expanduser(os.path.expandvars(env_cache_root)))
        if os.path.isabs(env_cache_root):
            return resolved_env
        print(
            f"Ignoring relative {RF_MODEL_CACHE_ENV}='{env_cache_root}'. "
            f"Using user-specific default instead."
        )

    user = user.strip().lower()

    if user == "nikolaj":
        cache_root = r"C:\Users\n_and\Documents\Data Science\Speciale\Shallow_learners\Artifacts"
    elif user == "christine - laptop":
        cache_root = r"C:\Users\Christine\Documents\Python\Personlig"
    elif user == "christine - desktop":
        cache_root = r"C:\Users\chris\Documents\Python\Personlig\Artifacts"
    else:
        raise ValueError(
            f"Unknown user '{user}'. Must be 'Nikolaj' or 'Christine'."
        )

    return os.path.abspath(
        os.path.expanduser(os.path.expandvars(cache_root))
    )


def load_rf_models(user: str, timeout: int = 60, cache_root: str | None = None) -> dict:
    """
    Download and load RF models for all forecast features once per Python session.
    
    Models are downloaded from W&B artifacts and cached in memory to avoid
    repeated downloads.

    Parameters
    ----------
    user : str
        The user for whom to load models.
    timeout : int, default 60
        Timeout in seconds for W&B API calls.
    cache_root : str | None, default None
        Optional local folder where W&B artifacts are downloaded.
        If None, tries WANDB_RF_MODEL_DIR only when it is an absolute path.
        Otherwise defaults to DEFAULT_RF_MODEL_CACHE_DIR.

    Returns
    -------
    dict
        Dictionary mapping feature names to pre-trained RF models.
        E.g., {"OffshoreWindPower_DK1": model, "OnshoreWindPower_DK1": model, ...}
        
    Raises
    ------
    FileNotFoundError
        If no .joblib model file is found in the W&B artifact.
        
    Examples
    --------
    >>> rf_models = load_rf_models(user="Nikolaj")
    >>> rf_models["OffshoreWindPower_DK1"]
    Pipeline(steps=[('scaler', StandardScaler()), ('model', RandomForestRegressor(...))])
    """
    global _model_cache

    if _model_cache is not None:
        return _model_cache

    cache_root = _resolve_cache_root(user, cache_root)

    if cache_root:
        os.makedirs(cache_root, exist_ok=True)

    api = wandb.Api(timeout=timeout)
    loaded_models = {}

    for key, artifact_path in RF_ARTIFACTS.items():
        try:
            artifact = api.artifact(artifact_path, type="model")
            if cache_root:
                target_dir = os.path.join(cache_root, key)
                os.makedirs(target_dir, exist_ok=True)
                artifact_dir = artifact.download(root=target_dir)
            else:
                artifact_dir = artifact.download()

            joblib_files = [
                f for f in os.listdir(artifact_dir)
                if f.endswith(".joblib")
            ]
            if not joblib_files:
                raise FileNotFoundError(f"No .joblib file found in artifact {artifact_path}")

            model_path = os.path.join(artifact_dir, joblib_files[0])
            loaded_models[key] = joblib.load(model_path)
            print(f"✓ Loaded RF model for {key} from {artifact_dir}")
        except Exception as e:
            print(f"✗ Failed to load RF model for {key}: {e}")
            raise

    _model_cache = loaded_models
    print(f"\n✓ Successfully loaded {len(loaded_models)} RF models")
    return _model_cache


def clear_model_cache():
    """Clear the cached RF models to force reload on next call."""
    global _model_cache
    _model_cache = None
    print("Model cache cleared")