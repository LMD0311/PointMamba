from utils import registry



MODELS = registry.Registry('models')

def build_model_from_cfg(cfg, **kwargs):
    return MODELS.build(cfg, **kwargs)
