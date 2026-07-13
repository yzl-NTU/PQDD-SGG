from .pqdd import build as build_pqdd
from .reltr import build as build_reltr


def build_model(args):
    model_name = getattr(args, "model", "pqdd")
    if model_name == "reltr":
        return build_reltr(args)
    return build_pqdd(args)
