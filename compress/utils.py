def extract_weights(model, cls_list=None):
    weights = []
    for name, module in model.named_modules():
        if cls_list is None or isinstance(module, tuple(cls_list)):
            if not hasattr(module, "weight"):
                continue
            weights.append(module.weight)
    return weights
