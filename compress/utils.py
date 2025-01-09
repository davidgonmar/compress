def extract_weights(
    model,
    cls_list=None,
    additional_check=lambda module: True,
    keywords={"weight"},
    ret_module=False,
):
    weights = []
    for name, module in model.named_modules():
        if cls_list is None or isinstance(module, tuple(cls_list)):
            if not additional_check(module):
                continue
            for keyword in keywords:
                if hasattr(module, keyword):
                    if not ret_module:
                        weights.append(getattr(module, keyword))
                    else:
                        # name should include weight attr
                        _name = name + "." + keyword
                        weights.append(((_name, module), getattr(module, keyword)))
    return weights
