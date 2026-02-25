import torch


def add_special_tokens():
    return None


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=False):
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue

        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names