import torch
from torch.nn.utils import prune
import sys

def prune_MPI(model, prune_unit="unstructured", pruning_method="l1", pruned_ratio=0.1):
    device = next(model.parameters()).device

    with torch.no_grad():
        fe_modules = []
        fe_modules += [model.w2v_model.feature_extractor.conv_layers[0][2]]
        fe_modules += [model.w2v_model.feature_extractor.conv_layers[i][0] for i in range(7)]
        fe_modules += [model.w2v_model.post_extract_proj]

        enc_modules = [model.w2v_model.encoder.layers[i].self_attn.k_proj for i in range(12)]
        enc_modules += [model.w2v_model.encoder.layers[i].self_attn.v_proj for i in range(12)]
        enc_modules += [model.w2v_model.encoder.layers[i].self_attn.q_proj for i in range(12)]
        enc_modules += [model.w2v_model.encoder.layers[i].self_attn.out_proj for i in range(12)]
        enc_modules += [model.w2v_model.encoder.layers[i].self_attn_layer_norm for i in range(12)]
        enc_modules += [model.w2v_model.encoder.layers[i].fc1 for i in range(12)]
        enc_modules += [model.w2v_model.encoder.layers[i].fc2 for i in range(12)]
        enc_modules += [model.w2v_model.encoder.layers[i].final_layer_norm for i in range(12)]
        enc_modules += [model.w2v_model.encoder.layer_norm]
        enc_modules += [model.w2v_model.layer_norm]

        modules = fe_modules + enc_modules

        parameters_to_prune = [(module, "weight") for module in modules]

        if prune_unit == "unstructured":
            if pruning_method == "l1":
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=pruned_ratio,
                )
            elif pruning_method == "random":
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.RandomUnstructured,
                    amount=pruned_ratio,
                )

        elif prune_unit == "structured":
            raise NotImplementedError

        for i in range(7):
            spar = 100 * float(torch.sum(model.w2v_model.feature_extractor.conv_layers[i][0].weight == 0)) \
            / float(model.w2v_model.feature_extractor.conv_layers[i][0].weight.nelement())
            spar = round(spar, 2)
            print(
                f"Sparsity in conv{i}.weight: {spar}"
                )
        spar = 100 * float(torch.sum(model.w2v_model.post_extract_proj.weight == 0)) \
        / float(model.w2v_model.post_extract_proj.weight.nelement())
        spar = round(spar, 2)
        print(
                f"Sparsity in conv.post_extract_proj.weight: {spar}"
                )
        for i in range(12):
            spar = 100 * float(torch.sum(model.w2v_model.encoder.layers[i].self_attn.k_proj.weight == 0)) \
            / float(model.w2v_model.encoder.layers[i].self_attn.k_proj.weight.nelement())
            spar = round(spar, 2)
            print(
                f"Sparsity in k_proj{i}.weight: {spar}"
                )
            spar = 100 * float(torch.sum(model.w2v_model.encoder.layers[i].self_attn.q_proj.weight == 0)) \
            / float(model.w2v_model.encoder.layers[i].self_attn.q_proj.weight.nelement())
            spar = round(spar, 2)
            print(
                f"Sparsity in q_proj{i}.weight: {spar}"
                )
            spar = 100 * float(torch.sum(model.w2v_model.encoder.layers[i].self_attn.v_proj.weight == 0)) \
            / float(model.w2v_model.encoder.layers[i].self_attn.v_proj.weight.nelement())
            spar = round(spar, 2)
            print(
                f"Sparsity in v_proj{i}.weight: {spar}"
                )
            spar = 100 * float(torch.sum(model.w2v_model.encoder.layers[i].self_attn.out_proj.weight == 0)) \
            / float(model.w2v_model.encoder.layers[i].self_attn.out_proj.weight.nelement())
            spar = round(spar, 2)
            print(
                f"Sparsity in out_proj{i}.weight: {spar}"
                )
            spar = 100 * float(torch.sum(model.w2v_model.encoder.layers[i].self_attn_layer_norm.weight == 0)) \
            / float(model.w2v_model.encoder.layers[i].self_attn_layer_norm.weight.nelement())
            spar = round(spar, 2)
            print(
                f"Sparsity in attn_ln{i}.weight: {spar}"
                )
            spar = 100 * float(torch.sum(model.w2v_model.encoder.layers[i].fc1.weight == 0)) \
            / float(model.w2v_model.encoder.layers[i].fc1.weight.nelement())
            spar = round(spar, 2)
            print(
                f"Sparsity in fc1{i}.weight: {spar}"
                )
            spar = 100 * float(torch.sum(model.w2v_model.encoder.layers[i].fc2.weight == 0)) \
            / float(model.w2v_model.encoder.layers[i].fc2.weight.nelement())
            spar = round(spar, 2)
            print(
                f"Sparsity in fc2{i}.weight: {spar}"
                )
            spar = 100 * float(torch.sum(model.w2v_model.encoder.layers[i].final_layer_norm.weight == 0)) \
            / float(model.w2v_model.encoder.layers[i].final_layer_norm.weight.nelement())
            spar = round(spar, 2)
            print(
                f"Sparsity in final_ln{i}.weight: {spar}"
                )
        spar = 100 * float(torch.sum(model.w2v_model.encoder.layer_norm.weight == 0)) \
        / float(model.w2v_model.encoder.layer_norm.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in encoder_ln.weight: {spar}"
            )
        spar = 100 * float(torch.sum(model.w2v_model.layer_norm.weight == 0)) \
        / float(model.w2v_model.layer_norm.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in ln.weight: {spar}"
            )

    return model

def move_weight_to_device(model, device):
    fe_modules = []
    fe_modules += [model.w2v_encoder.w2v_model.feature_extractor.conv_layers[0][2]]
    fe_modules += [model.w2v_encoder.w2v_model.feature_extractor.conv_layers[i][0] for i in range(7)]
    fe_modules += [model.w2v_encoder.w2v_model.post_extract_proj]

    enc_modules = [model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.k_proj for i in range(12)]
    enc_modules += [model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.v_proj for i in range(12)]
    enc_modules += [model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.q_proj for i in range(12)]
    enc_modules += [model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.out_proj for i in range(12)]
    enc_modules += [model.w2v_encoder.w2v_model.encoder.layers[i].self_attn_layer_norm for i in range(12)]
    enc_modules += [model.w2v_encoder.w2v_model.encoder.layers[i].fc1 for i in range(12)]
    enc_modules += [model.w2v_encoder.w2v_model.encoder.layers[i].fc2 for i in range(12)]
    enc_modules += [model.w2v_encoder.w2v_model.encoder.layers[i].final_layer_norm for i in range(12)]
    enc_modules += [model.w2v_encoder.w2v_model.encoder.layer_norm]
    enc_modules += [model.w2v_encoder.w2v_model.layer_norm]

    modules = fe_modules + enc_modules

    for module in modules:
        module.weight = module.weight.to(device)
        module.weight = module.weight.half()

    return model