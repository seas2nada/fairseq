# useage: python3 prune_omp.py finetuned trained_model save_model_name structured pruning_method 0.x

import torch
from torch.nn.utils import prune
import sys

from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc
from fairseq import checkpoint_utils, tasks, utils

mode = sys.argv[1]
trained_model = sys.argv[2]
pruned_model = sys.argv[3]
prune_unit = sys.argv[4] # (structured or unstructured)
pruning_method = sys.argv[5] # (l1 or random)
pruned_ratio = float(sys.argv[6])

if mode == "pretrained":
    # build model
    state = checkpoint_utils.load_checkpoint_to_cpu(trained_model)
    w2v_args = state.get("cfg", None)
    if w2v_args is None:
        w2v_args = convert_namespace_to_omegaconf(state["args"])
    w2v_args.criterion = None
    w2v_args.lr_scheduler = None

    task = tasks.setup_task(w2v_args.task, from_checkpoint=True)
    model = task.build_model(w2v_args.model, from_checkpoint=True)

    # copy trained parameters
    with torch.no_grad():
        old_dict = torch.load(trained_model)["model"]
        for name, p in model.named_parameters():
            if name not in old_dict.keys():
                raise NotImplementedError(f"{name} not in old_dict")
            p.copy_(old_dict[name])

    fe_modules = []
    fe_modules += [model.feature_extractor.conv_layers[0][2]]
    fe_modules += [model.feature_extractor.conv_layers[i][0] for i in range(7)]
    fe_modules += [model.post_extract_proj]

    enc_modules = [model.encoder.layers[i].self_attn.k_proj for i in range(12)]
    enc_modules += [model.encoder.layers[i].self_attn.v_proj for i in range(12)]
    enc_modules += [model.encoder.layers[i].self_attn.q_proj for i in range(12)]
    enc_modules += [model.encoder.layers[i].self_attn.out_proj for i in range(12)]
    enc_modules += [model.encoder.layers[i].self_attn_layer_norm for i in range(12)]
    enc_modules += [model.encoder.layers[i].fc1 for i in range(12)]
    enc_modules += [model.encoder.layers[i].fc2 for i in range(12)]
    enc_modules += [model.encoder.layers[i].final_layer_norm for i in range(12)]
    enc_modules += [model.encoder.layer_norm]
    enc_modules += [model.layer_norm]

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

    for module, name in parameters_to_prune:
        prune.remove(module, 'weight')

    for i in range(7):
        spar = 100 * float(torch.sum(model.feature_extractor.conv_layers[i][0].weight == 0)) \
        / float(model.feature_extractor.conv_layers[i][0].weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in conv{i}.weight: {spar}"
            )
    spar = 100 * float(torch.sum(model.post_extract_proj.weight == 0)) \
    / float(model.post_extract_proj.weight.nelement())
    spar = round(spar, 2)
    print(
            f"Sparsity in conv.post_extract_proj.weight: {spar}"
            )
    for i in range(12):
        spar = 100 * float(torch.sum(model.encoder.layers[i].self_attn.k_proj.weight == 0)) \
        / float(model.encoder.layers[i].self_attn.k_proj.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in k_proj{i}.weight: {spar}"
            )
        spar = 100 * float(torch.sum(model.encoder.layers[i].self_attn.q_proj.weight == 0)) \
        / float(model.encoder.layers[i].self_attn.q_proj.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in q_proj{i}.weight: {spar}"
            )
        spar = 100 * float(torch.sum(model.encoder.layers[i].self_attn.v_proj.weight == 0)) \
        / float(model.encoder.layers[i].self_attn.v_proj.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in v_proj{i}.weight: {spar}"
            )
        spar = 100 * float(torch.sum(model.encoder.layers[i].self_attn.out_proj.weight == 0)) \
        / float(model.encoder.layers[i].self_attn.out_proj.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in out_proj{i}.weight: {spar}"
            )
        spar = 100 * float(torch.sum(model.encoder.layers[i].self_attn_layer_norm.weight == 0)) \
        / float(model.encoder.layers[i].self_attn_layer_norm.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in attn_ln{i}.weight: {spar}"
            )
        spar = 100 * float(torch.sum(model.encoder.layers[i].fc1.weight == 0)) \
        / float(model.encoder.layers[i].fc1.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in fc1{i}.weight: {spar}"
            )
        spar = 100 * float(torch.sum(model.encoder.layers[i].fc2.weight == 0)) \
        / float(model.encoder.layers[i].fc2.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in fc2{i}.weight: {spar}"
            )
        spar = 100 * float(torch.sum(model.encoder.layers[i].final_layer_norm.weight == 0)) \
        / float(model.encoder.layers[i].final_layer_norm.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in final_ln{i}.weight: {spar}"
            )
    spar = 100 * float(torch.sum(model.encoder.layer_norm.weight == 0)) \
    / float(model.encoder.layer_norm.weight.nelement())
    spar = round(spar, 2)
    print(
        f"Sparsity in encoder_ln.weight: {spar}"
        )
    spar = 100 * float(torch.sum(model.layer_norm.weight == 0)) \
    / float(model.layer_norm.weight.nelement())
    spar = round(spar, 2)
    print(
        f"Sparsity in ln.weight: {spar}"
        )

elif mode == "finetuned":
    # build model
    state = checkpoint_utils.load_checkpoint_to_cpu(trained_model)
    w2v_args = state.get("cfg", None)
    if w2v_args is None:
        w2v_args = convert_namespace_to_omegaconf(state["args"])
    w2v_args.criterion = None
    w2v_args.lr_scheduler = None

    task = tasks.setup_task(w2v_args.task, from_checkpoint=True)
    model = task.build_model(w2v_args.model, from_checkpoint=True)

    # copy trained parameters
    with torch.no_grad():
        old_dict = torch.load(trained_model)["model"]
        for name, p in model.named_parameters():
            if name not in old_dict.keys():
                raise NotImplementedError(f"{name} not in old_dict")
            p.copy_(old_dict[name])

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
    enc_modules += [model.w2v_encoder.proj]

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
    
    for module, name in parameters_to_prune:
        prune.remove(module, 'weight')

    for i in range(7):
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.feature_extractor.conv_layers[i][0].weight == 0)) \
        / float(model.w2v_encoder.w2v_model.feature_extractor.conv_layers[i][0].weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in conv{i}.weight: {spar}"
            )
    spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.post_extract_proj.weight == 0)) \
    / float(model.w2v_encoder.w2v_model.post_extract_proj.weight.nelement())
    spar = round(spar, 2)
    print(
            f"Sparsity in conv.post_extract_proj.weight: {spar}"
            )
    for i in range(12):
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.k_proj.weight == 0)) \
        / float(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.k_proj.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in k_proj{i}.weight: {spar}"
            )
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.q_proj.weight == 0)) \
        / float(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.q_proj.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in q_proj{i}.weight: {spar}"
            )
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.v_proj.weight == 0)) \
        / float(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.v_proj.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in v_proj{i}.weight: {spar}"
            )
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.out_proj.weight == 0)) \
        / float(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.out_proj.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in out_proj{i}.weight: {spar}"
            )
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn_layer_norm.weight == 0)) \
        / float(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn_layer_norm.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in attn_ln{i}.weight: {spar}"
            )
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].fc1.weight == 0)) \
        / float(model.w2v_encoder.w2v_model.encoder.layers[i].fc1.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in fc1{i}.weight: {spar}"
            )
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].fc2.weight == 0)) \
        / float(model.w2v_encoder.w2v_model.encoder.layers[i].fc2.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in fc2{i}.weight: {spar}"
            )
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].final_layer_norm.weight == 0)) \
        / float(model.w2v_encoder.w2v_model.encoder.layers[i].final_layer_norm.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in final_ln{i}.weight: {spar}"
            )
    spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layer_norm.weight == 0)) \
    / float(model.w2v_encoder.w2v_model.encoder.layer_norm.weight.nelement())
    spar = round(spar, 2)
    print(
        f"Sparsity in encoder_ln.weight: {spar}"
        )
    spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.layer_norm.weight == 0)) \
    / float(model.w2v_encoder.w2v_model.layer_norm.weight.nelement())
    spar = round(spar, 2)
    print(
        f"Sparsity in ln.weight: {spar}"
        )
    spar = 100 * float(torch.sum(model.w2v_encoder.proj.weight == 0)) \
    / float(model.w2v_encoder.proj.weight.nelement())
    spar = round(spar, 2)
    print(
        f"Sparsity in proj.weight: {spar}"
        )

old_dict = torch.load(trained_model)
new_dict_model = model.state_dict()
old_dict['model'] = new_dict_model
torch.save(old_dict, pruned_model)