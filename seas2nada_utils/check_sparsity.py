# useage: python3 prune_omp.py finetuned trained_model

import torch
from torch.nn.utils import prune
import sys

from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc
from fairseq import checkpoint_utils, tasks, utils

mode = sys.argv[1]
trained_model = sys.argv[2]

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

    total_elements = 0
    pruned_elements = 0

    for i in range(6):
        total_elements += float(model.feature_extractor.conv_layers[i][0].weight.nelement())
        pruned_elements += float(torch.sum(model.feature_extractor.conv_layers[i][0].weight == 0))

        spar = 100 * float(torch.sum(model.feature_extractor.conv_layers[i][0].weight == 0)) \
        / float(model.feature_extractor.conv_layers[i][0].weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in conv{i}.weight: {spar}"
            )

    total_elements += float(model.post_extract_proj.weight.nelement())
    pruned_elements += float(torch.sum(model.post_extract_proj.weight == 0))
    spar = 100 * float(torch.sum(model.post_extract_proj.weight == 0)) \
    / float(model.post_extract_proj.weight.nelement())
    spar = round(spar, 2)
    print(
            f"Sparsity in conv.post_extract_proj.weight: {spar}"
            )
            
    for i in range(12):
        total_elements += float(model.encoder.layers[i].self_attn.k_proj.weight.nelement())
        pruned_elements += float(torch.sum(model.encoder.layers[i].self_attn.k_proj.weight == 0))
        spar = 100 * float(torch.sum(model.encoder.layers[i].self_attn.k_proj.weight == 0)) \
        / float(model.encoder.layers[i].self_attn.k_proj.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in k_proj{i}.weight: {spar}"
            )
        
        total_elements += float(model.encoder.layers[i].self_attn.q_proj.weight.nelement())
        pruned_elements += float(torch.sum(model.encoder.layers[i].self_attn.q_proj.weight == 0))
        spar = 100 * float(torch.sum(model.encoder.layers[i].self_attn.q_proj.weight == 0)) \
        / float(model.encoder.layers[i].self_attn.q_proj.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in q_proj{i}.weight: {spar}"
            )

        total_elements += float(model.encoder.layers[i].self_attn.v_proj.weight.nelement())
        pruned_elements += float(torch.sum(model.encoder.layers[i].self_attn.v_proj.weight == 0))
        spar = 100 * float(torch.sum(model.encoder.layers[i].self_attn.v_proj.weight == 0)) \
        / float(model.encoder.layers[i].self_attn.v_proj.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in v_proj{i}.weight: {spar}"
            )

        total_elements += float(model.encoder.layers[i].self_attn.out_proj.weight.nelement())
        pruned_elements += float(torch.sum(model.encoder.layers[i].self_attn.out_proj.weight == 0))
        spar = 100 * float(torch.sum(model.encoder.layers[i].self_attn.out_proj.weight == 0)) \
        / float(model.encoder.layers[i].self_attn.out_proj.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in out_proj{i}.weight: {spar}"
            )

        total_elements += float(model.encoder.layers[i].self_attn_layer_norm.weight.nelement())
        pruned_elements += float(torch.sum(model.encoder.layers[i].self_attn_layer_norm.weight == 0))
        spar = 100 * float(torch.sum(model.encoder.layers[i].self_attn_layer_norm.weight == 0)) \
        / float(model.encoder.layers[i].self_attn_layer_norm.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in attn_ln{i}.weight: {spar}"
            )

        total_elements += float(model.encoder.layers[i].fc1.weight.nelement())
        pruned_elements += float(torch.sum(model.encoder.layers[i].fc1.weight == 0))
        spar = 100 * float(torch.sum(model.encoder.layers[i].fc1.weight == 0)) \
        / float(model.encoder.layers[i].fc1.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in fc1{i}.weight: {spar}"
            )

        total_elements += float(model.encoder.layers[i].fc2.weight.nelement())
        pruned_elements += float(torch.sum(model.encoder.layers[i].fc2.weight == 0))
        spar = 100 * float(torch.sum(model.encoder.layers[i].fc2.weight == 0)) \
        / float(model.encoder.layers[i].fc2.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in fc2{i}.weight: {spar}"
            )

        total_elements += float(model.encoder.layers[i].final_layer_norm.weight.nelement())
        pruned_elements += float(torch.sum(model.encoder.layers[i].final_layer_norm.weight == 0))
        spar = 100 * float(torch.sum(model.encoder.layers[i].final_layer_norm.weight == 0)) \
        / float(model.encoder.layers[i].final_layer_norm.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in final_ln{i}.weight: {spar}"
            )

    total_elements += float(model.encoder.layer_norm.weight.nelement())
    pruned_elements += float(torch.sum(model.encoder.layer_norm.weight == 0))
    spar = 100 * float(torch.sum(model.encoder.layer_norm.weight == 0)) \
    / float(model.encoder.layer_norm.weight.nelement())
    spar = round(spar, 2)
    print(
        f"Sparsity in encoder_ln.weight: {spar}"
        )

    total_elements += float(model.layer_norm.weight.nelement())
    pruned_elements += float(torch.sum(model.layer_norm.weight == 0))
    spar = 100 * float(torch.sum(model.layer_norm.weight == 0)) \
    / float(model.layer_norm.weight.nelement())
    spar = round(spar, 2)
    print(
        f"Sparsity in ln.weight: {spar}"
        )

    spar = 100 * total_elements / pruned_elements
    spar = round(spar, 2)
    print(
        f"Sparsity total: {spar}"
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

    total_elements = 0
    pruned_elements = 0

    for i in range(6):
        total_elements += float(model.w2v_encoder.w2v_model.feature_extractor.conv_layers[i][0].weight.nelement())
        pruned_elements += float(torch.sum(model.w2v_encoder.w2v_model.feature_extractor.conv_layers[i][0].weight == 0))

        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.feature_extractor.conv_layers[i][0].weight == 0)) \
        / float(model.w2v_encoder.w2v_model.feature_extractor.conv_layers[i][0].weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in conv{i}.weight: {spar}"
            )

    total_elements += float(model.w2v_encoder.w2v_model.post_extract_proj.weight.nelement())
    pruned_elements += float(torch.sum(model.w2v_encoder.w2v_model.post_extract_proj.weight == 0))
    spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.post_extract_proj.weight == 0)) \
    / float(model.w2v_encoder.w2v_model.post_extract_proj.weight.nelement())
    spar = round(spar, 2)
    print(
            f"Sparsity in conv.post_extract_proj.weight: {spar}"
            )
            
    for i in range(12):
        total_elements += float(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.k_proj.weight.nelement())
        pruned_elements += float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.k_proj.weight == 0))
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.k_proj.weight == 0)) \
        / float(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.k_proj.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in k_proj{i}.weight: {spar}"
            )
        
        total_elements += float(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.q_proj.weight.nelement())
        pruned_elements += float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.q_proj.weight == 0))
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.q_proj.weight == 0)) \
        / float(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.q_proj.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in q_proj{i}.weight: {spar}"
            )

        total_elements += float(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.v_proj.weight.nelement())
        pruned_elements += float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.v_proj.weight == 0))
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.v_proj.weight == 0)) \
        / float(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.v_proj.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in v_proj{i}.weight: {spar}"
            )

        total_elements += float(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.out_proj.weight.nelement())
        pruned_elements += float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.out_proj.weight == 0))
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.out_proj.weight == 0)) \
        / float(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn.out_proj.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in out_proj{i}.weight: {spar}"
            )

        total_elements += float(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn_layer_norm.weight.nelement())
        pruned_elements += float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn_layer_norm.weight == 0))
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn_layer_norm.weight == 0)) \
        / float(model.w2v_encoder.w2v_model.encoder.layers[i].self_attn_layer_norm.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in attn_ln{i}.weight: {spar}"
            )

        total_elements += float(model.w2v_encoder.w2v_model.encoder.layers[i].fc1.weight.nelement())
        pruned_elements += float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].fc1.weight == 0))
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].fc1.weight == 0)) \
        / float(model.w2v_encoder.w2v_model.encoder.layers[i].fc1.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in fc1{i}.weight: {spar}"
            )

        total_elements += float(model.w2v_encoder.w2v_model.encoder.layers[i].fc2.weight.nelement())
        pruned_elements += float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].fc2.weight == 0))
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].fc2.weight == 0)) \
        / float(model.w2v_encoder.w2v_model.encoder.layers[i].fc2.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in fc2{i}.weight: {spar}"
            )

        total_elements += float(model.w2v_encoder.w2v_model.encoder.layers[i].final_layer_norm.weight.nelement())
        pruned_elements += float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].final_layer_norm.weight == 0))
        spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layers[i].final_layer_norm.weight == 0)) \
        / float(model.w2v_encoder.w2v_model.encoder.layers[i].final_layer_norm.weight.nelement())
        spar = round(spar, 2)
        print(
            f"Sparsity in final_ln{i}.weight: {spar}"
            )

    total_elements += float(model.w2v_encoder.w2v_model.encoder.layer_norm.weight.nelement())
    pruned_elements += float(torch.sum(model.w2v_encoder.w2v_model.encoder.layer_norm.weight == 0))
    spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.encoder.layer_norm.weight == 0)) \
    / float(model.w2v_encoder.w2v_model.encoder.layer_norm.weight.nelement())
    spar = round(spar, 2)
    print(
        f"Sparsity in encoder_ln.weight: {spar}"
        )

    total_elements += float(model.w2v_encoder.w2v_model.layer_norm.weight.nelement())
    pruned_elements += float(torch.sum(model.w2v_encoder.w2v_model.layer_norm.weight == 0))
    spar = 100 * float(torch.sum(model.w2v_encoder.w2v_model.layer_norm.weight == 0)) \
    / float(model.w2v_encoder.w2v_model.layer_norm.weight.nelement())
    spar = round(spar, 2)
    print(
        f"Sparsity in ln.weight: {spar}"
        )
    
    spar = 100 * pruned_elements / total_elements
    spar = round(spar, 2)
    print(
        f"Sparsity total: {spar}"
        )