import os
from os.path import join
import argparse
import torch
import json
from collections import defaultdict

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange

from diffmask.models.question_answering_rob_squad_diffmask import (
    RobertaQuestionAnsweringSquadDiffMask,
)
from diffmask.models.question_answering_rob_squad import (
    load_squad
)
from diffmask.utils.plot import plot_squad_attributions, print_attributions, plot_rob_squad_attributions

from collections import OrderedDict
import numpy as np
import itertools
from data.dataset_utils import merge_tokens_into_words
from common.utils import read_json

def condense_input(inputs_dict, feature, device):
    tokens = feature.tokens
    inputs_dict["mask"] = inputs_dict["attention_mask"][:,:len(feature.tokens)].to(device)
    del inputs_dict["attention_mask"]
    inputs_dict['input_ids'] = inputs_dict['input_ids'][:,:len(feature.tokens)].to(device)
    inputs_dict["token_type_ids"] = None

    return inputs_dict


def merge_token_attribution_by_segments(attributions, segments):
    new_val = []
    for a, b in segments:
        new_val.append(torch.sum(attributions[a:b,:], dim=0))
    attention = torch.stack(new_val)
    return attention

def condense_attributions(tokenizer, feature, attribution):
    words, segments = merge_tokens_into_words(tokenizer, feature)    
    aggregated_attribution = merge_token_attribution_by_segments(attribution, segments)
    return words, aggregated_attribution
# ----------- deprecated -----------------------
# def get_impacts_of_entity(tokenizer, feature, attribution, annotation):
#     annotation = list(set(annotation))
#     return aggregated_link_attribution_in_question(tokenizer, feature, attribution, annotation)
# def aggregated_link_attribution_in_question(tokenizer, feature, attribution, targets):
#     target_segments = []
#     for tok in targets:
#         target_segments.extend(extract_token_segments(tokenizer, feature, tok, include_question=False))
#     # for a, b in target_segments:
#     #     print(interp['feature'].tokens[a:b])

#     token_attribution = attribution.numpy()
    
#     selected_idx = list(itertools.chain(*[list(range(a,b)) for (a,b) in target_segments]))    
#     selected_attribution = token_attribution[selected_idx]

#     return_val = np.sum(selected_attribution)
#     return_val = np.sum(selected_attribution) / np.sum(token_attribution)
#     # print(target_segments)
#     # print(targets)
#     # print(np.sum(selected_attribution))
#     # print(np.sum(token_attribution))
#     return return_val
# ----------- deprecated ----------------------

def aggregated_token_attribution_in_context(tokenizer, feature, attribution, targets):
    target_segments = []
    for tok in targets:
        target_segments.extend(extract_token_segments(tokenizer, feature, tok, include_question=True))
    # print(targets)
    # for a, b in target_segments:
    #     print(interp['feature'].tokens[a:b])
    print(attribution)
    token_attribution = attribution.numpy()
    token_attribution[token_attribution < 0] = 0
    doc_tokens = feature.tokens
    context_start = doc_tokens.index(tokenizer.eos_token)

    selected_idx = list(itertools.chain(*[list(range(a,b)) for (a,b) in target_segments]))    
    selected_attribution = token_attribution[selected_idx]

    # return_val = np.sum(selected_attribution)
    # return_val = np.sum(selected_attribution) / np.sum(token_attribution[(context_start + 1):]) / len(selected_idx)
    # return_val = np.sum(selected_attribution) / np.sum(token_attribution)
    # return_val = np.sum(selected_attribution) / len(selected_idx) / np.sum(token_attribution)
    return_val = np.sum(selected_attribution) /len(selected_idx)
    return return_val

def get_impacts_of_property(tokenizer, feature, attribution, annotation):
    properties = annotation['perturb_property']['original_properties']
    properties = list(set(properties))
    return aggregated_token_attribution_in_context(tokenizer, feature, attribution, properties)
    


def list_whole_word_match(l, k, start):
    for p in range(len(k)):
        if l[start + p] != k[p]:
            return False

    # print(l[start + len(k)], l[start + len(k)][0], l[start + len(k)][0].isalnum())
    if (start + len(k)) >= len(l):
        return True
    leading_char =l[start + len(k)][0]
    if leading_char != 'Ġ' and leading_char.isalnum():
        return False    
    return True

def extract_token_segments(tokenizer, feature, tok, include_question=True, include_context=True):
    doc_tokens = feature.tokens
    sub_tokens = tokenizer.tokenize(tok, add_prefix_space=True)

    context_start = doc_tokens.index(tokenizer.eos_token)
    range_left = 0 if include_question else context_start
    range_right = len(doc_tokens) if include_context else context_start

    start_positions = [i for i in range(range_left, range_right) if list_whole_word_match(doc_tokens, sub_tokens, i)]

    segments = [(s, s + len(sub_tokens)) for s in start_positions]

    # special case
    sub_tokens = tokenizer.tokenize(tok)
    if list_whole_word_match(doc_tokens, sub_tokens, 1):
        segments = [(1, 1 + len(sub_tokens))] + segments
    return segments


def read_yesno_perturbations():
    prefix = 'yesno_perturb'
    fnames = os.listdir(prefix)
    fnames.sort(key=lambda x: int(x.split('-')[0]))    

    annotation_dict = OrderedDict()
    for fname in fnames:
        meta = read_json(join(prefix, fname))
        if not (meta['flag_ready'] and meta['flag_including']):
            continue
        quick_id = fname.split('-')[0]
        meta['quick_id'] = quick_id
        annotation_dict[meta['id']] = meta
    return annotation_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--model", type=str, default="checkpoints/squad_roberta-base")
    parser.add_argument(
        "--train_filename",
        type=str,
        default="./features/train_squad_roberta-base_512",
    )
    parser.add_argument(
        "--val_filename",
        type=str,
        default="./features/dev_squad_roberta-base_512",
    )
    parser.add_argument(
        "--test_filename",
        type=str,
        default="./features/yesno-mannual_hpqa_hpqa_roberta-base_512",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gate_bias", action="store_true")
    parser.add_argument("--seed", type=float, default=0)
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/squad-roberta-input-layer_pred=-1/epoch=0-val_acc=0.76-val_f1=0.00-val_l0=0.42.ckpt",
    )

    hparams, _ = parser.parse_known_args()
    torch.manual_seed(hparams.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = hparams.gpus

    device = "cuda:0"
    model = RobertaQuestionAnsweringSquadDiffMask.load_from_checkpoint(hparams.model_path).to(device)
    model.freeze()

    tokenizer = model.tokenizer
    inputs_dicts, features = load_squad(hparams.test_filename, tokenizer, return_dataset=False)


    # sample_id = '5a8c87aa554299653c1aa0ac'
    # inputs_dict = inputs_dicts[sample_id]
    # feature = features[sample_id]
    # inputs_dict = condense_input(inputs_dict, feature, device)
    # tokens = feature.tokens
    # logits_start_orig, logits_end_orig, expected_L0_full = model.forward_explainer(
    #     **inputs_dict, attribution=True
    # )
    # attributions = expected_L0_full.exp()[0,:len(tokens)].cpu()
    # tokens, attributions = condense_attributions(tokenizer, feature, attributions)
    # print(attributions.size())
    # plot_rob_squad_attributions(attributions, tokens, inputs_dict, logits_start_orig, logits_end_orig, 'vis.png', save=True)
    # print_attributions(tokens, attributions.sum(-1))

    annotations_dict = read_yesno_perturbations()
    for qid, annotation in annotations_dict.items():
        inputs_dict = inputs_dicts[qid]
        feature = features[qid]
        inputs_dict = condense_input(inputs_dict, feature, device)
        tokens = feature.tokens
        logits_start_orig, logits_end_orig, expected_L0_full = model.forward_explainer(
            **inputs_dict, attribution=True
        )
        attributions = expected_L0_full.exp()[0,:len(tokens)].cpu()
        # print(attributions)
        # print(attributions.size())
        # plot_rob_squad_attributions(attributions, tokens, inputs_dict, logits_start_orig, logits_end_orig, 'vis.png', save=True)
        token_attributions = attributions.sum(-1)
        print(get_impacts_of_property(tokenizer, feature, token_attributions, annotation))


# deprecated
    # for qid in inputs_dicts:
    #     inputs_dict = inputs_dicts[qid]
    #     feature = features[qid]
    #     inputs_dict = condense_input(inputs_dict, feature, device)
    #     tokens = feature.tokens
    #     logits_start_orig, logits_end_orig, expected_L0_full = model.forward_explainer(
    #         **inputs_dict, attribution=True
    #     )
    #     attributions = expected_L0_full.exp()[0,:len(tokens)].cpu()
    #     # print(attributions)
    #     # print(attributions.size())
    #     # plot_rob_squad_attributions(attributions, tokens, inputs_dict, logits_start_orig, logits_end_orig, 'vis.png', save=True)
    #     token_attributions = attributions.sum(-1)
    #     annotation = annotations[qid]
    #     print(get_impacts_of_entity(tokenizer, feature, token_attributions, annotation))        
