import json
import torch
import numpy as np
from sklearn.decomposition import PCA
import random

def process_paired_data(data, tokenizer, seed=0, ntrain=None):

    random.seed(seed)
    user_tag = "[INST]"
    assistant_tag = "[/INST]"
    template_str = "Consider the amount of {type} in this statement: {response}. The amount of {type} is:"

    ntrain = 100*16
    # Each element in data is [good, bad]
    good_statements = [good for good, _ in data[:ntrain]]
    bad_statements = [bad for _, bad in data[:ntrain]]

    violent = []
    nonviolent = []
    for stm in range(len(good_statements)):
        nonviolent.append(f"{user_tag} {template_str.format(type='violence', response=good_statements[stm])} {assistant_tag}")
        violent.append(f"{user_tag} {template_str.format(type='violence', response=bad_statements[stm])} {assistant_tag} ")

    grouped = [[nv, v] for nv, v in zip(nonviolent, violent)]

    train_pairs = grouped[:int(0.6 * ntrain)]
    test_pairs = grouped[int(0.6 * ntrain):]
   
    train_data = []
    train_labels = []
    for anti, biased in train_pairs:
        original = anti
        pair = [anti, biased]
        random.shuffle(pair)
        train_data.extend(pair)
        train_labels.append([s == original for s in pair])

    test_data = [s for pair in test_pairs for s in pair]

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1, 0] * (len(test_data) // 2)]}
    }

def extract_data_alert_dpo_repe(data_type=set(['crime_injury', 
                                      'crime_kidnapp', 
                                      'self_harm_suicide',
                                      'weapon_firearm', 
                                      'weapon_chemical', 
                                      'weapon_other',
                                      'weapon_biological',
                                      'weapon_radioactive'])):

    data = []
    with open('/home/shraya/mistral_circuit_tracing/alert_dpo.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                curr_line = json.loads(line)
                if curr_line['category'] in data_type:
                    good_line = curr_line['chosen']
                    bad_line = curr_line['rejected']
                    data.append([good_line, bad_line])
    return data

def extract_data_alert_dpo_caa(data_type=set(['crime_injury', 
                                      'crime_kidnapp', 
                                      'self_harm_suicide',
                                      'weapon_firearm', 
                                      'weapon_chemical', 
                                      'weapon_other',
                                      'weapon_biological',
                                      'weapon_radioactive'])):

    good_data = []
    bad_data = []
    with open('/home/shraya/mistral_circuit_tracing/alert_dpo.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                curr_line = json.loads(line)
                if curr_line['category'] in data_type:
                    curr_line['chosen'] += 'A'
                    curr_line['rejected'] += 'B'
                    good_line = curr_line['prompt'] + curr_line['chosen']
                    bad_line = curr_line['prompt'] + curr_line['rejected']
                    good_data.append(good_line)
                    bad_data.append(bad_line)
    good_data = good_data[:50*16]
    bad_data = bad_data[:50*16]
    return (good_data[:int(len(good_data) * 0.8)], 
            good_data[int(len(good_data) * 0.8):],
            bad_data[:int(len(bad_data) * 0.8)], 
            bad_data[int(len(bad_data) * 0.8):])


def derive_contrast_vector(layer_embs_mat, method='pca'):
    '''
    takes in a list of hidden embeddings positive / negative difference from a layer --> computes 'method' (e.g. PCA, average)
    '''
    if method == 'pca':
        layer_embs_mat_mean = torch.mean(layer_embs_mat, axis=0)
        layer_embs_mat -= layer_embs_mat_mean
        pca_model = PCA(n_components=1).fit(layer_embs_mat)
        return pca_model.components_, layer_embs_mat_mean
    elif method == 'avg':
        return torch.mean(layer_embs_mat, axis=0)

def modify_hidden_state(hidden_state, contrast_vector, coefficient):
    if isinstance(contrast_vector, np.ndarray):
        contrast_vector = torch.from_numpy(contrast_vector)
    contrast_vector = contrast_vector.view(1, 1, -1)
    contrast_vector = contrast_vector.to(hidden_state[0].device, dtype=hidden_state[0].dtype)
    modified = hidden_state[0].clone()
    modified += contrast_vector * coefficient
    return (modified,)

def hook_model(model, contrast_vectors, coefficient, desired_layer_idcs):
    # print('contrast vectors type from hook model function', type(contrast_vectors))
    handles = []
    def get_hook(layer_idx):
        def hook(module, input, output):
            return modify_hidden_state(output, contrast_vectors[layer_idx], coefficient)
        return hook
    
    n_layers = len(model.model.layers)
    for i, block in enumerate(model.model.layers):
        neg_idx = i - n_layers
        if neg_idx in desired_layer_idcs:
            handle = block.register_forward_hook(get_hook(neg_idx))
            handles.append(handle)
    return handles