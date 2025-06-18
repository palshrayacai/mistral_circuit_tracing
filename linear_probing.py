
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from repe import repe_pipeline_registry
import utils
from utils import extract_data_alert_dpo_repe, process_paired_data, extract_data_alert_dpo_caa
from utils import derive_contrast_vector, modify_hidden_state, hook_model
import einops
import wandb
import random
from collections import defaultdict


def linear_probe_repe(model, tokenizer):
    repe_pipeline_registry()
    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    n_difference = 1
    direction_method = 'pca'
    rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer)

    data = extract_data_alert_dpo_repe()
    dataset = process_paired_data(data, tokenizer)

    train_data = dataset['train']
    test_data = dataset['test']

    violence_rep_reader = rep_reading_pipeline.get_directions(
        dataset['train']['data'], 
        rep_token=rep_token, 
        hidden_layers=hidden_layers, 
        n_difference=n_difference, 
        train_labels=dataset['train']['labels'], 
        direction_method=direction_method,
        batch_size=32,
    )
    
    V_tests = rep_reading_pipeline(
        dataset['test']['data'], 
        rep_token=rep_token, 
        hidden_layers=hidden_layers, 
        rep_reader=violence_rep_reader,
        batch_size=32)

    results = {layer: {} for layer in hidden_layers}
    rep_readers_means = {}
    rep_readers_means['violence'] = {layer: 0 for layer in hidden_layers}

    for layer in hidden_layers:
        V_test = [V[layer] for V in V_tests]
        rep_readers_means['violence'][layer] = np.mean(V_test)
        V_test = [V_test[i:i+2] for i in range(0, len(V_test), 2)]
        
        sign = violence_rep_reader.direction_signs[layer]

        eval_func = min if sign == -1 else max
        cors = np.mean([eval_func(V) == V[0] for V in V_test])
        
        results[layer] = cors
    return violence_rep_reader


def steering_model_repe(model, tokenizer, rep_reader, desired_layer_idcs=[-36, -35, -34, -33, -31, -30, -28, -26, -25], user_inputs = ['When is it ok to murder someone?'], coefficient=-2):
    '''
    negative coefficient --> unsafe
    positive coefficient --> safe
    '''
    layer_ids = desired_layer_idcs
    block_name = "decoder_block"
    control_method = "reading_vec"

    rep_control_pipeline = pipeline(
        "rep-control",
        model=model,
        tokenizer=tokenizer,
        layers=layer_ids,
        block_name=block_name,
        control_method=control_method
    )

    user_tag = '''[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses do not propogate violence, or any other harmful behavior.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>

    '''

    assistant_tag = "[/INST]"

    inputs = [f"{user_tag} {inp} {assistant_tag}" for inp in user_inputs]

    coeff = coefficient
    activations = {
        layer: torch.tensor(
            coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]
        ).to(model.device).half()
        for layer in layer_ids
    }

    max_new_tokens = 100

    baseline_outputs = rep_control_pipeline(
        inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False
    )

    controlled_outputs = rep_control_pipeline(
        inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False
    )

    for inp, base, steered in zip(inputs, baseline_outputs, controlled_outputs):
        print("===== ORIGINAL PROMPT =====")
        print(inp)
        print("===== BASELINE OUTPUT =====")
        print(base[0]['generated_text'].replace(inp, "").strip())
        print("===== CONTROLLED OUTPUT =====")
        print(steered[0]['generated_text'].replace(inp, "").strip())
        print("\n" + "="*60 + "\n")


def linear_probe_caa(model, tokenizer):
    (good_data_train, good_data_test, bad_data_train, bad_data_test) = extract_data_alert_dpo_caa()
    batch_size = 16
    good_outputs = {} #indexed by layer, each value is a list of (B, S, H) tensors
    bad_outputs = {}
    good_outputs_hidden_states = defaultdict(list)
    bad_outputs_hidden_states = defaultdict(list)
    for idx in tqdm(range(0, len(good_data_train), batch_size), desc="forward pass on train"):
        good_inp, bad_inp = tokenizer(good_data_train[idx:idx+batch_size], return_tensors='pt', padding=True, truncation=True), tokenizer(bad_data_train[idx:idx+batch_size], return_tensors='pt',  padding=True, truncation=True)
        with torch.no_grad():
            good_outputs[idx] = model(**good_inp, output_hidden_states=True)
            bad_outputs[idx] = model(**bad_inp, output_hidden_states=True)
        for layer in range(-1, -model.config.num_hidden_layers, -1):
            good_outputs_hidden_states[layer].append(good_outputs[idx].hidden_states[layer]) #(B, S, H)
            bad_outputs_hidden_states[layer].append(bad_outputs[idx].hidden_states[layer])
    
    good_outputs_test = {}
    bad_outputs_test = {}
    good_outputs_hidden_states_test = defaultdict(list)
    bad_outputs_hidden_states_test = defaultdict(list)
    for idx in tqdm(range(0, len(good_data_test), batch_size), desc="forward pass on test"):
        good_inp, bad_inp = tokenizer(good_data_test[idx:idx+batch_size], return_tensors='pt', padding=True, truncation=True), tokenizer(bad_data_test[idx:idx+batch_size], return_tensors='pt',  padding=True, truncation=True)
        with torch.no_grad():
            good_outputs_test[idx] = model(**good_inp, output_hidden_states=True)
            bad_outputs_test[idx] = model(**bad_inp, output_hidden_states=True)
        for layer in range(-1, -model.config.num_hidden_layers, -1):
            good_outputs_hidden_states_test[layer].append(good_outputs_test[idx].hidden_states[layer])
            bad_outputs_hidden_states_test[layer].append(bad_outputs_test[idx].hidden_states[layer])

    differences = {}
    contrast_vectors = {}
    layer_accuracies = []
    for idx in tqdm(range(-1, -model.config.num_hidden_layers, -1), desc='compute contrast vec + classify'):
        #concatenate all hidden embeddings of last token embeddings of training data
        good_last_tokens_train, bad_last_tokens_train, good_last_tokens_test, bad_last_tokens_test  = [], [], [], []
        for gt in good_outputs_hidden_states[idx]:
            good_last_tokens_train.append(gt[:, -1, :])
        for bt in bad_outputs_hidden_states[idx]:
            bad_last_tokens_train.append(bt[:, -1, :]) #(B, H)
    
        good_outputs_hidden_states[idx] = torch.cat(good_last_tokens_train, dim=0) #(B, H)
        bad_outputs_hidden_states[idx] = torch.cat(bad_last_tokens_train, dim=0)
        
        #concatenate all hidden embeddings of last token embeddings of test data
        for gt in good_outputs_hidden_states_test[idx]:
            good_last_tokens_test.append(gt[:, -1, :])
        for bt in bad_outputs_hidden_states_test[idx]:
            bad_last_tokens_test.append(bt[:, -1, :]) #(B, H)

        good_outputs_hidden_states_test[idx] = torch.cat(good_last_tokens_test, dim=0) #(B, H)
        bad_outputs_hidden_states_test[idx] = torch.cat(bad_last_tokens_test, dim=0)


        #compute difference vector of all contrastive prompts of training embeddings + extract contrastive vector
        difference = good_outputs_hidden_states[idx] - bad_outputs_hidden_states[idx] #(B, H)
        differences[idx] =  difference #(B H)
        contrast_vec = derive_contrast_vector(difference, method='avg') #(1, H)
        contrast_vectors[idx] = contrast_vec

    return contrast_vectors, differences, layer_accuracies

def steering_model_caa(model, tokenizer, contrast_vectors, desired_layer_idcs, user_inputs = ['When is it ok to murder someone?'], coefficient=-2):
    user_tag = '''[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses do not propogate violence, or any other harmful behavior.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>

    '''
    # print('type model is', type(model))
    # print('type token is', type(tokenizer))
    # print('type contrast vectors', type(contrast_vectors))
    # print('type desired layer idcs', type(desired_layer_idcs))
    # print('user inps', type(user_inputs))

    assistant_tag = "[/INST]"
    inputs = [f"{user_tag} {inp} {assistant_tag}" for inp in user_inputs]
    controlled_outputs = []
    baseline_outputs = []
    for input_text in inputs:
        model_input = tokenizer(input_text, return_tensors="pt")
        model_input = {k: v.to(model.device) for k, v in model_input.items()}
        with torch.no_grad():
            generated_ids = model.generate(
                **model_input,
                max_new_tokens=100,
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id, 
            )
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        baseline_outputs.append(output_text)
    
    hook_handles = utils.hook_model(model, contrast_vectors, coefficient, desired_layer_idcs)
    for input_text in inputs:
        model_input = tokenizer(input_text, return_tensors="pt")
        model_input = {k: v.to(model.device) for k, v in model_input.items()}
        with torch.no_grad():
            generated_ids = model.generate(
                **model_input,
                max_new_tokens=100,
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id, 
            )

        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        controlled_outputs.append(output_text)
    for h in hook_handles:
        h.remove()
    
    for inp, base, steered in zip(inputs, baseline_outputs, controlled_outputs):
        print("===== ORIGINAL PROMPT =====")
        print(inp)
        print("===== BASELINE OUTPUT =====")
        print(base)
        print("===== CONTROLLED OUTPUT =====")
        print(steered)
        print("\n" + "="*60 + "\n")


