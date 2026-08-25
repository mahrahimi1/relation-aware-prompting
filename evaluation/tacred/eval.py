import json
import time
import sys
import csv

import openai
from openai import OpenAI
import backoff

from transformers import pipeline
import torch
from tqdm import tqdm
import numpy as np

from tacred_utils import get_example_fields, get_label_id, precision_recall_fscore_exclude_nota, get_relation_set, \
                         get_relation_set_string, get_clean_relation_label, get_definition, \
                         get_actual_relation_label, get_options, get_examples, Model_Family, Open_Source_API_Mode



@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_completion_openai(prompt, client, model):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


def get_completion_open_source(prompt, pipe):
    messages = [{"role": "user", "content": prompt}]

    outputs = pipe(
        messages,
        max_new_tokens=256,
        do_sample=False,
        temperature=None, top_k=None, top_p=None,
        pad_token_id=pipe.tokenizer.eos_token_id
    )

    response = outputs[0]["generated_text"][-1]["content"]
    return response


def get_completion_open_source_api(prompt, client, model):
    while True:
        try:
            messages = [{"role": "user", "content": prompt}]
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=256,
                temperature=0
            )
            response = completion.choices[0].message.content
            break
        except Exception as e:
            print("error:", str(e), "  retrying...")
            time.sleep(2)

    return response


def get_completion_mistral_api(prompt, client, model):
    messages = [{"role": "user", "content": prompt}]

    chat_response = client.chat.complete(
        model= model,
        messages = messages,
        temperature=0,
        max_tokens=256,
    )

    return chat_response.choices[0].message.content


def get_completion(open_source_api_mode, model_family, prompt, pipe, model):
    if model_family == Model_Family.OPENAI:
        return get_completion_openai(prompt, pipe, model)

    elif (model_family == Model_Family.OPEN_SOURCE) or (model_family == Model_Family.LLAMA_31_8B):
        if open_source_api_mode is None:
            return get_completion_open_source(prompt, pipe)

        elif open_source_api_mode == Open_Source_API_Mode.HUGGING_FACE:
            return get_completion_open_source_api(prompt, pipe, model)

        elif open_source_api_mode == Open_Source_API_Mode.MISTRAL:
            return get_completion_mistral_api(prompt, pipe, model)

        else:
            raise Exception(f"Open source API mode '{open_source_api_mode}' not supported.")

    else:
        raise Exception(f"Model Family '{model_family}' Not Supported.")


def get_clean_entity_type(entity_type):
    clean_entity_type = entity_type.replace("_", " ")
    return clean_entity_type


def trim_response(response):
    if response[:8] == "```json\n":
        return response[8:-4]
    else:
        return response


def match_response_to_label(response, relation_set):
    option_labels, options, _ = get_options(relation_set)

    if response in option_labels:
        return option_labels[response]

    if (response[-1] == '.') and (response[:-1] in option_labels):
        return option_labels[response[:-1]]

    if response in options:
        return options[response]

    option_labels_keys = list(option_labels.keys())
    options_keys = list(options.keys())
    for i in range(len(option_labels)):
        option_label = option_labels_keys[i]
        clean_relation = options_keys[i]
        if response == f"{option_label}. {clean_relation}":
            return options[clean_relation]

    split = response.split('.')
    if len(split) == 2:
        if split[0] in option_labels:
            try:
                return options[split[1].strip()]
            except:
                return option_labels[split[0]]

    if len(split) > 2:
        if split[0] in option_labels:
            return option_labels[split[0]]

    raise Exception("Response cannot be matched to any label.")


def get_patterns_string(patterns):
    if len(patterns) == 1:
        return f'"{patterns[0]}"'

    if len(patterns) == 2:
        return f'"{patterns[0]}" and "{patterns[1]}"'

    s = ""
    for pattern in patterns[:-1]:
        s += f'"{pattern}", '
    s += "and " + f'"{patterns[-1]}"'

    return s


def get_summarization_prompt(example):
    subj = example["subj"]
    obj = example["obj"]
    context = example["context"]

    prompt = f'Summarize the relation between the words "{subj}" and "{obj}" in the following sentence. ' \
             f'In the summary, ignore everything else other than the relation between "{subj}" and "{obj}". ' \
             f'Your summary must include the words "{subj}" and "{obj}". ' \
             f'In your summary, prefix "{subj}" with tag <ENT0> and suffix it with tag </ENT0>. ' \
             f'Also, prefix "{obj}" with tag <ENT1> and suffix it with </ENT1>. ' \
             f'Use no more than 10 words for the summary.\n\n' \
             f'Sentence: {context}'
    return prompt


def get_prompt(example, n, experiment_name):
    subj = example["subj"]
    obj = example["obj"]
    subj_type = example["subj_type"]
    obj_type = example["obj_type"]
    summarized_context = example["summarized_context"]

    relation_set = get_relation_set(subj_type, obj_type)
    _, _, option_list_string = get_options(relation_set)
    relation_set_string = get_relation_set_string(relation_set)

    prompt = "I would like you to perform the task of relation extraction. " \
             "In this task, you are given a sentence and a pair of entities in the sentence. " \
             "Your job is to select the relation between the two entities from a predefined set of candidate relations.\n\n" \
    
    prompt += f"The predefined relations are {relation_set_string}.\n\n"

    prompt += "The definition of each relation is as follows. Note that in relation examples or relation instances, <ENT0> is replaced with actual entity mention and is prefixed with tag <ENT0> and suffixed with tag </ENT0>. Also, <ENT1> is replaced with actual entity mention and is prefixed with tag <ENT1> and suffixed with </ENT1>.\n"

    for relation in relation_set:
        if relation == "no_relation":
            continue
        relation_label = get_clean_relation_label(relation)
        relation_definition = get_definition(relation)
        prompt += f'\n{relation_label}: The binary relation "{relation_label}" between ' \
                  f'entity placeholders <ENT0> and <ENT1> is defined by "{relation_definition}" '

    prompt += '\n\nIf none of the above relations holds between the two entities, you should output "no relation".'

    for relation in relation_set:
        if relation == "no_relation":
            continue
        relation_label = get_clean_relation_label(relation)

        if n > 0:
            if n == 1:
                prompt += f'\n\nBelow is an example sentence for the "{relation_label}" relation:'
            else:
                prompt += f'\n\nBelow are {n} example sentences for the "{relation_label}" relation:'

            file_name = f"json/relations_patterns_v4_{experiment_name}.json"
            patterns = get_examples(relation, file_name, n)
            for pattern in patterns:
                prompt += f"\n{pattern}"


    prompt += "\n\nNow, determine which option is the relation between <ENT0> and <ENT1> in the following sentence.\n\n"

    prompt += f"Sentence: {summarized_context}\n\n"

    prompt += f"Options:\n{option_list_string}\n"

    prompt += "Which option is the relation between <ENT0> and <ENT1> in the above sentence? Respond with a single letter without any additional things including your notes and explanations."

    return prompt


def evaluate(model_family, client, model, tokenizer, tacred_file, n, experiment_name, open_source_api_mode=None):
    sleep = False
    save = False

    if model_family == Model_Family.OPENAI:
        pipe = client
    else:
        if open_source_api_mode is None:
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
            )

        else:
            pipe = client


    tacred_path = "../../data/tacred/json/"
    with open(tacred_path + tacred_file, "r") as f:
        examples = json.load(f)


    labels = []
    outputs = []
    wrong_preds = []

    for example in tqdm(examples, total=len(examples)):
        if sleep:
            time.sleep(1)

        example = get_example_fields(example)

        subj = example["subj"]
        obj = example["obj"]
        if subj.lower().strip() != obj.lower().strip():
            summarization_prompt = get_summarization_prompt(example)
            summarized_context = get_completion(open_source_api_mode, model_family, summarization_prompt, pipe, model)
            example["summarized_context"] = summarized_context
        else:
            example["summarized_context"] = example["tagged_context"]

        prompt = get_prompt(example, n, experiment_name)

        response = get_completion(open_source_api_mode, model_family, prompt, pipe, model)

        if model_family == Model_Family.OPENAI:
            response = trim_response(response).strip()
        else:
            response = response.strip()

        subj_type = example["subj_type"]
        obj_type = example["obj_type"]
        relation_set = get_relation_set(subj_type, obj_type)

        try:
            predicted_label = match_response_to_label(response, relation_set)
            predicted_label_id = get_label_id(predicted_label)

        except:
            example_id = example["id"]
            label = example["label"]
            tagged_context = example["tagged_context"]
            relation_set_string = get_relation_set_string(relation_set)

            print("\nError:")
            print(f"id: {example_id}")
            print(f"context: {tagged_context}")
            print(f"relation set: {relation_set_string}")
            print(f"label: {label}")
            print(f"response: {response}\n")

            predicted_label_id = 0

        outputs.append(predicted_label_id)

        label = example["label"]
        label_id = get_label_id(label)
        labels.append(label_id)

        if predicted_label_id != label_id:
            relation_set_string = get_relation_set_string(relation_set)
            wrong_preds.append((example, predicted_label, response, relation_set_string))

    labels = np.array(labels)
    outputs = np.array(outputs)

    if save:
        np.save("experiments/labels.npy", labels)
        np.save("experiments/outputs.npy", outputs)
        with open("experiments/wrong_preds.json", "w") as f:
            json.dump(wrong_preds, f, indent=4)

    p, r, f1 = precision_recall_fscore_exclude_nota(labels, outputs)
    print(f"\np:{p} r:{r} f1:{f1}")

    return f1

