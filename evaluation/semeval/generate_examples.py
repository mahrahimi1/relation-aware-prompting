import json
import time

import openai
import backoff

from transformers import pipeline
import torch
from tqdm import tqdm

from semeval_utils import Model_Family, Open_Source_API_Mode


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
    messages = [{"role": "user", "content": prompt}]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=256,
        temperature=0
    )

    return completion.choices[0].message.content


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


def generate_prompt_orig(relation_label, relation_description, n):
    prompt = f'Please give {n} examples of the relation "{relation_label}" between X and Y ' \
             f'using a single subject-verb-object structure containing X and Y.\n' \
             f'Note that {relation_description}\n' \
             f'Do not overfit the pattern of the above definition. ' \
             f'Try as many different relation patterns or relation expressions as possible.\n' \
             f'Replace X and Y placeholders with actual words. ' \
             f'Produce your response as a list of strings in a json list object.'

    return prompt


def generate_prompt_llama31_8b(relation_label, relation_description, n):
    prompt = f'Please give {n} examples of the relation "{relation_label}" between X and Y ' \
             f'using a single subject-verb-object structure containing X and Y.\n' \
             f'Note that {relation_description}\n' \
             f'Do not overfit the pattern of the above definition. ' \
             f'Try as many different relation patterns or relation expressions as possible.\n' \
             f'Replace X and Y placeholders with actual words. ' \
             f'Produce your response as a list of strings in a json list object without any notes and explanations.'

    return prompt


def generate_prompt(model_family, relation_label, relation_description, n): 
    if model_family == Model_Family.LLAMA_31_8B:
        return generate_prompt_llama31_8b(relation_label, relation_description, n)

    else:
        return generate_prompt_orig(relation_label, relation_description, n)


def check(patterns, n):
    if (type(patterns) != list) and (type(patterns) != dict):
        print("check pattern failed")
        return False

    if len(patterns) != n:
        print("check pattern failed")
        return False

    return True


def convert_dict_to_list(patterns):
    if type(patterns) == dict:
        l = []
        for _ , pattern in patterns.items():
            l.append(pattern)
        patterns = l

    return patterns


def check_XY(patterns):
    for pattern in patterns:
        if "X" not in pattern:
            print("checkXY pattern failed")
            return False

        if "Y" not in pattern:
            print("checkXY pattern failed")
            return False

    return True


def trim_json(response):
    if response[:7] == "```json":
        return response[7:-3]
    else:
        return response


def generate_examples(model_family, client, model, tokenizer, experiment_name, open_source_api_mode=None):
    n = 10 #number of generated patterns per relation

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


    with open("json/relations_descriptions_simple_v0.json", mode='r', encoding="utf_8") as file:
        relations_descriptions = json.load(file)

    relations_patterns = {}

    for relation , relation_description in tqdm(relations_descriptions.items(), total=len(relations_descriptions)):
        relation_label = relation

        relations_patterns[relation] = []

        if (True):
            if (True):
                prompt = generate_prompt(model_family, relation_label, relation_description, n)
                orig_prompt = prompt

                counter = 0

                error_correction_prompt = ""
                while (True):
                    counter += 1
                    if counter > 10:
                        exit()

                    response = get_completion(open_source_api_mode, model_family, prompt + error_correction_prompt, pipe, model)
                    response = trim_json(response)

                    try:
                        patterns = json.loads(response)
                    except:
                        print("\nError!\n")
                        patterns = None

                    print(response,"\n")

                    if check(patterns, n) == False:
                        if model_family != Model_Family.OPENAI:
                            error_correction_prompt = " Only respond with a json list. Do not include notes and explanations in your response."
                        continue

                    patterns = convert_dict_to_list(patterns)
                    break

                relations_patterns[relation].append(patterns)

    with open(f"json/relations_patterns_v4_{experiment_name}.json", "w") as outfile:
        json.dump(relations_patterns, outfile, indent=4)

