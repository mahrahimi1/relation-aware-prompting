import json
from enum import Enum

from sklearn.metrics import precision_recall_fscore_support


class Model_Family(Enum):
    OPENAI = 1
    OPEN_SOURCE = 2
    LLAMA_31_8B = 3


class Open_Source_API_Mode(Enum):
    HUGGING_FACE = 1
    MISTRAL = 2


labels = [
    "Other",
    "Cause-Effect",
    "Instrument-Agency",
    "Product-Producer",
    "Content-Container",
    "Entity-Origin",
    "Entity-Destination",
    "Component-Whole",
    "Member-Collection",
    "Message-Topic"
]

labels_extended = [
    "Other",
    "Cause-Effect(e1,e2)",
    "Cause-Effect(e2,e1)",
    "Instrument-Agency(e1,e2)",
    "Instrument-Agency(e2,e1)",
    "Product-Producer(e1,e2)",
    "Product-Producer(e2,e1)",
    "Content-Container(e1,e2)",
    "Content-Container(e2,e1)",
    "Entity-Origin(e1,e2)",
    "Entity-Origin(e2,e1)",
    "Entity-Destination(e1,e2)",
    "Entity-Destination(e2,e1)",
    "Component-Whole(e1,e2)",
    "Component-Whole(e2,e1)",
    "Member-Collection(e1,e2)",
    "Member-Collection(e2,e1)",
    "Message-Topic(e1,e2)",
    "Message-Topic(e2,e1)"
]

labels_extended_XY = [
    "Other",
    "Cause-Effect(X,Y)",
    "Cause-Effect(Y,X)",
    "Instrument-Agency(X,Y)",
    "Instrument-Agency(Y,X)",
    "Product-Producer(X,Y)",
    "Product-Producer(Y,X)",
    "Content-Container(X,Y)",
    "Content-Container(Y,X)",
    "Entity-Origin(X,Y)",
    "Entity-Origin(Y,X)",
    "Entity-Destination(X,Y)",
    "Entity-Destination(Y,X)",
    "Component-Whole(X,Y)",
    "Component-Whole(Y,X)",
    "Member-Collection(X,Y)",
    "Member-Collection(Y,X)",
    "Message-Topic(X,Y)",
    "Message-Topic(Y,X)"
]


def get_relation_set_extended():
    return labels_extended


def get_relation_set_extended_XY():
    return labels_extended_XY


def get_options(relation_set):

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    letter_mode = False
    digit_mode = False
    option_list_string = ""
    option_labels = {}
    options = {}

    if len(relation_set) <= len(alphabet):
        letter_mode = True
    else:
        digit_mode = True

    for i in range(len(relation_set)):
        relation = relation_set[i]
        #clean_relation = get_clean_relation_label(relation)
        clean_relation = relation

        if letter_mode:
            option_label = alphabet[i]
        else:
            option_label = str(i)

        option_labels[option_label] = relation

        options[clean_relation] = relation

        line = f"{option_label}. {clean_relation}\n"
        option_list_string += line

    return option_labels, options, option_list_string


def get_relation_set_string(relation_set):

    if len(relation_set) == 0:
        raise Exception("The relation set is empty.")

    if len(relation_set) == 1:
        return add_double_quotes(get_clean_relation_label(relation_set[0]))

    if len(relation_set) == 2:
        return add_double_quotes(get_clean_relation_label(relation_set[0])) + \
               " and " + \
               add_double_quotes(get_clean_relation_label(relation_set[1]))

    s = ""
    for relation in relation_set[:-1]:
        s += add_double_quotes(get_clean_relation_label(relation)) + ", "

    s += "and " + add_double_quotes(get_clean_relation_label(relation_set[-1]))

    return s


def add_double_quotes(s):
    return '"' + s + '"'


def get_clean_relation_label(relation_label):
    return relation_label


def get_definition(relation, filename):
    with open(filename, "r") as f:
        relation_definitions = json.load(f)
    return relation_definitions[relation]


def get_relation_set():
    return labels


def precision_recall_fscore_(labels, preds, n_labels=19):
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=list(range(0, n_labels)), average="micro")
    return round(p*100,2), round(r*100,2), round(f*100,2)


def precision_recall_fscore_exclude_nota(labels, preds, n_labels=19):
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=list(range(1, n_labels)), average="micro")
    return round(p*100,2), round(r*100,2), round(f*100,2)


def get_label_id(relation):
    labels2id = {label: i for i, label in enumerate(labels_extended)}
    return labels2id[relation]


def get_examples(relation, file_name, n):
    with open(file_name, "r") as f:
        relation_patterns = json.load(f)

    patterns = relation_patterns[relation][0]

    return patterns[:n]


def load_examples_from_file(semeval_path, dataset):
    if dataset == "train":
        semeval_file = "SemEval2010_task8_training/TRAIN_FILE.TXT"

    elif dataset == "test":
        semeval_file = "SemEval2010_task8_testing/TEST_FILE.txt"

        test_keys = {}
        with open(semeval_path + "SemEval2010_task8_testing_keys/my_test_keys.txt", "r") as f:
            for line in f:
                split = line.split("\t", maxsplit=1)
                test_id = split[0]
                test_label = split[1].strip()
                test_keys[test_id] = test_label
                
    else:
        raise Exception("Dataset name not valid.")


    examples = []
    with open(semeval_path + semeval_file, "r") as input_file:
        while True:
            tokens_line = input_file.readline()
            if not tokens_line:
                break

            (index, tokens_string) = tokens_line.split('\t', maxsplit=1)  # separate index and tokens
            tokens_string = tokens_string.strip()[1:-1]  # remove quotation marks

            untagged_context = str(tokens_string)
            untagged_context = untagged_context.replace("<e1>", "")
            untagged_context = untagged_context.replace("</e1>", "")
            untagged_context = untagged_context.replace("<e2>", "")
            untagged_context = untagged_context.replace("</e2>", "")

            context = str(tokens_string)
            e1_start_idx = context.find("<e1>") + 4
            e1_end_idx   = context.find("</e1>")
            e1 = context[e1_start_idx : e1_end_idx]

            e2_start_idx = context.find("<e2>") + 4
            e2_end_idx   = context.find("</e2>")
            e2 = context[e2_start_idx : e2_end_idx]

            context = context.replace("<e1>", "<e1> ")
            context = context.replace("</e1>", " </e1>")
            context = context.replace("<e2>", "<e2> ")
            context = context.replace("</e2>", " </e2>")
            tagged_context = context

            if dataset == "train":
                relation_label = input_file.readline().strip()  # Remove trailing newline
                _ = input_file.readline()  # Comment string
                _ = input_file.readline()  # Empty line separator
            else:
                relation_label = test_keys[index]

            example = {
                "id": index,
                "e1": e1,
                "e2": e2,
                "context" : untagged_context,
                "tagged_context": tagged_context,
                "label": relation_label
            }
            examples.append(example)
    
    return examples

