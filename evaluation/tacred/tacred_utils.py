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
    "no_relation",
    "org:alternate_names",
    "org:city_of_headquarters",
    "org:country_of_headquarters",
    "org:dissolved",
    "org:founded",
    "org:founded_by",
    "org:member_of",
    "org:members",
    "org:number_of_employees/members",
    "org:parents",
    "org:political/religious_affiliation",
    "org:shareholders",
    "org:stateorprovince_of_headquarters",
    "org:subsidiaries",
    "org:top_members/employees",
    "org:website",
    "per:age",
    "per:alternate_names",
    "per:cause_of_death",
    "per:charges",
    "per:children",
    "per:cities_of_residence",
    "per:city_of_birth",
    "per:city_of_death",
    "per:countries_of_residence",
    "per:country_of_birth",
    "per:country_of_death",
    "per:date_of_birth",
    "per:date_of_death",
    "per:employee_of",
    "per:origin",
    "per:other_family",
    "per:parents",
    "per:religion",
    "per:schools_attended",
    "per:siblings",
    "per:spouse",
    "per:stateorprovince_of_birth",
    "per:stateorprovince_of_death",
    "per:stateorprovinces_of_residence",
    "per:title"
]


def get_clean_relation_label(relation_label):
    if relation_label == "no_relation":
        return "no relation"

    clean_relation_label = relation_label.split(":")[1]
    clean_relation_label = clean_relation_label.replace("_", " ")
    clean_relation_label = clean_relation_label.replace("/", " or ")
    clean_relation_label = clean_relation_label.replace("stateorprovince", "state or province")
    return clean_relation_label


def get_actual_relation_label(clean_relation_label, subj_type):
    clean_relation_label = clean_relation_label.strip().lower()

    if clean_relation_label == "no relation":
        return "no_relation"

    actual_relation_label = clean_relation_label.replace("state or province", "stateorprovince")
    actual_relation_label = actual_relation_label.replace(" or ", "/")
    actual_relation_label = actual_relation_label.replace(" ", "_")
    
    if subj_type == "PERSON":
        return "per:" + actual_relation_label
    elif subj_type == "ORGANIZATION":
        return "org:" + actual_relation_label
    else:
         raise Exception("Subject type is not valid.")


def get_relation_set(subj_type, obj_type):
    relation_types = get_relation_types()
    relation_set = []

    for relation in labels:
        if (relation == "no_relation") or \
           ((subj_type in relation_types[relation]["subj types"]) and (obj_type in relation_types[relation]["obj types"])):

            relation_set.append(relation)

    return relation_set


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


def get_example_fields(example):
    line = example

    example_id = line["id"]

    subj=" ".join(line["token"][line["subj_start"] : line["subj_end"] + 1])\
    .replace("-LRB-", "(")\
    .replace("-RRB-", ")")\
    .replace("-LSB-", "[")\
    .replace("-RSB-", "]")

    obj=" ".join(line["token"][line["obj_start"] : line["obj_end"] + 1])\
    .replace("-LRB-", "(")\
    .replace("-RRB-", ")")\
    .replace("-LSB-", "[")\
    .replace("-RSB-", "]")

    subj_type = line['subj_type']

    obj_type = line['obj_type']

    context=" ".join(line["token"])\
    .replace("-LRB-", "(")\
    .replace("-RRB-", ")")\
    .replace("-LSB-", "[")\
    .replace("-RSB-", "]")

    line["token"][line["subj_start"]] = "<ENT0> " + line["token"][line["subj_start"]]
    line["token"][line["subj_end"]] += " </ENT0>"
    line["token"][line["obj_start"]] = "<ENT1> " + line["token"][line["obj_start"]]
    line["token"][line["obj_end"]] += " </ENT1>"
    tagged_context=" ".join(line["token"])\
    .replace("-LRB-", "(")\
    .replace("-RRB-", ")")\
    .replace("-LSB-", "[")\
    .replace("-RSB-", "]")

    label=line["relation"]

    masked_context = tagged_context
    ent0_start = masked_context.find("<ENT0>")
    ent0_end = masked_context.find("</ENT0>")
    masked_context = masked_context[:ent0_start] + f"the {subj_type.lower()} <ENT0>" + masked_context[ent0_end + 7:]
    ent1_start = masked_context.find("<ENT1>")
    ent1_end = masked_context.find("</ENT1>")
    masked_context = masked_context[:ent1_start] + f"the {obj_type.lower()} <ENT1>" + masked_context[ent1_end + 7:]
    masked_context = masked_context[0].upper() + masked_context[1:]

    return {"subj": subj,
            "obj": obj,
            "subj_type": subj_type,
            "obj_type": obj_type,
            "context": context,
            "tagged_context": tagged_context,
            "masked_context": masked_context,
            "label": label,
            "id": example_id}


def get_label_id(relation):
    labels2id = {label: i for i, label in enumerate(labels)}
    #id2labels = {i: label for i, label in enumerate(labels)}

    return labels2id[relation]


def precision_recall_fscore_(labels, preds, n_labels=42):
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=list(range(0, n_labels)), average="micro")
    return round(p*100,2), round(r*100,2), round(f*100,2)


def precision_recall_fscore_exclude_nota(labels, preds, n_labels=42):
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=list(range(1, n_labels)), average="micro")
    return round(p*100,2), round(r*100,2), round(f*100,2)


def get_relation_types():
    with open("json/relation_types_all.json", "r") as f:
        relation_types = json.load(f)
    return relation_types


def get_definition(relation):
    with open("json/relations_descriptions_with_entities.json", "r") as f:
        relation_definitions = json.load(f)
    return relation_definitions[relation]


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
        clean_relation = get_clean_relation_label(relation)

        if letter_mode:
            option_label = alphabet[i]
        else:
            option_label = str(i)

        option_labels[option_label] = relation

        options[clean_relation] = relation

        line = f"{option_label}. {clean_relation}\n"
        option_list_string += line

    return option_labels, options, option_list_string


def get_patterns(relation, n):
    with open("json/relations_patterns.json", "r") as f:
        relation_patterns = json.load(f)

    per_type_patterns = relation_patterns[relation]

    num_types = len(per_type_patterns)

    #count_per_type = [round(n/num_types) for i in range(num_types-1)]
    #count_per_type.append(n - sum(count_per_type))
    #count_per_type.sort(reverse=True)

    count_per_type = [0] * num_types
    for i in range(n):
        count_per_type[i % num_types] += 1

    retval = []
    for i in range(len(per_type_patterns)):
        patterns = per_type_patterns[i][2]
        subj_type = per_type_patterns[i][0]
        obj_type  = per_type_patterns[i][1]

        left = count_per_type[i]
        if left == 0:
            continue

        for pattern in patterns:
            pattern = pattern.replace("X", f"the {subj_type} <ENT0>")
            pattern = pattern.replace("Y", f"the {obj_type} <ENT1>")
            pattern = pattern[0].upper() + pattern[1:]

            if pattern not in retval:
                retval.append(pattern)
                left -= 1
                if left == 0:
                    break

    return retval


def get_examples(relation, file_name, n):
    with open(file_name, "r") as f:
        relation_patterns = json.load(f)

    per_type_patterns = relation_patterns[relation]

    num_types = len(per_type_patterns)

    count_per_type = [0] * num_types
    for i in range(n):
        count_per_type[i % num_types] += 1

    retval = []
    for i in range(len(per_type_patterns)):
        patterns = per_type_patterns[i][2]
        subj_type = per_type_patterns[i][0]
        obj_type  = per_type_patterns[i][1]

        left = count_per_type[i]
        if left == 0:
            continue

        for pattern in patterns:
            #pattern = pattern.replace("X", f"the {subj_type} <ENT0>")
            #pattern = pattern.replace("Y", f"the {obj_type} <ENT1>")
            #pattern = pattern[0].upper() + pattern[1:]

            if pattern not in retval:
                retval.append(pattern)
                left -= 1
                if left == 0:
                    break

    return retval


def get_relation_set_all():
    relation_set = []
    for relation in labels:
        relation_set.append(relation)

    return relation_set

