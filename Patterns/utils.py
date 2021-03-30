import torch
from torch.autograd import Variable
import numpy as np
import os
import pathlib
import random
import sys
import pandas as pd
from CEP import CEP
from evaluation.EvaluationMechanismFactory import TreeBasedEvaluationMechanismParameters
from stream.Stream import OutputStream
from stream.FileStream import FileInputStream, FileOutputStream
from misc.Utils import generate_matches
from plan.TreePlanBuilderFactory import TreePlanBuilderParameters
from plan.TreeCostModels import TreeCostModels
from plan.TreePlanBuilderTypes import TreePlanBuilderTypes


# from plugin.ToyExample.Toy import DataFormatter
from plugin.Football.Football_processed import DataFormatter
from tree.PatternMatchStorage import TreeStorageParameters


from condition.Condition import Variable, TrueCondition, BinaryCondition
from condition.CompositeCondition import AndCondition, OrCondition
from condition.BaseRelationCondition import GreaterThanCondition, SmallerThanCondition, EqCondition, NotEqCondition, GreaterThanEqCondition, SmallerThanEqCondition
from base.PatternStructure import SeqOperator, PrimitiveEventStructure, NegationOperator
from base.Pattern import Pattern
import random
from plan.negation.NegationAlgorithmTypes import NegationAlgorithmTypes

from adaptive.optimizer.OptimizerFactory import OptimizerParameters
from adaptive.optimizer.OptimizerTypes import OptimizerTypes
from plan.multi.MultiPatternTreePlanMergeApproaches import MultiPatternTreePlanMergeApproaches


CSV_PATH = "Patterns/pattern12_temp.csv"
CSV_PATH_FINAL = "Patterns/pattern12.csv"

# #
# from base.Formula import (
#     GreaterThanFormula,
#     SmallerThanFormula,
#     SmallerThanEqFormula,
#     GreaterThanEqFormula,
#     MulTerm,
#     EqFormula,
#     IdentifierTerm,
#     AtomicTerm,
#     AndFormula,
#     TrueFormula,
#     NotEqFormula,
#     OrFormula,
#
# )
# from base.PatternStructure import (
#     AndOperator,
#     SeqOperator,
#     PrimitiveEventStructure,
#     NegationOperator,
# # )
# from base.Pattern import Pattern
from datetime import timedelta
import csv
import pickle

currentPath = pathlib.Path(os.path.dirname(__file__))
absolutePath = str(currentPath.parent)
sys.path.append(absolutePath)

INCLUDE_BENCHMARKS = False
# DEFAULT_TESTING_EVALUATION_MECHANISM_SETTINGS = TreeBasedEvaluationMechanismParameters(
#     TreePlanBuilderParameters(
#         TreePlanBuilderTypes.TRIVIAL_LEFT_DEEP_TREE,
#         TreeCostModels.INTERMEDIATE_RESULTS_TREE_COST_MODEL,
#     ),
#     TreeStorageParameters(
#         sort_storage=False, clean_up_interval=10, prioritize_sorting_by_timestamp=True
#     ),
# )

DEFAULT_TESTING_EVALUATION_MECHANISM_SETTINGS = TreeBasedEvaluationMechanismParameters(
    optimizer_params=OptimizerParameters(opt_type=OptimizerTypes.TRIVIAL_OPTIMIZER,
                                         tree_plan_params=TreePlanBuilderParameters(builder_type=TreePlanBuilderTypes.TRIVIAL_LEFT_DEEP_TREE,
                              cost_model_type=TreeCostModels.INTERMEDIATE_RESULTS_TREE_COST_MODEL,
                              tree_plan_merger_type=MultiPatternTreePlanMergeApproaches.TREE_PLAN_TRIVIAL_SHARING_LEAVES)),
    storage_params=TreeStorageParameters(sort_storage=False, clean_up_interval=10, prioritize_sorting_by_timestamp=True))


DEFAULT_TESTING_DATA_FORMATTER = DataFormatter()


def get_next_formula(bindings, curr_len, action_type, value, attribute, comp_target):
    """
    Creates a single condition in the formula of the pattern
    :param bindings: All bindings (events as symbols) remining
    :param curr_len: the match number of events

    :param action_type: current action type (comparison with next, comparison with value, ect.)
    :param value: current the values to compare with
    :param attribute: system attribute to create a condition in
    :param comp_target: what event to compate to
    :return: the next part of the formula
    """
    # This is dumb, but for now:
    if comp_target != "value":
        # if comp_target == "":
            # print(f"Comp FUCK")
            # print(f"action {action_type}")
            # print(f"Comp FUCK")
        if bindings[0] == chr(ord("a") + comp_target):
            return TrueCondition() # cant compare to itself
        elif comp_target >= curr_len:
            return TrueCondition()
        else:
            try:
                bindings[1] = chr(ord("a") + comp_target)
            except Exception as e:
                # end op list
                return TrueCondition()

    # try:
    if action_type == "nop":
        return TrueCondition()
    elif len(action_type.split("v")) == 2:
        action_type = action_type.split("v")[1]
        if action_type.startswith("<"):
            return SmallerThanCondition(
                Variable(bindings[0], lambda x: x[attribute]), value
                )
        elif action_type.startswith(">"):
            return GreaterThanCondition(
                Variable(bindings[0], lambda x: x[attribute]), value
            )
        elif action_type.startswith("="):
            return EqCondition(
                Variable(bindings[0], lambda x: x[attribute]), value
            )
        elif action_type.startswith("not <"):
            return GreaterThanEqCondition(
                Variable(bindings[0], lambda x: x[attribute]), value
            )
        elif action_type.startswith("not >"):
            return SmallerThanEqCondition(
                Variable(bindings[0], lambda x: x[attribute]), value
            )
        elif action_type.startswith("+"):
            return BinaryCondition(
                Variable(bindings[0], lambda x: x[attribute]),
                Variable(bindings[1], lambda x: x[attribute] + value),
                lambda x, y: x >= y + value,

            )
        elif action_type.startswith("-"):
            return BinaryCondition(
                Variable(bindings[0], lambda x: x[attribute]),
                Variable(bindings[1], lambda x: x[attribute] - value),
                lambda x, y: x >= y + value,

            )
        elif action_type.startswith("not +"):
            return BinaryCondition(
                Variable(bindings[0], lambda x: x[attribute]),
                Variable(bindings[1], lambda x: x[attribute]),
                lambda x, y: x < y + value,
            )
        elif action_type.startswith("not -"):
            return BinaryCondition(
                Variable(bindings[0], lambda x: x[attribute]),
                Variable(bindings[1], lambda x: x[attribute]),
                lambda x, y: x < y - value,
            )
        elif action_type.startswith("*="):
            return BinaryCondition(
                Variable(bindings[0], lambda x: x[attribute]),
                Variable(bindings[1], lambda x: x[attribute] + value),
                lambda x, y: int(x) == int(y * value),

            )
        elif action_type.startswith("not *"):
            return BinaryCondition(
                Variable(bindings[0], lambda x: x[attribute]),
                Variable(bindings[1], lambda x: x[attribute]),
                lambda x, y: int(x) != int(y * value),
            )
        else:  # action_type == "not ="
            return NotEqCondition(
                Variable(bindings[0], lambda x: x[attribute]), value
            )

    elif action_type == "<":
        return SmallerThanCondition(
            Variable(bindings[0], lambda x: x[attribute]),
            Variable(bindings[1], lambda x: x[attribute]),
        )
    elif action_type == ">":
        return GreaterThanCondition(
            Variable(bindings[0], lambda x: x[attribute]),
            Variable(bindings[1], lambda x: x[attribute]),
        )
    elif action_type == "=":
        return EqCondition(
            Variable(bindings[0], lambda x: x[attribute]),
            Variable(bindings[1], lambda x: x[attribute]),
        )
    elif action_type == "not <":
        return GreaterThanEqCondition(
            Variable(bindings[0], lambda x: x[attribute]),
            Variable(bindings[1], lambda x: x[attribute]),
        )
    elif action_type == "not >":
        return SmallerThanEqCondition(
            Variable(bindings[0], lambda x: x[attribute]),
            Variable(bindings[1], lambda x: x[attribute]),
        )
    else:  # action_type == "not ="
        return NotEqCondition(
            Variable(bindings[0], lambda x: x[attribute]),
            Variable(bindings[1], lambda x: x[attribute]),
        )
    # except Exception as e: #TODO: FIX!
    #     return TrueCondition()


def build_event_formula(bind, curr_len, actions, comps, cols, conds, targets, is_last=False):

    num_ops_remaining = sum([i != "nop" for i in actions])
    num_comps_remaining = sum([i != "nop" for i in comps])
    if num_comps_remaining == 0 and num_ops_remaining == 0:
        return TrueCondition()
    if is_last:  #TODO: remove this, for now need to change to if 0
        if num_comps_remaining == 0:
            return TrueCondition()
        elif comps[0] == "nop":
            return build_event_formula(
                bind, curr_len, actions[1:], comps[1:], cols[1:], conds[1:], targets[1:], is_last=True
            )
        else:
            return get_next_formula(bind, curr_len, actions[0], comps[0], cols[0], targets[0])

    elif num_ops_remaining == 1:
        if actions[0] == "nop":
            return build_event_formula(bind, curr_len,  actions[1:], comps[1:], cols[1:], conds[1:], targets[1:])
        else:
            return get_next_formula(bind, curr_len,  actions[0], comps[0], cols[0], targets[0])
    else:
        event_forumla = build_event_formula(bind, curr_len,  actions[1:], comps[1:], cols[1:], conds[1:], targets[1:])
        if actions[0] == "nop":
            return event_forumla
        else:
            next_formula = get_next_formula(bind, curr_len, actions[0], comps[0], cols[0], targets[0])

            if isinstance(event_forumla, TrueCondition):
                return next_formula
            elif isinstance(next_formula, TrueCondition):
                return event_forumla
            elif conds[0] == "and":
                return AndCondition(
                    next_formula,
                    event_forumla,
                )
            else:
                return OrCondition(
                    next_formula,
                    event_forumla,
                )

def build_formula(bindings, curr_len, action_types, comp_values, cols, conds, all_comps):
    """
    Build the condition formula of the pattern
    :param bindings: All bindings (events as symbols)
    :param curr_len: the match number of events
    :param action_types: all action type (comparison with next, comparison with value, ect.)
    :param comp_values: all the values to compare with
    :param cols: system attributes
    :param conds: and/or relations
    :param all_comps: compare to what event in match
    :return: The formula of the pattern
    """
    if len(bindings) == 1:
        return build_event_formula(
            bindings, curr_len, action_types[0], comp_values[0], cols[0], conds[0], all_comps[0], is_last=True
        )
    else:
        event_forumla = build_event_formula(
            bindings, curr_len, action_types[0], comp_values[0], cols[0], conds[0], all_comps[0]
        )
        next_formula = build_formula(bindings[1:], curr_len, action_types[1:], comp_values[1:], cols[1:], conds[1:], all_comps[1:])
        if isinstance(next_formula, TrueCondition):
            return event_forumla
        if isinstance(event_forumla, TrueCondition):
            return next_formula
        else:
            return AndCondition(
                event_forumla,
                next_formula
            )


def OpenCEP_pattern(actions, action_types, index, comp_values, cols, conds, all_comps):
    """
    Auxiliary function for running the CEP engine, build the pattern anc calls run_OpenCEP
    :param actions: all actions the model suggested
    :param action_types: all action type (comparison with next, comparison with value, ect.)
    :param index: episode number
    :param comp_values: all the values to compare with
    :param cols: system columns- attributes
    :param conds: all and / or relations
    :param all_comps: compare to what event in match
    :return: the condition of the pattern created
    """
    cols_rep = []
    [cols_rep.append(cols) for i in range(len(actions))]
    bindings = [chr(ord("a") + i) for i in range(len(actions))]
    action_types = np.array(action_types)
    # print(build_formula(bindings, len(bindings), action_types, comp_values, cols_rep, conds, all_comps))
    test1 = [PrimitiveEventStructure(event, chr(ord("a") + i)) for i, event in enumerate(actions)]
    # print(f"A: {test1}")
    # print(f"B: {test1[0]}")
    # print(*test1)
    # print(SeqOperator(*test1))
    # print(SeqOperator([PrimitiveEventStructure(event, chr(ord("a") + i)) for i, event in enumerate(actions)]))
    # input("next")

    pattern = Pattern(
        SeqOperator(*test1),
        build_formula(bindings, len(bindings), action_types, comp_values, cols_rep, conds, all_comps),
        timedelta(seconds=7),
    )
    run_OpenCEP(str(index), [pattern])
    return pattern


def after_epoch_test(
    pattern, eval_mechanism_params=DEFAULT_TESTING_EVALUATION_MECHANISM_SETTINGS
):
    cep = CEP([pattern], eval_mechanism_params)
    events = FileInputStream(os.path.join(absolutePath, "Data", "test_data_stream.txt"))
    base_matches_directory = os.path.join(absolutePath, "Data", "Matches")
    output_file_name = "%sMatches.txt" % "all"
    matches_stream = FileOutputStream(base_matches_directory, output_file_name)
    running_time = cep.run(events, matches_stream, DEFAULT_TESTING_DATA_FORMATTER)
    return running_time


def run_OpenCEP(
    test_name,
    patterns,
    eval_mechanism_params=DEFAULT_TESTING_EVALUATION_MECHANISM_SETTINGS,
):
    """
    This method receives the given pattern (could be used for several patterns) and runs the CEP engine, writing to
    output file all matches found
    :param test_name: Test Name, currently index of epsidoe (i.e. file name to run the engine on)
    :param patterns: list of at least one pattern to search for in the file
    :param eval_mechanism_params: unclear, need to ask Ilya
    :return: total run time of CEP engine, currently this value is unused
    """
    cep = CEP(patterns, eval_mechanism_params)
    events = FileInputStream(
        os.path.join(absolutePath, "Model", "training", "{}.txt".format(test_name))
    )
    base_matches_directory = os.path.join(absolutePath, "Data", "Matches")
    output_file_name = "%sMatches.txt" % test_name
    matches_stream = FileOutputStream(base_matches_directory, output_file_name)
    running_time = cep.run(events, matches_stream, DEFAULT_TESTING_DATA_FORMATTER)
    return running_time


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def get_event_type(event):
    return chr(ord("A") + event)


def mapping(num_events, value):
    # adding "not" support and value support, this value support must!!!! be changed
    # TODO: change model.value_option in a way that this mapping wont be hardcoded!
    if value >= num_events * 8:
        value, kind_of_action = mapping(num_events, value - num_events * 8)
        if kind_of_action != "nop":
            kind_of_action = "v" + kind_of_action + " value"
    elif value >= num_events * 4:
        value, kind_of_action = mapping(num_events, value - num_events * 4)
        if kind_of_action != "nop":
            kind_of_action = "not " + kind_of_action
    else:
        help_map = {i: chr(ord("A") + i) for i in range(num_events)}
        if value < num_events:
            kind_of_action = "nop"
        elif value < 2 * num_events:
            kind_of_action = "<"
        elif value < 3 * num_events:
            kind_of_action = ">"
        else:
            kind_of_action = "="
        value %= num_events
        value = help_map.get(value)
    return value, kind_of_action



def pattern_complexity(events, actions, comp_values, conds, max_events, max_ops, num_cols):
    """
    TODO: rebuild this function!
    This function is meant to determine the complexity of a pattern encouraging the model to predict complex
    patterns (long, include many events and comparison with values)
    :param events: all actions predicted
    :param actions: all action types predicted (none, <, =, >, not on op, op with value)
    :param comp_values: values to be compared with, this param should be changed
    :param cons: list of ANDs & ORs desicions
    :param max_events: maximum number of different events that can be predicted (used for normalization)
    :param max_ops: maximum number of different operations that can be predicted (used for normalization)
    :param num_cols: the number of columns the modle works on
    :return: the complexity of the given pattern
    """
    result = 1
    ops_per_col = [0] * num_cols
    for sublist in actions:
        for i, op in enumerate(sublist):
            if not op == 'nop':
                ops_per_col[i] += 1

    num_events = len(events)
    flatten = lambda list_list: [item for sublist in list_list for item in sublist]

    flat_actions = flatten(actions)
    flat_values = flatten(comp_values)
    flast_conds = flatten(conds)
    num_ops = sum([i != "nop" for i in flat_actions])
    num_comps = sum([i != "nop" for i in flat_values])
    num_ands = sum([i == "and" for i in flast_conds])
    num_ors = sum([i == "or" for i in flast_conds])
    num_unique_events = len(np.unique(events))
    num_unique_events_vals = len(np.unique(flat_values))
    num_actions = num_ops - num_comps
    if num_ors > num_ands:
        result += num_ands * 0.1
        result -= num_ors * 0.1
    result += num_actions * 0.1
    result += max(ops_per_col) * 0.2
    print(f"max_ops_col={max(ops_per_col)}")
    if num_ands + num_ors > 8:
        result -= 0.5
    result *= num_unique_events
    result += num_unique_events_vals * 0.1
    if num_events > 3 and num_unique_events == 1:
        return -0.0001
    else:
        unique, counts = np.unique(events, return_counts=True)
        if max(counts) > 5:
            return -0.1
        if max(counts) > 3:
            return -0.001

    print(result)
    return result


def new_mapping(event, reverse=False, random=False):
    """
    This should be replaced by real mapping!
    :param event: model's tagged event
    :return: the actual event
    """
    values = [98,  69,  19,  67,  66,  75,  65,  40,  47,  64,  44,  59,  68,
       106,  61,  49,  28,  99,  38,  58,  54, 100, 105,  73,  16,  97,
        14,  53,  23,  24,  74,  88,  63,  13,  71,  57,  62,  52,   8,
        10,   4]

    if random:
        weights = [5 if i in [4, 8, 10] else 1 for i in values]
        weights = np.array(weights)
        weights = weights/sum(weights)
        event = np.random.choice(len(values), p=weights)
    if reverse:
        return values.index(int(event))
    else:
        return values[event]



def get_action_type(mini_action, total_actions, actions, match_max_size):
    """
    refactoring of kind_of_action function.
    gets a-list of all selected mini-actions, the number of all possible options and all operator options
    :param mini_action: list of mini-actions selected in current step, each mini action is in a different column
    :param total_actions:
    :param actions:
    :param match_max_size: max len of match pattern
    :return:
    """
    not_flag = False
    if mini_action == total_actions:
        return "nop", "cond", ""
    if mini_action >= len(actions) * (match_max_size + 1) * 2:
        cond = "or"
        mini_action -= len(actions) * (match_max_size + 1) * 2
    else:
        cond = "and"
    if mini_action >= len(actions) * (match_max_size + 1):
        not_flag = True
        mini_action -= len(actions) * (match_max_size + 1)

    action = actions[mini_action % len(actions)]
    if not_flag and action != "nop":
        action = "not " + action

    comp_to = int(mini_action / len(actions) )
    if sum([i in action for i in ["+>", "->", "*="]]):
        if comp_to == match_max_size:
            action = "nop"
            comp_to = ""
        else:
            action = "v" + action + " value"
    elif comp_to == match_max_size:
        comp_to = "value"
        action = "v" + action + " value"


    return action, cond, comp_to


def create_pattern_str(events, actions, comp_vals, conds, cols, comp_target):
    str_pattern = ""
    for event_index in range(len(events)):
        event_char = chr(ord("a") + event_index)
        comps = actions[event_index]
        curr_conds = conds[event_index]
        for (i, action) in enumerate(comps):
            if action != 'nop':
                if action.startswith("v"):
                    action = action.split("v")[1]
                    if sum([i in action for i in ["+>", "->"]]):
                        if (event_index != len(events) - 1) and chr(ord("a") + comp_target[event_index][i]) != event_char and comp_target[event_index][i] < len(events):
                            str_pattern += f"{event_char}.{cols[i]} {action} {chr(ord('a') + comp_target[event_index][i])}.{cols[i]} + {comp_vals[event_index][i]}"
                        else:
                            str_pattern += "T"
                    elif "*=" in action:
                        if (event_index != len(events) - 1) and chr(ord("a") + comp_target[event_index][i]) != event_char and comp_target[event_index][i] < len(events):
                            if "not" in action:
                                str_pattern += f"{event_char}.{cols[i]} = {chr(ord('a') + comp_target[event_index][i])}.{cols[i]} * {comp_vals[event_index][i]}"
                            else:
                                str_pattern += f"{event_char}.{cols[i]} not = {chr(ord('a') + comp_target[event_index][i])}.{cols[i]} + {comp_vals[event_index][i]}"
                        else:
                            str_pattern += "T"
                    else:
                        str_pattern += f"{event_char}.{cols[i]} {action} {comp_vals[event_index][i]}"

                else:
                    if (event_index != len(events) - 1) and comp_target[event_index][i] != 'value' and chr(ord("a") + comp_target[event_index][i]) != event_char and comp_target[event_index][i] < len(events):
                        str_pattern += f"{event_char}.{cols[i]} {action} {chr(ord('a') + comp_target[event_index][i])}.{cols[i]}"
                    else:
                        str_pattern += "T"

                if i != len(comps) - 1:
                    str_pattern += f" {curr_conds[i]} "

        if (event_index != len(events) - 1):
            str_pattern += " AND "

    return simplify_pattern(str_pattern)
    # return str_pattern


def simplify_pattern(str_pattern):
    sub_patterns = str_pattern.split(" AND ")

    for i, sub_pattern in enumerate(sub_patterns):
        # print(f" Before {sub_pattern}")
        if sub_pattern.endswith(" and "):
            sub_pattern = sub_pattern[:-5]
        elif sub_pattern.endswith(" or "):
            sub_pattern = sub_pattern[:-4]
        sub_pattern = sub_pattern.replace("T and T", "T")
        sub_pattern = sub_pattern.replace("T and ", "")
        sub_pattern = sub_pattern.replace(" and T", "")
        sub_pattern = sub_pattern.replace("T or T", "T")
        sub_pattern = sub_pattern.replace("T or ", "")
        sub_pattern = sub_pattern.replace(" or T", "")
        sub_patterns[i] = sub_pattern
        # print(f" After {sub_pattern}")

    simple_str = ""
    for sub_pattern in sub_patterns:
        if sub_pattern != "T":
            simple_str += sub_pattern + " AND "

    if simple_str.endswith(" AND "):
        simple_str = simple_str[:-5]
    # print(simple_str)
    return simple_str




def store_patterns_and_rating_to_csv(pattern, user_rating, events, str_pattern):
    pattern_copy = pattern.detach().numpy()
    pattern_copy = [str(i) for i in pattern_copy]
    pattern_copy = ','.join(pattern_copy)
    if not os.path.isfile(CSV_PATH):
        modifier = "w"
    else:
        modifier = "a"
    with open(CSV_PATH, modifier) as csv_file:
        writer = csv.writer(csv_file)
        if modifier == "w":
            writer.writerow(["pattern", "rating", "events", "pattern_str"])
        writer.writerow([pattern_copy, user_rating, events, str_pattern])


def normalizeData():
    df = pd.read_csv(CSV_PATH)
    min_val = df['rating'].min()
    max_val = df['rating'].max()
    df['rating'] = [4 * (i-min_val)/(max_val - min_val) + 1 for i in df['rating']]
    df.to_csv(CSV_PATH_FINAL)
