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

from datetime import timedelta
import csv
import pickle

currentPath = pathlib.Path(os.path.dirname(__file__))
absolutePath = str(currentPath.parent)
sys.path.append(absolutePath)

INCLUDE_BENCHMARKS = False

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
    if comp_target != "value":

        if bindings[0] == chr(ord("a") + comp_target):
            return TrueCondition() # cant compare to itself
        elif comp_target >= curr_len:
            return TrueCondition()
        else:
            try:
                bindings[1] = chr(ord("a") + comp_target)
            except Exception as e:
                # end of list
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
        elif action_type.startswith("not<"):
            return GreaterThanEqCondition(
                Variable(bindings[0], lambda x: x[attribute]), value
            )
        elif action_type.startswith("not>"):
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
        elif action_type.startswith("not+"):
            return BinaryCondition(
                Variable(bindings[0], lambda x: x[attribute]),
                Variable(bindings[1], lambda x: x[attribute]),
                lambda x, y: x < y + value,
            )
        elif action_type.startswith("not-"):
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
        elif action_type.startswith("not*"):
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
    elif action_type == "not<":
        return GreaterThanEqCondition(
            Variable(bindings[0], lambda x: x[attribute]),
            Variable(bindings[1], lambda x: x[attribute]),
        )
    elif action_type == "not>":
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
    if is_last:
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
    :param curr_len: the number of events in pattern
    :param action_types: all action types (comparison with other attribute, comparison with value, ect.)
    :param comp_values: all the values to compare with
    :param cols: the attributes the model predict conditions on
    :param conds: and/or relations
    :param all_comps: list of comparison targets (e.g. second event in pattern, value)
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


def OpenCEP_pattern(actions, action_types, index, comp_values, cols, conds, all_comps, max_time):
    """
    Auxiliary function for running the CEP engine, build the pattern anc calls run_OpenCEP
    :param actions: all actions the model suggested
    :param action_types: all action types (comparison with other attribute, comparison with value, ect.)
    :param index: episode number
    :param comp_values: all the values to compare with
    :param cols: the attributes the model predict conditions on
    :param conds: all and / or relations
    :param all_comps: list of comparison targets
    :param max_time: max time (in seconds) for pattern duration (time from first event to last event)
    :return: the condition of the pattern created
    """
    cols_rep = []
    [cols_rep.append(cols) for i in range(len(actions))]
    bindings = [chr(ord("a") + i) for i in range(len(actions))]
    action_types = np.array(action_types)
    all_events = [PrimitiveEventStructure(event, chr(ord("a") + i)) for i, event in enumerate(actions)]
    pattern = Pattern(
        SeqOperator(*all_events),
        build_formula(bindings, len(bindings), action_types, comp_values, cols_rep, conds, all_comps),
        timedelta(seconds=max_time),
    )
    run_OpenCEP(str(index), [pattern])
    return pattern


def after_epoch_test(pattern, eval_mechanism_params=DEFAULT_TESTING_EVALUATION_MECHANISM_SETTINGS):
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




def new_mapping(event, events, reverse=False):
    """
    :param event: model's tagged event
    :param events: all events in the stream
    :param reverse: flag that indicates if we want event value or index in list
    :return: the actual event
    """
    if reverse:
        return (np.where(events == int(event))[0][0])
    else:
        return events[event]


def get_action_type(mini_action, total_actions, actions, match_max_size):
    """
    refactoring of kind_of_action function.
    gets a-list of all selected mini-actions, the number of all possible options and all operator options
    :param mini_action: list of mini-actions selected in current step, each mini action is in a different column
    :param total_actions:
    :param actions:
    :param match_max_size: max len of match pattern
    :return: tuple of (type, cond, target), where type is an action from param-actions,
    cond is in {and, or}, and target is in {"", value, event_i (for i in [0, max_match_size])}
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
        action = "not" + action

    comp_to = int(mini_action / len(actions) )
    if sum([i in action for i in ["+>", "->", "*="]]):
        if comp_to == match_max_size:
            action = "nop"
            comp_to = ""
        else:
            action = "v" + action + "value"
    elif comp_to == match_max_size:
        comp_to = "value"
        action = "v" + action + "value"


    return action, cond, comp_to

def create_pattern_str(events, actions, comp_vals, conds, cols, comp_target):
    """
    helper method that creates a string that describes the suggested pattern,
    deatiling it's events and conditions
    :param events: list of ther events that appear in the patterns
    :param actions: list of lists, the i-th inner list details the conditions on
    the attributes of i-th event in the pattern
    :param comp_val: list of lists, the i-th inner lists details the value that
    were chosen for comparisons with attributes of the i-th event in the pattern
    :param conds: list of lists, the i-th inner lists details the or/and relations
    that appears between conditions on attributes of the i-th event in the pattern
    :param cols: list of attributes the model works on
    :param comp_target: list of lists, the i-th inner lists details the comparison targets
    (values or event index) that were chosen for comparisons with attributes of
    the i-th event in the pattern
    :return: string that describes the pattern and it's components
    """

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


def simplify_pattern(str_pattern):
    """
    helper method to remove irrelavent parts from pattern-str
    :param str_pattern: output of (old) create_pattern_str function
    :return: a modified string where tautology parts are removed
    (e.g. A and (T and T) -> A)
    """
    sub_patterns = str_pattern.split(" AND ")

    for i, sub_pattern in enumerate(sub_patterns):
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

    simple_str = ""
    for sub_pattern in sub_patterns:
        if sub_pattern != "T":
            simple_str += sub_pattern + " AND "

    if simple_str.endswith(" AND "):
        simple_str = simple_str[:-5]
    return simple_str



def replace_values(comp_vals, selected_values):
    count = 0
    new_comp_vals = []
    for val in comp_vals:
        if not val == "nop":
            new_comp_vals.append(selected_values[count])
            count += 1
        else:
            new_comp_vals.append("nop")
    return new_comp_vals


def ball_patterns(events):
    if len(events) <= 2:
        return False
    ball_events = [4 if event in [4,8,10] else event for event in events]
    if len(np.unique(ball_events)) == 1 and events[0] in [4,8,10]:
        return True
    if ball_events.count(4) > int(len(events) / 2) + 1:
        return True
    return False


def store_to_file(actions, action_types, index, comp_values, cols, conds, new_comp_vals, targets, max_fine, max_time):
    """
    stores relavent info to txt files
    :param actions: list of actions (events)
    :param action_type: list of list, the i-th inner list contains the mini-actions of the i-th event in the pattern
    :param index: index of window in data
    :param comp_values: list of list, the i-th inner list contains the comparison values of the i-th event in the pattern
    :param cols: list of attributes the model works on
    :param new_comp_vals: same as comp_values but for the latests event in pattern
    :param conds: list of list, the i-th inner list contains the conds (and/or) of the i-th event in the pattern
    :param targets: list of list, the i-th inner list contains the comparison targets of the i-th event in the pattern
    :param max_fine: max leagal appearances of pattern in a single window
    :param max_time: max length (time wise) of pattern
    :return: has no return value
    """
    NAMES = ["actions", "action_type",  "index", "comp_values", "cols", "conds", "new_comp_vals", "targets", "max_fine", "max_time"]
    NAMES = [name + ".txt" for name in NAMES]
    TO_WRITE = [actions, action_types, index, comp_values, cols, conds, new_comp_vals, targets, max_fine, max_time]
    for file_name, file_content in zip(NAMES, TO_WRITE):
        with open(file_name, 'wb') as f:
            pickle.dump(file_content, f)




def set_values_bayesian(comp_vals, cols, eff_cols, mini_actions, event, conds, file, max_values, min_values):
    """
    finds ranges for bayesian serach
    :param comp_vals: list of values for comparison with attributes of the last event of the pattern
    :param cols: list of all attributes in data
    :param eff_cols: list of attributes the model works on
    :param mini_actions: list of conditions on the last event of the pattern
    :param event: last event of the pattern
    :param conds: list, contains the conds (and/or) of the last event of the pattern
    :param file: path to the data of the current window
    :param max_values: list of maximum leagl values to chose values from
    :param min_values: list of minimum leagl values to chose values from
    :return: list of ranges for eache bayesian serach
    """
    headers = ["event", "ts"] + cols
    return_dict = {}
    df = pd.read_csv(file, names=headers)
    keep_cols = eff_cols + ["event"]
    df = df[keep_cols]
    df = df.loc[df['event'] == event] # todo, try to remove
    count = 0
    for col_count, (val, col) in enumerate(zip(comp_vals, df.columns[1:])):
        if not val == "nop":
            return_dict.update({"x_" + str(count):
                (max([df[col].min() - 1000, min_values[col_count] + 1]),
                 min([df[col].max() + 1000, max_values[col_count] - 1]))
                 })
            count += 1

    return return_dict




def bayesian_function(**values):
    """
    list of values to do bayesian serach on, each value has it's predefined range
    :return: chosen value to compare with for each comparison with value in the pattern
    """
    NAMES = ["actions", "action_type",  "index", "comp_values", "cols", "conds", "new_comp_vals", "targets", "max_fine", "max_time"]
    NAMES = [name + ".txt" for name in NAMES]
    TO_READ = [[] for _ in range(len(NAMES))]
    for i, name in enumerate(NAMES):
        with open(name, 'rb') as f:
            TO_READ[i] = pickle.load(f)

    actions = TO_READ[0]
    action_types = TO_READ[1]
    index = TO_READ[2]
    comp_values = TO_READ[3]
    cols = TO_READ[4]
    conds = TO_READ[5]
    new_comp_vals = TO_READ[6]
    targets = TO_READ[7]
    max_fine = TO_READ[8]
    max_time = TO_READ[9]
    count = 0
    to_return_comp_vals = []

    values_keys = list(values.keys())
    for val in new_comp_vals:
        if not val == "nop":
            to_return_comp_vals.append(values[values_keys[count]])
            count += 1
        else:
            to_return_comp_vals.append("nop")

    try_comp_vals = comp_values
    try_comp_vals.append(to_return_comp_vals)

    # calls run_OpenCEP
    pattern = OpenCEP_pattern(
        actions, action_types, index, try_comp_vals, cols, conds, targets, max_time
    )
    # checks and return output

    with open("Data/Matches/{}Matches.txt".format(index), "r") as f:
        reward = int(f.read().count("\n") / (len(actions) + 1))
        if reward > max_fine:
            reward = 1
        return reward + random.uniform(1e-9, 1e-7) #epsilon added in case of 0
