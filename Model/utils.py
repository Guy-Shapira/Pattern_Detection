import torch
from torch.autograd import Variable
import numpy as np
import os
import pathlib
import sys
from CEP import CEP
from evaluation.EvaluationMechanismFactory import TreeBasedEvaluationMechanismParameters
from stream.Stream import OutputStream
from stream.FileStream import FileInputStream, FileOutputStream
from misc.Utils import generate_matches
from plan.TreePlanBuilderFactory import TreePlanBuilderParameters
from plan.TreeCostModels import TreeCostModels
from plan.TreePlanBuilderTypes import TreePlanBuilderTypes
from plugin.ToyExample.Toy import DataFormatter
from tree.PatternMatchStorage import TreeStorageParameters
from base.Formula import GreaterThanFormula, SmallerThanFormula, SmallerThanEqFormula, GreaterThanEqFormula, MulTerm, EqFormula, IdentifierTerm, \
    AtomicTerm, AndFormula, TrueFormula
from base.PatternStructure import AndOperator, SeqOperator, PrimitiveEventStructure, NegationOperator
from base.Pattern import Pattern
from datetime import timedelta

currentPath = pathlib.Path(os.path.dirname(__file__))
absolutePath = str(currentPath.parent)
sys.path.append(absolutePath)

INCLUDE_BENCHMARKS = False
DEFAULT_TESTING_EVALUATION_MECHANISM_SETTINGS = \
    TreeBasedEvaluationMechanismParameters(TreePlanBuilderParameters(TreePlanBuilderTypes.TRIVIAL_LEFT_DEEP_TREE,
                                                                     TreeCostModels.INTERMEDIATE_RESULTS_TREE_COST_MODEL),
                                           TreeStorageParameters(sort_storage=False,
                                                                 clean_up_interval=10,
                                                                 prioritize_sorting_by_timestamp=True))
DEFAULT_TESTING_DATA_FORMATTER = DataFormatter()


def get_next_formula(bindings, action_type):
    if action_type == "nop":
        return TrueFormula()
    elif action_type == "<":
        return SmallerThanFormula(IdentifierTerm(bindings[0], lambda x: x["Value"]),
                               IdentifierTerm(bindings[1], lambda x: x["Value"]))
    elif action_type == ">":
        return GreaterThanFormula(IdentifierTerm(bindings[0], lambda x: x["Value"]),
                               IdentifierTerm(bindings[1], lambda x: x["Value"]))
    else:
        return EqFormula(IdentifierTerm(bindings[0], lambda x: x["Value"]),
                                   IdentifierTerm(bindings[1], lambda x: x["Value"]))


def build_formula(bindings, action_types):
    num_ops_remainings = len(np.where(np.array(action_types) != 'nop')[0])
    if num_ops_remainings == 0 or len(bindings) == 1:
        return TrueFormula()
    elif num_ops_remainings == 1:
        if action_types[0] == "nop":
            return build_formula(bindings[1:], action_types[1:])
        else:
            return get_next_formula(bindings, action_types[0])
    else:
        if action_types[0] == "nop":
            return build_formula(bindings[1:], action_types[1:])
        else:
            return AndFormula(get_next_formula(bindings, action_types[0]), build_formula(bindings[1:], action_types[1:]))

def OpenCEP_pattern(actions, action_types, index):
    bindings = [chr(ord("a") + i) for i in range(len(actions))]
    action_types = np.array(action_types)
    pattern = Pattern(SeqOperator([PrimitiveEventStructure(event, chr(ord("a") + i)) for i, event in enumerate(actions)]),
                      build_formula(bindings, action_types),
                      timedelta(seconds=2))
    run_OpenCEP(str(index), [pattern])


def run_OpenCEP(test_name, patterns, eval_mechanism_params = DEFAULT_TESTING_EVALUATION_MECHANISM_SETTINGS):
    cep = CEP(patterns, eval_mechanism_params)
    events = FileInputStream(os.path.join(absolutePath, 'Model', 'Training', '{}.txt'.format(test_name)))
    base_matches_directory = os.path.join(absolutePath, 'Data', 'Matches')
    output_file_name = "%sMatches.txt" % test_name
    matches_stream = FileOutputStream(base_matches_directory, output_file_name)
    running_time = cep.run(events, matches_stream, DEFAULT_TESTING_DATA_FORMATTER)
    return running_time

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def prepare_loss_clac(index, window_size, match_size):
    if match_size == 0:
        match_size = 1
    print_list = []
    print_list.append('\t' + '"train_stream_path": ' + '\"' + "Model/training/{}".format(index) + '.txt",\n')
    print_list.append('\t' + '"test_stream_path": ' + '\"' + "Model/training/{}".format(index) + '.txt",\n')
    print_list.append('\t' + '"train_log_file": ' + '\"' + "Training/training_data/train_log_file.txt" + '",\n')
    print_list.append('\t' + '"train_matches": ' + '\"' + "Data/{}.txt".format(index) + '",\n')
    print_list.append('\t' + '"test_matches": ' + '\"' + "Data/{}.txt".format(index) + '",\n')
    print_list.append('\t' + '"test_size": ' + '\"' + str(window_size) + '",\n')
    print_list.append('\t' + '"train_size": ' + '\"' + str(window_size) + '",\n')
    print_list.append('\t' + '"pattern_window_size": ' + '\"' + str(window_size) + '",\n')
    print_list.append('\t' + '"window_size": ' + '\"' + str(window_size) + '",\n')
    print_list.append('\t' + '"match_size": ' + '\"' + str(match_size) + '",\n')
    print_list.append('\t' + '"event_types": ' + '["A", "B", "C", "D", "E", "F", "G"]' + "\n")
    with open("Data/constants.json", "w") as f:
        f.write('{\n')
        for l in print_list:
            f.write(l)
        f.write('}')



def prepare_pattern(forward_res, action_types,  index):
    action_types = np.array(action_types)
    num_speical_actions = len(np.where(action_types != "nop")[0])
    with open("pattern", "w") as f:
        count = 0
        assigments = []
        event_types = []
        for event in forward_res:
            count += 1
            event_types.append(event)
            # TODO: change to 10 to real value calculated from the model's output
            if len(assigments) == 0:
                assigments.append("from Event#length(10) as a{},\n".format(count))
            else:
                assigments.append("Event#length(10) as a{},\n".format(count))

        if count == 0:
            f.write("select * from Event#length(1) as a1 \nwhere \na1.type = '&' \ndone")
            return
        assigments[-1] = assigments[-1].split(",")[0]
        f.write("select *\n")
        for assign in assigments:
            f.write(assign)
        f.write('\nwhere\n')
        if count == 1:
            f.write('a{}.type = \'{}\'\n'.format(1, event_types[0]))
        else:
            for i in range(0, count):
                f.write('a{}.type = \'{}\' and\n'.format(i + 1, event_types[i]))
            for i in range(0, count - 2):
                f.write('a{}.count < a{}.count and \n'.format(i + 1, i + 2))
            if len(action_types) == num_speical_actions:
                f.write('a{}.count < a{}.count \n'.format(count - 1, count))
            else:
                f.write('a{}.count < a{}.count and \n'.format(count - 1, count))
                count_curr_speical = 0
                for i in range(0, count):
                    if action_types[i] != 'nop':
                        f.write('a{}.value {} a{}.value'.format(i, action_types[i], i + 1))
                        count_curr_speical += 1
                        if count_curr_speical == num_speical_actions:
                            f.write(" \n")
                            break
                        else:
                            f.write(" and \n")
        f.write("done")


def get_event_type(event):
    return chr(ord('A') + event)


def mapping(num_events, value):
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
    return help_map.get(value), kind_of_action


