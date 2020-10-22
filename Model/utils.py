import torch
from torch.autograd import Variable
import numpy as np

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


