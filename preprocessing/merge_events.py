import random



INPUT_FILE = "Football/game_first_part"
OUTPUT_FILE = "Football/merge_x00_new1"
GROUP_SIZE = 125
BALL_GROUP_SIZE = 1000

BALL = [4,8,10]
SAME_EVENTS = {13 : [13, 14, 97, 98], 47: [47, 16], 49: [49, 88], 19: [19, 52],
                53: [53, 54], 23: [23, 24], 57: [57, 58], 59: [59, 28], 61: [61, 62, 99, 100],
                63: [63, 64], 65: [65, 66], 67: [67, 68], 69: [69, 38], 71: [71,40],
                73: [73, 74], 75: [75, 44], 105: [105, 106], 4: [4], 8: [8], 10: [10], 12: [12]}
def main():
    counts = {}
    current_events = {}
    last_event = {}
    next_time = {}
    with open(OUTPUT_FILE, "w+") as out_f:
        with open(INPUT_FILE, "r") as read_f:
            for line in read_f.readlines():
                event = line.split(",")[0]
                for k, v in zip(SAME_EVENTS.keys(), SAME_EVENTS.values()):
                    if int(event) in v:
                        event = str(k)
                        break
                data = line.split("\n")[0].split(",")[1:]
                if event in current_events.keys():
                    current_events[event].append(data)
                else:
                    current_events[event] = [data]

                add_flag = False
                if not event in last_event:
                    last_event.update({event: current_events[event][-1][0]})
                    add_flag = True
                    if int(event) in BALL:
                        # print(event)
                        next_time.update({event: random.randint(2.5e11, 6e11)})
                    else:
                        next_time.update({event: random.randint(4e11, 7e11)})
                else:
                    old_ts = int(last_event[event])
                    curr_ts = int(current_events[event][-1][0])
                    if (curr_ts - old_ts) > next_time[event]:
                        last_event[event] = str(curr_ts)
                        if int(event) in BALL:
                            next_time.update({event: random.randint(2.5e11, 4e11)})
                        else:
                            next_time.update({event: random.randint(3e11, 5e11)})
                        add_flag = True
                if add_flag:
                    value = current_events[event]
                    location = value[-1][1:4]
                    speed = value[-1][6:9]
                    ts = value[-1][0]
                    event_str = event + "," + ts + "," + ",".join(location + speed) + "\n"
                    out_f.write(event_str)
                    if not event in counts.keys():
                        counts.update({event: 1})
                    else:
                        counts[event] += 1
                    current_events[event] = [current_events[event][-1]]

    counts = reversed(sorted(counts.items(), key=lambda x: x[1]))
    for (event, count) in counts:
        print(f" event: {event} - {count}")

if __name__ == "__main__":
    main()
