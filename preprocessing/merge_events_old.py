import random



INPUT_FILE = "Football/game_first_part"
OUTPUT_FILE = "Football/merge_x00_new"
GROUP_SIZE = 125
BALL_GROUP_SIZE = 1250

BALL = [4,8,10]

def main():
    current_events = {}
    counts = {}

    with open(OUTPUT_FILE, "w+") as out_f:
        with open(INPUT_FILE, "r") as read_f:
            for line in read_f.readlines():
                flag = False
                event = line.split(",")[0]
                data = line.split("\n")[0].split(",")[1:]
                if event in current_events.keys():
                    current_events[event].append(data)
                else:
                    current_events[event] = [data]

                if int(event) in BALL:
                    if len(current_events[event]) >= BALL_GROUP_SIZE:
                        flag = True
                elif len(current_events[event]) >= GROUP_SIZE:
                    flag = True
                if flag:
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
