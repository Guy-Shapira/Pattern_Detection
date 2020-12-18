INPUT_FILE = "Football/x02"
OUTPUT_FILE = "Football/merge_x02"
GROUP_SIZE = 200


def main():
    current_events = {}
    with open(OUTPUT_FILE, "w+") as out_f:
        with open(INPUT_FILE, "r") as read_f:
            for line in read_f.readlines():
                event = line.split(",")[0]
                data = line.split("\n")[0].split(",")[1:]
                if event in current_events.keys():
                    current_events[event].append(data)
                else:
                    current_events[event] = [data]

                if len(current_events[event]) == GROUP_SIZE:
                    value = current_events[event]
                    start_location = value[0][1:4]
                    start_speed = value[0][6:9]
                    ts = value[-1][0]
                    end_location = value[-1][1:4]
                    end_speed = value[-1][6:9]
                    event_str = event + "," + ts + "," + ",".join(start_location + start_speed + end_location + end_speed) + "\n"
                    out_f.write(event_str)
                    del current_events[event]

if __name__ == "__main__":
    main()