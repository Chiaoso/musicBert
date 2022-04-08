with open('train.jojo', 'r') as f:
    with open('train.txt', 'w') as jo:
        for j in range(4):
            line = f.readline()
            line = line.split()
            if len(line) < 256:
                line = line + \
                            ['</s>']*(256-len(line))
            for i in line:
                jo.write(str(i)+" ")
            jo.write("\n")