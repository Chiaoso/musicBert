with open('train.txt', 'r') as f:
    for j in range(4):
        line = f.readline()
        line = line.split()
        print(len(line))