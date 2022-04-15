with open('train.txt', 'r') as f:
    with open('test.txt', 'w') as t:
        len_txt = []
        len_label = []
        for j in range(5):
            line = f.readline()
            t.write(line)
