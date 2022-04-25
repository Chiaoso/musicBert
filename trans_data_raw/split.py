_id = open('train.id', 'r').read().strip().split('\n')
label = open('train.label', 'r').read().strip().split('\n')
txt = open('train.txt', 'r').read().strip().split('\n')

n = len(txt)
train_n = n // 10 * 9
test_n = n - train_n

train_id = '\n'.join(_id[:train_n])
train_label = '\n'.join(label[:train_n])
train_txt = '\n'.join(txt[:train_n])

test_id = '\n'.join(_id[train_n:])
test_label = '\n'.join(label[train_n:])
test_txt = '\n'.join(txt[train_n:])

with open('train.id', 'w') as f:
        f.write(train_id)
with open('train.label', 'w') as f:
        f.write(train_label)
with open('train.txt', 'w') as f:
        f.write(train_txt)

with open('test.id', 'w') as f:
        f.write(test_id)
with open('test.label', 'w') as f:
        f.write(test_label)
with open('test.txt', 'w') as f:
        f.write(test_txt)
