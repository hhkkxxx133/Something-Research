stat_val = {}
stat_test = {}
with open('stat.txt','r') as f:
    for line in f.readlines():
        videoid, label, ratio = line.split('\t')
        active, total = ratio.split('/')
        if videoid[:4] == 'test':
            if int(label) in stat_test:
                stat_test[int(label)][0] += int(active)
                stat_test[int(label)][1] += int(total)
            else:
                stat_test[int(label)] = [int(active), int(total)]
        elif videoid[:3] == 'val':
            if int(label) in stat_val:
                stat_val[int(label)][0] += int(active)
                stat_val[int(label)][1] += int(total)
            else:
                stat_val[int(label)] = [int(active), int(total)]
for key in stat_val:
    stat_val[key] = [stat_val[key][0], stat_val[key][1]]
    stat_test[key] = [stat_test[key][0], stat_test[key][1]]


print('>>> validation')
print(stat_val)
active = 0
total = 0
for key in stat_val:
    active += stat_val[key][0]
    total += stat_val[key][1]
print(active)
print(total)
print(active/total)
active = 0
total = 0
print('>>> test')
print(stat_test)
for key in stat_test:
    active += stat_test[key][0]
    total += stat_test[key][1]
print(active)
print(total)
print(active/total)
