# coding: utf-8

net = {}
node_count = 0
map_ = {}

with open("out.dblp-cite", "r") as f:
    line = f.readline()
    line = f.readline()
    line = f.readline()

    while line != "":
        n1, n2 = map(int, line.strip().split(" "))
        if n1 not in map_:
            map_[n1] = node_count
            node_count += 1
        if n2 not in map_:
            map_[n2] = node_count
            node_count += 1

        line = f.readline()

    print("node count:", node_count)

with open("out.dblp-cite", "r") as f:
    line = f.readline()
    line = f.readline()
    line = f.readline()

    while line != "":
        n1, n2 = map(int, line.strip().split(" "))
        try:
            net[map_[n1]][map_[n2]] = 1
        except:
            net[map_[n1]] = {map_[n2]: 1}
        try:
            net[map_[n2]][map_[n1]] = 1
        except:
            net[map_[n2]] = {map_[n1]: 1}
        line = f.readline()

    print("edge count:", sum([len(net[i]) for i in net]))

with open("dblp-net.txt", "w") as f:
    for n1 in net:
        for n2 in net[n1]:
            f.write("%d %d 1\n" % (n1, n2))
