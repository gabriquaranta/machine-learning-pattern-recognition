import math

# flags
flag1 = "-b"
flag2 = "-l"
param1 = "1976"
param2 = "4"

# set here instead of terminal arguments
flag = flag1
param = param1
flag = flag2
param = param2


class BusStop:
    def __init__(self, busid, lineid, x, y, time):
        self.busid = busid
        self.lineid = lineid
        self.x = x
        self.y = y
        self.time = time


class BusRide:
    def __init__(self, busid, lineid, distance, time):
        self.busid = busid
        self.lineid = lineid
        self.distance = distance
        self.time = time
        self.speed = distance / time


def distance2D(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# open file
with open(
    "/home/gabri/Polito/ii - Magistrale/0-repos/MachineLearningPatternRecognition/lab1/ex2_transports.txt"
) as inp:
    inputlines = inp.readlines()  # inputlines list of strings

# generate list of BusStop objects
busses = []
for line in inputlines:
    # split line to fields
    fields = line.split(" ")
    # create BusStop object and appen to list
    busid = fields[0]
    lineid = fields[1]
    x = int(fields[2])
    y = int(fields[3])
    time = int(fields[4])
    busses.append(BusStop(busid, lineid, x, y, time))

# generate list og BusRide objects
# calculatig for each bus id the total distance and time
busrides = []
for bus in busses:
    coor_x = []
    coor_y = []
    times = []
    totaldistance = 0
    totaltime = 0

    # create list of all coordinates for stops for each bus id
    for bus1 in filter(lambda b: b.busid == bus.busid, busses):
        coor_x.append(bus1.x)
        coor_y.append(bus1.y)
        times.append(bus1.time)
        busses.remove(bus1)  # removes so its not counted multiple times

    # calculate toal distacne for each bus id as sum of distances betweeen stops
    for i in range(len(coor_x) - 1):
        x1 = coor_x[i]
        y1 = coor_y[i]
        j = i + 1
        x2 = coor_x[j]
        y2 = coor_y[j]
        totaldistance += distance2D(x1, y1, x2, y2)

    # calculate total time as the last time logged minus the first
    totaltime = times[-1] - times[0]
    # create and appen a BusRide object
    busrides.append(BusRide(bus.busid, bus.lineid, totaldistance, totaltime))


# output
if flag == "-b":
    for ride in busrides:
        if ride.busid == param:
            print(ride.busid, ride.distance)

if flag == "-l":
    c = 0
    s = 0
    for ride in filter(lambda r: r.lineid == param, busrides):
        s += ride.speed
        c = c + 1
    print(ride.lineid, s / c)
