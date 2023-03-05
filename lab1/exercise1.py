# open file and read to list of strings
with open(
    "lab1/ex1_score.txt"
) as inputfile:
    filecontent = inputfile.readlines()

# split each string in the list to a list of strings
fields = (s.split(" ") for s in filecontent)

# list of [player_n player_s country score ...]
playerscores = list(fields)
ps = []
cs = []

# generates two list of list:
#   ps = list of [playername, playersocre] one per player
#   cs = list of [country, playerscore] one per player
for player in playerscores:
    name = player[0] + " " + player[1]
    country = player[2]

    maxi = 0
    mini = 200
    for i in range(3, 8):
        tmp = float(player[i])
        if tmp > maxi:
            maxi = i
        if tmp <= mini:
            mini = i
        else:
            player[i] = tmp
    player[maxi] = 0.0
    player[mini] = 0.0

    score = 0
    for i in range(3, 8):
        score += player[i]

    ps.append([name, score])
    cs.append([country, score])

# sort ps and print podium aka first 3
print()
print("podium:")
ps.sort(key=lambda x: x[1], reverse=True)
for player in ps[:3]:
    print(player[0] + ": " + str(player[1]))

# calc total for each country and print max one
print()
print("best country:")
for cstuple in cs:
    for cstuple2 in cs:
        if cstuple[0] == cstuple2[0] and cstuple[1] != cstuple2[1]:
            cstuple[1] += cstuple2[1]
            cstuple2[1] = 0.0
cs.sort(key=lambda x: x[1], reverse=True)
print(cs[0][0] + ": " + str(cs[0][1]))
print()
