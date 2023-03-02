#open file and read to list of strings
with open("/home/gabri/Polito/ii - Magistrale/0-repos/MachineLearningPatternRecognition/lab1/score.txt") as inputfile:
    filecontent=inputfile.readlines()

#split each string in the list to a list of strings
fields=map(lambda s:s.split(" "),filecontent)

#empty list of strings for output
ranking=[]

# for each partecipants
for i in fields:

    #removes newline char in the last score
    i[-1]=i[-1].replace("\n","")

    max=0
    min=200
    # evaluate max and min score and sets them to 0 so they dont 
    # count for the rank
    for j in range(3,8):
        i[j]=float(i[j])
        if i[j]>max: max=j
        if i[j]<min : min=j 
    i[max]=0
    i[min]=0

    # calc final score for player i
    sum=0
    for j in range(3,7):
        sum+=i[j]

    # save value to new list
    totalscore=[]
    totalscore.append(i[0]+" "+i[1])
    totalscore.append(i[2])
    totalscore.append(sum)

    # append to a list:
    # ranking is a list of list in the shape:
    # [ [player,contry,score], [], [] ]
    ranking.append(totalscore)

# sort ranking by total score
ranking.sort(key=lambda pl:-pl[2])

# top 3 players
podium=ranking[0:3]


print(podium)