#open file and read to list of strings
with open("/home/gabri/Polito/ii - Magistrale/0-repos/MachineLearningPatternRecognition/lab1/score.txt") as inputfile:
    filecontent=inputfile.readlines()

#split each string in the list to a list of strings
fields=map(lambda s:s.split(" "),filecontent)

#empty list of strings for output
ranking=[]

