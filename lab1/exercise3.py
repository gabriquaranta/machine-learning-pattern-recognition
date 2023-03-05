
""" 
    A text file contains informations on a group of people born in a given year. 

    The format is the following:
        <name> <surname> <birthplace> <birthdate>
    The first three fields are strings (with no blanks), 
    <birthdate> is a string with format DD/MM/YYYY/

    Each line corresponds to a person, and births are not sorted. 

    Write a program that computes:
    • The number of births for each city
    • The number of births for each month
    • The average number of births per city (number of births over number of cities)
"""

#open file
with open("lab1/ex3_people.txt") as inputfile:
    lines=inputfile.readlines()

#months dict
month_births_count={i:0 for i in range(1,13)}
city_births_count={}

for line in lines:
    fields=line.split(" ")
    city=fields[2]
    month=int(fields[3].split("/")[1])

    # number of births per month
    month_births_count[month]=+1

    #number of births per city :  increment existsing city count or innit
    if city in city_births_count:
        city_births_count[city]+=1
    else:
        city_births_count[city]=1


#births per month
print("\n",month_births_count)
#births per city
print("\n",city_births_count)
#avg births per city
s=0 
for i in city_births_count.values():
    s+=i
print("\nAverage number of births by city: ",s/len(city_births_count.values()))