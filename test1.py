import csv
import pandas as pd
import numpy as np


match=[]
win=[]
gold=[]

with open('stats_10000.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            #print(f'Column names are {", ".join(row)}' + "\n")
            line_count += 1
        else:
            line_count += 1
            match.append(row[1])
            win.append(row[2])
            gold.append(row[42])
            
    print(f'Processed {line_count} lines.' + "\n")

##calculate average open price
sum=0
count=0
for g in gold:
    sum += int(g) 
    count+=1
mean = sum/count
print(str(count) + " data. " + str(sum) + " in sum and average gold earned is: " + str(mean) + "\n")

##calculate variance
sum_sq = 0
for g in gold:
    sum_sq += (int(g)-mean)**2
var = sum_sq/(count-1)
print("the gold earned variance is: " + str(var) + "\n")

print("length of match list " + str(len(match)) + "\n")

#print 5 rows to confirm the data set is correct.
for i in range(5):
    print(match[i])
    print(win[i])
    print(gold[i])
    print("----")
    
print("")

##calculate the gold variance of each team.
df = pd.DataFrame({"match":match,"win":win,"gold":gold},columns =['match','win','gold'], dtype=int)
print(df.info())
print("")
print(df.tail()) #confirm the 'df' dataframe is correct.
print("")

df_tgold = df.groupby(['match','win']) #categorize df
df_var_tgold = df_tgold['gold'].agg('var')

print(df_var_tgold.tail(10)) #confrim the df_var_tgold.

df_var_tgold.to_csv('rank_var_test.csv')
#output the data into a new csv




    
