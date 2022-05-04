#!/bin/env python

import sys, string, os

minval = 0.000001

infile = open(sys.argv[1]+".csv","r")
outfile = open(sys.argv[1]+"_std.csv","w")

c = 0
for line in infile:
 tokens = line.split(',')
 if len(tokens) > 1:
  if c > 0:
    tokens = [float(a) for a in tokens]
    tokens2 = []
    for i in range(1):
      FP = tokens[4*i]
      TP = tokens[4*i+1]
      FPstd = tokens[4*i+2]
      TPstd = tokens[4*i+3]
      tokens2.extend([FP,TP,FP-FPstd,TP+TPstd,FP+FPstd,TP-TPstd])
    tokens2.extend([tokens[4],tokens[5]])
    tokens2 = [min(a,1) for a in tokens2]
    tokens2 = [max(a,minval) for a in tokens2]
  else:
    tokens2 = ["FP1","TP1","FP1UB","TP1UB","FP1LB","TP1LB","RANDOM_FP","RANDOM_TP"]
  #if c == 2:
  #  print(tokens2)
  for i in range(len(tokens2)):
    outfile.write(str(tokens2[i]))
    if i < len(tokens2) - 1:
      outfile.write(',')
    else:
      outfile.write('\n')
  c = c + 1

infile.close()
outfile.close()