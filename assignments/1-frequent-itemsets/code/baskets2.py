"""
Created on Fri, Feb  2, 2018
"""
import operator

IN_FILE = './browsingdata.txt'
OUT_FILE = './output.txt'
SUPPORT = 100
TOP = 5

#Store baskets as list
baskets = list()
f = open(IN_FILE, 'r')
lines = f.read().splitlines()
for line in lines:
    baskets.append(set(line.split(" ")))
f.close()

#Store count of each item as dictionary
items = dict()
for basket in baskets:
    for item in basket:
        items[item] = items.get(item, 0) + 1
items.pop('', None)

#Store frequent items as dictionary
freq_items = dict()
for key in items:
    value = items[key]
    if (value >= SUPPORT):
        freq_items[key] = value
        
#Store count of candidate itemsets as dictionary
candidates = dict()
for basket in baskets:
    for index_i, item_i in enumerate(basket):
        if item_i in freq_items:
            for index_j, item_j in enumerate(basket):
                if index_j > index_i:
                    if item_j in freq_items:
                        if item_i > item_j:
                            candidates[item_i, item_j] = candidates.get((item_i, item_j), 0) + 1
                        else:
                            candidates[item_j, item_i] = candidates.get((item_j, item_i), 0) + 1

#Store count of frequent candidate itemsets as dictionary
freq_candidates = dict()
for key in candidates:
    value = candidates[key]
    if (value >= SUPPORT):
        freq_candidates[key] = value

#Store count of candidate itemsets as dictionary
candidates3 = dict()
for basket in baskets:
    for index_i, item_i in enumerate(basket):
        if item_i in freq_items:
            for index_j, item_j in enumerate(basket):
                if index_j > index_i:
                    if item_j in freq_items:
                        if (item_i,item_j) in freq_candidates or (item_j,item_i) in freq_candidates: 
                            for index_k, item_k in enumerate(basket):
                                if index_k > index_j:
                                    if item_k in freq_items:
                                        if (item_i,item_k) in freq_candidates or (item_k,item_i) in freq_candidates:
                                            if (item_j,item_k) in freq_candidates or (item_k,item_j) in freq_candidates:
                                                if item_i > item_j:
                                                    if item_k > item_i:
                                                        candidates3[item_k, item_i, item_j] = candidates3.get((item_k, item_i, item_j), 0) + 1
                                                    elif item_k > item_j:
                                                        candidates3[item_i, item_k, item_j] = candidates3.get((item_i, item_k, item_j), 0) + 1
                                                    else:
                                                        candidates3[item_i, item_j, item_k] = candidates3.get((item_i, item_j, item_k), 0) + 1
                                                else:
                                                    if item_k > item_j:
                                                        candidates3[item_k, item_j, item_i] = candidates3.get((item_k, item_j, item_i), 0) + 1
                                                    elif item_k > item_i:
                                                        candidates3[item_j, item_k, item_i] = candidates3.get((item_j, item_k, item_i), 0) + 1
                                                    else:
                                                        candidates3[item_j, item_i, item_k] = candidates3.get((item_j, item_i, item_k), 0) + 1

#Store count of frequent candidate itemsets as dictionary
freq_candidates3 = dict()
for key in candidates3:
    value = candidates3[key]
    if (value >= SUPPORT):
        freq_candidates3[key] = value      
                                                                                                                                                                                                
#Store rules with confidence scores as dictionary
rules = dict()
for key in freq_candidates:
    rules[key[0], key[1]] = freq_candidates[key]/freq_items[key[0]]
    rules[key[1], key[0]] = freq_candidates[key]/freq_items[key[1]]
rules3 = dict()
for key in freq_candidates3:
    if (key[0], key[1]) in freq_candidates:
        rules3[key[0], key[1], key[2]] = freq_candidates3[key]/freq_candidates[key[0], key[1]]
    else:
        rules3[key[0], key[1], key[2]] = freq_candidates3[key]/freq_candidates[key[1], key[0]]
    if (key[0], key[2]) in freq_candidates:    
        rules3[key[0], key[2], key[1]] = freq_candidates3[key]/freq_candidates[key[0], key[2]]
    else:
        rules3[key[0], key[2], key[1]] = freq_candidates3[key]/freq_candidates[key[2], key[0]]
    if (key[1], key[2]) in freq_candidates:
        rules3[key[1], key[2], key[0]] = freq_candidates3[key]/freq_candidates[key[1], key[2]]
    else:
        rules3[key[1], key[2], key[0]] = freq_candidates3[key]/freq_candidates[key[2], key[1]]
        
#Convert rules to sorted list of tuples
sorted_rules = sorted(rules.items(), key=operator.itemgetter(1), reverse=True)
sorted_rules3 = sorted(rules3.items(), key=operator.itemgetter(1), reverse=True)

#Subset rules with confidence equal to or greater than TOP=5 highest rule 
top = list()
tie = list()
threshold_confidence = sorted_rules[TOP-1][1] 
for rule in sorted_rules:
    if (rule[1] > threshold_confidence):
        top.append(rule)
    elif (rule[1] == threshold_confidence):
        tie.append(rule)
    else:
        break
tie.sort(key=operator.itemgetter(0))
top = top + tie[0:(TOP-len(top))]    

top3 = list()
tie3 = list()
threshold_confidence = sorted_rules3[TOP-1][1] 
for rule in sorted_rules3:
    if (rule[1] > threshold_confidence):
        top3.append(rule)
    elif (rule[1] == threshold_confidence):
        tie3.append(rule)
    else:
        break
tie3.sort(key=operator.itemgetter(0))
top3 = top3 + tie3[0:(TOP-len(top3))] 
 
#Write TOP rules to file
f = open(OUT_FILE, 'w')
f.write('OUTPUT A' + '\n')
for rule in top:
    f.write(rule[0][0] + ' ' + rule[0][1] + ' ' + '{:.4f}'.format(rule[1]) + '\n')
f.write('OUTPUT B' + '\n')
for rule in top3:
    f.write(rule[0][0] + ' ' + rule[0][1] + ' ' + rule[0][2] + ' ' + '{:.4f}'.format(rule[1]) + '\n')
f.close()