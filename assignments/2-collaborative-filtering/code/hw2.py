import operator
import numpy

IN_FILE = './ratings.csv'
OUT_FILE = './output.txt'
NEIGHBORHOOD = 5
TOP = 5

#Store file as list of ratings
ratings = list()
f = open(IN_FILE, 'r')
lines = f.read().splitlines()
lines.pop(0)
for line in lines:
    line = line.split(",")
    userid = int(line[0])
    movieid = int(line[1])
    rating = float(line[2])    
    ratings.append((userid,movieid,rating))
f.close()
del lines

#Collect disinct userids, movieids
#Find max userid, movieid
userids = set()
movieids = set()
for rating in ratings:
    userids.add(rating[0])
    movieids.add(rating[1])
userids = sorted(userids)
movieids = sorted(movieids)
max_userid = max(userids)
max_movieid = max(movieids)
        
#Store ratings as matrix
copy = numpy.zeros((max_movieid+1,max_userid+1))
for rating in ratings:
    copy[rating[1]][rating[0]] = rating[2]
           
#Normalize movie ratings in matrix
for movieid in movieids:
    num_ratings = 0
    for userid in userids:
        if copy[movieid][userid] != 0:
            num_ratings += 1
    avg_rating = numpy.sum(copy[movieid][:]) / num_ratings
    for userid in userids:
        if copy[movieid][userid] != 0:
            copy[movieid][userid] = copy[movieid][userid] - avg_rating
            
#Compute norm2 for each movie and store as dictionary:
#  key=movie, value=norm2
norm2 = dict()
for movieid in movieids:
    norm2[movieid] = numpy.linalg.norm(copy[movieid][:])

#Compute movie-movie similarity scores and store as dictionary
#  key=(movie1,movie2), value=norm2
scores = dict()
for i in movieids:
    for j in movieids:
        if j > i:
            dot_product = numpy.sum(copy[i][:] * copy[j][:])
            norm2_product = norm2[i] * norm2[j] 
            if norm2_product != 0:
                scores[(i,j)] = dot_product / norm2_product
            else:
                scores[(i,j)] = -1 
del copy
del norm2 
         
#Compute movie neighborhood as dictionary:
#  key=movie, value=list of (movie,score) sorted by score
neighbors = dict()
for i in movieids:
    el = list()
    for j in movieids:
        if i < j:
            el.append((j,scores[(i,j)]))
        elif i > j: 
            el.append((j,scores[(j,i)]))
    neighbors[i] = sorted(el, key=operator.itemgetter(1), reverse=True)
del scores
        
#Limit neighborhood  
for movieid in movieids:
    top = list()
    tie = list()
    el = neighbors[movieid]
    if len(el) > NEIGHBORHOOD:
        threshold = el[NEIGHBORHOOD-1][1]
        for score in el:
            if score[1] > threshold:
                top.append(score)
            elif score[1] == threshold:
                tie.append(score)
            else:
                break
        tie.sort(key=operator.itemgetter(0))
        top = top + tie[0:(NEIGHBORHOOD-len(top))]
    else:
        top = el
    neighbors[movieid] = top

#Store ratings as matrix
matrix = numpy.zeros((max_movieid+1,max_userid+1))
for rating in ratings:
    matrix[rating[1]][rating[0]] = rating[2]
del ratings

#Compute ratings from movie neighborhood
computed = dict()
for userid in userids:
    el = list()
    for movieid in movieids:
        if matrix[movieid][userid] == 0:
            num = 0
            denom = 0
            for n in neighbors[movieid]:
                if matrix[n[0]][userid] != 0:
                    num += n[1] * matrix[n[0]][userid]
                    denom += n[1]
            if denom > 0:
                el.append((movieid, num / denom))
    computed[userid] = sorted(el, key=operator.itemgetter(1), reverse=True)
del matrix   
           
#Limit recommendations
recs = dict()
for userid in userids:
    top = list()
    tie = list()
    el = computed[userid]
    if len(el) > TOP:
        threshold = el[TOP-1][1]
        for rating in el:
            if rating[1] > threshold:
                top.append(rating)
            elif rating[1] == threshold:
                tie.append(rating)
            else:
                break
        tie.sort(key=operator.itemgetter(0))
        top = top + tie[0:(TOP-len(top))]
    else:
        threshold = el[len(el)-1][1]
        for rating in el:
            if rating[1] > threshold:
                top.append(rating)
            else:
                tie.append(rating)
        tie.sort(key=operator.itemgetter(0))
        top = top + tie
    recs[userid] = top    
del computed  

#Write TOP recs for each user to file
f = open(OUT_FILE, 'w')
for key, value in sorted(recs.items()):
    f.write(str(key))
    for v in value:
        f.write(' ' + str(v[0]))
    f.write('\n')
f.close()