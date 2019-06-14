import numpy

'''Constants'''
TRAINING_FILE = './fortunecookiedata/traindata.txt'
TRAINING_LABEL_FILE = './fortunecookiedata/trainlabels.txt'
TESTING_FILE = './fortunecookiedata/testdata.txt'
TESTING_LABEL_FILE = './fortunecookiedata/testlabels.txt'
STOP_WORDS_FILE = './fortunecookiedata/stoplist.txt'
OUT_FILE = './output.txt'
LEARNING_RATE = 1
ITERATIONS = 20

'''Functions'''
#Compute accuracy of training or testing data
# 
def accuracy(weight,examples,labels):
    correct = 0
    S = numpy.shape(examples)
    for index in range(0,S[0]):
        predicted = numpy.dot(examples[index],numpy.transpose(weight))
        if ((predicted[0] > 0 and labels[index] > 0) or \
            (predicted[0] <= 0 and labels[index] < 0)):
            correct += 1           
    return correct / S[0]

'''Binary classifier'''
#Read words in training file into set
training_words = set()
f = open(TRAINING_FILE, 'r')
lines = f.read().split('\n')
for line in lines:
    words = line.split(' ')
    for word in words:
        training_words.add(word)        
f.close()

#Remove stop words from training words to generate vocabulary
f = open(STOP_WORDS_FILE, 'r')
stop_words = f.read().split('\n')
for word in stop_words:
    training_words.discard(word)        
f.close()
del stop_words

#Index sorted vocabulary as dictionary
vocabulary = dict()
M = len(training_words)
for index, word in enumerate(sorted(training_words)):
    vocabulary[word] = index
del training_words

#Transform each fortune into a feature vector, where each feature corresponds
#  to the presence (1) or absence (0) of a vocabulary word
fortunes = numpy.zeros((len(lines),M+1))
for index, line in enumerate(lines):
    words = line.split(' ')
    for word in words:
        if vocabulary.get(word) is not None:
            fortunes[index][vocabulary[word]] = 1
    fortunes[index][M] = 1
del lines

#Read training labels into list
f = open(TRAINING_LABEL_FILE, 'r')
train_labels = f.read().split('\n')
for index,label in enumerate(train_labels):
    if int(label) == 0:
        train_labels[index] = -1
    else:
        train_labels[index] = 1
f.close()

#Read and transform testing data into feature vectors 
f = open(TESTING_FILE, 'r')
lines = f.read().split('\n')
lines.pop()
test_data = numpy.zeros((len(lines),M+1))
for index, line in enumerate(lines):
    words = line.split(' ')
    for word in words:
        if vocabulary.get(word) is not None:
            test_data[index][vocabulary[word]] = 1
    test_data[index][M] = 1        
f.close()
del lines

#Read testing labels into list
f = open(TESTING_LABEL_FILE, 'r')
test_labels = f.read().split('\n')
test_labels.pop()
for index,label in enumerate(test_labels):
    if int(label) == 0:
        test_labels[index] = -1
    else:
        test_labels[index] = 1
f.close()

#Perceptron
w = numpy.zeros((1,M+1))
el_mistakes = list()
el_train = list()
el_test = list()
S = numpy.shape(fortunes)
for i in range(1,ITERATIONS+1):
    mistakes = 0
    for index in range(0,S[0]):
        predicted = numpy.dot(fortunes[index],numpy.transpose(w))
        if (predicted[0] * train_labels[index] <= 0):
            mistakes += 1
            w = w + LEARNING_RATE * train_labels[index] * fortunes[index]
    el_mistakes.append(mistakes)
    el_train.append(accuracy(w,fortunes,train_labels))
    el_test.append(accuracy(w,test_data,test_labels))
    
#AvgPerceptron
w = numpy.zeros((1,M+1))
u = numpy.zeros((1,M+1))
c = 1
S = numpy.shape(fortunes)
for i in range(1,ITERATIONS+1):
    for index in range(0,S[0]):
        predicted = numpy.dot(fortunes[index],numpy.transpose(w))
        if (predicted[0] * train_labels[index] <= 0):
            w = w + LEARNING_RATE * train_labels[index] * fortunes[index]
            u = u + c * LEARNING_RATE * train_labels[index] * fortunes[index]
        c += 1
w = w - u * (1/c)
avg_train_acc = accuracy(w,fortunes,train_labels)
avg_test_acc = accuracy(w,test_data,test_labels)
    
#Write mistakes, training accuracy, and testing accuracy to outfile
f = open(OUT_FILE, 'w')
for i in range(1,ITERATIONS+1):
    f.write('iteration-' + str(i) + ' ' + str(el_mistakes[i-1]) + '\n')
for i in range(1,ITERATIONS+1):
    f.write('iteration-' + str(i) + ' ' + str(el_train[i-1]) + ' ' + str(el_test[i-1]) + '\n')
f.write(str(el_train[ITERATIONS-1]) + ' ' + str(el_test[ITERATIONS-1]) + '\n')
f.write(str(avg_train_acc) + ' ' + str(avg_test_acc) + '\n\n')
f.close()


'''Constants'''
OCR_TRAINING_FILE = './OCRdata/ocr_train.txt'
OCR_TESTING_FILE = './OCRdata/ocr_test.txt'
OUT_FILE = './output.txt'
LEARNING_RATE = 1
ITERATIONS = 20
CLASSES = 26

'''Functions'''
#Compute accuracy of training or testing data
# 
def accuracy2(weight,examples,labels,d):
    correct = 0
    S = numpy.shape(examples)
    for i in range(0,S[0]):
        predicted = numpy.zeros((1,CLASSES))
        for j in range(0,CLASSES):
            predicted[0][j] = numpy.dot(examples[i],numpy.transpose(weight[j]))
        if d[numpy.argmax(predicted)] == labels[i]:
            correct += 1           
    return correct / S[0]

'''Multi-class classifier'''
#Read training data and labels into separate lists
ocr_train = list()
ocr_train_label = list()
f = open(OCR_TRAINING_FILE, 'r')
lines = f.read().split('\n')
for line in lines:
    elements = line.split('\t')
    if (len(elements) > 3) and (elements[3] == '_'):
        ocr_train.append(elements[1].lstrip('im'))
        ocr_train_label.append(elements[2])
f.close()

#Read testing data and labels into separate lists
ocr_test = list()
ocr_test_label = list()
f = open(OCR_TESTING_FILE, 'r')
lines = f.read().split('\n')
for line in lines:
    elements = line.split('\t')
    if (len(elements) > 3) and (elements[3] == '_'):
        ocr_test.append(elements[1].lstrip('im'))
        ocr_test_label.append(elements[2])
f.close()
del lines

#Transform training data into feature vectors
features = len(ocr_train[0])
train_data = numpy.zeros((len(ocr_train),features+1))
for index,example in enumerate(ocr_train):
    for i,digit in enumerate(example):
        train_data[index][i] = int(digit)
    train_data[index][features] = 1
del ocr_train
    
#Transform testing data into feature vectors
test_data = numpy.zeros((len(ocr_test),features+1))
for index,example in enumerate(ocr_test):
    for i,digit in enumerate(example):
        test_data[index][i] = int(digit)
    test_data[index][features] = 1
del ocr_test

#Build dictionaries to map letter to number
letter_to_index = dict()
index_to_letter = dict()
letters = sorted(list(set(ocr_train_label)))
for index,letter in enumerate(letters):
    letter_to_index[letter] = index 
    index_to_letter[index] = letter
    
#Perceptron
w = numpy.zeros((CLASSES,features+1))
el_mistakes = list()
el_train = list()
el_test = list()
S = numpy.shape(train_data)
for i in range(1,ITERATIONS+1):
    mistakes = 0
    for j in range(0,S[0]):
        predicted = numpy.zeros((1,CLASSES))
        for k in range(0,CLASSES):
            predicted[0][k] = numpy.dot(train_data[j],numpy.transpose(w[k]))
        p_index = numpy.argmax(predicted)
        a_index = letter_to_index[ocr_train_label[j]]
        if p_index != a_index:
            mistakes += 1
            w[p_index] = w[p_index] - LEARNING_RATE * train_data[j]
            w[a_index] = w[a_index] + LEARNING_RATE * train_data[j]
    el_mistakes.append(mistakes)
    el_train.append(accuracy2(w,train_data,ocr_train_label,index_to_letter))
    el_test.append(accuracy2(w,test_data,ocr_test_label,index_to_letter))

 
#AvgPerceptron
w = numpy.zeros((CLASSES,features+1))
u = numpy.zeros((CLASSES,features+1))
c = 1
for i in range(1,ITERATIONS+1):
    for j in range(0,S[0]):
        predicted = numpy.zeros((1,CLASSES))
        for k in range(0,CLASSES):
            predicted[0][k] = numpy.dot(train_data[j],numpy.transpose(w[k]))
        p_index = numpy.argmax(predicted)
        a_index = letter_to_index[ocr_train_label[j]]
        if p_index != a_index:
            w[p_index] = w[p_index] - LEARNING_RATE * train_data[j]
            w[a_index] = w[a_index] + LEARNING_RATE * train_data[j]           
            u[p_index] = u[p_index] - c * LEARNING_RATE * train_data[j]
            u[a_index] = u[a_index] + c * LEARNING_RATE * train_data[j]
        c += 1
w = w - u * (1/c)
avg_train_acc = accuracy2(w,train_data,ocr_train_label,index_to_letter)
avg_test_acc = accuracy2(w,test_data,ocr_test_label,index_to_letter)
 
#Write mistakes, training accuracy, and testing accuracy to outfile
f = open(OUT_FILE, 'a')
for i in range(1,ITERATIONS+1):
    f.write('iteration-' + str(i) + ' ' + str(el_mistakes[i-1]) + '\n')
for i in range(1,ITERATIONS+1):
    f.write('iteration-' + str(i) + ' ' + str(el_train[i-1]) + ' ' + str(el_test[i-1]) + '\n')
f.write(str(el_train[ITERATIONS-1]) + ' ' + str(el_test[ITERATIONS-1]) + '\n')
f.write(str(avg_train_acc) + ' ' + str(avg_test_acc) + '\n')
f.close()

