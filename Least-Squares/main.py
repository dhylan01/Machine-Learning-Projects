
#List of imported packages to help with commands
import numpy
import scipy.io
import scipy.io as sio
import numpy as np
from itertools import combinations
from pathlib import Path
from matplotlib import pyplot as plt




#script for the main method of my function
if __name__ == '__main__':
    #command to import data
    data = scipy.io.loadmat("mnist.mat")

    #seperate data and cast as floats then normalize image pixel values
    testX = data['testX'].astype(float) / 255
    trainX = data['trainX'].astype(float) / 255
    testY = data['testY']
    trainY = data['trainY']


    #add in the a end value of 1 for each image
    testX = np.append(testX, np.ones([len(testX), 1]), 1)
    trainX = np.append(trainX, np.ones([len(trainX), 1]), 1)

    #a function made to remove all columns with less than 600 non-zero values
    def remove_nonZeros(to_remove1,to_remove2):
        i = 0
        while i < to_remove1.shape[1]:
            #we want to get the column
            temp = to_remove1[:, i]
            if np.count_nonzero(temp) < 600:
                to_remove1 = np.delete(to_remove1,i,axis=1)
                to_remove2 = np.delete(to_remove2, i, axis=1)
            else:
                i += 1
        return to_remove1,to_remove2

    #function called to remove zeroes on all of the images but made little difference in test data so it was not used
    #trainX,testX = remove_nonZeros(trainX,testX)

    #unused function to get classifier with the regular inverse
    def get_classifier(fullRank_x, changed_y):

        fullRank_x_transpose = np.transpose(fullRank_x)
        train_inv = np.linalg.inv(np.matmul(fullRank_x_transpose, fullRank_x))
        changed_y_transpose = np.transpose(changed_y)
        #print("sizes",train_inv.shape,fullRank_x_transpose.shape,)
        theta = np.matmul(np.matmul(train_inv,fullRank_x_transpose), changed_y)
        return theta
    #function to get theta from a given x and a matrix of adjusted labels to problem
    def get_classifier_pinv(x_prime, adjusted_y):
        #use pinv function to get a pseudo inverse of x
        train_inv = np.linalg.pinv(x_prime)
        #find our theta/weights matrix by multiplying the pseudo inverse of x with y
        theta = np.matmul(train_inv, adjusted_y)
        return theta
    #function to generate all one vs all weights/classfieirs for a set of training data and a set of numbers(range from start to end - inclusive)
    def oneVsAll_classfiers(start,end,training_X,training_Y):

        #tX, training_data = remove_nonZeros(training_X, training_Y) - Note: did not make a difference

        #make copies of given data in the case that they might be edited
        tX, training_data = training_X.copy(),training_Y.copy()
        #temperary array to store weights until returned
        oneVsAll_temp = []
        #run for each range of values
        for i in range(start,end+1):
            # if label = one for all value then set to 1, otherwise set to 0
            tempY = training_data
            tempY = (tempY == i)
            tempY = tempY*2 - 1
            tempY = np.transpose(tempY)
            #use the pinv to get the right classifier
            classifier = get_classifier_pinv(tX,tempY)
            #store classifier
            oneVsAll_temp.append(classifier)
        return oneVsAll_temp

    # function to return wrong_predictions,error_rate, and confusion matrix for one versus all
    def test_oneVsAll(start,end,test_images,test_data,classifiers):

        #matrix to store all predicted labels from each classifier
        results = []
        #for each classifier store the label
        for classifier in classifiers:
            results.append(np.matmul(test_images, classifier))
        #empty confusion matrix with each row = a value 0-9 of given data lavel and each colum = a value 0-9 of predicted data label
        oneVsALL_confusion_test = [[0 for i in range(start,end+1)] for j in range(start,end+1)]
        #use argmax to only use label for each image form the classifier with highest confidence
        results = np.argmax(results,axis=0)

        #go through each correct label
        for idx,result in enumerate(test_data[0]):
            #check predicted and given results and add to confusion matrix accordingly
            oneVsALL_confusion_test[int(result)][int(results[idx][0])] += 1
        #calculate wrong predictions and error rate
        wrong_predictions = np.sum(oneVsALL_confusion_test) - np.trace(oneVsALL_confusion_test)
        error_rate = wrong_predictions / max(test_data.shape)
        return wrong_predictions,error_rate,oneVsALL_confusion_test


    #function to generate all one verson one classifiers
    def one_Vs_one_classifiers(start,end,training_x,training_y):
        #array to store all weights/classifiers
        ret = []
        #go through each combination of numbers in the range of numbers given
        for first,second in list(combinations(range(start,end+1),2)):

            #make a compy of x,y data due to manipulation of data
            tempX = training_x.copy()
            tempY = training_y.copy()

            #set labels equal to 1 if first part of pair and -1 if second. Any non-relevent images takn out from dataset
            tempY_first = tempY == first
            tempY_second = tempY == second
            tempY = tempY_first.astype(int) - tempY_second.astype(int)
            to_delete = np.where(tempY==0)
            tempY = np.delete(tempY,to_delete,axis=1)
            tempX = np.delete(tempX,to_delete,axis=0)

            #get classifier using classifier function
            theta = get_classifier_pinv(tempX,np.transpose(tempY.astype(float)))
            #store weight to be returned
            ret.append(theta)
        return ret

    # function to return wrong_predictions,error_rate, and confusion matrix for one versus one classifiers
    def test_one_Vs_one(classifiers, x_given,y_given,start,end):
        #matrix to store all predicted labels from each classifier
        results = []
        #look trough every image in given set
        for row in x_given:
            #matrix to store votes
            v = [0 for i in range(0,10)]
            #index of current classifier
            i = 0
            #iterate through each classifier and store votes accordingly
            for first,second in list(combinations(range(start,end+1),2)):
                if np.matmul(np.transpose(classifiers[i]),row) > 0:
                    v[first] += 1
                else:
                    v[second] += 1
                i += 1
            #get results with most votes(in tie will use the first element)
            results.append(np.argmax(v))
        #initialize confusion matrix
        confusion_test = [[0 for i in range(start, end + 1)] for j in range(start, end + 1)]
        #populate confusion matrix using predicted results
        for idx, result in enumerate(y_given[0]):
            confusion_test[int(result)][int(results[idx])] += 1
        #get error_rate,wrong predictions
        wrong_predictions = np.sum(confusion_test) - np.trace(confusion_test)
        error_rate = wrong_predictions / max(y_given.shape)
        #print(confusion_test)
        return wrong_predictions, error_rate, confusion_test




    # given an L, generate or apply map depending on inputs
    def gen_random_W(L,given_X,w=[],b=[]):
        #generate dimension of d
        d = given_X.shape[1]
        #create w if needed
        if len(w) == 0:
            w = np.random.normal(size=(d, L))
        #apply map and store to return
        ret = np.matmul(given_X,w)
        #generate bias if needed
        if len(b) == 0:
            b = np.random.normal(size=(1, L))
        #apply bias(to every row of current return matrix)
        ret += b
        return ret,w,b


    #function to run problem 2 code and store
    def problem2(training_X,training_Y,testing_X,testing_Y):
        #create one versus all classifiers and test them -> save results
        oneVsAll = oneVsAll_classfiers(0, 9, training_X, training_Y)
        np.save("problem_2_oneVall_test_confusion",test_oneVsAll(0, 9, testing_X, testing_Y, oneVsAll)[2])
        np.save("problem_2_oneVall_train_confusion",test_oneVsAll(0, 9, training_X, training_Y, oneVsAll)[2])

        #create one versus one classifiers and test them -> save results
        oneVSOne = one_Vs_one_classifiers(0, 9, training_X, training_Y)
        np.save("problem_2_oneVone_test_confusion",test_one_Vs_one(oneVSOne, testing_X, testing_Y, 0, 9)[2])
        np.save("problem_2_oneVone_train_confusion",test_one_Vs_one(oneVSOne, training_X, training_Y, 0, 9)[2])

    #code to run problem 2 code
    problem2(trainX,trainY,testX,testY)

    #given maps for features
    def identity(x):
        return x
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    def sinusoidal(x):
        return np.sin(np.radians(x))
    def ReLU(x):
        return np.maximum(0,x)
    #array to store each feature function
    functions = [identity,sigmoid,sinusoidal,ReLU]
    #array of L values to test
    ls = numpy.linspace(100,1200,12).astype(int)
    #errors for types of Ls
    err1 = []
    err2 = []

    #problem3 information
    conf = []
    def problem3(training_X,training_Y,testing_X,testing_Y):
        #will get data for each L
        for L in ls:

            #apply map on current data
            temp_tr_X,W,B = gen_random_W(L,training_X)
            temp_te_X,w,q = gen_random_W(L,testing_X,W,B)
            #store error rates for this L
            temp1 = []
            temp2 = []
            #collect data for each feature function
            for f in functions:
                temp = []
                #apply specific feature
                tr_x = f(temp_tr_X)
                te_x = f(temp_te_X)
                oneVsAll = oneVsAll_classfiers(0, 9, tr_x, training_Y)
                #temp1.append(test_oneVsAll(oneVsAll,te_x,testing_Y,0,9)[1]*100)
                temp.append(test_oneVsAll(0,9,te_x,testing_Y,oneVsAll)[2])
                oneVSOne = one_Vs_one_classifiers(0, 9, tr_x, training_Y)
                #temp2.append(test_one_Vs_one(oneVSOne, te_x, testing_Y, 0, 9)[1]*100)
                temp.append(test_one_Vs_one(oneVSOne, te_x, testing_Y, 0, 9)[2])
                conf.append(temp)
            #append errors
            err1.append(temp1)
            err2.append(temp2)


    #plot a given file as a dot plot
    def plotL(file_name,title_1):
        all_errors = np.load(file_name)
        for idx,function in enumerate(functions):
            errors = all_errors[:,idx]
            print(errors.shape)
            plt.title("L vs Error rates - " + title_1 + " " + function.__name__)
            plt.xlabel("L")
            plt.ylabel("Error Rate(%)")
            plt.plot(ls,errors,"ob")
            plt.show()


    #plotL("OnevAll_erros.npy","One v All Error Rates")
    #("OnevOne_erros.npy", "One v One  Error Rates")

    #plot a matrice
    def plot(file_name,title):
        fig, ax = plt.subplots()
        to_plot = numpy.load(file_name)

        for i in range(0,len(to_plot)):
            for j in range(0,len(to_plot[0])):
                c = to_plot[j, i]
                ax.text(i+.5, j+.5, str(c), va='center', ha='center')
        #plot and correctly add labels
        plt.matshow(to_plot, cmap=plt.cm.Blues)
        #ax.grid()
        print("hap")
        #row is y and col is my x
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.grid()
        ax.set_xlabel("My classifier Values")
        ax.set_ylabel("Given classifier Values")
        ax.set_title(title)
        print(to_plot)
        plt.show()

    #plot("problem_2_oneVall_test_confusion.npy","One Versus All Test Data Confusion Matrix")
    #plot("problem_2_oneVall_train_confusion.npy","One Versus All Train Data Confusion Matrix")
    #plot("problem_2_oneVone_test_confusion.npy","One Versus One Test Data Confusion Matrix")
    #plot("problem_2_oneVone_train_confusion.npy","One Versus One Train Data Confusion Matrix")

    #alternative to plotting -> use excel
    #numpy.savetxt("foo.csv", np.load("conf-L=1000-rle.npy")[1], delimiter=",")

    problem3(trainX, trainY, testX, testY)

    '''
    commands to save data
    numpy.save("conf-L=1000-iden",conf[0])
    numpy.save("conf-L=1000-sig", conf[1])
    numpy.save("conf-L=1000-sin", conf[2])
    numpy.save("conf-L=1000-rle", conf[3])
    '''
    #numpy.save("Ls",ls)
    #numpy.save("OnevAll_errors",err1)
    #numpy.save("OnevOne_errors", err2)

