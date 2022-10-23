from decimal import ROUND_HALF_DOWN
import preprocessing as prep
from data_splitting import data_splitting
from decision_tree import DecisionTree
from nltk.tokenize import word_tokenize
from classification import Classifier
import my_decision_tree as dt
from naive_bayes import NaiveBayes
from svm import SVM
import matplotlib.pyplot as plt
import numpy as np
import time

def main():
    #preProcessing the data
    userInput = input('Give 3 files: ').split()
    data = prep.preProcessing
    data.merge(userInput)
    K = 1500
    rows, cols = (len(data.line),len(data.l))
    words_list = data.l
    #D =data.init_matrix(data.line, data.l)

    # split the data into training, validation, testing
    raw = data_splitting(data.line, data.y)
    X_train, X_val, X_test, y_train, y_val, y_test = raw.data_splitting()

  
    feature = data.init_matrix(X_train, words_list)
    
    # feature selection
    # choose top k words
    top_k_words = data.top_k_words(words_list, K)
    langs = ['DT', 'DT(implemented)', 'MultinomialNB', 'SVM']
    rawFeatureTime = []
    rawFeatureTrainAccuracy =[]
    rawFeatureValidAccuracy =[]
    rawFeatureTestAccuracy =[]
    reducedFeatureTime = []
    reducedFeatureTrainAccuracy =[]
    reducedFeatureValidAccuracy =[]
    reducedFeatureTestAccuracy =[]
    print('---------------------------Decision Tree---------------------------')   
    #apply classification
    start_time = time.time()
    classifier = Classifier(DecisionTree(2, 11))
    classifier.train(feature, y_train)
    train_accuracy = classifier.getAccuracy(feature, y_train)
    val_accuracy = classifier.getAccuracy(data.init_matrix(X_val,  words_list), y_val)
    test_accuracy = classifier.getAccuracy(data.init_matrix(X_test, words_list), y_test)
    print(f'The Accuracy of Training data is {train_accuracy}')
    print(f'The Accuracy of Validation data is {val_accuracy}')
    print(f'The Accuracy of Testing data is {test_accuracy}')
    print("sklearn decision tree time--- %s seconds ---" % (time.time() - start_time))
    #add data
    rawFeatureTime.append(time.time() - start_time)
    rawFeatureTrainAccuracy.append(train_accuracy)
    rawFeatureValidAccuracy.append(val_accuracy)
    rawFeatureTestAccuracy.append(test_accuracy)

    for i, row in enumerate(feature):
        row.append(y_train[i])
    start_time = time.time()
    root =dt.build_tree(feature, 2, 11)
    train_accuracy = dt.getAccuracy(root, data.init_matrix(X_train,  words_list), y_train)
    val_accuracy = dt.getAccuracy(root, data.init_matrix(X_val,  words_list), y_val)
    test_accuracy = dt.getAccuracy(root, data.init_matrix(X_test, words_list), y_test)
    print(f'The Accuracy of Training data is {train_accuracy}')
    print(f'The Accuracy of Validation data is {val_accuracy}')
    print(f'The Accuracy of Testing data is {test_accuracy}')
    print("My decision tree time--- %s seconds ---" % (time.time() - start_time))
    rawFeatureTime.append(time.time() - start_time)
    rawFeatureTrainAccuracy.append(train_accuracy)
    rawFeatureValidAccuracy.append(val_accuracy)
    rawFeatureTestAccuracy.append(test_accuracy)

    print('---------------------------Naive Bayes---------------------------')
    feature = data.init_matrix(X_train, words_list)
    start_time = time.time()
    classifier = Classifier(NaiveBayes())
    classifier.train(feature, y_train)
    train_accuracy = classifier.getAccuracy(feature, y_train)
    val_accuracy = classifier.getAccuracy(data.init_matrix(X_val,  words_list), y_val)
    test_accuracy = classifier.getAccuracy(data.init_matrix(X_test, words_list), y_test)
    print(f'The Accuracy of Training data is {train_accuracy}')
    print(f'The Accuracy of Validation data is {val_accuracy}')
    print(f'The Accuracy of Testing data is {test_accuracy}')
    print("sklearn Naive Bayes time--- %s seconds ---" % (time.time() - start_time))
    rawFeatureTime.append(time.time() - start_time)
    rawFeatureTrainAccuracy.append(train_accuracy)
    rawFeatureValidAccuracy.append(val_accuracy)
    rawFeatureTestAccuracy.append(test_accuracy)

    print('---------------------------SVM---------------------------')
    start_time = time.time()
    classifier = Classifier(SVM())
    classifier.train(feature, y_train)
    train_accuracy = classifier.getAccuracy(feature, y_train)
    val_accuracy = classifier.getAccuracy(data.init_matrix(X_val,  words_list), y_val)
    test_accuracy = classifier.getAccuracy(data.init_matrix(X_test, words_list), y_test)
    print(f'The Accuracy of Training data is {train_accuracy}')
    print(f'The Accuracy of Validation data is {val_accuracy}')
    print(f'The Accuracy of Testing data is {test_accuracy}')
    print("sklearn SVM time--- %s seconds ---" % (time.time() - start_time))
    rawFeatureTime.append(time.time() - start_time)
    rawFeatureTrainAccuracy.append(train_accuracy)
    rawFeatureValidAccuracy.append(val_accuracy)
    rawFeatureTestAccuracy.append(test_accuracy)
    
        # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
      
    # Set position of bar on X axis
    br1 = np.arange(len(rawFeatureTrainAccuracy))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    
    # Make the plot
    plt.bar(br1, rawFeatureTrainAccuracy, color ='r', width = barWidth,
            edgecolor ='grey', label ='orgFtAccuracy-Train')
    plt.bar(br2, rawFeatureValidAccuracy, color ='g', width = barWidth,
            edgecolor ='grey', label ='orgFtAccuracy-Valid')
    plt.bar(br3, rawFeatureTestAccuracy, color ='b', width = barWidth,
            edgecolor ='grey', label ='orgFtAccuracy-Test')
    
    # Adding Xticks
    plt.xlabel('Classification Algorithms', fontweight ='bold', fontsize = 15)
    plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(rawFeatureTrainAccuracy))],
            langs)
    
    plt.legend()
    plt.show()

    plt.close()

    words_list = top_k_words
    feature = data.init_matrix(X_train, words_list)
    print('---------------------------Decision Tree (Reduced Feature)---------------------------')   
    #apply classification
    start_time = time.time()
    classifier = Classifier(DecisionTree(2, 11))
    classifier.train(feature, y_train)
    train_accuracy = classifier.getAccuracy(feature, y_train)
    val_accuracy = classifier.getAccuracy(data.init_matrix(X_val,  words_list), y_val)
    test_accuracy = classifier.getAccuracy(data.init_matrix(X_test, words_list), y_test)
    print(f'The Accuracy of Training data is {train_accuracy}')
    print(f'The Accuracy of Validation data is {val_accuracy}')
    print(f'The Accuracy of Testing data is {test_accuracy}')
    print("sklearn decision tree time--- %s seconds ---" % (time.time() - start_time))
    #add data
    reducedFeatureTime.append(time.time() - start_time)
    reducedFeatureTrainAccuracy.append(train_accuracy)
    reducedFeatureValidAccuracy.append(val_accuracy)
    reducedFeatureTestAccuracy.append(test_accuracy)
    for i, row in enumerate(feature):
        row.append(y_train[i])
    start_time = time.time()
    root =dt.build_tree(feature, 2, 11)
    train_accuracy = dt.getAccuracy(root, data.init_matrix(X_train,  words_list), y_train)
    val_accuracy = dt.getAccuracy(root, data.init_matrix(X_val,  words_list), y_val)
    test_accuracy = dt.getAccuracy(root, data.init_matrix(X_test, words_list), y_test)
    print(f'The Accuracy of Training data is {train_accuracy}')
    print(f'The Accuracy of Validation data is {val_accuracy}')
    print(f'The Accuracy of Testing data is {test_accuracy}')
    print("My decision tree time--- %s seconds ---" % (time.time() - start_time))

    reducedFeatureTime.append(time.time() - start_time)
    reducedFeatureTrainAccuracy.append(train_accuracy)
    reducedFeatureValidAccuracy.append(val_accuracy)
    reducedFeatureTestAccuracy.append(test_accuracy)
    print('---------------------------Naive Bayes (Reduced Feature)---------------------------')
    feature = data.init_matrix(X_train, words_list)
    start_time = time.time()
    classifier = Classifier(NaiveBayes())
    classifier.train(feature, y_train)
    train_accuracy = classifier.getAccuracy(feature, y_train)
    val_accuracy = classifier.getAccuracy(data.init_matrix(X_val,  words_list), y_val)
    test_accuracy = classifier.getAccuracy(data.init_matrix(X_test, words_list), y_test)
    print(f'The Accuracy of Training data is {train_accuracy}')
    print(f'The Accuracy of Validation data is {val_accuracy}')
    print(f'The Accuracy of Testing data is {test_accuracy}')
    print("sklearn Naive Bayes time--- %s seconds ---" % (time.time() - start_time))

    reducedFeatureTime.append(time.time() - start_time)
    reducedFeatureTrainAccuracy.append(train_accuracy)
    reducedFeatureValidAccuracy.append(val_accuracy)
    reducedFeatureTestAccuracy.append(test_accuracy)
    print('---------------------------SVM (Reduced Feature)---------------------------')
    start_time = time.time()
    classifier = Classifier(SVM())
    classifier.train(feature, y_train)
    train_accuracy = classifier.getAccuracy(feature, y_train)
    val_accuracy = classifier.getAccuracy(data.init_matrix(X_val,  words_list), y_val)
    test_accuracy = classifier.getAccuracy(data.init_matrix(X_test, words_list), y_test)
    print(f'The Accuracy of Training data is {train_accuracy}')
    print(f'The Accuracy of Validation data is {val_accuracy}')
    print(f'The Accuracy of Testing data is {test_accuracy}')
    print("sklearn SVM time--- %s seconds ---" % (time.time() - start_time))

    reducedFeatureTime.append(time.time() - start_time)
    reducedFeatureTrainAccuracy.append(train_accuracy)
    reducedFeatureValidAccuracy.append(val_accuracy)
    reducedFeatureTestAccuracy.append(test_accuracy)

        # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
      
    # Set position of bar on X axis
    br1 = np.arange(len(reducedFeatureTrainAccuracy))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    
    # Make the plot
    plt.bar(br1, reducedFeatureTrainAccuracy, color ='r', width = barWidth,
            edgecolor ='grey', label ='reducedFtAccuracy-Train')
    plt.bar(br2, reducedFeatureValidAccuracy, color ='g', width = barWidth,
            edgecolor ='grey', label ='reducedFtAccuracy-Valid')
    plt.bar(br3, reducedFeatureTestAccuracy, color ='b', width = barWidth,
            edgecolor ='grey', label ='reducedFtAccuracy-Test')
    
    # Adding Xticks
    plt.xlabel('Classification Algorithms', fontweight ='bold', fontsize = 15)
    plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(rawFeatureTrainAccuracy))],
            langs)
    
    plt.legend()
    plt.show()
    plt.close()
    
    br1 = np.arange(len(rawFeatureTime))
    br2 = [x + barWidth for x in br1]
    
    # Make the plot
    plt.bar(br1, rawFeatureTime, color ='r', width = barWidth,
            edgecolor ='grey', label ='rawFtTime')
    plt.bar(br2, reducedFeatureTime, color ='g', width = barWidth,
            edgecolor ='grey', label ='reducedFtTime')

    
    # Adding Xticks
    plt.xlabel('Classification Algorithms', fontweight ='bold', fontsize = 15)
    plt.ylabel('Time (seconds)', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(rawFeatureTime))],
            langs)
    
    plt.legend()
    plt.show()
    plt.close()
 
if __name__ == '__main__':
    main()

