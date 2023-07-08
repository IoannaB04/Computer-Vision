import os
import cv2 as cv
import numpy as np

" INPUTS "
train_directory = "imagedb"
test_directory  = "imagedb_test"

print(train_directory)

number_of_clusters  = [3, 6, 25, 50, 100, 200, 500, 800]
number_of_neighbors = [2, 4, 10, 15,  25,  50]


# ----------------------------------------------- WE CODE HERE --------------------------------------------------------
sift = cv.xfeatures2d_SIFT.create()

# printing in file and in console
def custom_print(message_to_print):
    print(message_to_print)
    log_file = (file_name + '.txt')
    with open(log_file, 'a') as of:
        of.write(str(message_to_print) + '\n')

# ------------------- Create Vocabulary -------------------------
def create_vocabulary(train_directory, number_of_clusters):
    """
    I used k-means cluster in order to create the vocabulary.
    The vocabulary may contain different number of clusters according to number_of_clusters parameter.
    """
    print('---- Creating vocabulary ----')

    filename = str(number_of_clusters) + '_vocabulary.npy'

    if os.path.exists(filename):
        print("The vocabulary already exists.\n")
    else:
        # running for the first time
        train_descs = extract_features(train_directory)

        print('Creating clusters with K-Means alorithm ...')
        term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)  # stops when the centers move 0.1
        trainer = cv.BOWKMeansTrainer(number_of_clusters, term_crit, 1, cv.KMEANS_PP_CENTERS)
                                    # number_of_clusters, Termination_critiria, attempts, flags
        vocabulary = trainer.cluster(train_descs.astype(np.float32))

        filename = str(number_of_clusters) + '_vocabulary.npy'
        np.save(filename, vocabulary)
        print("The vocabulary has been created.\n")

def extract_features(train_directory):
    train_folders = os.listdir(train_directory)
    # print(train_folders)  # to make sure the algorith is working as expected

    print('Extracting features...')
    train_descs = np.zeros((0, 128))  # 128 because we use SIFT algorithm for finding keypoints and local descriprtors
    for folder in train_folders:
        folder_path = os.path.join(train_directory, folder)
        files = os.listdir(folder_path)
        for file in files:
            # print(file)
            path = os.path.join(folder_path, file)  # path for every image
            desc = extract_local_features(path)
            if desc is None:  # prevent bugs
                print("\t\tNONE")
                custom_print(str(file) + str(": no descriptor found"))
                continue
            train_descs = np.concatenate((train_descs, desc), axis=0)  # ένωση με την train_descs
        print('\t' + folder + ': DONE')
    return train_descs

def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    return desc[1]

# ------------------- Creating BOVW Model -------------------------
def extract_global_descriptors(type, number_of_clusters, train_directory, test_directory, number_of_neighbors):  # LAB def index

    # BOVW coding
    vocabulary = np.load(str(number_of_clusters) + '_vocabulary.npy')
    descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
    descriptor_extractor.setVocabulary(vocabulary)

    img_paths = []
    bow_descs = np.zeros((0, vocabulary.shape[0]))

    if type == 'training':
        print('---- Extracting Global Descriptors in Training Set----')
        filename_database = str(number_of_clusters) + '_bow_descs.npy'
        if os.path.exists(filename_database):
            print("bow_descs, labels and path have already been created")
        else:
            """
            Creates the BOW, or in other words an array of every global_descriptor of each image in training set
            # a database according to the features (global_descriptors) of each image in training set
            """
            train_folders = os.listdir(train_directory)

            train_labels = np.zeros((0, 1))
            temp_label = np.zeros((1, 1))

            for folder in train_folders:
                folder_path = os.path.join(train_directory, folder)
                files = os.listdir(folder_path)

                for file in files:
                    path = os.path.join(folder_path, file)
                    img_paths.append(path)

                    # getting the global descriptor for each image in training set
                    img = cv.imread(path)
                    kp = sift.detect(img)
                    bow_desc = descriptor_extractor.compute(img, kp)

                    # collecting every global descriptor creating the Bag Of Words
                    bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)

                    train_labels = np.concatenate((train_labels, temp_label), axis=0)
                temp_label[0] = temp_label[0] + 1

                print("\t" + str(folder) + ": DONE")

            print('Train Database (global descriptors) created.')
            filename_database = str(number_of_clusters) + '_bow_descs.npy'
            np.save(filename_database, bow_descs)

            print('Train labels created.')
            filename_labels = 'labels.npy'
            np.save(filename_labels, train_labels)

            print('Created paths for images in training set.\n')
            np.save('paths', img_paths)  # returns path for each image in training set
    elif type == 'testing':
        test_folders = os.listdir(test_directory)
        print('---- Extracting Global Descriptors in Testing Set----')

        filename_database = str(number_of_clusters) + '_bow_descs_test.npy'
        if os.path.exists(filename_database):
            print("bow_descs, labels and path have already been created")
        else:
            print('Creating global descriptors and labels for testing set...')
            test_labels = np.zeros((0, 1))
            temp_label = np.zeros((1, 1))

            for folder in test_folders:
                folder_path = os.path.join(test_directory, folder)
                files = os.listdir(folder_path)
                for file in files:
                    path = os.path.join(folder_path, file)
                    img_paths.append(path)

                    # getting the global descriptor for each image in training set
                    img = cv.imread(path)
                    kp = sift.detect(img)
                    bow_desc = descriptor_extractor.compute(img, kp)

                    # getting every global descriptor --> BOVW
                    bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)

                    test_labels = np.concatenate((test_labels, temp_label), axis=0)
                temp_label[0] = temp_label[0] + 1
                print("\t" + str(folder) + ": DONE")

            print('Test Database (global descriptors) created.')
            filename_database = str(number_of_clusters) + '_bow_descs_test.npy'
            np.save(filename_database, bow_descs)

            print('Test labels created.')
            filename_labels = 'labels_test.npy'
            np.save(filename_labels, test_labels)

            print('Created paths for images in testing set.\n')
            np.save('paths_test', img_paths)  # returns path for each image in testing set
    else:
        print("Typo in the final parameter in extract_global_descriptors.")
        print("\tIt can be 'training' or 'testing' ")
        exit(-1)

# ------------------- Classification in Testing Set -------------------------
" KNN CLASSIFIER"
def euclidean_distance(d1, d2):  # Euclidean Distance = sqrt(sum i to N (x1_i – x2_i)^2)
    distances = []
    for i in range(d2.shape[0]):
        distance = np.sqrt(np.sum((d1 - d2[i]) ** 2))
        distances.append(distance)
    return np.asarray(distances)  # turning list into numpy array

def knn_classifier(number_of_neighbors, number_of_clusters, number_of_classes):
    bow_descs_training = np.load(str(number_of_clusters) + '_bow_descs.npy')
    bow_descs_testing = np.load(str(number_of_clusters) + '_bow_descs_test.npy')

    labels_training = np.load('labels.npy')

    predictions_final = []

    for i in range(bow_descs_testing.shape[0]):
        distances = euclidean_distance(bow_descs_testing[i], bow_descs_training)

        idx = np.argpartition(distances, number_of_neighbors)
        idx = idx[:number_of_neighbors]
        """ np.argpartition returns the indexes of number_of_neighbors smallest values in the given array. 
            It's not guaranteed to be in order from smallest to largest.    """

        predictions = np.zeros(number_of_classes)
        for m in range(len(idx)):
            p = idx[m]
            if labels_training[p] >= 0:
                c = int(labels_training[p])
                predictions[c] += 1
        predictions_final.append(np.argmax(predictions))

    return predictions_final  # returns the class of every global_descriptor, aka of every image in testing set

" SVM "
def svm_create(img_ind, bow_descs, kernel, name_of_class, number_of_clusters, number_of_neighbors):
    filename = str(number_of_clusters) + '_svm_' + str(kernel) + '_' + str(name_of_class)
    if os.path.exists(filename):
        print(str(name_of_class) + " has already been trained with SVM.")
    else:
        svm = cv.ml.SVM_create()
        svm.setType(cv.ml.SVM_C_SVC)
        svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))

        if   kernel == 'RBF':       svm.setKernel(cv.ml.SVM_RBF)
        elif kernel == 'CHI2':      svm.setKernel(cv.ml.SVM_CHI2)
        elif kernel == 'LINEAR':    svm.setKernel(cv.ml.SVM_LINEAR)
        elif kernel == 'INTER':     svm.setKernel(cv.ml.SVM_INTER)
        else:
            print("Typo in the kernel parameter in svm_classifier.")
            print("\tIt can be 'RBF' or 'CHI2' or 'LINEAR' or 'POLY' or 'INTER' or 'SIGMOID' ")
            exit(-1)

        svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, img_ind)
        svm.save(str(number_of_clusters) + '_svm_' + str(kernel) + '_' + str(name_of_class))
        print("\t" + str(name_of_class) + ": Done")

def svm_1vsAll(kernel, train_folders, number_of_clusters, number_of_neighbors):
    svm = cv.ml.SVM_create()
    bow_descs = np.load(str(number_of_clusters) + '_bow_descs.npy')
    paths = np.load('paths.npy')

    " use as many svm classifiers as the classes in which the training set is divided "
    for i in range(len(train_folders)):
        name_of_class = train_folders[i]

        img_ind = np.array([name_of_class in a for a in paths], np.int32)

        svm_create(img_ind, bow_descs, kernel, name_of_class, number_of_clusters, number_of_neighbors)
        svm.load(str(number_of_clusters) + '_svm_' + str(kernel) + '_' + str(name_of_class))

def svm_1vsAll_classifier(bow_desc, test_folders, train_folders, number_of_clusters, number_of_neighbors, kernel):
    " has to be called for each image in testing set"
    svm = cv.ml.SVM_create()

    prediction = []
    for i in range(len(test_folders)):
        svm = svm.load(str(number_of_clusters) + '_svm_' + str(kernel) + '_' + str(train_folders[i]))
        prediction.append(svm.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)[1][0][0])

    return prediction


# ------------------- Testing Classifiers -------------------------
def knn_classifier_test(number_of_neighbors, number_of_clusters, number_of_classes, train_folders, test_folders):
    labels_test = np.load('labels_test.npy')  # used to test the accuracy
    predictions = knn_classifier(number_of_neighbors, number_of_clusters, number_of_classes)

    # get the size of every folder in test directory
    test_folder_size = []
    for folder in test_folders:
        folder_path = os.path.join(test_directory, folder)
        files = os.listdir(folder_path)
        test_folder_size.append(len(files))

    accuracy = []

    failure_names = []
    failure_names_index = []
    paths = np.load('paths_test.npy') # used to find the failed images
    for i in range(labels_test.shape[0]):
        _, a = os.path.split(paths[i])
        if predictions[i] == labels_test[i]:
            accuracy.append(1)
        else:
            accuracy.append(0)
            _, failed_file_name = os.path.split(paths[i])
            failure_names.append(failed_file_name)
            failure_names_index.append(i)


    # Print results
    custom_print("Number of neighbors: " + str(number_of_neighbors))
    accuracy_tot = sum(accuracy) / labels_test.shape[0] * 100
    custom_print("Accuracy: " + str("%.2f" % accuracy_tot) + "%")
    first = 0
    for x in range(len(test_folder_size)):
        last = first + test_folder_size[x]
        temp = accuracy[first : last]
        accuracy_in_folder = sum(temp) / labels_test.shape[0] * 100
        first = last

        custom_print("\t" + str(test_folders[x]) + "\t" + str("%.2f" % accuracy_in_folder) + "%")

    custom_print("Failed attempts: " + str(labels_test.shape[0] - sum(accuracy)))
    for x in range(len(failure_names)):
        index = failure_names_index[x]
        training_class = train_folders[predictions[index]]
        test_class = test_folders[int(labels_test[ index ])]
        custom_print("\t" + str(failure_names[x]) + "\tAlgorithm: " + str(training_class) + "\t\tTrue: " + str(test_class))

    return accuracy

def svm_1vsAll_classifier_test(test_directory, train_folders, number_of_clusters, number_of_neighbors, kernel):
    vocabulary = np.load(str(number_of_clusters) + '_vocabulary.npy')
    descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
    descriptor_extractor.setVocabulary(vocabulary)

    test_folders = os.listdir(test_directory)
    class_folder = 0

    accuracy = []
    failure = 0
    failure_names = []
    failure_predicted = []
    failure_test = []
    thesi = 0
    for folder in test_folders:
        folder_path = os.path.join(test_directory, folder)
        files = os.listdir(folder_path)

        success = 0
        for file in files:
            path = os.path.join(folder_path, file)

            # getting the global descriptor for each image in testing set
            img = cv.imread(path)
            kp = sift.detect(img)
            bow_desc = descriptor_extractor.compute(img, kp)

            # classifying with svm 1 vs all
            result = np.asarray(svm_1vsAll_classifier(bow_desc, test_folders, train_folders, number_of_clusters, number_of_neighbors, kernel))

            class_predicted = np.argmin(result)
            # testing the outcome of my algorithm
            if class_folder == class_predicted:
                success += 1
            else:
                failure += 1
                failure_names.append(file)
                failure_predicted.append(class_predicted)
                failure_test.append(class_folder)
            thesi += 1

        class_folder += 1
        accuracy.append(success)


    accuracy_tot = sum(accuracy) / (sum(accuracy) + failure) * 100
    custom_print("Accuracy: " + str("%.2f" % accuracy_tot) + "%")
    for i in range(len(accuracy)):
        accurasy_class = accuracy[i] / (sum(accuracy) + failure) * 100
        custom_print("\t" + str(train_folders[i]) + ": \t" + str( "%.2f" % accurasy_class) + "%")


    custom_print("Failed attempts:" + str(failure))
    for x in range(len(failure_names)):
        training_class = train_folders[failure_predicted[x]]
        test_class = test_folders[int(failure_test[x])]
        custom_print("\t" + str(failure_names[x]) + "\tAlgorithm: " + str(training_class) + "\t\tTrue: " + str(test_class))

    return accuracy_tot

# ------------------- main -------------------------
def my_code(number_of_clusters, number_of_neighbors, kernel, train_directory, test_directory):
    train_folders_outer = os.listdir(train_directory)
    test_folders_outer = os.listdir(test_directory)

    " BOVW Creation "
    create_vocabulary(train_directory, number_of_clusters)
    extract_global_descriptors('training', number_of_clusters, train_directory, test_directory, number_of_neighbors)
    extract_global_descriptors('testing', number_of_clusters, train_directory, test_directory, number_of_neighbors)

    " KNN Classifier "
    custom_print('Vocabulary size: ' + str(number_of_clusters))
    custom_print('\n---- Testing KNN Classification: ----')
    for n in range(len(number_of_neighbors)):
        knn_classifier(number_of_neighbors[n], number_of_clusters, len(train_folders_outer))
        knn_classifier_test(number_of_neighbors[n], number_of_clusters, len(train_folders_outer), train_folders_outer, test_folders_outer)
        custom_print('\n')

    custom_print("\n")

    " SVM "
    custom_print('---- Testing SVM 1 VS ALL Classification: ----')
    for k in range(len(kernels)):
        custom_print("KERNEL: " + str(kernel[k]))
        svm_1vsAll(kernel[k], train_folders_outer, number_of_clusters, number_of_neighbors)
        svm_1vsAll_classifier_test(test_directory, train_folders_outer, number_of_clusters, number_of_neighbors, kernel[k])
        custom_print("\n")

kernels = ['RBF', 'CHI2', 'LINEAR', 'INTER']

for c in range(len(number_of_clusters)):

    file_name = str('console_' + str(number_of_clusters[c]))
    f = open((file_name + '.txt'), 'w')

    my_code(number_of_clusters[c], number_of_neighbors, kernels, train_directory, test_directory)

    f.close()
    print("***********************************************************************************************************")
    print("\t\t\t\t\t\t\tSaved outputs in txt file: " + file_name)
    print("***********************************************************************************************************\n\n")


# Choose
# clusters =
# neighbors =
# kernels =
# file_name = str('console_' + str(clusters))
# my_code(clusters, neighbors, kernels, train_directory, test_directory)

print("\n-------------------------------------------------------------------------------------------------------------")
