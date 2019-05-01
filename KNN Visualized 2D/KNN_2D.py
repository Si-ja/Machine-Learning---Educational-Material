def KNN_2D_plot(data, targets, test_size = 0.33, random_state = 42, knn_limit = True, step_size_mesh = 0.02):
    """
    Created by: Si_ja
    https://github.com/Si-ja
    Date: 2019-05-01
    Last build: 2019-05-01
    
    This is a one in all function that takes your data, trains a KNN model and visualizes it. For vizualisation
    it uses both the training and test data that go into the model. However it only provides the accuracy of the test
    set.
    
    What you need to know and give a model to work with:
    
    *data = indicate data with which you are working. It should have only 2 variables. In short: there should be
    at least and also at max 2 columns of data. Number of rows does not matter.
    
    *targets = basically labes for your data. This should be prepared in a numerical fashion preferably.
    
    *test_size = how much of your data will go into the training and into the testing set. By default 33% of data
    is going into the testing set and 66% into the training set.
    
    *random_state = indicate a seed for spliting the data and training it. Good for replicability sake. By default
    the value is set to 42.
    
    *knn_limit = this will allow for the system to experiment with how many k values can be tried for preparing the 
    knn model. By default value is set to True, meaning experimentation will happen only on 13 values:
    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]. If it is changed to False, the conditions will change and the
    experimentation set might decrease or increase. Alternative method uses 10% of datas info to create a list of KNNs.
    #E.g. if there are 150 entries then neighbors 1,3,5...15 are checked;
    #E.g. if there are 1500 entries then neigbors 1,3,5...150 are checked.
    
    *step_size_mesh = in essence will influence how smooth your graph and colors in it will be. By default set to 0.02.
    0.01 Might make a smoother graph, but it will take longer to execute.
    
    Additional notes: the function will save the final graph and the model into your working directory."""
    
    
    #Inserting only the essential function
    #from sklearn import datasets <- commented out as this is ment for you to try on your own data
    from sklearn.neighbors import KNeighborsClassifier    
    from sklearn.model_selection import train_test_split                   
    import matplotlib.pyplot as plt                       
    from matplotlib.colors import ListedColormap          
    import numpy as np                                    
    from joblib import dump
    import sys #<- This is a new one. A basically safety net to not crash anything. If it never triggers, be happy.
               #Essence - it will allow for stoping of executing of a function if it is noticed that data is not prepared
               #in the correct fashion. An error will signify of that.
  
    #Assign correct information into required variables to train the data
    X = data
    y = targets
    test_size = test_size
    random_state = random_state
    targets_amount = len(set(y))
    h = step_size_mesh
    #Experimentation is bound to happen only with 13 alternative KNNs, unless specified otherwise
    n_neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    #Alternative method that uses 10% of datas info to create a list of KNNs.
    #E.g. if there are 150 entries then neighbors 1,3,5...15 are checked
    #E.g. if there are 1500 entries then neigbors 1,3,5...150 are checked
    alt_n_neighbors = np.arange(1, (X.shape[0]/10)+1, 2, dtype = int)      
    light_colors = ["#FFAAAA", "#AAFFAA", "#AAAAFF", "#FFD0AA", "#FFC1FC","#FFC1DC", "#9EFFF5", "#E1FFA0", "#A8FFE1", "#E593FF"]
    bold_colors = ["#FF0000", "#00FF00", "#0000FF", "#FC7000", "#FC00F3", "#FF006F", "#00FFE4", "#AEFF00", "#00FFA9", "#C300FF"]
    
    #1st safety check:
    if len(X) != len(y):
        print("Your data and targets do not match. Data has {} entries and the target {}".format(len(X), len(y)))
        print("The execution will not stop!")
        sys.exit()
    #2nd safety check:
    if X.shape[1] != 2:
        print("You will not be able to visualize a not 2 dimensional space. You are working with {} variables. Please indicate only 2.".format(X.shape[1]))
        print("The execution will not stop!")
        sys.exit()
    #3rd safety check:
    if test_size > 1.0:
        print("You cannot allocate less than 0% into the testing set. Please chance the test_size value.")
        print("The execution will not stop!")
        sys.exit()
    #4th safety checker:
    if targets_amount > 10:
        print("There are too many classes to visualize. This function can only handle up to 10. You have {}".format(targets_amount))
        print("The execution will not stop!")
        sys.exit()
    
    #Split and shuffle the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    
    #Limiter checker on how many knn will have to be assessed. Naturally affecting the time of model creation for bigger
    #sets of data
    if knn_limit == True:
        n_n = n_neighbors
    else:
        n_n = alt_n_neighbors
        
    #Optimizing and preparing the model
    results = {} 
    for n_value in n_n:
        neigh = KNeighborsClassifier(n_neighbors = n_value)
        neigh.fit(X_train, y_train)
        final_score = np.round(100*neigh.score(X_test, y_test),2)
        results[n_value] = final_score
    maximum_val = max(results, key=results.get)
    neigh = KNeighborsClassifier(n_neighbors = maximum_val)
    neigh.fit(X_train, y_train)
    
    #Preparing the colors
    cmap_light = ListedColormap(light_colors[0:targets_amount])
    cmap_bold = ListedColormap(bold_colors[0:targets_amount])
    
    #Preparing the plane
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    #Preparing the grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = neigh.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig = plt.figure() #this will save your image into an internal variable
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("{}-Class classification (K = {}; Accuracy = {}%)".format(targets_amount, maximum_val, results[maximum_val] ))
    plt.show()
    
    dump(neigh, 'NEW_knn.joblib')
    fig.savefig('NEW_KNN_plot.png') 
