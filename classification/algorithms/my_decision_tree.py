
def build_tree(x,  min_sample_split, max_depth):
    root = best_split(x)
    split(root, min_sample_split, max_depth, 1)
    return root
'''
Find the best splitting attribute
'''  
def best_split(x):
    class_values = list(set(row[-1] for row in x))
    best_index, best_value, best_score, best_groups = 999, 999, 999, None
    for i in range(len(x[0])-1):
        #compute possible splitting values
        means = get_means(x, i)
        for mean in means:
            # split the col list to in to left and right
            groups = test_split(i, mean, x)
            # compute gini index
            gini = gini_index(groups, class_values)
            if gini < best_score:
                 best_index, best_value, best_score, best_groups= i, mean, gini, groups
    return {'index': best_index, 'value': best_value,  'groups': best_groups}
'''
recursively find the decision node
'''             
def split(root, min_sample_split, maxDepth, depth):
    left, right = root['groups']
    del(root['groups'])
    if not left or not right:
        root['left'] = root['right'] = terminate(left + right)
        return
    if depth >= maxDepth:
        root['left'], root['right'] = terminate(left), terminate(right)
        return
    if len(left)<= min_sample_split:
        root['left'] = terminate(left)
    else:
        root['left'] = best_split(left)
        split(root['left'], min_sample_split, maxDepth, depth+1)
        
    if len(right) <= min_sample_split:
        root['right'] = terminate(right)
    else:
        root['right'] = best_split(right)  
        split(root['right'], min_sample_split, maxDepth, depth+1)     
'''
compute set of means
'''    
def get_means(m, col):
        l = []
        # get all val of the current attributes
        for r in m:
            l.append(r[col])
        l = list(set(l))
        l.sort
        means = []
        # find mean of every two consective element in the sorted array
        for i in range(len(l)-1):
            means.append((l[i]+l[i+1])/(2.0))
        return means
    
'''
separate the data to left and right
based on the splitting value
'''          
def test_split(col, val, x):
      
    left, right = list(), list()
    for r in x:
        # append the classes to the column
        if r[col] <= val:
             left.append(r)
        else:
            right.append(r)
    return left, right
'''
compute gini index
'''   
def gini_index(groups, classes):
    n = len(classes)
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size  == 0:
            continue
        score =0.0
        for val in classes:
            p = [row[-1] for row in group].count(val) / size
            score += p * p
        gini += (1.0 - score) * (size / n)
    return gini

def terminate(group):
    res = [row[-1] for row in group]
    return max(set(res), key=res.count)

def predict(root, row):
	if row[root['index']] <= root['value']:
		if isinstance(root['left'], dict):
			return predict(root['left'], row)
		else:
			return root['left']
	else:
		if isinstance(root['right'], dict):
			return predict(root['right'], row)
		else:
			return root['right']

def getAccuracy(root, x, y):
    correct =0
    for i, yi in enumerate(y):
        y_predicted = predict(root,x[i])
        if(y_predicted == yi):
            correct+=1
    return (correct+0.0) / len(y)   