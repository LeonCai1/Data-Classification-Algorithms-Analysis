# Classification Algorithms analysis

Implementing Decision Tree and compared with other classification algorithms in sklearn library.

## Structure
Main programming files are:
```bash
main.py preprocessing.py data_splitting.py classification.py 
decision_tree.py my_decision_tree.py naive_bayes.py svm.py
```
Input files are
```bash
amazon_cells_labelled.txt imdb_labelled.txt yelp_labelled.txt
```
Extra files:
- rawFtAccuracy.png and reducedFtAccuracy.png saved output bar charts
- HW 3 Report.pdf
output:
 - console ouput: each algorithm's time and accuracy perfomance
 -  bar charts to visualize the output data

## Usage

1. Run the main.py file
2. give the input **amazon_cells_labelled.txt imdb_labelled.txt yelp_labelled.txt** when see the prompt

## Result bar charts
 Using raw feature matrix:
 
 ![Accuracy before using feature selection](rawFtAccuracy.png)
 
 After Applying feature selection algorithm (top K frequent words)
 
 ![Accuracy after using feature selection](reducedFtAccuracy.png)
 Compare the performance on time:
 
 ![Time comparision](classificationTime.png)
 
## License and Authority
Leon Cai (https://github.com/LeonCai1)
