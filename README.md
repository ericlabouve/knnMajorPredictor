# CPE466 Final Project

## Summary
To use supervised and unsupervised methods in order to classify senior projects. The supervised algorithm is k nearest neighbors and euclidean distance used as the distance method. The algorithm works by calculating the distance between a single datapoint against all other points in the data set, comparing the closest "k" points to our datapoint, and then assigning a class to the datapoint according to the majority class represented by the "k" closest datapoints. The unsupervised algorithm is hierarchical agglomerative clustering with single link distance used as the distance method. The hierarchical clustering algorithm works by making each datapoint in the dataset an individual cluster, then combining clusters that are closest to one another based on the distance function provided. You repreat this process untill all points are contained under one large superset.

## Input Data
Senior projects were taken from the following source: http://digitalcommons.calpoly.edu/seniorprojects/ 
Senior projects were first downloaded as pdfs and converted to text files by hand.
There are nine different engineering disciplines used for our data: 
Aerospace
Computer
Materials
Architectural
Electrical
Mechanical
Bioresource and Agricultural
Industrial and Manufacturing
Liberal Arts

## Resulting Data and Analysis
The systems were evaluated using cross validation since we only had 90 data points.

The following table is our resulting data for our KNN system:

              Precision   Recall    F-Score
Electrical    0.53        0.80      0.64\n
Aerospace     0.50        0.20      0.29
Computer      0.14        0.30      0.19
Mechanical    0.66        0.40      0.50
Bio and Agr   0.63        0.50      0.55
Materials     1.0         0.1       0.18
Liberal Arts  1.0         0.1       0.18
Architectural 0.26       0.60      0.36
Average       0.59        0.38      0.46

With a precision close to 0.60 and a recall of close to 0.40, the system is not reliable but performs significantly better than picking a category at random. The system could be improved if the text vectors were normalized or a more accurate distance function was used such as Okapi25. 

No analysis was conducted on the agglomerative clustering algorithm because as the algorithm ran, we noticed that most of the data points were collected underneath one large cluster. In short, the algorithm was not able to classify any of the majors correctly to any degree of accuracy. This might be due to the nature of our data. Our data is not hierarchical so running our agglomerative clustering was expected to fail.

Our findings are useful because it can also be used to identify papers that individuals might find interesting to read. It is commonly the case that individuals are interested in their accademic discipline. The system can be ran on news outlets such as the NYT and scrape articles that the reader might be interested in. This fixes the issue of sifting through articles that the reader might not be interested in reading.

## Work Log
Eric LaBouve:
- Collected test and training data for aerospace and architectural entingeering.
- Wrote the following functions inside generateVectors.py: getTfVectors, generateTrainingVectors, generateTestingVectors, and generateVectors.
- Wrote the following functions inside knn.py: getIntersection, euclideanDistance, and computeKnn.
- Wrote the cross validation code inside main.py
- Made slides 1 - 7 on the presentation.

Jason Chin:
- Collected test and training data for liberal arts, materials, and mechanical engineering.
- Implemented the Matrix class and all methods inside agglomerative.py.
- Refactored the design of the code: separated the main function into its own file, moved functions around to make more logical sense.

Minjie Fang:
- Collected test and training data for bioresource & agricultural, computer, electrical, and industrial & manufacturing engineering.
- Made slides 8 - 9 on the presentation
