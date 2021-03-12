# Metric-Evaluator
 
A python module to evaluate some custom metrics on some custom visual explanation methods

The main classes of interest are: 

### Metric

a class to manage a general type of metric to be evaluated

### MetricOnAllDataset

a class that inherits from Metric, built to manage the general type of metric evaluated on all the examples from a dataset 

### MetricOnSingleExample

a class, that inherits from Metric, built to manage the general type of metric evaluated on each single example.

These are all abstract classes. One can implement their own specific classes (like AverageDrop, Deletion, and the others I implemented as an example) to build their own metrics. Each one of them must inherit from one of either MetricOnAllDataset or MetricOnSingleExample and implement the method update() and final_step(). These are the methods used from the algorithm to evaluate each metric.

Then the central class is: 

### MetricEvaluator

which is a class containing all the suite to manage the flow to evaluate each metric on the data provided. The method that gets the job done is evaluate_metrics(), containing all the instructions for its purpose.

### Usage examples
# Few parameters are required:
1) "-rp" to specify the path to save the results in, str
2) "-cid" to specify the chunk id, int
3) "-cdim" to specify how many images each chunk is taking into account, int
4) "-cnn" the backbone to be used, str
5) "-m" the metrics to evaluate, list of str (an undescore has to be put between each pair of words: "average_increase")
6) "-am" the attribution methods, list of str

# Examples:
1) python MAIN_TO_RUN.py -rp "path/of/life" -cid 0 -cdim 150 -cnn "resnet50" -m average_drop average_increase average_coherency -am integratedgradients saliency
2) python MAIN_TO_RUN.py -rp "path/of/life" -cid 2 -cdim 3 -cnn "vgg16" -m average_complexity average_coherency -am ScoreCAM
3) python MAIN_TO_RUN.py -rp "path/of/life" -cid 4 -cdim 1000 -cnn "resnet18" -m average_complexity average_score_variance deletion insertion -am gradcam GradCAM++ integratedgradients saliency

# List of backbones implemented:
1) resnet18
2) resnet50
3) vgg16

# List of attribution methods implemented
1) GradCAM
2) GradCAM++
3) SmoothGradCAM++
4) ScoreCAM
5) IntegratedGradients
6) Saliency
7) Occlusion
8) FakeCAM

# List of metrics implemented
1) Average drop
2) Average increase/increase in confidence
3) Deletion
4) Insertion
5) Average complexity
6) Average coherency
7) Average score variance
8) Elapsed time

#### Still under review
