# Metrics Evaluator
 A python module to evaluate some custom metrics on some custom visual explanation methods
 
 The main classes of interest are: 
  Metric: a class to manage a general type of metric to evaluate
  MetricOnAllDataset: a class that inherits from Metric, built to manage the general type of metric evaluated on all the examples from a dataset
  MetricOnSingleExample: a class, that inherits from Metric, built to manage the general type of metric evaluated on each single example
 These are all abstract classes. One can implement their own specific classes (like AverageDrop, Deletion, and the others I implemented as an example) to build their own metrics. Each one of them must inherit from one of either MetricOnAllDataset or MetricOnSingleExample and implement the method update() and final_step(). 
 These are the methods used from the algorithm to evaluate each metric.
 
 Then the central class is: MetricEvaluator, which is a class containing all the suits to manage the flow to evaluate each metric on the data provided. The method that gets the job done is evaluate_metrics(), containing all the instructions for its purpose.

