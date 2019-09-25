# Machine Learning Engineer Nanodegree

## Capstone Proposal

Timo Meiendresch
September 23rd, 2019

## Proposal

### Domain Background

- Background information on the domain
- Historical information relevant to the project should be included
- Why this problem in the domain can or should be solved
- Academic research

Time Series Forecasting is a key challenge in business, economics, and many other areas. Having an advantage by increasing forecast accuracy for future development of sales, demand, storage capacities, or many other target quantities is of great benefit.

Unlike in many other areas of statistical applications, such as image classification or text analysis, deep learning did not take over in the area of time series forecasting. Traditional methods, in particular the *autoregressive integrated moving-average (ARIMA)* and *exponential smoothing (ES)* methods outperformed newly developed, and often more complex, methods. It is not that long ago that researchers perceived that highly complex methods for time series forecasting were not performing better than traditional ones (e.g. Hyndman, 2019). Among others, Makridakis et al. (2018) noted that there is only limited scientific evidence that suggests that neural networks for time series forecasting are an essential tool for time series forecasting.

Nevertheless, advances in deep learning for time series forecasting in the last two years seem to challenge this notion. In the widely recognized M4 forecasting competition a hybrid model based on a recurrent neural network (RNN) outperformed all other submitted methods and, thus, showed the potential of RNN-based methods in this area (Smyl, 2019, Makridakis et al., 2019).

[from local to global methods]

Moreover, the M4 competition emphasized an ongoing paradigm shift in the forecasting community. Traditionally, methods are applied to individual time series, i.e. one series and one model at a time. These *local* methods often focus on estimating a limited number of parameters within a fixed space of the given model and are referred to as model-based (Wang et al., 2019). The focus here lies on estimating an individual series independently of other available data or features.
With the increased availability of large datasets a new type of forecasting problem has emerged. Instead of forecasting a single series independently, it is frequently necessary to forecast big collections of related time series. Examples of these type of data can be found in diverse areas, such as web traffic, household electricity consumption, or product demand of online retailers.

RNN-based methods enable the use of *cross-series learning*, i.e. one *global* model is trained on the entire dataset of related time series, thus using possible dependencies between series. This approach yields a *global* representation of the entire dataset which may improve forecasting accuracy depending on size and relatedness of the series in the dataset.  

(Actual assumptions regarding size and relatedness are not yet well established)
(In the aftermath of the M4 competition)

In the aftermath of the M4 competition, various RNN-based methods have been published that are based on the idea of *cross-series learning* and/or combining local approaches with recurrent networks.  

using a RNN-based method to benefit from *cross-series learning* but only few have been tested on the M4 data yet. The with exception of the winning method. One of the most prominent one is the *DeepAR* forecasting algorithm (Salinas et al., 2017) that is the primary time series algorithm available in the Amazon Web Services (AWS) SageMaker. It trains a preferably large dataset of related series using a recurrent network model and including previous observations as features ().   

To my knowledge,  results of these methods applied to the M4 dataset are not published yet. Moreover, these methods are published using vague assumptions of *large dataset* and *relatedness* of series within the dataset. The goal of this project is to study some of these aforementioned aspects.

[other global methods]
[M4 as a trigger]
[M4 competition, Paradigm shift]

[Personal motivation for investigating a particular problem in the domain is encouraged]

My personal motivation for this project is my background in Statistics & Econometrics, in particular time series analysis. During my studies I covered the traditional methods, such as ARIMA, exponential smoothing methods and various other exotic models. The superiority of local methods was never questioned by the lecturer and machine learning models for time series forecasting was not covered at all. Given my interest and background, I followed the developments the M4 competition closely and take this project as a chance to keep up with current research, recently developed methods, as well as new frameworks for time series forecasting (i.e. GluonTS).

### Problem Statement

[Describe the problem that is to be solved]  

A key problem is that recently developed algorithms are not yet tested on a wide variety of datasets. Whether these methods are applicable in a certain environment and if they add value to previous methods is subject to speculation. The M4 has delivered a vast amount of benchmarks for a dataset that can be separated by frequency and domain and results of the three RNN-based algorithms should serve a useful purpose in practice.

Moreover, the M4 competition data were designed for the purpose of resembling "real-world" forecasting practice. Hence, we can we can leverage these data and answer some of the many questions regarding the RNN-based algorithms.

The main problem can be summarized as:

1. Are the RNN-based algorithms (DeepAR, DeepFactor, DeepState) appropriate to use in a *real-world* forecasting scenario? Results  on M4 data can be used to compare their performance with available benchmarks.

In particular, how do these algorithms perform on the M4 dataset which can be seen as close to a real-world setting as it gets (more on this in the section about the data). Also, are there significant performance differences across *frequencies* (Yearly, Quarterly, Monthly, Weekly, Daily, Hourly) or *domain* (Micro, Industry, Macro, Finance, Demographic, Other).

Other questions regarding this algorithm come from the authors themselves as they state that data should be *related* and the dataset itself be *large* (for example Salinas et al., 2017; Wang et al., 2019). Unfortunately, there are no guidelines or suggestions about what is considered *large* or what is considered to be *related*. Using the M4 data can help to answer these questions:

2. Do models that are  trained on larger series of the same frequency perform better? Experimental Design: Using random subsets of N={100, 500, 1000, n_i}
3. Do models that are trained on the same domain perform better compared to models that are trained cross-sectoral keeping N constant.
4. Among the three RNN-based methods (DeepAR, DeepState, DeepFactor): Which one performs better? Also, what is the additional benefit of these three methods compared to the three benchmark methods ARIMA,  exponential smoothing, and the so-called Comb benchmark?

Note that "better" in the context of this project refers to "more accurate" according to the accuracy metrics outlined in the competition.

Also, please note that answering all these questions is well beyond the extent of such a project. I will therefore focus on some of the outlined aspects, in particular question 1. and 4.

Spiliotis et al. (2019) argue that a subset of 1,000 series should be sufficient to reach similar conclusions about the data. Hence, I will use smaller subsets of the M4 data to alleviate time and computational capacity restrictions.  

[...have at least one relevant potential solution]

One relevant solution is to answer which of the algorithms of interest performs better on the data in terms of accuracy metrics and the specific characteristics of the data (more on the data in the next section).

Moreover, I will compare the three algorithms to simple local benchmarks (automatic ARIMA, ETS and Comb) directly. Results for the *local* benchmark methods (i.e. methods are independently trained on one series) have been published by organizers.

[Problem quantifiable, measurable, and replicable]

Performance will be measured by comparing estimated forecasts with the realized observations (ground truth) using evaluation metrics of the competition. These metrics will be described in a later section.

All code and datasets will be provided on github making this project fully replicable.


### Datasets and Inputs

- Describe the datasets for the Project
- References and citations
- It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

The dataset that was used in the M4 competition (Makridakis et al., 2019) contains 100,000 real-world time series with a frequency-specific lower limit of available observations. Also, the data is divided according to six domains (Economic, Finance, Demographics, Industry, and Other), as well as by frequency (yearly, quarterly, monthly, weekly, daily, and hourly).

These data are readily supplied in the GluonTS API and are divided in a train-and test dataset. The test data have a length equal to the frequency-specific forecasting horizon. In the M4 competition the forecasting horizons were given as follows:

- Yearly (frequency) - 6  (forecast horizon)
- Quarterly - 8
- Monthly - 18
- Weekly - 13
- Daily - 14
- Hourly - 48

According to Spiliotis et al. (2019) the M4 data is diverse and the closest to what can be perceived as "real-world" among a wide variety of competition datasets. Moreover, results suggest that random samples of 1,000 series could be enough to resemble the overall feature space of the entire dataset. Accordingly, due to computational resource limitations, I will restrict myself in this project to a subset of 1,000 series per domain or frequency at maximum.


### Solution Statement

- Describe a solution to the problem
- Applicable to the project domain and appropriate for the dataset(s) or input(s) given.
- Describe the solution thoroughly such that it is clear that the solution is *quantifiable*, *measurable*, and *replicable*.

The main problem is to indicate whether aforementioned RNN-based algorithms are appropriate to use in a *real-world* forecasting scenario. As a proxy for *real-world* conditions I will use a subset of the M4 competition.

The project clearly quantifies the performances of each algorithm using the following evaluation metrics:

- Symmetric mean absolute percentage error (sMAPE)
- Mean absolute scaled error (MASE)
- Overall weighted average (OWA)

Therefore, the goal of the project is to clearly quantify the performance of the metrics using the appropriate measures of the competition. Replicability will be secured by making the code available using github.

A solution includes primarily basic takeaways and results on the accuracy of the RNN-based models applied to the M4 data. Additionally, the effects of size, relatedness, or frequency are under investigation if applicable.


### Benchmark Model

- Details for a benchmark model or result that relates to the domain, problem statement, and intended solution.
- Existing methods or known information in the domain? compare to solution
- Describe how the benchmark model or result is measurable

The M4 competition employed eight statistical benchmarks with three variations of naive forecasting methods and three exponential smoothing methods. Moreover, the winner of the M3 competition, namely theta method, and a combination method called Comb was included. Comb, the arithmetic average of simple exponential smoothing, Holt method, and damped Holt method was used to express relative performance of the respective methods.

In this project I will use the results for Comb, Auto ARIMA (automatic framework for ARIMA methods), and ETS (automatic framework for exponential smoothing methods) as benchmarks for the RNN-based algorithms. The relative performance will be measured against the Comb benchmarks similar to what the competition did.


### Evaluation Metrics

- Propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model.

Both, train/test data are available in the project. The train data is used during the training of the model and its performance is tested on the test data that will be held out until the inference stage. To compare our algorithms with the results I will use the same evaluation metrics that were used during the competition.

Ranks in the competition were determined by the overall weighted average (OWA) which is a composite measure of the symmetric mean absolute percentage error (sMAPE) and mean absolute scaled error (MASE). The formulas are described in Makridakis et al. (2019).


### Project Design

- Summarize theoretical workflow for approaching a solution given the problem.
- Discussion for what strategies you may consider employing, what analysis of the data might be required before being used
- Outline your intended workflow of the capstone project.

The workflow will be focused around the interaction between the GluonTS API in Python and using the computational power of GPU-powered AWS instances. Gluon Time Series (GluonTS) toolkit for probabilistic time series modeling is based on the Apache MXNet deep learning framework. This project requires to extend the knowledge on model deployment using PyTorch and SageMaker to working with MXNet on an AWS instance to train the models on a GPU instance.

The algorithms are built-in algorithms of GluonTS or AWS SageMaker. Data preprocessing will be a crucial part of the project as these algorithms require the data in a specific format. Based on Spiliotis et al. (2019) and computational complexity of these models I will use subsets of the data with an upper limit of 1,000 time series per experiment. Therefore, I need to preprocess the data to randomly choose 1,000 series. Moreover, the data needs to be divided by frequency and domain to research

Additionally, calculating the respective benchmarks requires additional coding using a set of modules.

- Step 1 (Data processing) - Preprocess the data in aforementioned way
- Step 2 (Data modeling) - Use GluonTS and AWS SageMaker to train the models on the subsets of the M4 data and get forecast estimations
- Step 3 (Inference) - Evaluate the forecasts using the outlined accuracy measures.


### References

- Hyndman, Rob J. "A brief history of forecasting competitions." *International Journal of Forecasting (2019).*
- Salinas, David, Valentin Flunkert, and Jan Gasthaus. "DeepAR: Probabilistic forecasting with autoregressive recurrent networks." *arXiv preprint arXiv:1704.04110 (2017).*
- Smyl, Slawek. "A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting." *International Journal of Forecasting (2019).*
- Spiliotis, Evangelos, et al. "Are forecasting competitions data representative of the reality?." *International Journal of Forecasting (2019).*
- Makridakis, Spyros, Evangelos Spiliotis, and Vassilios Assimakopoulos. "Statistical and Machine Learning forecasting methods: Concerns and ways forward." *PloS one 13.3 (2018): e0194889.*
- Makridakis, Spyros, Evangelos Spiliotis, and Vassilios Assimakopoulos. "The M4 competition: 100,000 time series and 61 forecasting methods." *International Journal of Forecasting (2019).*
- Wang, Yuyang, et al. "Deep Factors for Forecasting." *arXiv preprint arXiv:1905.12417 (2019).*
