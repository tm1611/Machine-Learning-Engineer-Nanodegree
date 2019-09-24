# Machine Learning Engineer Nanodegree

## Capstone Proposal

Timo Meiendresch
September 23rd, 2019

## Proposal

### Domain Background

- Background information of the domain:
- Historical information relevant to the project should be included:
- Why this problem in the domain can or should be solved:  
- Academic research:
- Personal motivation for investigating a particular problem in the domain is encouraged:

Time Series Forecasting is a key challenge in business, economics, and many other areas. Having a marginal advantage in increasing forecast accuracy for future development of sales, demand, storage capacities, or many other target quantities may be of great benefit.

Unlike in many other areas of statistical applications, such as image classification or text analysis, deep learning did not take over in the realm of time series forecasting. Traditional methods, in particular the Autoregressive Integrated Moving-average (ARIMA) and exponential smoothing (ES) methods outperformed newly developed, more complex methods. It is not all that long ago that there was a widespread consensus that highly complex methods for time series forecasting were not performing better than traditional ones. Among others, Makridakis et al. (2018) noted that there is only limited scientific evidence that suggests that neural networks for time series forecasting are an essential tool for time series forecasting.

However, advances in deep learning for time series forecasting in the last two years seem to challenge this long lasting notion. In the widely recognized M4 forecasting competition a hybrid model that uses exponential smoothing and a recurrent neural network (RNN) showed the potential of RNN-based methods for time series forecasting by outperforming all other submitted methods (Smyl, 2019, Makridakis et al., 2019).

[from local to global methods]

The M4 competition not only showed the potential of RNN-based methods but in addition emphasized an ongoing paradigm shift in the forecasting community. Traditional methods are applied to individual time series, i.e. one series and one model at a time. These *local* methods often focus on estimating a limited number of parameters within a fixed space of the given model and can therefore be referred to as model-based (Wang et al., 2019).
With the increased availability of large datasets a new type of forecasting problem has emerged. Instead of forecasting a single series independently using a single model, it is frequently necessary to forecast big collections of time series. Examples of these type of data can be found in diverse areas such as web traffic, electricity consumption of individual households, or product demand of online retailers.
Using RNN-based methods enables *cross-series* learning, i.e. one *global* model is trained on the entire dataset of related time series, thus exploiting possible dependencies between series. This approach yields a *global* representation of the entire dataset which hypothetically which can then be used on individual series to improve the forecasting accuracy.

(In the aftermath of the M4 competition)

Various methods have been published that are based on the idea of using a RNN-based method that uses *cross-learning* but only few have been tested on the M4 data yet with exception of the winning method. One of the most prominent one is the *DeepAR* forecasting algorithm (Salinas et al., 2017) that is the primary time series algorithm available in the Amazon Web Services (AWS) SageMaker. It trains a preferably large dataset of related series using a recurrent network model and including previous observations as features ().   


[other global methods]

[M4 as a trigger]

[M4 competition, Paradigm shift]


### Problem Statement

[Describe the problem that is to be solved]  

A problem for practitioners is that those newly developed methods are not yet tested on many datasets for which benchmarks are known and that resemble forecasting in practice. As the M4 competition data were designed for this purpose we can leverage these data and answer some of the many questions regarding the RNN-based algorithms.

The main problem can therefore be summarized as:

1. Are the RNN-based algorithms (DeepAR, DeepFactor, DeepState) appropriate to use in a *real-world* forecasting scenario?

In particular, how do these algorithms perform on the M4 dataset which can be seen as close to a real-world setting as it gets (more on this in the section about the data). Also, are there significant performance differences across *frequencies* (Yearly, Quarterly, Monthly, Weekly, Daily, Hourly) or *domain* (Micro, Industry, Macro, Finance, Demographic, Other).

Other questions regarding this algorithm come from the authors themselves as they state that data should be *related* and the dataset itself be *large* (for example Salinas et al., 2017; Wang et al., 2019). Unfortunately, there are no guidelines or suggestions about what is considered *large* or what is considered to be *related*. Using the M4 data we can check both of these questions to a certain extent, i.e.:

2. Do models that are  trained on larger series of the same frequency perform better? Experimental Design: Using random subsets of N={100, 500, 1000, n_i}
3. Do models that are trained on the same domain perform better compared to models that are trained cross-sectoral keeping N constant.
4. Among the three RNN-based methods (DeepAR, DeepState, DeepFactor): Which one performs better? Also, what is the additional benefit of these three methods compared to the three benchmark methods ARIMA,  exponential smoothing, and the so-called COMB benchmark?

Note that "better" in the context of this project refers to "more accurate" according to the accuracy metrics outlined below.

It should be noted that answering all these questions is well beyond the scope of the project and I will therefore focus on question 1. and 4. Moreover, subsets of the M4 data is used due to time and computational capacity. Spiliotis et al. (2019) argue that a subset of 1,000 series should be sufficient to reach similar conclusions about the data.

[...have at least one relevant potential solution]

One relevant potential solution is answering the question which of these 3 algorithms works best in terms of accuracy on the described data (described in the next section). Moreover, whether these newly developed global methods outperform simple local benchmarks (automatic ARIMA, ETS and Comb). Results for the benchmark methods that consist solely of *local* methods, i.e. are trained on a single series individually are made available by the organizers of the competition.

[Problem quantifiable, measurable, and replicable]

The problem can be quantified by measuring the respective evaluation metrics of the competition (sMAPE, MASE, OWA) which are described below. Using seeds for determining the respective subsets (if a subset is used) and on this public dataset and providing respective code on github makes the project replicable.

### Datasets and Inputs

- Describe the datasets for the Project
- References and citations
- It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

The dataset that was used in the M4 competition (Makridakis et al., 2019) contains 100,000 real-world time series with a frequency-specific lower limit of available observations. Data is subdivided according six domains (Economic, Finance, Demographics, Industry, and Other) and by frequency (yearly, quarterly, monthly, weekly, daily, and hourly).

These data are readily supplied in the GluonTS API and are divided in a train-and test dataset. The test data have a length equal to the frequency-specific forecasting horizon. In the M4 competition the forecasting horizons were given as follows:

- Yearly (frequency) - 6  (forecast horizon)
- Quarterly - 8
- Monthly - 18
- Weekly - 13
- Daily - 14
- Hourly - 48

According Spiliotis et al. (2019) the M4 data is diverse and the closest to what can be perceived as "real-world" among a wide variety of competition data. Moreover, results suggest that a random sample of 1,000 series could be enough to resemble the overall feature space of the entire dataset. Accordingly, due to computational resource limitations, I will restrict myself in this project to a subset of 1,000 per domain or frequency at maximum.

### Solution Statement

- Describe a solution to the problem
- Applicable to the project domain and appropriate for the dataset(s) or input(s) given.
- Describe the solution thoroughly such that it is clear that the solution is *quantifiable*, *measurable*, and *replicable*.

The main problem is to indicate whether aforementioned RNN-based algorithms are appropriate to use in a *real-world* forecasting scenario. As a proxy for *real-world* conditions I will use a subset of the M4 competition.

The project clearly quantifies the performance of these algorithms using described

### Benchmark Model

- Details for a benchmark model or result that relates to the domain, problem statement, and intended solution.
- Existing methods or known information in the domain? compare to solution
- Describe how the benchmark model or result is measurable



### Evaluation Metrics

- Propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model.

### Project Design

- Summarize theoretical workflow for approaching a solution given the problem.
- Discussion for what strategies you may consider employing, what analysis of the data might be required before being used
- Outline your intended workflow of the capstone project.

### References

- Smyl, Slawek. "A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting." *International Journal of Forecasting (2019).*
- Makridakis, Spyros, Evangelos Spiliotis, and Vassilios Assimakopoulos. "Statistical and Machine Learning forecasting methods: Concerns and ways forward." *PloS one 13.3 (2018): e0194889.*
- Makridakis, Spyros, Evangelos Spiliotis, and Vassilios Assimakopoulos. "The M4 competition: 100,000 time series and 61 forecasting methods." *International Journal of Forecasting (2019).*
- Wang, Yuyang, et al. "Deep Factors for Forecasting." *arXiv preprint arXiv:1905.12417 (2019).*
- Salinas, David, Valentin Flunkert, and Jan Gasthaus. "DeepAR: Probabilistic forecasting with autoregressive recurrent networks." *arXiv preprint arXiv:1704.04110 (2017).*
- Spiliotis, Evangelos, et al. "Are forecasting competitions data representative of the reality?." *International Journal of Forecasting (2019).*
