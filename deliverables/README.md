> [!IMPORTANT]
> If any links or files require a password, please use my **UTAS Application Number**.

## Abstract

**Title**: Application of Casper Neural Network in Predicting User Satisfaction with Web Searches Based on Task-Related and User Behavioural Features & Evaluating the Efficiency of Genetic Algorithms in Hyperparameter Tuning and Feature Selection for Casper

**Keywords**: Optimisation, Genetic Algorithm, Artificial Neural Networks, Casper, User Satisfaction Prediction

**Type**: Independent Project

**Duration**: Aug 2023 – Oct 2023

**Course Name**: [COMP4660](https://programsandcourses.anu.edu.au/2023/course/comp4660) – Neural Networks, Deep Learning and Bio-inspired Computing

**Course Outline**:

- Each student in this course was assigned a unique combination of an artificial neural network model or technique and a dataset. Following that, they were required to identify a specific research topic and undertake the research independently. In the next stage, students were required to extend the research by integrating either Deep Learning or Evolutionary Algorithm. Work was assessed by both traditional marking and double-blind peer review. ([reference](https://1drv.ms/f/c/4f49bb445ba8ff14/IgC-wOsVFEZoSpTf3QKQgPQlATR43PxNM1Op5wL73sWP8w8?e=gFK3oo))

**Course Final Mark**: 85 / 100

**Project Weight**: [50%](https://programsandcourses.anu.edu.au/2023/course/COMP4660/Second%20Semester/7520) of COMP4660

**Project Mark**: [88](https://1drv.ms/f/c/4f49bb445ba8ff14/IgC-wOsVFEZoSpTf3QKQgPQlATR43PxNM1Op5wL73sWP8w8?e=gFK3oo) / 100

**Deliverables**: [paper (application)](https://github.com/glowing-sea/genetic-algorithm-for-casper-optimisation/blob/main/deliverables/paper-application.pdf), [paper (genetic algorithm extension)](https://github.com/glowing-sea/genetic-algorithm-for-casper-optimisation/blob/main/deliverables/paper-genetic-algorithm.pdf), code

**Description**:

- This project explores the efficacy of an early Constructive Artificial Neural Network architecture, Casper, for predicting user satisfaction in web searches, while using and evaluating Genetic Algorithms (GAs) for Hyperparameter Tuning and Feature Selection.
- Automatic user satisfaction estimation is significant for web search engine providers, enabling them to receive real-time, instant feedback on their products and overcoming the latency and dishonesty issues inherent in traditional user surveys.
- Although user satisfaction prediction can be achieved with modern Deep Neural Network models, exploring the use of a classical artificial neural network remains important for resource-limited domains.
- The model is trained and evaluated using n-repeat k-fold validation on Kim et al. ’s dataset, with a set of user behavioural (e.g., eye-tracking) and task-related (e.g., different snippet sizes) features.
- The optimisation search space is designed to consist of 4096 hyperparameter combinations and 32768 feature subsets. Grid Search is first performed to pre-calculate the Fitness Table, that is, the mappings between all hyperparameter or feature subset setups (also called Chromosomes in the GA setting) and their MSE (also called Fitness). In this way, GA can be performed offline in just a few seconds.
- Results show that GA is statistically faster, in terms of the number of iterations, than Random Search (RS) if the goal is to obtain any one of the top 10 hyperparameter combinations and feature subsets. However, GA does not statistically outperform RS at finding the global optimum, potentially due to the randomness, complexity, and dynamics of Casper's hyperparameter space.
- The final optimised model achieves a MAE of 1.127 (Std: 0.153) on satisfaction score prediction (scaling from 1-7) and highlights the core features of “time to first click”, “snippet length”, and “time spent”.
- Future work includes training and evaluating Casper on more user satisfaction datasets and extending the GA evaluation to a variety of GA setups and a larger search space, such as using the float chromosome encoding and representing hyperparameter combinations and feature subsets in the same space.