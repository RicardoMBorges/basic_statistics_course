
# Basic Statistics Course – Interactive App

An interactive **Streamlit application** designed to teach the fundamental concepts of statistics through visualization, simulation, and experimentation.

This tool allows students to explore statistical ideas such as:

* descriptive statistics
* probability distributions
* hypothesis testing
* ANOVA
* regression
* analytical measurement error

The app was designed for **teaching statistics to students in chemistry, biology, and analytical sciences**.

---

# Overview

The application provides interactive simulations and visualizations that help students understand statistical concepts intuitively.

Instead of learning statistics only from formulas, students can **manipulate parameters and observe how distributions and results change in real time**.

The app includes modules covering:

* Data exploration
* Distributions
* Hypothesis testing
* ANOVA
* Regression
* Analytical errors

---

# Features

## Import and Explore Data

Students can:

* Upload their own datasets (CSV)
* Load example datasets
* Explore variables interactively

Includes:

* descriptive statistics
* histograms
* boxplots

---

## Distribution Visualization

Explore how distributions behave.

Features:

* histogram visualization
* group comparison
* normal distribution overlay
* rug plots for data points

Students can observe:

* symmetry
* skewness
* dispersion

---

## Normal Distribution Simulator

Interactive simulator allowing control of:

* mean (μ)
* standard deviation (σ)
* sample size

Students can see how:

* the distribution shifts when the mean changes
* the curve becomes narrower when σ decreases
* very small samples produce irregular distributions

The simulator displays:

* histogram of sampled data
* theoretical normal distribution
* empirical distribution estimate
* mean and standard deviation indicators

---

## Hypothesis Testing

Compare two groups using statistical tests.

Features:

* t-test
* group comparison
* visualization of distributions

Students can observe how:

* sample differences influence p-values
* statistical significance is determined

---

## ANOVA

Analysis of variance across multiple groups.

The module automatically detects the number of groups in the dataset.

Outputs include:

* F statistic
* p-value (scientific notation)
* group comparison boxplots
* descriptive statistics per group

This section helps students understand:

* between-group variance
* within-group variance
* hypothesis testing across multiple groups

---

## Correlation and Linear Regression

Explore relationships between two variables.

Features:

* scatter plots
* regression line
* regression statistics

The regression model used is:

$$
y = \beta_0 + \beta_1 x
$$

The app reports:

* slope
* intercept
* R²
* p-value (scientific notation)
* standard error

Students can observe how data patterns influence correlation strength and statistical significance.

---

## Errors in Analytical Chemistry

A dedicated module to illustrate **measurement error** concepts used in analytical chemistry.

Simulates repeated measurements while controlling:

* true value
* random error
* systematic error
* number of replicates

Outputs include:

* measurement distribution
* mean measured value
* standard deviation
* absolute error
* relative error
* coefficient of variation

Students can visualize the difference between:

### Accuracy

How close the measured value is to the true value.

Affected mainly by **systematic error**.

### Precision

How close repeated measurements are to each other.

Affected mainly by **random error**.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/basic-statistics-course.git
```

Enter the project folder:

```bash
cd basic-statistics-course
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Requirements

Typical dependencies:

* streamlit
* pandas
* numpy
* scipy
* plotly
* pillow

Example requirements file:

```txt
streamlit
pandas
numpy
scipy
plotly
pillow
```

---

# Running the App

Launch the Streamlit application:

```bash
streamlit run app_basic_statistics.py
```

The app will open automatically in your browser:

```
http://localhost:8501
```
---

# Educational Goals

This project aims to help students:

* develop intuition for statistical concepts
* visualize statistical distributions
* understand hypothesis testing
* interpret regression results
* distinguish between precision and accuracy

The interactive nature of the tool allows students to **experiment and learn through exploration**.

---

# Intended Audience

This tool is especially useful for:

* undergraduate chemistry courses
* analytical chemistry courses
* life science statistics courses
* data science introductions

---

# Author

Ricardo M. Borges
Professor – Institute of Natural Products Research (IPPN)
Federal University of Rio de Janeiro

---

# License

MIT License

---

# Contributing

Contributions are welcome.

Possible improvements include:

* additional statistical simulations
* bootstrap methods
* Bayesian inference examples
* power analysis tools
* additional datasets

---

# Acknowledgments

Developed for educational purposes to support the teaching of statistics in experimental sciences.
