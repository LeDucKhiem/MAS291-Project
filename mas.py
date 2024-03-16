import numpy
import statsmodels.api 
import matplotlib.pyplot as plt
from tabulate import tabulate

with open('c:/Users/leduc/Desktop/New folder/Python/x.txt', 'r') as x_file:
    x = numpy.array([float(line.strip()) for line in x_file])

with open('c:/Users/leduc/Desktop/New folder/Python/y.txt', 'r') as y_file:
    y = numpy.array([float(line.strip()) for line in y_file])
 
#Fit a regression model
X = statsmodels.api.add_constant(x)  # Add a constant term for the intercept
model = statsmodels.api.OLS(y, X).fit()
#Test for significance of regression with alpha=0.05
alpha = 0.05
#Test with slope(B1) of regression (H0: B1=0, H1: B1!=0) 
if model.tvalues[1] > 2.011:
    regression_significant_slope="With alpha=0.05, The Slope is significant."
else:
    regression_significant_slope="With alpha=0.05, The Slope is not significant."
#Test with intercept(B0) of regression (H0: B0=0, H1: B0!=0) 
if model.pvalues[0] < alpha:
    regression_significant_intercept="With alpha=0.05, The Intercept is significant."
else:
    regression_significant_intercept="With alpha=0.05, The Intercept is not significant."
#Calculate R^2
r_squared = model.rsquared
correlation_coefficient = numpy.sqrt(r_squared)
#Test the hypothesis that p=0, using alpha=0.05
test_statistic=(correlation_coefficient*numpy.sqrt(50-2))/numpy.sqrt(1-numpy.sqrt(model.rsquared))
print(test_statistic)
if test_statistic > 2.011:
    correlation_hypothesis="The predictor variable is significant."
else:
    correlation_hypothesis="The predictor variable is not significant."

#Construct a 95% confidence interval for the correlation coefficient
n = len(x)
confidence_interval_lower = correlation_coefficient - 1.96*((1-r_squared)/numpy.sqrt(50))
confidence_interval_upper = correlation_coefficient + 1.96*((1-r_squared)/numpy.sqrt(50))

table_data = [
    ['Regression Analysis', 'Value'],
    ['Slope', model.params[1]],
    ['Intercept', model.params[0]],
    ['Correlation Coefficient', correlation_coefficient],
    ['R_squared', r_squared],
    ['Regression Significance Status', regression_significant_slope],
    ['Regression Significance Status', regression_significant_intercept],
    ['Hypothesis', correlation_hypothesis],
    ['Confidence Interval ', (confidence_interval_lower, confidence_interval_upper)]
]

table=tabulate(table_data, tablefmt='grid')
print(model.summary())
print(table) 
# print linear regression 
plt.scatter(x, y, label="Data Points")
plt.plot(x, model.predict(X), color='green', label="Linear Regression")
plt.xlabel("Total Monthly Energy Usage (kWh)")
plt.ylabel("Peak Hour Demand (kW)")
plt.title("Linear Regression Model: Peak Hour Demand vs. Total Monthly Energy Usage")
plt.legend()
plt.grid(True)
plt.show()
