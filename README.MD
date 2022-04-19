# Abalone Project  
This Repository uses the Abalone dataset to work on a regression project.

## Dataset
The dataset I have used can be found here: [Abalone dataset](https://github.com/icarocamelo/AbaloneRegression/blob/main/Training_Data.csv)

## Purpose of project
The dataset contains data to predict the age of abalone, one of the many sea snails, from physical measurements such as sex, length, diameter, height, total weight, weight of meat, the weight of the viscera and the weight of the shell. The age of an abalone is determined by cutting the shell through the cone, coloring it, and counting the number of rings using a microscope - a tedious and lengthy task. Your client wants to automate this process using a learning algorithm. Your client provides you with the following explanations of available data:
- Sex represents the sex: M, F, or I (infant) / nominal value
- Length represents the length of the abalone, in mm / continuous value
- Diameter is the diameter of the abalone (perpendicular to the length), in mm / continuous value
- Height represents height, in mm / continuous value
- Whole weight represents the total weight, in grams / continuous value
- Shucked weight represents the weight of the meat, in grams / continuous value
- Viscera weight represents the weight of the viscera, in grams / continuous value
- Shell weight represents the weight of the shell, in grams / continuous value
- Rings represents the number of rings / integer value (for information only: by adding 1.5 to this number we can compute the age of the abalone, in years)
The client also tells you that in the dataset, data with missing values has already been deleted, and continuous values have been scaled by normalization (division by 200)

## Models utilized
- Linear Regression
- Decision Tree
- Random Forest

## Techniques
- EDA
- RandomizedSearch CV
- GridSearch CV
  
## Conclusion
Random Forest was the algorithm with the best performance even before optimizing it with `RandomizedSearch` and `GridSearch`. I started by calculating the correlation to identify the features that were highly correlated with [`Rings`]. The [`Sex`] feature `corr = 0.36` didn’t make a lot of difference in the overall performance as it decreased by `~2%` among the three models. Next interaction, I removed the second least correlated feature [`Shucked Weight`] with `corr = 0.42` and the performance dramatically dropped. So, to avoid overfitting, I decided to drop the [`Sex`] feature hoping for the model to generalize better and perform well in the new dataset.