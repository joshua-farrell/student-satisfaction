#
# NOTE: Set this according to your own environment:
# setwd("~/Projects/work/joshua-farrell/students/")
#

source("./packages.R")
source("./shared.R")
source("./import.R")
source("./select.R")
source("./clean.R")
source("./train.R")
source("./graph.R")

#
# Data setup
#
# The import, cleaning, and separation processes are all applied in the
# following line. To see the details of what's happening take a look at the
# `import.R` and `clean.R` files, but generally we're importing the CSV without
# converting string variables into vectors, then we do the following:
#
# 1. Convert invalid values (e.g. 0, 48) to NAs
# 2. Impute NAs using the most common value for a variable
# 3. Convert ordinal variables were appropriate
# 4. Convert categorical variables were appropriate
# 5. Collapse categories in `SAT_*` variables into two groups
#

data <- clean(import("./data/data.csv"))

#
# We have a very unbalanced predictive problem: in all three cases, the ratio is
# around 0.90 to 0.95 in the `Satisfied` category. Such a large disproportion
# can have large negative effects in the predictive power of the models, and we
# therefore use the `Kappa` metric instead of the `Accuracy` metric to better
# measure the predictive power of our models (to change this configuration
# settting you should make the adjustment in the `constants.R` file).
#
# To help reduce the impact of such imbalance we do two things. We use Synthetic
# Minority Over-sampling TEchnique (SMOTE) to oversample the minority class and
# undersample the majority class, which will produce a more balanced dataset.
# The following code shows the difference in the proportions before and after
# the re-sampling technique is used.
#

category_proportions(data, "SAT_OVERALL")
category_proportions(data, "SAT_COMMUNITY")
category_proportions(data, "SAT_INSTRUCTION")

mosaic(data, "SAT_OVERALL", "SAT_COMMUNITY")
mosaic(data, "SAT_OVERALL", "SAT_INSTRUCTION")
mosaic(data, "SAT_COMMUNITY", "SAT_INSTRUCTION")

balloon(data, "SAT_OVERALL", "SAT_COMMUNITY")
balloon(data, "SAT_OVERALL", "SAT_INSTRUCTION")
balloon(data, "SAT_COMMUNITY", "SAT_INSTRUCTION")

data <- balance(data)

category_proportions(data, "SAT_OVERALL")
category_proportions(data, "SAT_COMMUNITY")
category_proportions(data, "SAT_INSTRUCTION")

mosaic(data, "SAT_OVERALL", "SAT_COMMUNITY")
mosaic(data, "SAT_OVERALL", "SAT_INSTRUCTION")
mosaic(data, "SAT_COMMUNITY", "SAT_INSTRUCTION")

balloon(data, "SAT_OVERALL", "SAT_COMMUNITY")
balloon(data, "SAT_OVERALL", "SAT_INSTRUCTION")
balloon(data, "SAT_COMMUNITY", "SAT_INSTRUCTION")

#
# We apply stratified sampling to make sure we keep the proportions reached by
# the SMOTE technique while creating the two data subsets (`train` and `test`),
# plus the original one saved in the `full` attribute. To access the three
# different datasets we created (i.e. `full`, `train`, and `test`) you can
# access the respective elements of the `data` object as shown in the examples
# below.
#
data <- separate(data)

# Examples:
# nrow(data[["full"]])
# nrow(data[["train"]])
# nrow(data[["test"]])
# str(data)
# head(data[["full"]])
# names(data[["full"]])
# summary(data[["full"]])

#
# Visualizations
#
# Since almost all variables are categorical, we don't have too much room to
# play with with visualizations, but we use two main types of graphs: mosaics
# and correspondence analysis. Mosaic graphs allow us to visualize the
# conditional distributions in the data we're analyzing, and to understand where
# subgroups are deviating from the implied random distributions. Correspondence
# analysis helps understand the relationships among the variable values. It's
# similar to Principal Component Analysis (PCA) but focuses on relative rather
# than absolute differences, which is what we want in this case to understand
# relative distributions, which helps for variable selection.
#
# For the mosaic graphs, the area of the mosaic reflects the relative magnitude
# of the values. Blue colors indicate observed values are higher than expected
# values, if the data were random. Red colors indicate observed values are lower
# than expected values, if the data were random.
#
# The the correspondence analysis, all the blue values correspond to one
# variable, and the red ones to the other. The closer they are in the graph, the
# more likely they occur together in the data, and viceversa. Keep in mind that
# thes graphs will only be created when there are at least three categories for
# each of the two categorical variables used in each case. This is a
# requirements of the correspondence analysis.
#
# There are lots of combinations to try, so we automate the process of creating
# all of the possible combinations and saving the graphs to disk so that it's
# easier to go through them using the operating system's image explorer (look
# for them in side the `graphs/` directory). There are interesting findings when
# going through those graphs.
#
# If you want to create an individual mosaic or correspondence graph, you can
# use the following code as reference:
#
# mosaic(data, "INCOME", "FINCON") correspondence(data, "INCOME", "FINCON")
#

#
# CAUTION: Executing this lines can take a long time because there are thousands
#          of graphs being created for each of the three types of graphs.
#
create_and_save_all_graphs(data, "correspondence")
create_and_save_all_graphs(data, "balloon")
create_and_save_all_graphs(data, "mosaic")

#
# Variable selection
#
# Variable selection is performed with Recursive Feature Elimination (RFE). The
# `selection_*` objects contain the following attributes: `data` which is the
# data with the variable selection applied, `graph` which shows the accuracy
# change as more variables are added, `n_variables` which is the number of
# variables selected, and `variables` which are the variables actually selected.
# Internally, Random Forests are used to measure variable importance.
#

selection_overall <- variable_selection(data, "SAT_OVERALL")
selection_community <- variable_selection(data, "SAT_COMMUNITY")
selection_instruction <- variable_selection(data, "SAT_INSTRUCTION")

#
# For example, to explore the results for the selection for "overall", you can
# use the following to see the filtered datasets, variables, and optimization
# graph.
#
head(selection_overall[["data"]][["full"]])
head(selection_overall[["data"]][["train"]])
head(selection_overall[["data"]][["test"]])

selection_overall[["n_variables"]]
selection_overall[["variables"]]

selection_overall[["graph"]]

#
# Model application
#
# Feel free to add other models, you can find the full list here:
# https://topepo.github.io/caret/available-models.html
#
# These models depend on external packages. If you don't have installed in your
# system, when you execute the code, you'll be notified that you don't have
# them, and you'll be asked whether you want to install them. If you do,
# execution will continue normally after the corresponding installations.
#

#
# At times, model application becomes problematic because there's a very low
# number of observations per year for some samples. To avoid that problem, I'm
# currently setting the `YEAR_FS` value to `ALL` for all observations, we can
# relax this later to test what recategorization makes more sense.
#
val <- "ALL"
var <- "YEAR_FS"
data <- recode_variable(data, var, val)
selection_overall[["data"]] <- recode_variable(
    selection_overall[["data"]], var, val
)
selection_community[["data"]] <- recode_variable(
    selection_community[["data"]], var, val
)
selection_instruction[["data"]] <- recode_variable(
    selection_instruction[["data"]], var, val
)

#
# Here is where ML models are actually applied:
#

models <- list(
    "random_forest" = "rf",
    "support_vector_machine" = "svmRadial",
    "naive_bayes" = "naive_bayes"
)

overall_results <- best_results_per_year(
    selection_overall[["data"]],  # or just `data`
    "SAT_OVERALL",
    models
)
community_results <- best_results_per_year(
    selection_community[["data"]],  # or just `data`
    "SAT_COMMUNITY",
    models
)
instruction_results <- best_results_per_year(
    selection_instruction[["data"]],  # or just `data`
    "SAT_INSTRUCTION",
    models
)

#
# Results exploration
#
# To explore other years, you should change the `Y_ALL` value to other years.
# Note that we had to use the `Y_` prefix because R does not allow for names to
# start with numbers. You can similarly explore other dependent variables or
# models. These are just some examples.
#
# To compare the predictive power, you may want to look at the predictive power
# of each model for each year and for each dependent variable. You can find
# these values in the `metrics` attribute in each case.
#
# Note that in some cases there may be no variable importance information (it
# depends on whether or not the model used is able to produce it), and there may
# also not be a ROC curve to show (in the case were the sample did not have at
# least one case for one of the categories). To fix this last case we would have
# to use stratified-sampling (TODO: do stratified sampling to avoid missing
# variable problems)
#

# Years available
names(overall_results)

# Models available for a given year
names(overall_results[["Y_ALL"]])

# Result objects available for year/model selection
names(overall_results[["Y_ALL"]][["random_forest"]])

# See the `predictor` object, for a given year and Random Forest
overall_results[["Y_ALL"]][["random_forest"]][["predictor"]]

# See the `metrics` objects, for a given year and all models
# The results I'm seeing (may be different for you) are:
#
# - Support Vector Machine: Accuracy: 0.6637, Kappa: 0.3274
# - Random Forest:          Accuracy: 0.7876, Kappa: 0.5752
# - Naive Bayes:            Accuracy: 0.6327, Kappa: 0.2655
#
overall_results[["Y_ALL"]][["support_vector_machine"]][["metrics"]]
overall_results[["Y_ALL"]][["random_forest"]][["metrics"]]
overall_results[["Y_ALL"]][["naive_bayes"]][["metrics"]]

# See the "optimization" `graph`, for a given year and Random Forest
overall_results[["Y_ALL"]][["random_forest"]][["graph"]]

# See the "variable" `importance`, for a given year and Random Forest
# (From these three models, only Random Forests provide `importance`)
overall_results[["Y_ALL"]][["random_forest"]][["importance"]]

# See the "test" `roc`, for a given year and Random Forest
predictor <- overall_results[["Y_ALL"]][["random_forest"]][["predictor"]]
roc(predictor, data[["test"]], "SAT_OVERALL")
