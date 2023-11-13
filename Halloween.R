library(vroom)
library(tidyverse)
library(tidymodels)
library(bonsai)
library(lightgbm)

halloween <- vroom("./trainWithMissingValues.csv")
train_data <- vroom("./train.csv")
test_data <- vroom("./test.csv")

folds <- vfold_cv(train_data, v = 5, repeats = 1)

# Main Recipe
nn_recipe <- recipe(type ~., data = train_data) %>%
  step_mutate(color = as.factor(color)) %>%
  step_mutate(type = as.factor(type), skip = TRUE) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1) %>%
  prep()


################################
### imputation
################################

my_rec <- recipe(type ~ ., data=halloween) %>% # Set model formula and dataset
  step_impute_linear(bone_length, impute_with=imp_vars('has_soul', 'color', 'type')) %>%
  step_impute_linear(rotting_flesh, impute_with=imp_vars('has_soul', 'color', 'type', 'bone_length')) %>%
  step_impute_linear(hair_length, impute_with=imp_vars('has_soul', 'color', 'type', 'bone_length', 'rotting_flesh')) %>%
  prep()

testbake <- bake(my_rec, new_data=halloween)

rmse_vec(train_data[is.na(halloween)], testbake[is.na(halloween)])

sapply(halloween, class)

################################
### Neural Network
################################

nn_model <- mlp(hidden_units = tune(),
                epochs = 300) %>%
  set_engine("nnet") %>%
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tune_grid <- grid_regular(hidden_units(range = c(1, 15)),
                             levels = 10)

nn_results_tune <- nn_wf %>%
  tune_grid(resamples = folds,
            grid = nn_tune_grid,
            metrics = metric_set(accuracy))

nn_results_tune %>% collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

bestTune_nn <- nn_results_tune %>%
  select_best("accuracy")

final_nn_wf <- nn_wf %>%
  finalize_workflow(bestTune_nn) %>%
  fit(data = train_data)

nn_predictions <- final_nn_wf %>%
  predict(new_data = test_data, type = "class")

ggg_predictions_nn <- nn_predictions %>%
  bind_cols(., test_data) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)


vroom_write(x=ggg_predictions_nn, file="./nn.csv", delim=",")

################################
### Boost
################################

boost_model <- boost_tree(tree_depth=tune(),
trees=tune(),
learn_rate=tune()) %>%
set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
set_mode("classification")

## CV tune, finalize and predict here and save results

boost_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(boost_model)

boost_tune_grid <- grid_regular(tree_depth()
                                ,trees()
                                ,learn_rate())


boost_results_tune <- boost_wf %>%
  tune_grid(resamples = folds,
            grid = boost_tune_grid,
            metrics = metric_set(accuracy))

boost_results_tune %>% collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

bestTune_boost <- boost_results_tune %>%
  select_best("accuracy")

final_boost_wf <- boost_wf %>%
  finalize_workflow(bestTune_boost) %>%
  fit(data = train_data)

boost_predictions <- final_boost_wf %>%
  predict(new_data = test_data, type = "class")

ggg_predictions_boost <- boost_predictions %>%
  bind_cols(., test_data) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)


vroom_write(x=ggg_predictions_boost, file="./boost.csv", delim=",")


################################
### Bart
################################

bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
set_engine("dbarts") %>% # might need to install
set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(bart_model)

bart_tune_grid <- grid_regular(trees())

bart_results_tune <- bart_wf %>%
  tune_grid(resamples = folds,
            grid = bart_tune_grid,
            metrics = metric_set(accuracy))

bestTune_bart <- bart_results_tune %>%
  select_best("accuracy")

final_bart_wf <- bart_wf %>%
  finalize_workflow(bestTune_bart) %>%
  fit(data = train_data)

bart_predictions <- final_bart_wf %>%
  predict(new_data = test_data, type = "class")

ggg_predictions_bart <- bart_predictions %>%
  bind_cols(., test_data) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)


vroom_write(x=ggg_predictions_bart, file="./bart.csv", delim=",")


