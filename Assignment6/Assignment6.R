# Assignment 6: Bank Marketing Predictive and Prescriptive Analytics

# ---- Setup ----
required_packages <- c(
  "tidyverse",
  "tidymodels",
  "vip",
  "skimr",
  "GGally",
  "cluster",
  "factoextra"
)

installed_packages <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!pkg %in% installed_packages) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

library(tidyverse)
library(tidymodels)
library(vip)
library(skimr)
library(GGally)
library(cluster)
library(factoextra)

set.seed(2025)

# ---- Directories ----
output_dir <- "Assignment6"
fig_dir <- file.path(output_dir, "figures")
results_dir <- file.path(output_dir, "results")
data_dir <- file.path(output_dir, "data")

if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)
if (!dir.exists(data_dir)) dir.create(data_dir, recursive = TRUE)

# ---- Data Ingestion ----
data_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"
local_data <- file.path(data_dir, "bank-additional-full.csv")

if (!file.exists(local_data)) {
  download_attempt <- try(
    download.file(data_url, destfile = local_data, mode = "wb"),
    silent = TRUE
  )

  if (inherits(download_attempt, "try-error")) {
    stop(
      "Unable to download the dataset automatically. ",
      "Please download 'bank-additional-full.csv' from the UCI repository ",
      "and place it in ", normalizePath(data_dir), "."
    )
  }
}

bank_raw <- read_delim(local_data, delim = ";", na = c("unknown", ""), show_col_types = FALSE)

write_csv(bank_raw, file.path(results_dir, "bank_marketing_raw.csv"))

# ---- Exploratory Data Analysis ----
# Summary statistics
overall_summary <- skim(bank_raw)
write_csv(as_tibble(overall_summary), file.path(results_dir, "overall_summary.csv"))

# Class balance
y_distribution <- bank_raw %>%
  count(y) %>%
  mutate(share = n / sum(n))
write_csv(y_distribution, file.path(results_dir, "y_distribution.csv"))

ggplot(y_distribution, aes(x = y, y = n, fill = y)) +
  geom_col(show.legend = FALSE) +
  labs(title = "Subscription Outcome Distribution",
       x = "Term Deposit Subscription",
       y = "Count") +
  scale_fill_brewer(palette = "Set2")

ggsave(filename = file.path(fig_dir, "class_balance.png"), width = 6, height = 4, dpi = 300)

# Numeric feature distributions
numeric_cols <- bank_raw %>%
  select(where(is.numeric)) %>%
  names()

bank_raw %>%
  pivot_longer(all_of(numeric_cols), names_to = "feature", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30, fill = "#3182bd", color = "white") +
  facet_wrap(~feature, scales = "free") +
  labs(title = "Numeric Feature Distributions")

ggsave(filename = file.path(fig_dir, "numeric_distributions.png"), width = 10, height = 8, dpi = 300)

# Correlation matrix for numeric variables
cor_matrix <- bank_raw %>%
  select(where(is.numeric)) %>%
  cor(use = "pairwise.complete.obs")

ggcorr_plot <- ggcorr(bank_raw %>% select(where(is.numeric)), label = TRUE, label_size = 3, nbreaks = 6)

ggsave(ggcorr_plot,
       filename = file.path(fig_dir, "numeric_correlation.png"),
       width = 8,
       height = 7,
       dpi = 300)

write_csv(as_tibble(cor_matrix, rownames = "feature"), file.path(results_dir, "numeric_correlation.csv"))

# ---- Data Preparation ----
bank_clean <- bank_raw %>%
  mutate(y = factor(y))

set.seed(2025)
data_split <- initial_split(bank_clean, prop = 0.8, strata = y)
train_data <- training(data_split)
test_data <- testing(data_split)

cv_folds <- vfold_cv(train_data, v = 5, strata = y)

base_recipe <- recipe(y ~ ., data = train_data) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

pca_recipe <- base_recipe %>%
  step_pca(all_numeric_predictors(), threshold = 0.9)

prepped_recipe <- prep(base_recipe)
predictor_count <- prepped_recipe %>%
  juice() %>%
  select(-y) %>%
  ncol()

# ---- Models ----
logistic_model <- logistic_reg(mode = "classification") %>%
  set_engine("glm")

rf_model <- rand_forest(mode = "classification", trees = 1000, mtry = tune(), min_n = tune()) %>%
  set_engine("ranger", importance = "impurity")

xgb_model <- boost_tree(
  mode = "classification",
  trees = 1000,
  learn_rate = tune(),
  tree_depth = tune(),
  mtry = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost")

pca_logistic_model <- logistic_reg(mode = "classification") %>%
  set_engine("glm")

# Workflows
logistic_wf <- workflow() %>%
  add_model(logistic_model) %>%
  add_recipe(base_recipe)

rf_wf <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(base_recipe)

xgb_wf <- workflow() %>%
  add_model(xgb_model) %>%
  add_recipe(base_recipe)

pca_logistic_wf <- workflow() %>%
  add_model(pca_logistic_model) %>%
  add_recipe(pca_recipe)

# ---- Hyperparameter Tuning ----
rf_grid <- grid_regular(
  mtry(range = c(5, min(30, predictor_count))),
  min_n(range = c(2, 20)),
  levels = 5
)

xgb_grid <- grid_latin_hypercube(
  learn_rate(range = c(-3, -1)),
  tree_depth(range = c(2, 8)),
  mtry(range = c(5, min(30, predictor_count))),
  min_n(range = c(2, 20)),
  size = 20
)

rf_res <- tune_grid(
  rf_wf,
  resamples = cv_folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc, accuracy, kap)
)

xgb_res <- tune_grid(
  xgb_wf,
  resamples = cv_folds,
  grid = xgb_grid,
  metrics = metric_set(roc_auc, accuracy, pr_auc)
)

saveRDS(rf_res, file.path(results_dir, "rf_tuning.rds"))
saveRDS(xgb_res, file.path(results_dir, "xgb_tuning.rds"))

best_rf <- select_best(rf_res, "roc_auc")
best_xgb <- select_best(xgb_res, "roc_auc")

# Finalize workflows
final_rf_wf <- finalize_workflow(rf_wf, best_rf)
final_xgb_wf <- finalize_workflow(xgb_wf, best_xgb)

# Fit models on training data
logistic_fit <- fit(logistic_wf, data = train_data)
pca_logistic_fit <- fit(pca_logistic_wf, data = train_data)
rf_fit <- fit(final_rf_wf, data = train_data)
xgb_fit <- fit(final_xgb_wf, data = train_data)

# Evaluate on test data
test_results <- list(
  logistic = logistic_fit,
  pca_logistic = pca_logistic_fit,
  random_forest = rf_fit,
  xgboost = xgb_fit
) %>%
  imap_dfr(function(fit, model_name) {
    preds_prob <- predict(fit, test_data, type = "prob")
    preds_class <- predict(fit, test_data)
    preds <- bind_cols(test_data %>% select(y), preds_prob, preds_class)

    metric_values <- bind_rows(
      roc_auc(preds, truth = y, .pred_yes),
      pr_auc(preds, truth = y, .pred_yes),
      accuracy(preds, truth = y, .pred_class),
      sens(preds, truth = y, .pred_class),
      spec(preds, truth = y, .pred_class)
    ) %>%
      mutate(model = model_name)

    metric_values
  })

write_csv(test_results, file.path(results_dir, "model_test_metrics.csv"))

# Variable importance for best-performing model (assume xgboost)
try({
  vip <- vip::vip(xgb_fit %>% extract_fit_parsnip(), num_features = 15)
  ggsave(vip, filename = file.path(fig_dir, "xgb_variable_importance.png"), width = 8, height = 6, dpi = 300)
})

# Confusion matrices
conf_mat_list <- list(
  logistic = logistic_fit,
  pca_logistic = pca_logistic_fit,
  random_forest = rf_fit,
  xgboost = xgb_fit
) %>%
  imap(function(fit, model_name) {
    preds <- predict(fit, test_data) %>%
      bind_cols(test_data %>% select(y))
    conf <- yardstick::conf_mat(preds, truth = y, estimate = .pred_class)
    conf
  })

walk2(conf_mat_list, names(conf_mat_list), function(conf, name) {
  write_csv(as_tibble(conf$table), file.path(results_dir, paste0(name, "_confusion_matrix.csv")))
})

# ---- Clustering Analysis ----
# Use scaled numeric features for clustering
numeric_train <- train_data %>% select(where(is.numeric))
scaled_numeric <- scale(numeric_train)

# Determine optimal clusters using silhouette width
sil_widths <- map_dfr(2:6, function(k) {
  km <- kmeans(scaled_numeric, centers = k, nstart = 25)
  ss <- silhouette(km$cluster, dist(scaled_numeric))
  tibble(k = k, silhouette = mean(ss[, 3]))
})

write_csv(sil_widths, file.path(results_dir, "silhouette_scores.csv"))

best_k <- sil_widths %>% arrange(desc(silhouette)) %>% slice(1) %>% pull(k)

km_best <- kmeans(scaled_numeric, centers = best_k, nstart = 25)
cluster_summary <- train_data %>%
  mutate(cluster = factor(km_best$cluster)) %>%
  group_by(cluster) %>%
  summarise(across(where(is.numeric), list(mean = mean, sd = sd), .names = "{.col}_{.fn}"),
            count = n())

write_csv(cluster_summary, file.path(results_dir, "cluster_summary.csv"))

fviz_cluster(list(data = scaled_numeric, cluster = km_best$cluster), geom = "point", show.clust.cent = TRUE) +
  labs(title = paste("K-Means Clustering (k =", best_k, ")"))

ggsave(filename = file.path(fig_dir, "kmeans_clusters.png"), width = 7, height = 6, dpi = 300)

# ---- Save Model Artifacts ----
saveRDS(logistic_fit, file.path(results_dir, "logistic_fit.rds"))
saveRDS(pca_logistic_fit, file.path(results_dir, "pca_logistic_fit.rds"))
saveRDS(rf_fit, file.path(results_dir, "rf_fit.rds"))
saveRDS(xgb_fit, file.path(results_dir, "xgb_fit.rds"))
saveRDS(km_best, file.path(results_dir, "kmeans_model.rds"))

# Session info for reproducibility
writeLines(capture.output(sessionInfo()), file.path(results_dir, "session_info.txt"))

message("Analysis complete. Outputs saved to Assignment6/results and Assignment6/figures.")
