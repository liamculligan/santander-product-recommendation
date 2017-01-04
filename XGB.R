#Santander Product Recommendation

#Bagged gradient boosted decision trees

#Authors: Tyrone Cragg & Liam Culligan
#Date: 20 December 2016

#Install the required packages
packages_required = c("data.table", "dplyr", "dtplyr", "lubridate", "Matrix", "xgboost", "Metrics")
new_packages = packages_required[!(packages_required %in% installed.packages()[,"Package"])]
if(length(new_packages) > 0) {
  install.packages(new_packages)
}

#Load required packages
library(data.table)
library(dplyr)
library(dtplyr)
library(lubridate)
library(Matrix)
library(xgboost) #Important - Version 0.6 required
library(Metrics)

# #Load the pre-processed data
load("data_pre_process.RData")

#Add a placeholder target variable to test
test[, product := NA]

#Combine train and test
data = rbind(train, test)
rm(train, test)
gc()

#Convert ult_fec_cli_1t to date
data[, ult_fec_cli_1t := ymd(ult_fec_cli_1t)]

#Convert fecha_dato to date
data[, fecha_dato := ymd(fecha_dato)]

#Create time difference feature
data[, time_diff := as.numeric(ult_fec_cli_1t - fecha_dato)]

#Remove unecessary features
data[, fecha_dato := NULL]
data[, fecha_alta := NULL] #antiguedad is similar
data[, ult_fec_cli_1t := NULL]
data[, cod_prov := NULL]

#Get train and test indices
train_indices = data[, .I[train_test == 1]]
test_indices = data[, .I[train_test == 2]]

#Remove indicator variable
data[, train_test := NULL]

#Extract id
train_ncodpers = data[!is.na(product), ncodpers]
test_ncodpers = data[is.na(product), ncodpers]

#Remove id from data
data[, ncodpers := NULL]

#Extract target variable
train_product = data[!is.na(product), product]

#Convert target variable to numeric
train_product = recode(train_product,"ind_cco_fin_ult1" = 0, "ind_nom_pens_ult1" = 1, "ind_recibo_ult1" = 2, "ind_reca_fin_ult1" = 3,
                       "ind_nomina_ult1" = 4, "ind_tjcr_fin_ult1" = 5, "ind_cno_fin_ult1" = 6, "ind_hip_fin_ult1" = 7,
                       "ind_ecue_fin_ult1" = 8, "ind_fond_fin_ult1" = 9, "ind_ctop_fin_ult1" = 10, "ind_valo_fin_ult1" = 11,
                       "ind_dela_fin_ult1" = 12, "ind_cder_fin_ult1" = 13, "ind_viv_fin_ult1" = 14, "ind_deme_fin_ult1" = 15,
                       "ind_plan_fin_ult1" = 16, "ind_ctma_fin_ult1" = 17, "ind_deco_fin_ult1" = 18, "ind_ctpp_fin_ult1" = 19,
                       "ind_pres_fin_ult1" = 20, "ind_ctju_fin_ult1" = 21)

#Remove target variable from data
data[, product := NULL]

#Replace NAs with  -999999
f_dowle3 = function(DT, missing_value) {
  for (j in seq_len(ncol(DT)))
    set(DT,which(is.na(DT[[j]])),j ,missing_value)
}

f_dowle3(data,  -999999)

#Convert characters to factors
data = data %>%
  mutate_if(is.character, as.factor)

#Convert data.table to sparse model matrix
data_m = sparse.model.matrix(~ . -1, data = data)

#Split data_m into train and test
train = data_m[train_indices, ]
test = data_m[test_indices, ]

#Save the sparse matrices
save(train, test, file = "data_processed.RData")

#Remove data and data_m
rm(data)
rm(data_m)
gc()

#Convert the feature set and target variable to an xgb.DMatrix - required as an input to xgboost
dtrain = xgb.DMatrix(data = train, label = train_product, missing = -999999)

#Initialise the data frame Results - will be used to save the results of a grid search
Results = data.frame(iteration = NA, eta = NA, max_depth = NA, subsample = NA, colsample_bytree = NA, gamma = NA, min_child_weight = NA,
                     alpha = NA, mlogloss = NA, st_dev = NA, n_rounds = NA)

#Initialise the list OOSPreds - will be used to save the out-of-fold model predictions - needed if a stacked generalisation is required later
OOSPreds = list()

#Save the version number for the stacked generalisation
Version = 1

#Fix the early stop round for boosting
early_stop_round = 40

#Set the parameters to be used within the grid search - only best set of parameters found via tuning is retained here
eta_values = c(0.01)
max_depth_values = c(3)
subsample_values = c(0.95)
colsample_bytree_values = c(0.5)
min_child_weight_values = c(1)
gamma_values = c(0.5)
alpha_values = c(1)

#Initialise iteration
iteration = 0

#Loop through all combinations of the parameters to be used in the above grid
for (eta in eta_values) {
  for (max_depth in max_depth_values) {
    for (subsample in subsample_values) {
      for (colsample_bytree in colsample_bytree_values) {
        for (gamma in gamma_values) {
          for (min_child_weight in min_child_weight_values) {
            for (alpha in alpha_values) {
              
              iteration = iteration + 1
              
              #Set the model parameters for the current iteration
              
              param = list(objective = "multi:softprob",
                           num_class = 22,
                           booster = "gbtree",
                           eval_metric = "mlogloss",
                           eta = eta,
                           max_depth = max_depth,
                           subsample = subsample,
                           colsample_bytree = colsample_bytree,
                           gamma = gamma,
                           min_child_weight = min_child_weight,
                           alpha = alpha
              )
              
              #Train the model using 5-fold cross validation with the given parameters
              set.seed(44)
              XGBcv = xgb.cv(params = param,
                             data = dtrain,
                             nrounds = 10000,
                             verbose = T,
                             nfold = 5,
                             early_stopping_rounds = early_stop_round,
                             prediction = T
              )
              
              #Save the number of boosting rounds for this set of parameters
              n_rounds = XGBcv$best_iteration
              
              #Save the logloss score obtained using 5-fold cross validation for this set of parameters
              mlogloss = XGBcv$evaluation_log$test_mlogloss_mean[n_rounds]
              
              #Save the standard deviation of the scoring metric for this set of parameters
              st_dev = XGBcv$evaluation_log$test_mlogloss_std[n_rounds]
              
              #Save the set of parameters and model tuning results in the data frame Results
              Results = rbind(Results, c(iteration, eta, max_depth, subsample, colsample_bytree, gamma, min_child_weight, alpha, mlogloss, 
                                         st_dev, n_rounds))
              
              #Add out-of-fold predictions to OOSPreds for this set of parameters
              OOSPreds[[iteration]] = XGBcv$pred
              
              rm(XGBcv)
              gc()
            }
          }
        }
      }
    }
  }
}

#Order Results in ascending order according to the scoring metric
Results = na.omit(Results)
Results = Results[order(Results$mlogloss),]

#Select only the out-of-fold predictions for the best set of parameters found above
OOSPreds = OOSPreds[[Results$iteration[1]]]

#Create a matrix, where each of the 22 rows represents the probability that a customer will purchase that product
OOSPreds = as.data.frame(matrix(OOSPreds, nrow = 22))

#Rename rows to product codes
rownames(OOSPreds) = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)

#For each customer, extract the 7 most likely products - required for mean average precision @ 7
OOSPreds = as.vector(apply(OOSPreds, 2, function(x) names(sort(x)[22:16])))

#Create a data frame with 7 most likely products for each customer
OOSPreds = as.data.frame(t(matrix(OOSPreds, nrow = 7)))

#Combine 7 most likely products into one string
OOSPreds$combined = paste(OOSPreds$V1, OOSPreds$V2, OOSPreds$V3, OOSPreds$V4, OOSPreds$V5, OOSPreds$V6, OOSPreds$V7, sep = " ")

#Set the best tuning paramters obtained using the previous grid search
paramTuned = list(objective = "multi:softprob",
                  num_class = 22,
                  booster = "gbtree",
                  eval_metric = "mlogloss",
                  eta = Results$eta[1],
                  max_depth = Results$max_depth[1],
                  subsample = Results$subsample[1],
                  colsample_bytree = Results$colsample_bytree[1],
                  gamma = Results$gamma[1],
                  min_child_weight = Results$min_child_weight[1],
                  alpha = Results$alpha[1]
)

#Initialise an empty list to store model predictions
TestPreds = list()

#Initialise seeds in a vector
seeds = c(8, 17, 25, 44, 99)

#Initialise iteration
iteration = 0

#Train each model, with a different seed, using the full training set
for (seed in seeds) {
  
  iteration = iteration + 1
  
  #Train the model on the full training set using the best parameters
  set.seed(seed)
  XGB = xgboost(params = paramTuned, 
                data = dtrain,
                nrounds = Results$n_rounds[1],
                verbose = 1
  )
  
  #Save the model predictions for this seed to a list
  TestPreds[[iteration]] = predict(XGB, test, missing = -999999)
  
  rm(XGB)
  gc()
  
}

#Using each of the five models, obtain test set predictions by bagging predictions
pred_test = Reduce("+", TestPreds)/length(TestPreds)

#Create a matrix, where each of the 22 rows represents the probability that a customer will purchase that product
pred_test = matrix(pred_test, nrow = 22)

#Set rownames
rownames(pred_test) = c("ind_cco_fin_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1", "ind_reca_fin_ult1",
                        "ind_nomina_ult1", "ind_tjcr_fin_ult1", "ind_cno_fin_ult1", "ind_hip_fin_ult1",
                        "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_ctop_fin_ult1", "ind_valo_fin_ult1",
                        "ind_dela_fin_ult1", "ind_cder_fin_ult1", "ind_viv_fin_ult1", "ind_deme_fin_ult1",
                        "ind_plan_fin_ult1", "ind_ctma_fin_ult1", "ind_deco_fin_ult1", "ind_ctpp_fin_ult1",
                        "ind_pres_fin_ult1", "ind_ctju_fin_ult1")

#Check that id order is identical
identical(test_ncodpers, may_2016_products[, ncodpers])

#Remove id from existing customer products
may_2016_products[, ncodpers := NULL]

#Transpose existing customer products
may_2016 = t(may_2016_products)

#Multiply June 2016 predictions with existing customer products from May 2016 - element-wise matrix multiplciation
#Needed to identify which products a customer already had
#The aim of the model is to predict only new products that a customer purchased for the month
pred_test = pred_test*may_2016

pred_test = as.data.frame(pred_test)

#For each customer, extract the 7 most likely products - required for mean average precision @ 7
pred_test = as.vector(apply(pred_test, 2, function(x) names(sort(x)[22:16])))

#Convert to a data frame with columns for each of the top 7 products by customer
pred_test = as.data.frame(t(matrix(pred_test, nrow = 7)))

#Combine top 7 predictions into 1 variable
pred_test$combined = paste(pred_test$V1, pred_test$V2, pred_test$V3, pred_test$V4, pred_test$V5, pred_test$V6, pred_test$V7, sep = " ")

#Create submission file
submission = data.frame(ncodpers = test_ncodpers, added_products = pred_test$combined)

#Save submission as csv file for Kaggle submission
write.csv(submission, "XGB.csv", row.names = F)

