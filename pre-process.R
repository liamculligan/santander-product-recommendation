#Santander Product Recommendation

#Pre-process data

#Authors: Tyrone Cragg & Liam Culligan
#Date: 15 December 2016

#Install the required packages
packages_required = c("data.table", "dplyr", "dtplyr", "lubridate", "tidyr")
new_packages = packages_required[!(packages_required %in% installed.packages()[,"Package"])]
if(length(new_packages) > 0) {
  install.packages(new_packages)
}

#Load required packages
library(data.table)
library(dplyr)
library(dtplyr)
library(lubridate)
library(tidyr)

#Read in data
train = fread("train_ver2.csv")
test = fread("test_ver2.csv")
gc()

#Convert fecha_dato to date format
train[, fecha_dato := ymd(fecha_dato)]
test[, fecha_dato := ymd(fecha_dato)]

#Convert fecha_alta to date format
train[, fecha_alta := ymd(fecha_alta)]
test[, fecha_alta := ymd(fecha_alta)]

#Only require training data up to and including June 2015
train = train[month(fecha_dato) <= 6]
gc()

#Get test variable names
test_names = names(test)

#Count of NAs per row
train[, na_count := rowSums(is.na(train[, .SD, .SDcols = test_names]))]
test[, na_count := rowSums(is.na(test[, .SD, .SDcols = test_names]))]

#Remove rows from train where NA count = 8 - all data missing - doesn't occur in test
train = train[na_count < 8]

#Feature names
target_names = setdiff(names(train), names(test))

#Add a placeholder target variables to test
test[, (target_names) := NA]

#Add indicator variables for training and testing data
train[, train_test := 1]
test[, train_test := 2]

#Combine train and test
data = rbind(train, test)
rm(train, test)
gc()

#Variables only update in January for training set but update every month for testing set
#Therefore replace antiguedad with variable that is calculated each month
data[, client_days := as.numeric(fecha_dato -fecha_alta)]

#Remove antiguedad
data[, antiguedad := NULL]

#Add indicator for whether the row is real
data[, real := 1]

#Create all possible combinations of ncodpers and fecha_dato
ncodpers_fecha_dato = as.data.table(expand.grid(unique(data$ncodpers), unique(data$fecha_dato)))
names(ncodpers_fecha_dato) = c("ncodpers", "fecha_dato")

#Set keys prior to joining
setkeyv(data, c("ncodpers", "fecha_dato"))
setkeyv(ncodpers_fecha_dato, c("ncodpers", "fecha_dato"))

#Merge data with all possible combinations to fill in missing rows
data = data[ncodpers_fecha_dato]

#Sort data for lagging
data = data %>%
  arrange(fecha_dato)

#Tidy the factor names of indrel_1mes
data[indrel_1mes == "1.0", indrel_1mes := "1"]
data[indrel_1mes == "3.0", indrel_1mes := "3"]

#Calculate 1-5 month lag features for target variables and sum of products per lag
for (i in 1:5) {
  lag_names = paste(target_names, "lag", i, sep = "_")
  data[, (lag_names) := shift(.SD, i, type = "lag"), by = ncodpers, .SDcols = target_names]
  
  lag_sum_name = paste("lag", i, "sum", sep = "_")
  lag_var = paste("lag", i, sep="_")
  data[, (lag_sum_name) := rowSums(.SD), .SDcols = grep(lag_var, names(data))]
}

#Calculate differences between each lag by product
for (i in 1:4) {
  for (j in 2:5) {
    if (i < j) {
      for (name in target_names) {
        lag_name_i = paste(name, "lag", i, sep = "_")
        lag_name_j = paste(name, "lag", j, sep = "_")
        difference_name = paste(name, "lag", i, j, "diff", sep = "_")
        data[, (difference_name) := data[[lag_name_i]] - data[[lag_name_j]]]
      }
    }
  }
}

#Calculate differences between each lag sum by product
for (i in 1:4) {
  for (j in 2:5) {
    if (i < j) {
      lag_sum_name_i = paste("lag", i, "sum", sep = "_")
      lag_sum_name_j = paste("lag", j, "sum", sep = "_")
      difference_sum_name = paste("lag", i, j, "sum_diff", sep = "_")
      data[, (difference_sum_name) := data[[lag_sum_name_i]] - data[[lag_sum_name_j]]]
    }
  }
}

#Calculate rolling features for target variables
for (i in 1:4) {
  for (j in 2:5) {
    if (i < j) {
      for (name in target_names) {
        
        #Initialise lag_names with i lag
        lag_names = paste(name, "lag", i, sep = "_")
        
        #Add (i+1):j lags to lag_names
        for (k in (i+1):j) {
          lag_names = c(lag_names, paste(name, "lag", k, sep = "_"))
        }
        
        #Define names (one removing NA values, other including them)
        rolling_name = paste(name, "rolling", i, j, sep = "_")
        rolling_name_na = paste(name, "rolling_na", i, j, sep = "_")
        
        #Calculate features (one removing NA values, other including them)
        data[, (rolling_name) := rowMeans(.SD, na.rm = T), .SDcols = lag_names]
        data[, (rolling_name_na) := rowMeans(.SD, na.rm = F), .SDcols = lag_names]
        
      }
    }
  }
}

gc()

#Characters
#Calculate median_renta by group and differences between renta and median_renta by customer
vars = c("indrel_1mes", "tiprel_1mes", "indresi", "conyuemp", "segmento", "indrel", "nomprov", "pais_residencia")
for (var in vars) {
  
  #Define names
  median_renta_var = paste(var, "median_renta", sep = "_")
  data[, (median_renta_var) := median(renta, na.rm = T), by = var]
  
  #Calculate values
  median_renta_difference_var = paste(var, "median_renta_difference", sep = "_")
  data[, (median_renta_difference_var) := data[["renta"]] - data[[median_renta_var]]]
}

#Filter data to only include the months of May and June
data = data[month(fecha_dato) == 5 | month(fecha_dato) == 6]
gc()

#Only keep real rows and remove indicator variable
data = data[!is.na(real)]
data[, real := NULL]

#Split data into train and test
train = data[train_test == 1]
test = data[train_test == 2]

#Remove data
rm(data)
gc()

#Remove target_names from test
test[, (target_names) := NULL]

#Data for May 2016
train_may_2016 = train[month(fecha_dato) == 5 & year(fecha_dato) == 2016]

#Function to swap 1s and 0s
manipulate_dummies = function(x) {
  ifelse(x == 1, 0,
         ifelse(x == 0, 1, 
                x))
}

#Swap 1s and 0s for May 2016
#Will be multiplied with June 2016 predictions to remove any predicted products that customer already
#had in May 2016

train_may_2016 = train_may_2016 %>%
  select(ncodpers, ind_ahor_fin_ult1:ind_recibo_ult1) %>%
  mutate_if(is.integer, manipulate_dummies) 

#Get the test ids for June 2016
test_id = test %>%
  select(ncodpers)

#Set keys prior to joining
setkey(test_id, ncodpers)
setkey(train_may_2016, ncodpers)

#Left outer join test_id on train_may_2016, select only columns that exist in reduced training data
#and arrange by id
may_2016_products = train_may_2016[test_id] %>%
  select(ncodpers, ind_cco_fin_ult1, ind_nom_pens_ult1, ind_recibo_ult1, ind_reca_fin_ult1, ind_nomina_ult1, ind_tjcr_fin_ult1,
         ind_cno_fin_ult1, ind_hip_fin_ult1, ind_ecue_fin_ult1, ind_fond_fin_ult1, ind_ctop_fin_ult1, ind_valo_fin_ult1,
         ind_dela_fin_ult1, ind_cder_fin_ult1, ind_viv_fin_ult1, ind_deme_fin_ult1, ind_plan_fin_ult1, ind_ctma_fin_ult1,
         ind_deco_fin_ult1, ind_ctpp_fin_ult1, ind_pres_fin_ult1, ind_ctju_fin_ult1) %>%
  arrange(ncodpers)

#Arrange test by id
test = test %>%
  arrange(ncodpers)

#Data for June 2015
train_june = train[month(fecha_dato) == 6 & year(fecha_dato) == 2015, .SD]

#Gather the 24 feature columns into 1 column
train_june_long = train_june %>%
  gather(product, product_value, ind_ahor_fin_ult1:ind_recibo_ult1) %>%
  filter(product_value != 0) %>%
  select(-product_value)

#Select only 2 columns needed for the next step
train_june_long_key = train_june_long %>%
  select(ncodpers, product)

#Data for May 2015
train_may = train[month(fecha_dato) == 5 & year(fecha_dato) == 2015, .SD]

#Gather the 24 feature columns into 1 column
train_may_long = train_may %>%
  gather(product, product_value, ind_ahor_fin_ult1:ind_recibo_ult1) %>%
  filter(product_value != 0) %>%
  select(-product_value)

#Select only 2 columns needed for the next step
train_may_long_key = train_may_long %>%
  select(ncodpers, product)

#Find the combinations of id and product that exist in June 2015 but not May 2015
#These are new products for which a user signed up in June 2015
sub = dplyr::setdiff(train_june_long_key, train_may_long_key) %>%
  as.data.table()

#Set appropriate keys prior to joining
setkeyv(sub, c("ncodpers", "product"))
train_june_long = as.data.table(train_june_long)
setkeyv(train_june_long, c("ncodpers", "product"))

#Left outer join sub on train_june_long
train = train_june_long[sub]

#Save train and test as RData
save(train, test, may_2016_products, file = "data_pre_process.RData")
