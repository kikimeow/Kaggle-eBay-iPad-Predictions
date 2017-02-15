#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Run Ensemble using H2O
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Ensemble performance (AUC): 0.905116279069767  "h2o.glm.wrapper"
# Ensemble performance (AUC): 0.900930232558139  "h2o.deeplearning.wrapper"
# Ensemble performance (AUC): 0.895116279069767  "h2o.deeplearning.wrapper"

#RF1: use average price diff 0.85757, 0.86788 (public, private) rfFit.rda
#RF2: use average price and predict price diff 0.85958, 0.86374 rfFit2.rda
#RF3: use predict price 0.86220, 0.85943 rfFit3.rda
#RF4: use average price, predict price, storage price 0.86064, 0.87220  rfFit4.rda
#RF5: PCA + center, scale word count 0.85584, 0.86892  rfFit5.rda
#RF6: no PCA, center, scale word count 0.86342, 0.87320 rfFit6.rda
#RF7: no PCA, center, scale word count (1 word count), H2O to predict price 0.86323, 0.87381 rfFit7.rda
#H2Odeep: no PCA, center, scale word count 0.85356, 0.88313
#H2OGBM: no PCA, center, scale word count 0.85468, 0.87999
#H2Odeep2: no PCA, center, scale word count (word count was counted twice, removed one) 0.85086, 0.87969
#H2Odeep3: no PCA, center, scale, 1 word count, predict price using H2O 0.85983, 0.88040


# Load preprocessed data
setwd("C:/Users/kikimeow/Documents/Classes/Class- MIT Analytics (R)/Kaggle/Kaggle- ebay") #set working directory
load("~/Classes/Class- MIT Analytics (R)/Kaggle/Kaggle- ebay/data/preprocessedData.RData")
fileName <- "submitH2ODeep3"


library(h2oEnsemble)  # Requires version >=0.0.4 of h2oEnsemble
localH2O <- h2o.init(nthreads = -1, max_mem_size = "8G")  # Start an H2O cluster with nthreads = num cores on your machine.  Use all cores available to you (-1)
h2o.removeAll() # Clean slate - just in case the cluster was already running

# create h2o dataframe for model training
X_train <- train[split,!(colnames(train)) %in% colExclude]
X_eval  <- train[-split, !(colnames(train)) %in% colExclude]
X_test <- test[,!(colnames(test)) %in% colExclude]

train_h2o <- as.h2o(X_train,  destination_frame = "train_h2o")
eval_h2o <- as.h2o(X_eval,  destination_frame = "eval_h2o")

X_test$sold <- NULL
test_h2o <- as.h2o(X_test,  destination_frame = "test_h2o")

# Setup X (predictors) & Y
Namey <- "sold"
Namesx <- setdiff(names(X_train), Namey)

# H20 Deep learning as metalearner
learner <- c(#"h2o.deeplearning.wrapper",
             "h2o.glm.wrapper", 
             "h2o.randomForest.wrapper", 
             "h2o.gbm.wrapper")

metalearner <-  "h2o.glm.wrapper" #"h2o.deeplearning.wrapper"

family <- "binomial"

h2oFit <- h2o.ensemble(x = Namesx, y = Namey,   # can only use column names
                       training_frame = train_h2o, 
                       validation_frame = eval_h2o,
                       family = family, 
                       learner = learner, 
                       metalearner = metalearner,
                       cvControl = list(V = 10, shuffle = TRUE)) 

# Evaluate ensemble model performance
h2oPerf <- h2o.ensemble_performance(h2oFit, newdata = eval_h2o )
h2oPerf

predictH2O <- predict(h2oFit, test_h2o)
head(predictH2O)
predictions <- as.data.frame(predictH2O$pred)[,3] # probability of "Sold"
head(predictions)

submitH2O <- cbind(test$UniqueID, predictions)
submitH2O <- as.data.frame(submitH2O)
names(submitH2O)[1] <- "UniqueID"
names(submitH2O)[2] <- "Probability1"
write.csv(submitH2O, paste0("submission/", fileName, ".csv", quote=FALSE, row.names = FALSE) 

h2o.shutdown()
