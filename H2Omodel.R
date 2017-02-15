#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Run Ensemble using H2O
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Load preprocessed data if not loaded 
# load("data/preprocessedData.RData")
fileName <- "submitH2ODeep3"  # This is for naming the file for submission

library(h2oEnsemble)  
localH2O <- h2o.init(nthreads = -1, max_mem_size = "8G") 

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
