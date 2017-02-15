#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Random Forest Model
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Load preprocessed data
load("data/preprocessedData.RData")

modelName <- "rfFit7" # for saving model output
fileName <- "submitRF7" # for submission

# Caret Train Control
ctrl <- trainControl(method = "CV",
                     number = 10,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = "final",
                     allowParallel = TRUE)
###
# Random Forest
###

mtryValues <- c(10, 50, 100) #Number of variables randomly sampled as candidates at each split. mtry <- sqrt(ncol(X_train))
set.seed(100)
rfFit <- train(x = X_train, 
               y = y_train,
               method = "rf",
               ntree = 500,
               tuneGrid = data.frame(mtry = mtryValues),
               importance = TRUE,
               metric = "ROC",
               trControl = ctrl)
rfFit #mtry = 50
rfFit$times$everything
rfFitCM <- confusionMatrix((predict(rfFit, X_eval)), y_eval)
rfFitCM # accuracy: 0.8333, kappa:0.6606  (reasonable kappa around 0.3-0.5)
confusionMatrix(rfFit$final)
varImp(rfFit)
save(rfFit, file = paste0("model output/", modelName, ".rda"))

predictRF <- predict(rfFit, test, type = "prob")
predictions <- as.data.frame(predictRF$Sold) # probability of "Sold"
head(predictions)

submitRF <- cbind(test$UniqueID, predictions[1])
names(submitRF)[1] <- "UniqueID"
names(submitRF)[2] <- "Probability1"
write.csv(submitRF , paste0("submission/", fileName, ".csv"), quote=FALSE, row.names = FALSE) 
