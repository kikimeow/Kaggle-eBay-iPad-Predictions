#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Random Forest Model
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

modelName <- "rfFit7" 
fileName <- "submitRF7"


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