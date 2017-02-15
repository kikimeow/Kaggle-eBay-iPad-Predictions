libraries <-c("plyr", "dplyr", "tm", "Hmisc", "SnowballC", "caret")

lapply(libraries, FUN = function(X) {
  do.call("require", list(X)) 
})
rm(libraries)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Read data
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
train <- read.csv("data/eBayiPadTrain.csv", stringsAsFactors=FALSE)
test <- read.csv("data/eBayiPadTest.csv", stringsAsFactors=FALSE)

# first look at data
#names(test)
#names(train)
#summary(data)
#describe(data)
#str(data)

# check for unknown products
#data$description <- tolower(data$description)
#unknown <- filter(data, productline == "Unknown")
#unknown$description

# Check for NA
#indx <- apply(data, 2, function(x) any(is.na(x)))
#NACols <- colnames(data)[which(indx == TRUE)]
#NACols

# reorganize columes before merging two datasets for pre-processing
# which(names(train)=="sold")
train <- train[,c(1:(ncol(train)-2),ncol(train),which(names(train)=="sold"))]
test$sold <- 0
train$set <- "train"
test$set <- "test"
data <- rbind(train, test)

# Data structure
#str(data)
data$productline <- as.factor(data$productline)
data$condition <- as.factor(data$condition)
data$cellular <- as.factor(data$cellular)
data$carrier <- as.factor(data$carrier)
data$color <- as.factor(data$color)
data$storage <- as.factor(data$storage)
data$UniqueID <- as.character(data$UniqueID)
data$sold <- as.integer(data$sold)
data$startprice <- as.numeric(data$startprice)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Adding Features
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

####
# Predict the price of the item using non-biddable sold items using Random Forest Model
# diffPredict:  Difference between the predicted price - start price
####

# add new column for summarising average price
data$simpleCondition <- ifelse(data$condition == "New", "New", "Not New")
data$simpleCondition <- as.factor(data$simpleCondition)

# filter for only non-biddable sold items
soldTrain <- filter(data, set == "train", sold == 1 & biddable == 0)
idSoldTrain <- soldTrain$UniqueID

# create training set
colExclude <- c("biddable", "UniqueID", "sold", "set", "description")

## Predict using Random Forest

# X_train <- soldTrain[,!(colnames(soldTrain)) %in% colExclude]
# y_train <- X_train$startprice
# X_train$startprice <- NULL
# 
# # use random forest model for prediction
# ctrl <- trainControl(method = "CV",
#                      number = 10)
# 
# set.seed(100)
# rfTune <- train(x = X_train,
#                 y = y_train,
#                 method = "rf",
#                 trControl = ctrl,
#                 importance = TRUE)
# 
# rfTune
# save(rfTune, file = "model output/priceRFTune.rda")
# predictPrice  <- predict(rfTune, data[, !(colnames(soldTrain)) %in% colExclude])
# data$predictPrice <- predictPrice
# data$diffPredict <- data$startprice - data$predictPrice


## Predict using H2O ensemble

library(h2oEnsemble)  # Requires version >=0.0.4 of h2oEnsemble
localH2O <- h2o.init(nthreads = -1, max_mem_size = "8G")  # Start an H2O cluster with nthreads = num cores on your machine.  Use all cores available to you (-1)
h2o.removeAll() # Clean slate - just in case the cluster was already running

# create h2o dataframe for model training
train_h2o <- as.h2o(soldTrain[,!(colnames(soldTrain)) %in% colExclude],  destination_frame = "train_h2o")
test_h2o <- as.h2o(data[, !(colnames(soldTrain)) %in% colExclude],  destination_frame = "test_h2o")

# Setup X (predictors) & Y
Namey <- "startprice"
Namesx <- setdiff(names(train_h2o), Namey)

# H20 Deep learning as metalearner
learner <- c("h2o.deeplearning.wrapper",
             "h2o.glm.wrapper", 
             "h2o.randomForest.wrapper", 
             "h2o.gbm.wrapper")

metalearner <- "h2o.glm.wrapper" #"h2o.deeplearning.wrapper" #"h2o.glm.wrapper"
family <- "gaussian"

h2oFit <- h2o.ensemble(x = Namesx, y = Namey,   # can only use column names
                       training_frame = train_h2o, 
                       #classification=FALSE,
                       family = family, 
                       learner = learner, 
                       metalearner = metalearner,
                       cvControl = list(V = 10, shuffle = TRUE)) 

predictH2O <- predict(h2oFit, test_h2o)
head(predictH2O)
predictions <- as.data.frame(predictH2O$pred)[,1]
head(predictions)
predictPrice <- predictions

data$predictPrice <- predictPrice
data$diffPredict <- data$predictPrice - data$startprice



####
# Compute average price of item using non-biddable sold items grouped by "productline + simpleCondition"
# If no data is available for non-biddable sold item, use average of all listed items.  
# diffPrice: Difference between the computed average price - start price
####

# Obtain list of unique productline + simpleCondition for the dataset
avgPriceList <- data %>%
  select(productline, simpleCondition) %>%
  group_by(productline, simpleCondition)%>%
  distinct()

# Calculate average sold price of non-biddable item by productline
avgPrice <- data %>%
  select(biddable, startprice, productline, sold, simpleCondition)%>%
  filter(sold == 1 & biddable == 0)%>%
  group_by(productline, simpleCondition ) %>%
  summarise(avgPrice = mean(startprice), numObs = n())

# add average price to avgPriceList
avgPriceList <- avgPriceList %>%
  left_join(avgPrice, by = c("productline", "simpleCondition"))

# look at rows where there is no average price based on (Sold + Unbiddable) for the (condition + productline).  
# Compute "Listing" average price
missingPrices <- avgPriceList %>%
  filter(is.na(avgPrice))

avgMissingPrice <- data %>%
  select(startprice, productline, simpleCondition)%>%
  inner_join(missingPrices, by = c("productline", "simpleCondition")) %>%
  group_by(productline, simpleCondition) %>%
  summarise(avgPrice = mean(startprice), numObs = n())

# merge back to avgPriceList
avgPriceList <- avgPriceList %>%
  filter(!is.na(avgPrice))
avgPriceList <- rbind(avgPriceList , avgMissingPrice)

rm(avgPrice)
rm(missingPrices)
rm(avgMissingPrice)

# merge average price with data
data <- data %>%
  left_join(avgPriceList, by = c("productline", "simpleCondition"))
data$numObs <- NULL

# Calculate biddable - avgPrice
data$diffPrice <- data$startprice - data$avgPrice

# Check for NA
indx <- apply(data, 2, function(x) any(is.na(x)))
NACols <- colnames(data)[which(indx == TRUE)]
NACols

####
# Compute average price of item using non-biddable sold items grouped by "productline + simpleCondition + storage"
# If no data is available for non-biddable sold item, use average of all listed items.  
# diffStorage: Difference between the computed average price - start price
####

# Obtain list of unique productline + simpleCondition + storage for the dataset
avgPriceList <- data %>%
  select(productline, simpleCondition, storage) %>%
  group_by(productline, simpleCondition, storage)%>%
  distinct()

# Calculate average sold price of non-biddable item by productline
avgStorage <- data %>%
  select(biddable, startprice, productline, sold, simpleCondition, storage)%>%
  filter(sold == 1 & biddable == 0)%>%
  group_by(productline, simpleCondition, storage) %>%
  summarise(avgStorage = mean(startprice), numObs = n())

# add average price to avgPriceList
avgPriceList <- avgPriceList %>%
  left_join(avgStorage, by = c("productline", "simpleCondition", "storage"))

# look at rows where there is no average price based on (Sold + Unbiddable) for the (condition + productline + storage).  
# Compute "Listing" average price
missingPrices <- avgPriceList %>%
  filter(is.na(avgStorage))

avgMissingPrice <- data %>%
  select(startprice, productline, simpleCondition, storage)%>%
  inner_join(missingPrices, by = c("productline", "simpleCondition", "storage")) %>%
  group_by(productline, simpleCondition, storage) %>%
  summarise(avgStorage = mean(startprice), numObs = n())

# merge back to avgPriceList
avgPriceList <- avgPriceList %>%
  filter(!is.na(avgStorage))
avgPriceList <- rbind(avgPriceList , avgMissingPrice)

rm(avgStorage)
rm(missingPrices)
rm(avgMissingPrice)

# merge average price with data
data <- data %>%
  left_join(avgPriceList, by = c("productline", "simpleCondition", "storage"))
data$numObs <- NULL

# Calculate biddable - avgStorage
data$diffStorage <- data$startprice - data$avgStorage

# Check for NA
indx <- apply(data, 2, function(x) any(is.na(x)))
NACols <- colnames(data)[which(indx == TRUE)]
NACols 


#####
# Text Analytics
#####

stemDocumentfix <- function(x)
{
  PlainTextDocument(paste(stemDocument(unlist(strsplit(as.character(x), " "))),collapse=' '))
}

# get rid of non-ascii characters
data$description <- iconv(data$description, "latin1", "ASCII", sub="")
data$description <- gsub("ui", "", data$description)


corpus <- Corpus(VectorSource(data$description))

corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)
corpus <- tm_map(corpus, stemDocumentfix)
corpus <- tm_map(corpus, PlainTextDocument) #  Lazy mappings are mappings which are delayed until the content is accessed
dtm <- DocumentTermMatrix(corpus)
#inspect(dtm[1:5, 1:20])
freq <- colSums(as.matrix(dtm))
freq[order(-freq)]

# add wordCount to data
data$wordCount <- rowSums(as.matrix(dtm))

# remove sparse terms
SparseFactor = 0.999
sparse <- removeSparseTerms(dtm, SparseFactor) 
DescriptionWords <- as.data.frame(as.matrix(sparse))
colnames(DescriptionWords) <- make.names(colnames(DescriptionWords))
#sparse

# check which terms are frequent
# newFreq <- colSums(as.matrix(sparse))
# newFreq <- newFreq[order(-newFreq)]
# newFreq <- data.frame(word=names(newFreq), freq=newFreq)
# p <- ggplot(subset(newFreq, freq >= quantile(freq,0.90)), aes(word, freq))
# p <- p + geom_bar(stat="identity")
# p <- p + theme(axis.text.x=element_text(angle=45, hjust=1))
# p

# add terms back dataset
DescriptionWords [,names(DescriptionWords)] <- 
  lapply(DescriptionWords[,names(DescriptionWords),drop=FALSE],as.integer) # convert from num to integer

# PCA for DescriptionWords
# row.names(DescriptionWords)<- NULL   # change from "character(0)" to sequential
# wordPcaModel<-preProcess(DescriptionWords, method = c("pca"))
# #wordPcaModel #PCA needed 250 components to capture 95 percent of the variance
# wordPCA <- predict(wordPcaModel, DescriptionWords) 

data <- cbind(data, DescriptionWords) 
#data <- cbind(data, wordPCA)   #wordPCA not used because performance lowers


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Create Dummy Variables, Standardize Numeric Variables, Split dataset
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

detach("package:tm", unload=TRUE)
detach("package:SnowballC", unload=TRUE)

# remove variables
colExclude <- c("simpleCondition", "description", "avgPrice", "predictPrice", "avgStorage")
#dataExclude <- data[,names(data) %in% colExclude]
data <- data[,!names(data) %in% colExclude]

# create dummy variables for factor variables
factorVars <- names(data[,sapply(data,is.factor)])
factorVars

factorData <- data[,factorVars]
dummiesModel <- dummyVars( ~ ., data = factorData)
factorData <- as.data.frame(predict(dummiesModel, newdata = factorData))
rm(dummiesModel)

# rename columns of factor variables
recodeValues <- function(x)   
{
  x <- gsub("-", ".", x)
  x <- gsub("[,'\\(\\)]", "", x)
  x <- gsub(" ", ".", x)
  x <- gsub("\\(", "", x)
  x <- gsub("\\)", "", x)
  x <- gsub("/", ".", x) 
}

for (i in names(factorData)) 
  names(factorData)[names(factorData)== i ] <- recodeValues(i)

# Standardize (Center & Scale) Numeric variables
numericVars <- c("startprice", "wordCount", "diffPrice", "diffPredict", "diffStorage")
preprocessParams <- preProcess(data[,numericVars], method=c("center", "scale")) 
numericData <- predict(preprocessParams, data[,numericVars])

# Binding processed data together
data <- data[,!names(data) %in% numericVars]
data <- cbind(data, numericData)
data <- data[,!names(data) %in% factorVars]
data <- cbind(data, factorData)

# Split into training, evaluation, test set
data$sold <- as.factor(data$sold)
levels(data$sold)[levels(data$sold)=="0"] <- "NotSold"
levels(data$sold)[levels(data$sold)=="1"] <- "Sold"
train <- subset(data, set == "train")
test <- subset(data, set == "test")
set.seed(100)
split <- createDataPartition(train$sold, p = .90)[[1]]

# Fill in column to be ignored by model
colExclude <- c("set", "UniqueID")

X_train <- train[split,!(colnames(train)) %in% colExclude]
X_eval  <- train[-split, !(colnames(train)) %in% colExclude]
X_test <- test[,!(colnames(test)) %in% colExclude]
y_train <- X_train$sold
y_eval <- X_eval$sold
X_train$sold <- NULL
X_eval$sold <- NULL
X_test$sold <- NULL

rm(factorData)
rm(numericData)
rm(soldTrain)
rm(corpus)
gc()

save.image("data/preprocessedData.RData")
