
# **************************************
# Setup
# **************************************

# set your working directory
path <- "enter path here"
setwd(path)

# create directory
dir.create(paste0("data")) # Please put data into this folder
dir.create(paste0("model output"))
dir.create(paste0("submission"))

# **************************************
# run scripts
# **************************************

source("script/Preprocess.R")
source("script/H2O_model.R")
source("script/RF.R")
