data(iris)
ind <- sample(2,nrow(iris),replace=TRUE,prob=c(0.7,0.3))
trainData <- iris[ind==1,]
testData <- iris[ind==2,]

library(randomForest)

args <- commandArgs(trailingOnly=TRUE)

args.ntree <- strtoi(unlist(strsplit(args[1], '='))[2])
args.nodesize <- strtoi(unlist(strsplit(args[2], '='))[2])

iris_rf <- randomForest(Species~.,data=trainData,ntree=args.ntree,proximity=TRUE,nodesize=args.nodesize)
irisPred<-predict(iris_rf,newdata=testData)

output.directory <- Sys.getenv("POLYAXON_RUN_OUTPUTS_PATH")
output.path <- file.path(output.directory, "iris_pred.csv")
write.csv(irisPred, file = output.path)

model.path <- file.path(output.directory, "model.Rds")
saveRDS(iris_rf, file = model.path, compress = TRUE)
