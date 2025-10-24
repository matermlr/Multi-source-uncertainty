#相比于directboots.R，特征更换为新筛选的
#相比于directboots2.R，使用5特征；bootstrap使用不重复训练集

setwd("~/ECEdemo/directboots")
load("~/ECEdemo/directboots/3.RData")
# setwd("F:/机器学习/Multi-fidelity/ECE demo/directboots")

# library(RMariaDB)
library(dplyr)
library(MuFiCokriging)
library(parallel)
library(gbm)           #for gradient boosting
library(e1071)         #for SVR.rbf
library(randomForest)
# library(ggplot2)
# 
# ##** input data and preprocessing **##
# 
# ##** data **##
# #* input data *#
# #Connecting to MySQL DB
# db_ec<-dbConnect(MariaDB(), user = "root", password = "mlmater123", dbname = "db_ec_bto", host = "localhost")
# 
# dt_direct<-dbReadTable(db_ec, "dt_direct", check.names=F)[, -1]  #import data 不考虑主键
# #！注意：该工作中使用的数据为数据库的前1659条
# #列名包含运算符, check.name=F否则运算符会被替换成.
# processing<-dbReadTable(db_ec, "processing", check.names=F)
# composition<-dbReadTable(db_ec, "composition", check.names=F)
# # 拼表格 #
# comp<-composition[1,-c(1,12,13)]
# for (i in 1:dim(dt_direct)[1]){
#   compiid<-processing[which(processing$BTO_proc_id==dt_direct$BTO_proc_id[i]),]$BTO_id
#   compi<-subset(composition, BTO_id==compiid)[,-c(1,12,13)]
#   comp<-rbind(comp,compi)
# }
# comp<-comp[-1,]
# row.names(comp)<-c(1:dim(comp)[1])
# proc<-processing[1,c(3,4,6,7,13)]
# for (i in 1:dim(dt_direct)[1]){
#   proci<-subset(processing, BTO_proc_id==dt_direct$BTO_proc_id[i])[,c(3,4,6,7,13)]
#   proc<-rbind(proc,proci)
# }
# proc<-proc[-1,]
# row.names(proc)<-c(1:dim(proc)[1])
# direct<-cbind(dt_direct,comp,proc)
# 
# #* rearrange data *#
# direct<-direct[,c(12:25,1,2,4,26)]
# colnames(direct)[11]<-"calcinT"
# colnames(direct)[12]<-"calcint"
# colnames(direct)[13]<-"sinterT"
# colnames(direct)[14]<-"sintert"
# colnames(direct)[15]<-"E"
# colnames(direct)[16]<-"T_K"
# colnames(direct)[17]<-"ECs"
# direct$T_K<-round(direct$T_K,0)
# direct$E<-round(direct$E,1)
# direct$ECs<-round(direct$ECs,4)
# 
# # calculate features #
# #run electrocal-fea.R
# dirf<-fn.data.features(direct)
# scalecT<-(dirf$calcinT-min(dirf$calcinT,na.rm = T))/(max(dirf$calcinT,na.rm = T)-min(dirf$calcinT,na.rm = T))
# scalect<-(dirf$calcint-min(dirf$calcint,na.rm = T))/(max(dirf$calcint,na.rm = T)-min(dirf$calcint,na.rm = T))
# scalesT<-(dirf$sinterT-min(dirf$sinterT,na.rm = T))/(max(dirf$sinterT,na.rm = T)-min(dirf$sinterT,na.rm = T))
# scalest<-(dirf$sintert-min(dirf$sintert,na.rm = T))/(max(dirf$sintert,na.rm = T)-min(dirf$sintert,na.rm = T))
# dirf$cTpt<-scalecT+scalect
# dirf$cTmt<-scalecT*scalect
# dirf$cTdt<-dirf$calcinT/dirf$calcint
# dirf$sTpt<-scalesT+scalest
# dirf$sTmt<-scalesT*scalest
# dirf$sTdt<-dirf$sinterT/dirf$sintert
# 
# # features selection #
# # 5 best features, results of features_selection.R:
# # E, T, NCT, zeff, sTmt
# 
# #* data for model *#
# dirfs<-cbind(dirf[,c(1:10,15,16,17)],dirf[,c("NCT","zeff","sTmt","sinterT","sintert","reference_id")])
# 
# # rm.NA #
# for(i in 1:18){
#   dirfs<-subset(dirfs,!is.na(dirfs[,i]))
# }
# 
# # unique #
# dirfs<-anti_join(dirfs,dirfs[duplicated(dirfs[,-c(13,19)]),],by=colnames(dirfs)[-c(13,19)])
# 
# ##****************##
# 
# 
# 
# ##** bootstrap **##
# testsamp<-matrix(nrow=500,ncol=1609)
# for(i in 1:500){
#   set.seed(i)
#   testsamp[i,]<-sample(dim(dirfs)[1], dim(dirfs)[1], replace = TRUE)
# }
# duplicated(testsamp)
# 
# set.seed(1)
# samp1<-sample(1609,1609,replace = T)
# sampstat<-list(samp1[!duplicated(samp1)])
# for(i in 2:500){
#   set.seed(i)
#   sampi<-sample(1609,1609,replace = T)
#   sampstat[[i]]<-sampi[!duplicated(sampi)]
# }
# sampcount<-table(unlist(sampstat))
# samptime<-table(sampcount)
# write.csv(samptime,"samptime.csv")
#   
# #* GB *#
# # tune #
# #Input: parameters vector  Output: parameters, cve, r2
# gbpd<-function(paras){
#   gbr2<-c()
#   gbcv<-c()
#   for (i in 3:7){
#     set.seed(11+20*i)
#     gbdt<-try(gbm(ECs~., data = dirfs[,11:16], n.trees = paras[1],
#                   interaction.depth = paras[2], shrinkage = paras[3], cv.folds = 10))
#     if ('try-error' %in% class(gbdt)) {
#       gbr2<-c(gbr2,NA)
#     }else{
#       gbr2<-c(gbr2,1-sum((dirfs[,13]-as.numeric(gbdt$fit))^2)/sum((dirfs[,13]-mean(unlist(dirfs[,13])))^2))
#     }
#     dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),11:16] #random order
#     gbcve<-c()
#     for (j in 0:9){
#       dim10<-dim(dirfr)[1]%/%10
#       dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),]
#       dirfcv2<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#       gbcvm<-try(gbm(ECs~., data = dirfcv2, n.trees = paras[1],
#                      interaction.depth = paras[2], shrinkage = paras[3], cv.folds = 10))
#       if ('try-error' %in% class(gbcvm)) {
#         gbcve<-c(gbcve,NA)
#       }else{
#         gbcve<-c(gbcve,sum((dirfcv1[,3]-predict(gbcvm,dirfcv1))^2)/dim10)
#       }
#     }
#     gbcv<-c(gbcv,mean(gbcve, na.rm = T))
#   }
#   print(c(paras,mean(gbcv, na.rm = T),mean(gbr2, na.rm = T)))
#   write.table(c(paras,mean(gbcv, na.rm = T),mean(gbr2, na.rm = T)),"gbdtune.csv",append = TRUE,sep = ",")
#   return(c(paras,mean(gbcv, na.rm = T),mean(gbr2, na.rm = T)))
# }
# 
# gbin<-matrix(ncol=3)
# for(nt in c(500,1500,2000,3000,5000)){
#   for(id in c(1,4,5,8)){
#     for(sh in c(0.001,0.1))
#     gbin<-rbind(gbin,c(nt,id,sh))
#   }
# }
# gbin<-gbin[-1,]
# 
# gbin2<-matrix(ncol=3)
# for(nt in c(4000,5000,8000)){
#   for(id in c(8,10,20)){
#     for(sh in c(0.1))
#       gbin2<-rbind(gbin2,c(nt,id,sh))
#   }
# }
# gbin2<-gbin2[-1,]
# 
# #cluster initialization
# cores<-detectCores()
# cl<-makeCluster(detectCores(),outfile="gbdtune.txt")
# clusterExport(cl, list("gbpd","dirfs"))
# clusterEvalQ(cl,{library(gbm)})
# 
# system.time(
#   gbdtune2<-parApply(cl, gbin2, 1, gbpd)  #gbtune is a matrix
# )
# 
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# gbdbp<-gbdtune2[,which.min(gbdtune2[4,])]
# 
# save.image("3.RData")
# 
# # Input: seed(bootstrap number)  Output: GB predictions
# gbbt<-function(B){
#   set.seed(B)
#   dirfbt<-dirfs[sample(dim(dirfs)[1], dim(dirfs)[1], replace = TRUE),11:16]
#   dirfbt<-dirfbt[!duplicated(dirfbt),]
#   dirfbttest<-anti_join(dirfs,dirfbt)
#   btgbm<-gbm(ECs~., data = dirfbt, n.trees = 5000,
#              interaction.depth = 20, shrinkage = 0.1, cv.folds = 10)
#   prebttest<-predict(btgbm,dirfbttest)
#   pretestna<-c()
#   for(i in 1:dim(dirfs)[1]){
#     ind<-which(dirfbttest$E==dirfs$E[i]&dirfbttest$T_K==dirfs$T_K[i]&
#                  dirfbttest$zeff==dirfs$zeff[i]&dirfbttest$sTmt==dirfs$sTmt[i])
#     if(length(ind)==0){
#       pretestna<-c(pretestna,NA)
#     }else{
#       pretestna<-c(pretestna,prebttest[ind])
#     }
#   }
#   gc()
#   return(data.frame(pretestna,predict(btgbm,dirfs)))
# }
# 
# # B=500 #
# cl<-makeCluster(detectCores(),outfile="3_gbbt5.txt")
# clusterExport(cl, list("gbbt","dirfs"))
# clusterEvalQ(cl,{library(gbm);library(dplyr)})
# 
# system.time(
#   gbbtpre5<-parLapply(cl, 1:500, gbbt)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# gbbt5all<-gbbtpre5[[1]][,2]
# for (i in 2:length(gbbtpre5)){
#   gbbt5all<-rbind(gbbt5all,gbbtpre5[[i]][,2])
# }
# btvar5a<-apply(gbbt5all,2,var)
# 
# gbbt5test<-gbbtpre5[[1]][,1]
# for (i in 2:length(gbbtpre5)){
#   gbbt5test<-rbind(gbbt5test,gbbtpre5[[i]][,1])
# }
# btvar5t<-c()
# for(i in 1:dim(gbbt5test)[2]){
#   btvar5t<-c(btvar5t,var(gbbt5test[,i],na.rm=T))
# }
# 
# dirfs<-cbind(dirfs,btvar5a,btvar5t)
# 
# 
# # B=1000 #
# cl<-makeCluster(detectCores(),outfile="gbbt10.txt")
# clusterExport(cl, list("gbbt","dirfs"))
# clusterEvalQ(cl,{library(gbm)})
# 
# system.time(
#   gbbtpre10<-parLapply(cl, 1:1000, gbbt)
# )
# stopCluster(cl)
# 
# save.image("2.RData")
# 
# gbbt10all<-gbbtpre10[[1]]
# for (i in 2:length(gbbtpre10)){
#   gbbt10all<-rbind(gbbt10all,gbbtpre10[[i]])
# }
# btvar10<-apply(gbbt10all,2,var)
# 
# dirfs<-cbind(dirfs,btvar10)
# 
# # B=200 #
# cl<-makeCluster(detectCores(),outfile="gbbt2.txt")
# clusterExport(cl, list("gbbt","dirfs"))
# clusterEvalQ(cl,{library(gbm)})
# 
# system.time(
#   gbbtpre2<-parLapply(cl, 1:200, gbbt)
# )
# stopCluster(cl)
# 
# save.image("2.RData")
# 
# gbbt2all<-gbbtpre2[[1]]
# for (i in 2:length(gbbtpre2)){
#   gbbt2all<-rbind(gbbt2all,gbbtpre2[[i]])
# }
# btvar2<-apply(gbbt2all,2,var)
# 
# dirfs<-cbind(dirfs,btvar2)
# 
# 
# #* SVR.rbf *#
# # tune #
# # Input: cost and gamma  Output: CVE and R2
# svrpd<-function(cg){
#   svrr2<-c()
#   svrcv<-c()
#   for (i in 3:7){
#     set.seed(11+20*i)
#     svrm<-try(svm(ECs~.,data=dirfs[,11:16],type="eps-regression",kernel="radial",
#                   cost=cg[1],gamma=cg[2]))
#     if ('try-error' %in% class(svrm)) {
#       svrr2<-c(svrr2,NA)
#     }else{
#       svrr2<-c(svrr2,1-sum((predict(svrm,dirfs)-dirfs[,13])^2)/sum((dirfs[,13]-mean(unlist(dirfs[,13])))^2))
#     }
#     dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),11:16] #random order
#     svrcve<-c()
#     for (j in 0:9){
#       dim10<-dim(dirfr)[1]%/%10
#       dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),]
#       dirfcv2<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#       svrcvm<-try(svm(ECs~.,data=dirfcv2,type="eps-regression",kernel="radial",
#                       cost=cg[1],gamma=cg[2]))
#       if ('try-error' %in% class(svrcvm)) {
#         svrcve<-c(svrcve,NA)
#       }else{
#         svrcve<-c(svrcve,sum((dirfcv1[,3]-predict(svrcvm,dirfcv1))^2)/dim10)
#       }
#     }
#     svrcv<-c(svrcv,mean(svrcve, na.rm = T))
#   }
#   print(c(cg,mean(svrcv, na.rm = T),mean(svrr2, na.rm = T)))
#   write.table(c(cg,mean(svrcv, na.rm = T),mean(svrr2, na.rm = T)),"svrdtune.csv",append = TRUE,sep = ",")
#   return(c(cg,mean(svrcv, na.rm = T),mean(svrr2, na.rm = T)))
# }
# 
# #Input of parallel
# svrin<-matrix(ncol=2)
# for(c in c(10,50,100,200,500)){
#   for(g in c(0.001,0.1,1,10,20)){
#     svrin<-rbind(svrin,c(c,g))
#   }
# }
# svrin<-svrin[-1,]
# 
# svrin2<-matrix(ncol=2)
# for(c in c(30,50,70)){
#   for(g in c(0.1,0.5,1,1.5,2,5,10)){
#     svrin2<-rbind(svrin2,c(c,g))
#   }
# }
# svrin2<-svrin2[-1,]
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="svrdtune2.txt")
# clusterExport(cl, list("svrpd","dirfs"))
# clusterEvalQ(cl,{library(e1071)})
# 
# system.time(
#   svrdtune2<-parApply(cl, svrin2, 1, svrpd)
# )
# 
# stopCluster(cl)
# save.image("3.RData")
# svrdbp<-svrdtune2[,which.min(svrdtune2[3,])]
# save.image("3.RData")
# 
# # Input: seed(bootstrap number)  Output: SVR.r predictions
# svrbt<-function(B){
#   set.seed(B)
#   dirfbt<-dirfs[sample(dim(dirfs)[1], dim(dirfs)[1], replace = TRUE),11:16]
#   dirfbt<-dirfbt[!duplicated(dirfbt),]
#   dirfbttest<-anti_join(dirfs,dirfbt)
#   btsvrm<-svm(ECs~.,data=dirfbt,type="eps-regression",kernel="radial",
#               cost=50,gamma=2)
#   prebttest<-predict(btsvrm,dirfbttest)
#   pretestna<-c()
#   for(i in 1:dim(dirfs)[1]){
#     ind<-which(dirfbttest$E==dirfs$E[i]&dirfbttest$T_K==dirfs$T_K[i]&
#                  dirfbttest$zeff==dirfs$zeff[i]&dirfbttest$sTmt==dirfs$sTmt[i])
#     if(length(ind)==0){
#       pretestna<-c(pretestna,NA)
#     }else{
#       pretestna<-c(pretestna,prebttest[ind])
#     }
#   }
#   gc()
#   return(data.frame(pretestna,predict(btsvrm,dirfs)))
# }
# 
# # B=500 #
# cl<-makeCluster(detectCores(),outfile="3svrbt5.txt")
# clusterExport(cl, list("svrbt","dirfs"))
# clusterEvalQ(cl,{library(e1071);library(dplyr)})
# 
# system.time(
#   svrbtpre5<-parLapply(cl, 1:500, svrbt)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# svrbt5all<-svrbtpre5[[1]][,2]
# for (i in 2:length(svrbtpre5)){
#   svrbt5all<-rbind(svrbt5all,svrbtpre5[[i]][,2])
# }
# svrvar5a<-apply(svrbt5all,2,var)
# 
# svrbt5test<-svrbtpre5[[1]][,1]
# for (i in 2:length(svrbtpre5)){
#   svrbt5test<-rbind(svrbt5test,svrbtpre5[[i]][,1])
# }
# svrvar5t<-c()
# for(i in 1:dim(svrbt5test)[2]){
#   svrvar5t<-c(svrvar5t,var(svrbt5test[,i],na.rm=T))
# }
# 
# dirfs<-cbind(dirfs,svrvar5a,svrvar5t)
# 
# 
# # B=1000 #
# cl<-makeCluster(detectCores(),outfile="svrbt10.txt")
# clusterExport(cl, list("svrbt","dirfs"))
# clusterEvalQ(cl,{library(svrm)})
# 
# system.time(
#   svrbtpre10<-parLapply(cl, 1:1000, svrbt)
# )
# stopCluster(cl)
# 
# save.image("2.RData")
# 
# svrbt10all<-svrbtpre10[[1]]
# for (i in 2:length(svrbtpre10)){
#   svrbt10all<-rbind(svrbt10all,svrbtpre10[[i]])
# }
# svrvar10<-apply(svrbt10all,2,var)
# 
# dirfs<-cbind(dirfs,svrvar10)
# 
# # B=200 #
# cl<-makeCluster(detectCores(),outfile="svrbt2.txt")
# clusterExport(cl, list("svrbt","dirfs"))
# clusterEvalQ(cl,{library(svrm)})
# 
# system.time(
#   svrbtpre2<-parLapply(cl, 1:200, svrbt)
# )
# stopCluster(cl)
# 
# save.image("2.RData")
# 
# svrbt2all<-svrbtpre2[[1]]
# for (i in 2:length(svrbtpre2)){
#   svrbt2all<-rbind(svrbt2all,svrbtpre2[[i]])
# }
# svrvar2<-apply(svrbt2all,2,var)
# 
# dirfs<-cbind(dirfs,svrvar2)
# 
# 
# #* RF *#
# # tune #
# # Input: ntree and mtree  Output: CVE and R2
# rfpd<-function(nm){
#   rfr2<-c()
#   rfcv<-c()
#   for (i in 3:7){
#     set.seed(11+20*i)
#     rfm<-try(randomForest(ECs~.,data=dirfs[,11:16],ntree=nm[1],mtry=nm[2]))
#     if ('try-error' %in% class(rfm)) {
#       rfr2<-c(rfr2,NA)
#     }else{
#       rfr2<-c(rfr2,1-sum((rfm$y-rfm$predicted)^2)/sum((dirfs[,13]-mean(unlist(dirfs[,13])))^2))
#     }
#     dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),11:16] #random order
#     rfcve<-c()
#     for (j in 0:9){
#       dim10<-dim(dirfr)[1]%/%10
#       dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),]
#       dirfcv2<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#       rfcvm<-try(randomForest(ECs~.,data=dirfcv2,ntree=nm[1],mtry=nm[2]))
#       if ('try-error' %in% class(rfcvm)) {
#         rfcve<-c(rfcve,NA)
#       }else{
#         rfcve<-c(rfcve,sum((dirfcv1[,3]-predict(rfcvm,dirfcv1))^2)/dim10)
#       }
#     }
#     rfcv<-c(rfcv,mean(rfcve, na.rm = T))
#   }
#   print(c(nm,mean(rfcv, na.rm = T),mean(rfr2, na.rm = T)))
#   write.table(c(nm,mean(rfcv, na.rm = T),mean(rfr2, na.rm = T)),"rfdtune.csv",append = TRUE,sep = ",")
#   return(c(nm,mean(rfcv, na.rm = T),mean(rfr2, na.rm = T)))
# }
# 
# rfin<-matrix(ncol=2)
# for(nt in c(200,500,1000,1200,1500,1800,2000,5000)){
#   for(mt in c(2,4,5)){
#     rfin<-rbind(rfin,c(nt,mt))
#   }
# }
# rfin<-rfin[-1,]
# 
# #cluster initialization
# cores<-detectCores()
# cl<-makeCluster(detectCores(),outfile="rfdtune.txt")
# clusterExport(cl, list("rfpd","dirfs"))
# clusterEvalQ(cl,{library(randomForest)})
# 
# system.time(
#   rfdtune<-parApply(cl, rfin, 1, rfpd)
# )
# 
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# rfdpall<-t(rfdtune[,1])
# for (i in 2:dim(rfdtune)[2]){
#   rfdpall<-rbind(rfdpall,t(rfdtune[,i]))
# }
# rfdbp<-as.numeric(rfdpall[which(rfdpall[,3]==min(rfdpall[,3])),])
# #tuned result rfdbp=c(1200,4,0.000841,0.95814)
# save.image("3.RData")
# 
# # Input: seed(bootstrap number)  Output: rf.r predictions
# rfbt<-function(B){
#   set.seed(B)
#   dirfbt<-dirfs[sample(dim(dirfs)[1], dim(dirfs)[1], replace = TRUE),11:16]
#   dirfbt<-dirfbt[!duplicated(dirfbt),]
#   dirfbttest<-anti_join(dirfs,dirfbt)
#   btrfm<-randomForest(ECs~.,data=dirfbt,ntree=1200,mtry=4)
#   prebttest<-predict(btrfm,dirfbttest)
#   pretestna<-c()
#   for(i in 1:dim(dirfs)[1]){
#     ind<-which(dirfbttest$E==dirfs$E[i]&dirfbttest$T_K==dirfs$T_K[i]&
#                  dirfbttest$zeff==dirfs$zeff[i]&dirfbttest$sTmt==dirfs$sTmt[i])
#     if(length(ind)==0){
#       pretestna<-c(pretestna,NA)
#     }else{
#       pretestna<-c(pretestna,prebttest[ind])
#     }
#   }
#   gc()
#   return(data.frame(pretestna,predict(btrfm,dirfs)))
# }
# 
# # B=500 #
# cl<-makeCluster(detectCores(),outfile="3rfbt5.txt")
# clusterExport(cl, list("rfbt","dirfs"))
# clusterEvalQ(cl,{library(randomForest);library(dplyr)})
# 
# system.time(
#   rfbtpre5<-parLapply(cl, 1:500, rfbt)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# rfbt5all<-rfbtpre5[[1]][,2]
# for (i in 2:length(rfbtpre5)){
#   rfbt5all<-rbind(rfbt5all,rfbtpre5[[i]][,2])
# }
# rfvar5a<-apply(rfbt5all,2,var)
# 
# rfbt5test<-rfbtpre5[[1]][,1]
# for (i in 2:length(rfbtpre5)){
#   rfbt5test<-rbind(rfbt5test,rfbtpre5[[i]][,1])
# }
# rfvar5t<-c()
# for(i in 1:dim(rfbt5test)[2]){
#   rfvar5t<-c(rfvar5t,var(rfbt5test[,i],na.rm=T))
# }
# 
# rfvar5a<-as.numeric(unlist(read.csv("rfvar5a.csv",header=T)))
# rfvar5t<-as.numeric(unlist(read.csv("rfvar5t.csv",header=T)))
# 
# dirfs<-cbind(dirfs,rfvar5a,rfvar5t)
# 
# rfmd<-randomForest(ECs~.,data=dirfs[,11:16],ntree=1200,mtry=4)
# rfpvar<-apply(predict(rfmd,dirfs[,11:16],predict.all=T)[["individual"]],1,var)
# dirfs<-cbind(dirfs,rfpvar)
# 
# # plot distribution of var estimated by different models #
# pmvar<-data.frame(V1=c(rep("GB bootstrap",1609),rep("SVR bootstrap",1609),
#                        rep("RF bootstrap",1609)),
#                   V2=c(btvar5t,svrvar5t,rfvar5t))
# pmvar$V1<-ordered(pmvar$V1,levels = c("GB bootstrap","SVR bootstrap",
#                                       "RF bootstrap"))
# ggplot(pmvar, aes(x=V1, y=V2)) +
#   stat_boxplot(geom="errorbar",width=0.3,size=1,position=position_dodge(0.6),color="grey")+
#   geom_boxplot(aes(fill=V1),size=1,alpha=0.5,position=position_dodge(0.8),width=0.8,
#                color="grey",outlier.colour = "grey")+
#   ylim(0,0.0012)+
#   labs(y="Estimated variances",x="")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"))
# 
# # plot var vs ref #
# ggplot(dirfs, aes(x=as.ordered(reference_id), y=rfvar5t)) +
#   stat_boxplot(geom="errorbar",width=0.3,size=1,position=position_dodge(0.6),color="grey")+
#   geom_boxplot(aes(fill=as.ordered(reference_id)),size=1,alpha=0.5,position=position_dodge(0.8),width=0.8,
#                color="grey",outlier.colour = "grey")+
#   #ylim(0,0.00069)+
#   labs(y="Estimated variances",x="Reference ID")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         #axis.text.x = element_text(angle = 25,hjust = 1),
#         axis.title=element_text(face="bold"))
# 
# # plot var vs data size of ref #
# refds<-c()
# for(i in 3:28){
#   dirfsp<-dirfs[which(dirfs$reference_id==i),]
#   refds<-c(refds,rep(dim(dirfsp)[1],dim(dirfsp)[1]))
# }
# dirfs<-cbind(dirfs,refds)
# ggplot(dirfs, aes(x=refds, y=rfvar5t)) +
#   stat_boxplot(aes(group=refds),geom="errorbar",width=1.6,size=1.2,position=position_dodge(0.6),color="grey")+
#   geom_boxplot(aes(group=refds,fill=as.ordered(refds)),size=0.5,alpha=0.5,position=position_dodge(0.8),width=1.2,
#                color="grey",outlier.colour = "grey")+
#   #ylim(0,3200)+
#   labs(y="Estimated variances",x="Data size of reference")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         #axis.text.x = element_text(angle = 25,hjust = 1),
#         axis.title=element_text(face="bold"))
# 
# #MIC
# library(minerva)
# vrmic<-mine(x=dirfs[,c("rfvar5a","reference_id")],normalization=T,var.thr=1e-10,use="pairwise.complete.obs")$MIC
# 
# # ggplot(dirfs, aes(x=as.ordered(dirfs$reference_id), y=ECs)) +
# #   geom_violin(aes(linetype=NA,fill=as.ordered(dirfs$reference_id)),alpha=0.5,position=position_dodge(0.8),width=1)+
# #   #coord_flip()+
# #   #ylim(0,0.001)+
# #   labs(y="ECs",x="Reference_id")+
# #   theme_bw(base_size = 20)+
# #   theme(legend.position = "none",panel.grid.major = element_blank(),
# #         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
# #         axis.title=element_text(face="bold"))
# ##*********##
# 
# 
# 
# ##** test method estimate var **##
# test<-matrix(nrow=500,ncol=1009)
# for(i in 1:500){
#   set.seed(i)
#   test[i,]<-sample(dim(dirfs)[1], 1009, replace = F)
# }
# duplicated(test)
# 
# #* GB *#
# # Input: seed(bootstrap number)  Output: GB predictions
# gbtv<-function(B){
#   set.seed(B)
#   dirftv<-dirfs[sample(dim(dirfs)[1], 1009, replace = F),11:16]
#   dirftvtest<-anti_join(dirfs,dirftv)
#   tvgbm<-gbm(ECs~., data = dirftv, n.trees = 5000,
#              interaction.depth = 20, shrinkage = 0.1, cv.folds = 10)
#   pretvtest<-predict(tvgbm,dirftvtest)
#   pretestna<-c()
#   for(i in 1:dim(dirfs)[1]){
#     ind<-which(dirftvtest$E==dirfs$E[i]&dirftvtest$T_K==dirfs$T_K[i]&
#                  dirftvtest$zeff==dirfs$zeff[i]&dirftvtest$sTmt==dirfs$sTmt[i])
#     if(length(ind)==0){
#       pretestna<-c(pretestna,NA)
#     }else{
#       pretestna<-c(pretestna,pretvtest[ind])
#     }
#   }
#   gc()
#   return(pretestna)
# }
# 
# # B=500 #
# cl<-makeCluster(detectCores(),outfile="3_gbtv5.txt")
# clusterExport(cl, list("gbtv","dirfs"))
# clusterEvalQ(cl,{library(gbm);library(dplyr)})
# 
# system.time(
#   gbtvpre5<-parLapply(cl, 1:500, gbtv)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# gbtv5test<-as.matrix(gbtvpre5[[1]])
# for (i in 2:length(gbtvpre5)){
#   gbtv5test<-cbind(gbtv5test,gbtvpre5[[i]])
# }
# tvgbvar5<-c()
# for(i in 1:dim(gbtv5test)[1]){
#   tvgbvar5<-c(tvgbvar5,var(gbtv5test[i,],na.rm=T))
# }
# 
# dirfs<-cbind(dirfs,tvgbvar5)
# 
# #* SVR.rbf *#
# # Input: seed(bootstrap number)  Output: SVR.r predictions
# svrtv<-function(B){
#   set.seed(B)
#   dirftv<-dirfs[sample(dim(dirfs)[1], 1009, replace = F),11:16]
#   dirftvtest<-anti_join(dirfs,dirftv)
#   tvsvrm<-svm(ECs~.,data=dirftv,type="eps-regression",kernel="radial",
#               cost=50,gamma=2)
#   pretvtest<-predict(tvsvrm,dirftvtest)
#   pretestna<-c()
#   for(i in 1:dim(dirfs)[1]){
#     ind<-which(dirftvtest$E==dirfs$E[i]&dirftvtest$T_K==dirfs$T_K[i]&
#                  dirftvtest$zeff==dirfs$zeff[i]&dirftvtest$sTmt==dirfs$sTmt[i])
#     if(length(ind)==0){
#       pretestna<-c(pretestna,NA)
#     }else{
#       pretestna<-c(pretestna,pretvtest[ind])
#     }
#   }
#   gc()
#   return(pretestna)
# }
# 
# # B=500 #
# cl<-makeCluster(detectCores(),outfile="3svrtv5.txt")
# clusterExport(cl, list("svrtv","dirfs"))
# clusterEvalQ(cl,{library(e1071);library(dplyr)})
# 
# system.time(
#   svrtvpre5<-parLapply(cl, 1:500, svrtv)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# svrtv5test<-as.matrix(svrtvpre5[[1]])
# for (i in 2:length(svrtvpre5)){
#   svrtv5test<-cbind(svrtv5test,svrtvpre5[[i]])
# }
# tvsvrvar5<-c()
# for(i in 1:dim(svrtv5test)[1]){
#   tvsvrvar5<-c(tvsvrvar5,var(svrtv5test[i,],na.rm=T))
# }
# 
# dirfs<-cbind(dirfs,tvsvrvar5)
# 
# 
# #* RF *#
# # Input: seed(bootstrap number) and sample train size  Output: rf.r predictions
# rftv<-function(Bts){
#   set.seed(Bts[1])
#   dirftv<-dirfs[sample(dim(dirfs)[1], Bts[2], replace = F),11:16]
#   dirftvtest<-anti_join(dirfs,dirftv)
#   tvrfm<-randomForest(ECs~.,data=dirftv,ntree=1200,mtry=4)
#   pretvtest<-predict(tvrfm,dirftvtest)
#   pretestna<-c()
#   for(i in 1:dim(dirfs)[1]){
#     ind<-which(dirftvtest$E==dirfs$E[i]&dirftvtest$T_K==dirfs$T_K[i]&
#                  dirftvtest$zeff==dirfs$zeff[i]&dirftvtest$sTmt==dirfs$sTmt[i])
#     if(length(ind)==0){
#       pretestna<-c(pretestna,NA)
#     }else{
#       pretestna<-c(pretestna,pretvtest[ind])
#     }
#   }
#   gc()
#   return(pretestna)
# }
# 
# # B=500 #
# Btsin<-matrix(ncol=2)
# for(ts in c(809,1009,1209)){
#   for(B in 1:500){
#     Btsin<-rbind(Btsin,c(B,ts))
#   }
# }
# Btsin<-Btsin[-1,]
# 
# Btsin2<-matrix(ncol=2)
# for(ts in c(609,1409,1509)){
#   for(B in 1:500){
#     Btsin2<-rbind(Btsin2,c(B,ts))
#   }
# }
# Btsin2<-Btsin2[-1,]
# 
# Btsin3<-matrix(ncol=2)
# for(ts in c(1559,1589)){
#   for(B in 1:500){
#     Btsin3<-rbind(Btsin3,c(B,ts))
#   }
# }
# Btsin3<-Btsin3[-1,]
# 
# cl<-makeCluster(detectCores(),outfile="3rftv5.txt")
# clusterExport(cl, list("rftv","dirfs"))
# clusterEvalQ(cl,{library(randomForest);library(dplyr)})
# 
# system.time(
#   rftvpre5_3<-parApply(cl, Btsin3, 1, rftv)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# tv8rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv8rfvar5<-c(tv8rfvar5,var(rftvpre5[i,1:500],na.rm=T))
# }
# tv10rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv10rfvar5<-c(tv10rfvar5,var(rftvpre5[i,501:1000],na.rm=T))
# }
# tv12rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv12rfvar5<-c(tv12rfvar5,var(rftvpre5[i,1001:1500],na.rm=T))
# }
# tv6rfvar5<-c()
# for(i in 1:dim(rftvpre5_2)[1]){
#   tv6rfvar5<-c(tv6rfvar5,var(rftvpre5_2[i,1:500],na.rm=T))
# }
# tv14rfvar5<-c()
# for(i in 1:dim(rftvpre5_2)[1]){
#   tv14rfvar5<-c(tv14rfvar5,var(rftvpre5_2[i,501:1000],na.rm=T))
# }
# tv15rfvar5<-c()
# for(i in 1:dim(rftvpre5_2)[1]){
#   tv15rfvar5<-c(tv15rfvar5,var(rftvpre5_2[i,1001:1500],na.rm=T))
# }
# tv155rfvar5<-c()
# for(i in 1:dim(rftvpre5_3)[1]){
#   tv155rfvar5<-c(tv155rfvar5,var(rftvpre5_3[i,1:500],na.rm=T))
# }
# tv158rfvar5<-c()
# for(i in 1:dim(rftvpre5_3)[1]){
#   tv158rfvar5<-c(tv158rfvar5,var(rftvpre5_3[i,501:1000],na.rm=T))
# }
# 
# tv15rfvar5m<-c()
# for(i in 1:28){
#   tv15rfvar5p<-dirfs[which(dirfs$reference_id==i),]$tv15rfvar5
#   tv15rfvar5m<-c(tv15rfvar5m,rep(mean(tv15rfvar5p),length(tv15rfvar5p)))
# }
# 
# dirfs<-cbind(dirfs,tv8rfvar5,tv10rfvar5,tv12rfvar5)
# dirfs<-cbind(dirfs,tv6rfvar5,tv14rfvar5,tv15rfvar5)
# dirfs<-cbind(dirfs,tv155rfvar5)
# dirfs<-cbind(dirfs,tv15rfvar5m)
# 
# #kriging estimated nugget
# knug<-krigm(dirfs[,11:16],t,p,0,64)@covariance@nugget
# 
# # plot distribution of var estimated by different models #
# pmvar<-data.frame(V1=c(rep("RF bootstrap",1609),rep("1000 RF test",1609),
#                        rep("800 RF test",1609),rep("600 RF test",1609),
#                        rep("400 RF test",1609),rep("200 RF test",1609),
#                        rep("100 RF test",1609),rep("50 RF test",1609),"estimated nugget"),
#                   V2=c(rfvar5t,tv6rfvar5,tv8rfvar5,tv10rfvar5,
#                        tv12rfvar5,tv14rfvar5,tv15rfvar5,tv155rfvar5,knug))
# pmvar$V1<-ordered(pmvar$V1,levels = c("RF bootstrap","1000 RF test",
#                                       "800 RF test","600 RF test",
#                                       "400 RF test","200 RF test",
#                                       "100 RF test","50 RF test","estimated nugget"))
# ggplot(pmvar, aes(x=V1, y=V2)) +
#   stat_boxplot(geom="errorbar",width=0.3,size=1,position=position_dodge(0.6),color="grey")+
#   geom_boxplot(aes(fill=V1),size=1,alpha=0.5,position=position_dodge(0.8),width=0.8,
#                color="grey",outlier.colour = "grey")+
#   ylim(0,0.0008)+
#   labs(y="Estimated variances",x="")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.text.x = element_text(angle = 25,hjust = 1),
#         axis.title=element_text(face="bold"))
# 
# 
# # plot test var vs boots test
# plot(x=btvar5t,y=tvgbvar5,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0,0.01), ylim = c(0,0.01),
#      xlab="Bootstrap test variances",ylab="Test variances",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="GB base learner",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# 
# plot(x=svrvar5t,y=tvsvrvar5,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0,0.036), ylim = c(0,0.036),
#      xlab="Bootstrap test variances",ylab="Test variances",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="SVR base learner",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# 
# plot(x=rfvar5t,y=tvrfvar5,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0,0.005), ylim = c(0,0.005),
#      xlab="Bootstrap test variances",ylab="Test variances",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="RF base learner",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# 
# ##**********##
# 
# 
# 
# ##** kriging model without noise **##
# # kriging model without noise #
# # Input: data (ECs at col 3), scale of θ, p
# krigmo<-function(dat,t,p,sseed){
#   set.seed(sseed)
#   for(n in c(1e-17,1e-15,1e-12,1e-10,1e-8)){
#     cvhkm<-try(km(formula = ~1,design = dat[,-3],
#                   response = dat[,3],covtype = "powexp",
#                   control = list(trace=F),
#                   #lower = c(t,1e-10),
#                   nugget = n))
#     if ('try-error' %in% class(cvhkm)) {
#       next()
#     }else{
#       break()
#     }
#   }
#   return(cvhkm)
# }
# 
# #* repeat seed 10-fold CVE(MAE) *#
# krigocve<-function(i){
#   krigcv<-c()
#   set.seed(11+20*i)
#   dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),11:16] #random order
#   for (j in 0:9){
#     dim10<-dim(dirfr)[1]%/%10
#     dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),]
#     dirfcv2<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigmo(dirfcv2,t,p,64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcv<-c(krigcv,NA)
#     }else{
#       krigcv<-c(krigcv,sum(abs(dirfcv1[,3]-predict(krigcvm,dirfcv1[,-3],type="SK")$mean))/dim10)
#     }
#     #}
#   }
#   print(c(i,krigcv))
#   return(c(i,krigcv))
# }
# 
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="3krigocvs.txt")
# clusterExport(cl, list("krigocve","krigmo","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krigocvs<-parLapply(cl, 1:100, krigocve)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# krigocvsm<-c()
# for(i in 1:length(krigocvs)){
#   krigocvsm<-c(krigocvsm,mean(krigocvs[[i]][-1]))
# }
# 
kcvin<-matrix(ncol = 2)
for(i in 1:100){
  for(j in 0:9){
    kcvin<-rbind(kcvin,c(i,j))
  }
}
kcvin<-kcvin[-1,]

# return all results
krigocve2<-function(ij){    #3:j为折数0:9，放在cvin里
  set.seed(11+20*ij[1])
  dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),11:16] #random order
  dim10<-dim(dirfr)[1]%/%10
  dirfcv1<-dirfr[(1+dim10*ij[2]):(dim10*(ij[2]+1)),]
  dirfcv2<-dirfr[-((1+dim10*ij[2]):(dim10*(ij[2]+1))),]
  krigcvm<-try(krigmo(dirfcv2,t,p,64))   #sseed可替换为8*k
  if ('try-error' %in% class(krigcvm)) {
    prete<-rep(NA,dim(dirfcv1)[1]);pretr<-rep(NA,dim(dirfcv2)[1])
    preteva<-rep(NA,dim(dirfcv1)[1]);pretrva<-rep(NA,dim(dirfcv2)[1])
  }else{
    pretr<-predict(krigcvm,dirfcv2[,-3],type="SK")$mean
    pretrva<-predict(krigcvm,dirfcv2[,-3],type="SK")$sd
    prete<-predict(krigcvm,dirfcv1[,-3],type="SK")$mean
    preteva<-predict(krigcvm,dirfcv1[,-3],type="SK")$sd
  }
  res<-list(data.frame(dirfcv1[,3],prete,preteva),data.frame(dirfcv2[,3],pretr,pretrva))
  fnm1<-paste("cv10o/cv10pre3_", ij[1], "_", ij[2], ".csv", sep = "")  #3包括parameters
  fnm2<-paste("cv10o/cv10ptr3_", ij[1], "_", ij[2], ".csv", sep = "")
  write.table(res[[1]], fnm1, append = TRUE, sep = ",")
  write.table(res[[2]], fnm2, append = TRUE, sep = ",")
  return(res)
}

#cluster initialization
cl<-makeCluster(detectCores(),outfile="3krigocvs.txt")
clusterExport(cl, list("krigmo","t","p","dirfs"))
clusterEvalQ(cl,{library(DiceKriging)})

system.time(
  krigocvs2<-parApply(cl, kcvin, 1, krigocve2)
)
stopCluster(cl)

save.image("3.RData")

cv10omae<-c(); cv10or2<-c()
for(i in 1:100){
  te<-c(); pte<-c()
  for(j in 1:10){
    te<-c(te,krigocvs2[[((i-1)*10+j)]][[1]][,1])
    pte<-c(pte,krigocvs2[[((i-1)*10+j)]][[1]][,2])
  }
  ptemae<-mean(abs(pte - te),na.rm=T)
  pter2<-round(1-sum((pte-te)^2,na.rm=T)/sum((te-mean(te,na.rm=T))^2,na.rm=T),3)
  cv10omae<-c(cv10omae,ptemae)
  cv10or2<-c(cv10or2,pter2)
}
cv10or2s<-c()
for(i in 1:1000){
  te<-krigocvs2[[i]][[1]][,1]
  pte<-krigocvs2[[i]][[1]][,2]
  pter2<-round(1-sum((pte-te)^2,na.rm=T)/sum((te-mean(te,na.rm=T))^2,na.rm=T),3)
  cv10or2s<-c(cv10or2s,pter2)
}
# ##***********##
# 
# 
# 
# ##** kriging model with nugget **##
# # Kriging model with nugget or noise #
# # Input: data (ECs at col 3), scale of θ, p, noise (#33 or nugget)
# krigm<-function(dat,t,p,n,sseed){
#   set.seed(sseed)
#   if(length(n)==1){
#     cvhkm<-km(formula = ~1,design = dat[,-3],
#           response = dat[,3],covtype = "powexp",
#           control = list(trace=F),
#           #lower = c(t,1e-10),
#           nugget.estim = T)
#   }else{
#     cvhkm<-km(formula = ~1,design = dat[,-3],
#               response = dat[,3],covtype = "powexp",
#               control = list(trace=F),
#               #lower = c(t,1e-10),
#               noise.var = n)
#   }
#   return(cvhkm)
# }
# 
# #* one model for test *#
# #随机选300个作测试集，diagonal plot
# set.seed(57)
# test.dirfs<-dirfs[sample(1:dim(dirfs)[1],300),]
# tt.dirfs<-anti_join(dirfs, test.dirfs)
# test.ECs<-test.dirfs[,13]
# 
# tkrig<-krigm(tt.dirfs[,11:16],t,p,0,567)
# tprekrig<-predict(tkrig,test.dirfs[,c(11,12,14,15,16)],type="SK")$mean
# plot(x=test.ECs,y=tprekrig,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0,0.55), ylim = c(0,0.55),
#      xlab="Measured ECs (1e-6K▪m/V)",ylab="Predicted ECs (1e-6K▪m/V)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="Kriging on the test set",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# tkrigr2<-1-sum((test.ECs-tprekrig)^2)/sum((test.ECs-mean(test.ECs))^2)
# 
# 
# #* repeat seed 10-fold CVE(MAE) *#
# krigcve<-function(i){
#   krigcv<-c()
#   set.seed(11+20*i)
#   dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),11:16] #random order
#   for (j in 0:9){
#     dim10<-dim(dirfr)[1]%/%10
#     dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),]
#     dirfcv2<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#       krigcvm<-try(krigm(dirfcv2,t,p,0,64))   #sseed可替换为8*k
#       if ('try-error' %in% class(krigcvm)) {
#         krigcv<-c(krigcv,NA)
#       }else{
#         krigcv<-c(krigcv,sum(abs(dirfcv1[,3]-predict(krigcvm,dirfcv1[,-3],type="SK")$mean))/dim10)
#       }
#     #}
#   }
#   print(c(i,krigcv))
#   return(c(i,krigcv))
# }
# 
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="3krigcvs.txt")
# clusterExport(cl, list("krigcve","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krigcvs<-parLapply(cl, 1:100, krigcve)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# krigcvsm<-c()
# for(i in 1:length(krigcvs)){
#   krigcvsm<-c(krigcvsm,mean(krigcvs[[i]][-1]))
# }
# 
# 
# ##***********##
# 
# 
# 
# ##** kriging model with noise **##
# #* one model for test *#
# # with var by GB with compositions #
# tkrign<-krigm(tt.dirfs[,11:16],t,p,tt.dirfs$btvar,567)
# tprekrign<-predict(tkrign,test.dirfs[,c(11,12,14,15,16)],type="SK")$mean
# plot(x=test.ECs,y=tprekrign,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0,0.55), ylim = c(0,0.55),
#      xlab="Measured ECs (1e-6K▪m/V)",ylab="Predicted ECs (1e-6K▪m/V)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="Noisy kriging on the test set",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# tkrignr2<-1-sum((test.ECs-tprekrign)^2)/sum((test.ECs-mean(test.ECs))^2)
# 
# # with var by GB #
# tkrign5<-krigm(tt.dirfs[,11:15],t,p,tt.dirfs$btvar5,567)
# tprekrign5<-predict(tkrign5,test.dirfs[,c(11,12,14,15)],type="SK")$mean
# plot(x=test.ECs,y=tprekrign5,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0,0.55), ylim = c(0,0.55),
#      xlab="Measured ECs (1e-6K▪m/V)",ylab="Predicted ECs (1e-6K▪m/V)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="Noisy kriging (GB) on the test set",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# tkrign5r2<-1-sum((test.ECs-tprekrign5)^2)/sum((test.ECs-mean(test.ECs))^2)
# 
# # with var by SVR #
# tkrignsv<-krigm(tt.dirfs[,11:15],t,p,tt.dirfs$svrvar,567)
# tprekrignsv<-predict(tkrignsv,test.dirfs[,c(11,12,14,15)],type="SK")$mean
# plot(x=test.ECs,y=tprekrignsv,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0,0.55), ylim = c(0,0.55),
#      xlab="Measured ECs (1e-6K▪m/V)",ylab="Predicted ECs (1e-6K▪m/V)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="Noisy kriging (SVR) on the test set",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# tkrignsvr2<-1-sum((test.ECs-tprekrignsv)^2)/sum((test.ECs-mean(test.ECs))^2)
# 
# 
# #* var models' performances *#
# #GB
# tgb<-gbm(ECs~., data = tt.dirfs[,11:16], n.trees = 5000,
#          interaction.depth = 20, shrinkage = 0.1, cv.folds = 10)
# tpregb<-predict(tgb,test.dirfs[,11:16])
# plot(x=test.ECs,y=tpregb,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0,0.55), ylim = c(0,0.55),
#      xlab="Measured ECs (1e-6K▪m/V)",ylab="Predicted ECs (1e-6K▪m/V)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="GB on the test set",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# tgbr2<-1-sum((test.ECs-tpregb)^2)/sum((test.ECs-mean(test.ECs))^2)
# 
# #SVR
# tsv<-svm(ECs~.,data=tt.dirfs[,11:16],type="eps-regression",kernel="radial",
#          cost=50,gamma=2)
# tpresv<-predict(tsv,test.dirfs[,11:16])
# plot(x=test.ECs,y=tpresv,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0,0.55), ylim = c(0,0.55),
#      xlab="Measured ECs (1e-6K▪m/V)",ylab="Predicted ECs (1e-6K▪m/V)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="SVR on the test set",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# tsvr2<-1-sum((test.ECs-tpresv)^2)/sum((test.ECs-mean(test.ECs))^2)
# 
# #RF
# trf<-randomForest(ECs~.,data=tt.dirfs[,11:16],ntree=1200,mtry=4)
# tprerf<-predict(trf,test.dirfs[,11:16])
# plot(x=test.ECs,y=tprerf,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0,0.55), ylim = c(0,0.55),
#      xlab="Measured ECs (1e-6K▪m/V)",ylab="Predicted ECs (1e-6K▪m/V)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="RF on the test set",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# trfr2<-1-sum((test.ECs-tprerf)^2)/sum((test.ECs-mean(test.ECs))^2)
# 
# 
# #* repeat seed 10-fold CVE(MAE) with 1000 bootstrap var*#
# krign10cve<-function(i){
#   krigcv<-c()
#   set.seed(11+20*i)
#   dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),] #random order
#   krigcve<-c()
#   for (j in 0:9){
#     dim10<-dim(dirfr)[1]%/%10
#     dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),11:15]
#     dirfcv2<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigm(dirfcv2[,11:15],t,p,dirfcv2$btvar10,64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcve<-c(krigcve,NA)
#     }else{
#       krigcve<-c(krigcve,sum(abs(dirfcv1[,3]-predict(krigcvm,dirfcv1[,-3],type="SK")$mean))/dim10)
#     }
#     #}
#     krigcv<-c(krigcv,mean(krigcve, na.rm = T))
#   }
#   print(c(i,krigcv))
#   return(c(i,krigcv))
# }
# 
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krign10cvs.txt")
# clusterExport(cl, list("krign10cve","krigm","t","p","btvar","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krign10cvs<-parLapply(cl, c(4,5,6,7), krign10cve)
# )
# stopCluster(cl)
# 
# save.image("2.RData")
# 
# krign10cvsm<-c()
# for(i in 1:length(krign10cvs)){
#   krign10cvsm<-c(krign10cvsm,mean(krign10cvs[[i]][-1]))
# }
# 
# 
# #* repeat seed 10-fold CVE(MAE) with 200 bootstrap var*#
# krign2cve<-function(i){
#   krigcv<-c()
#   set.seed(11+20*i)
#   dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),] #random order
#   krigcve<-c()
#   for (j in 0:9){
#     dim10<-dim(dirfr)[1]%/%10
#     dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),11:15]
#     dirfcv2<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigm(dirfcv2[,11:15],t,p,dirfcv2$btvar2,64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcve<-c(krigcve,NA)
#     }else{
#       krigcve<-c(krigcve,sum(abs(dirfcv1[,3]-predict(krigcvm,dirfcv1[,-3],type="SK")$mean))/dim10)
#     }
#     #}
#     krigcv<-c(krigcv,mean(krigcve, na.rm = T))
#   }
#   print(c(i,krigcv))
#   return(c(i,krigcv))
# }
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krign2cvs.txt")
# clusterExport(cl, list("krign2cve","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krign2cvs<-parLapply(cl, c(4,5,6,7), krign2cve)
# )
# stopCluster(cl)
# 
# save.image("2.RData")
# 
# krign2cvsm<-c()
# for(i in 1:length(krign2cvs)){
#   krign2cvsm<-c(krign2cvsm,mean(krign2cvs[[i]][-1]))
# }
# 
# #* repeat seed 10-fold CVE(MAE) with tbtvar5a*#
# krignt5acve<-function(i){
#   krigcv<-c()
#   set.seed(11+20*i)
#   dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),] #random order
#   krigcve<-c()
#   for (j in 0:9){
#     dim10<-dim(dirfr)[1]%/%10
#     dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),11:16]
#     dirfcv5<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigm(dirfcv5[,11:16],t,p,dirfcv5$tbtvar5a,64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcve<-c(krigcve,NA)
#     }else{
#       krigcve<-c(krigcve,sum(abs(dirfcv1[,3]-predict(krigcvm,dirfcv1[,-3],type="SK")$mean))/dim10)
#     }
#     #}
#     krigcv<-c(krigcv,mean(krigcve, na.rm = T))
#   }
#   print(c(i,krigcv))
#   return(c(i,krigcv))
# }
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krignt5acvs.txt")
# clusterExport(cl, list("krignt5acve","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krignt5acvs<-parLapply(cl, c(4,5,6,7), krignt5acve)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# krignt5acvsm<-c()
# for(i in 1:length(krignt5acvs)){
#   krignt5acvsm<-c(krignt5acvsm,mean(krignt5acvs[[i]][-1]))
# }
# # 
# # #* repeat seed 10-fold CVE(MAE) with 0.2*btvar5a*#
# # krig2n5acve<-function(i){
# #   krigcv<-c()
# #   set.seed(11+20*i)
# #   dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),] #random order
# #   krigcve<-c()
# #   for (j in 0:9){
# #     dim10<-dim(dirfr)[1]%/%10
# #     dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),11:16]
# #     dirfcv5<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
# #     #for(k in 7:9){
# #     krigcvm<-try(krigm(dirfcv5[,11:16],t,p,dirfcv5$btvar5a*0.2,64))   #sseed可替换为8*k
# #     if ('try-error' %in% class(krigcvm)) {
# #       krigcve<-c(krigcve,NA)
# #     }else{
# #       krigcve<-c(krigcve,sum(abs(dirfcv1[,3]-predict(krigcvm,dirfcv1[,-3],type="SK")$mean))/dim10)
# #     }
# #     #}
# #     krigcv<-c(krigcv,mean(krigcve, na.rm = T))
# #   }
# #   print(c(i,krigcv))
# #   return(c(i,krigcv))
# # }
# # 
# # #cluster initialization
# # cl<-makeCluster(detectCores(),outfile="krig2n5acvs.txt")
# # clusterExport(cl, list("krig2n5acve","krigm","t","p","dirfs"))
# # clusterEvalQ(cl,{library(DiceKriging)})
# # 
# # system.time(
# #   krig2n5acvs<-parLapply(cl, c(4,5,6,7), krig2n5acve)
# # )
# # stopCluster(cl)
# # 
# # save.image("3.RData")
# # 
# # krig2n5acvsm<-c()
# # for(i in 1:length(krig2n5acvs)){
# #   krig2n5acvsm<-c(krig2n5acvsm,mean(krig2n5acvs[[i]][-1]))
# # }
# # 
# #* repeat seed 10-fold CVE(MAE) with tbtvar5t*#
# krignt5tcve<-function(i){
#   krigcv<-c()
#   set.seed(11+20*i)
#   dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),] #random order
#   krigcve<-c()
#   for (j in 0:9){
#     dim10<-dim(dirfr)[1]%/%10
#     dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),11:16]
#     dirfcv5<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigm(dirfcv5[,11:16],t,p,dirfcv5$tbtvar5t,64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcve<-c(krigcve,NA)
#     }else{
#       krigcve<-c(krigcve,sum(abs(dirfcv1[,3]-predict(krigcvm,dirfcv1[,-3],type="SK")$mean))/dim10)
#     }
#     #}
#     krigcv<-c(krigcv,mean(krigcve, na.rm = T))
#   }
#   print(c(i,krigcv))
#   return(c(i,krigcv))
# }
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krignt5tcvs.txt")
# clusterExport(cl, list("krignt5tcve","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krignt5tcvs<-parLapply(cl, c(4,5,6,7), krignt5tcve)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# krignt5tcvsm<-c()
# for(i in 1:length(krignt5tcvs)){
#   krignt5tcvsm<-c(krignt5tcvsm,mean(krignt5tcvs[[i]][-1]))
# }
# 
# 
# #* repeat seed 10-fold CVE(MAE) with svrvar5a*#
# krignsv5acve<-function(i){
#   krigcv<-c()
#   set.seed(11+20*i)
#   dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),] #random order
#   krigcve<-c()
#   for (j in 0:9){
#     dim10<-dim(dirfr)[1]%/%10
#     dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),11:16]
#     dirfcv5<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigm(dirfcv5[,11:16],t,p,dirfcv5$svrvar5a,64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcve<-c(krigcve,NA)
#     }else{
#       krigcve<-c(krigcve,sum(abs(dirfcv1[,3]-predict(krigcvm,dirfcv1[,-3],type="SK")$mean))/dim10)
#     }
#     #}
#     krigcv<-c(krigcv,mean(krigcve, na.rm = T))
#   }
#   print(c(i,krigcv))
#   return(c(i,krigcv))
# }
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krignsv5acvs.txt")
# clusterExport(cl, list("krignsv5acve","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krignsv5acvs<-parLapply(cl, c(4,5,6,7), krignsv5acve)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# krignsv5acvsm<-c()
# for(i in 1:length(krignsv5acvs)){
#   krignsv5acvsm<-c(krignsv5acvsm,mean(krignsv5acvs[[i]][-1]))
# }
# 
# #* repeat seed 10-fold CVE(MAE) with svrvar5t*#
# krignsv5tcve<-function(i){
#   krigcv<-c()
#   set.seed(11+20*i)
#   dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),] #random order
#   krigcve<-c()
#   for (j in 0:9){
#     dim10<-dim(dirfr)[1]%/%10
#     dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),11:16]
#     dirfcv5<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigm(dirfcv5[,11:16],t,p,dirfcv5$svrvar5t,64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcve<-c(krigcve,NA)
#     }else{
#       krigcve<-c(krigcve,sum(abs(dirfcv1[,3]-predict(krigcvm,dirfcv1[,-3],type="SK")$mean))/dim10)
#     }
#     #}
#     krigcv<-c(krigcv,mean(krigcve, na.rm = T))
#   }
#   print(c(i,krigcv))
#   return(c(i,krigcv))
# }
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krignsv5tcvs.txt")
# clusterExport(cl, list("krignsv5tcve","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krignsv5tcvs<-parLapply(cl, c(4,5,6,7), krignsv5tcve)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# krignsv5tcvsm<-c()
# for(i in 1:length(krignsv5tcvs)){
#   krignsv5tcvsm<-c(krignsv5tcvsm,mean(krignsv5tcvs[[i]][-1]))
# }
# 
# #* repeat seed 10-fold CVE(MAE) with rfvar5a*#
# krignrf5acve<-function(i){
#   krigcv<-c()
#   set.seed(11+20*i)
#   dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),] #random order
#   krigcve<-c()
#   for (j in 0:9){
#     dim10<-dim(dirfr)[1]%/%10
#     dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),11:16]
#     dirfcv5<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigm(dirfcv5[,11:16],t,p,dirfcv5$rfvar5a,64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcve<-c(krigcve,NA)
#     }else{
#       krigcve<-c(krigcve,sum(abs(dirfcv1[,3]-predict(krigcvm,dirfcv1[,-3],type="SK")$mean))/dim10)
#     }
#     #}
#     krigcv<-c(krigcv,mean(krigcve, na.rm = T))
#   }
#   print(c(i,krigcv))
#   return(c(i,krigcv))
# }
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krignrf5acvs.txt")
# clusterExport(cl, list("krignrf5acve","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krignrf5acvs<-parLapply(cl, c(4,5,6,7), krignrf5acve)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# krignrf5acvsm<-c()
# for(i in 1:length(krignrf5acvs)){
#   krignrf5acvsm<-c(krignrf5acvsm,mean(krignrf5acvs[[i]][-1]))
# }
# 
# #* repeat seed 10-fold CVE(MAE) with rfvar5t*#
# krignrf5tcve<-function(i){
#   krigcv<-c()
#   set.seed(11+20*i)
#   dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),] #random order
#   krigcve<-c()
#   for (j in 0:9){
#     dim10<-dim(dirfr)[1]%/%10
#     dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),11:16]
#     dirfcv5<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigm(dirfcv5[,11:16],t,p,dirfcv5$rfvar5t,64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcve<-c(krigcve,NA)
#     }else{
#       krigcve<-c(krigcve,sum(abs(dirfcv1[,3]-predict(krigcvm,dirfcv1[,-3],type="SK")$mean))/dim10)
#     }
#     #}
#     krigcv<-c(krigcv,mean(krigcve, na.rm = T))
#   }
#   print(c(i,krigcv))
#   return(c(i,krigcv))
# }
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krignrf5tcvs.txt")
# clusterExport(cl, list("krignrf5tcve","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krignrf5tcvs<-parLapply(cl, c(4,5,6,7), krignrf5tcve)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# krignrf5tcvsm<-c()
# for(i in 1:length(krignrf5tcvs)){
#   krignrf5tcvsm<-c(krignrf5tcvsm,mean(krignrf5tcvs[[i]][-1]))
# }
# 
# #* repeat seed 10-fold CVE(MAE) with rfpvar*#
# krignrfpcve<-function(i){
#   krigcv<-c()
#   set.seed(11+20*i)
#   dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),] #random order
#   krigcve<-c()
#   for (j in 0:9){
#     dim10<-dim(dirfr)[1]%/%10
#     dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),11:16]
#     dirfcv5<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigm(dirfcv5[,11:16],t,p,dirfcv5$rfpvar,64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcve<-c(krigcve,NA)
#     }else{
#       krigcve<-c(krigcve,sum(abs(dirfcv1[,3]-predict(krigcvm,dirfcv1[,-3],type="SK")$mean))/dim10)
#     }
#     #}
#     krigcv<-c(krigcv,mean(krigcve, na.rm = T))
#   }
#   print(c(i,krigcv))
#   return(c(i,krigcv))
# }
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krignrfpcvs.txt")
# clusterExport(cl, list("krignrfpcve","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krignrfpcvs<-parLapply(cl, c(4,5,6,7), krignrfpcve)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# krignrfpcvsm<-c()
# for(i in 1:length(krignrfpcvs)){
#   krignrfpcvsm<-c(krignrfpcvsm,mean(krignrfpcvs[[i]][-1]))
# }
# 
# # plot distribution of CVEs for different models #
# pmcv<-data.frame(V1=c(rep("kriging",4),rep("noisy kriging (GB all)",4),
#                       rep("noisy kriging (GB test)",4),rep("noisy kriging (0.2 * GB all)",4),
#                       rep("noisy kriging (tuned GB all)",4),rep("noisy kriging (tuned GB test)",4)),
#                        V2=c(krigcvsm,krign5acvsm,krign5tcvsm,krig2n5acvsm,
#                             krignt5acvsm,krignt5tcvsm))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "noisy kriging (GB all)",
#                                     "noisy kriging (GB test)", "noisy kriging (0.2 * GB all)",
#                                     "noisy kriging (tuned GB all)", "noisy kriging (tuned GB test)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   coord_flip()+
#   labs(y="CVE (10-fold MAE)",x="")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"))
# 
# # plot distribution of CVEs for different models #
# pmcv<-data.frame(V1=c(rep("kriging",4),rep("kriging with nugget",4),rep("noisy kriging (GB bootstrap)",4),
#                       rep("noisy kriging (SVR bootstrap)",4),rep("noisy kriging (RF bootstrap)",4)),
#                  V2=c(krigocvsm,krigcvsm,krignt5tcvsm,
#                       krignsv5tcvsm,krignrf5tcvsm),
#                  V3=c(c(mean(krigocvsm),rep(NA,3)),c(mean(krigcvsm),rep(NA,3)),c(mean(krignt5tcvsm),rep(NA,3)),
#                       c(mean(krignsv5tcvsm),rep(NA,3)),c(mean(krignrf5tcvsm),rep(NA,3))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging","kriging with nugget","noisy kriging (GB bootstrap)",
#                                     "noisy kriging (SVR bootstrap)","noisy kriging (RF bootstrap)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="CVE (10-fold MAE)",x="")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"))
# 
# # plot var estimated by different models #
# plot(x=btvar5,y=btvar,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(59, 0, 159, maxColorValue = 255), cex = 1.2,
#      xlim = c(0,0.006), ylim = c(0,0.006),
#      xlab="Variance by GB (B=500)",ylab="Variance by GB with compositions (B=500)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0), font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 34, 139, maxColorValue = 255))
# 
# plot(x=btvar5,y=svrvar5,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(59, 0, 159, maxColorValue = 255), cex = 1.2,
#      xlim = c(0,0.006), ylim = c(0,0.006),
#      xlab="Variance by GB (B=500)",ylab="Variance by SVR (B=500)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0), font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 34, 139, maxColorValue = 255))
# 
# 
# #* repeat seed 10-fold CVE(MAE) with ratio rfvar5a*#
# #Input: ratio for rfvar5a & seed  Output: input and CVE
# krignrrf5acve<-function(ri){
#   krigcv<-c()
#   set.seed(11+20*ri[2])
#   dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),] #random order
#   krigcve<-c()
#   for (j in 0:9){
#     dim10<-dim(dirfr)[1]%/%10
#     dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),11:16]
#     dirfcv5<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigm(dirfcv5[,11:16],t,p,ri[1]*dirfcv5$rfvar5a,64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcve<-c(krigcve,NA)
#     }else{
#       krigcve<-c(krigcve,sum(abs(dirfcv1[,3]-predict(krigcvm,dirfcv1[,-3],type="SK")$mean))/dim10)
#     }
#     #}
#     krigcv<-c(krigcv,mean(krigcve, na.rm = T))
#   }
#   print(c(ri,krigcv))
#   return(c(ri,krigcv))
# }
# 
# rim<-matrix(ncol=2)
# for(r in c(0.01,0.05,0.1,0.2,0.3,0.5,0.8,1.5)){
#   for(i in c(4,5,6,7)){
#     rim<-rbind(rim,c(r,i))
#   }
# }
# rim<-rim[-1,]
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krignrrf5acvs.txt")
# clusterExport(cl, list("krignrrf5acve","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krignrrf5acvs<-parApply(cl, rim, 1, krignrrf5acve)
# )
# stopCluster(cl)
# 
# save.image("~/ECEdemo/directboots/3.RData")
# 
# knrrf5acvsm<-data.frame(rep(NA,4))
# for(j in 1:8){
#   knrrf5acvse<-c()
#   for(i in ((j-1)*4+1):(j*4)){
#     knrrf5acvse<-c(knrrf5acvse,mean(krignrrf5acvs[,i][-c(1,2)]))
#   }
#   knrrf5acvsm<-cbind(knrrf5acvsm,knrrf5acvse)
# }
# knrrf5acvsm<-knrrf5acvsm[,-1]
# colnames(knrrf5acvsm)<-c(0.01,0.05,0.1,0.2,0.3,0.5,0.8,1.5)
# save.image("~/ECEdemo/directboots/3.RData")
# 
# #* repeat seed 10-fold CVE(MAE) with ratio rfvar5t*#
# #Input: ratio for rvar5t & seed  Output: input and CVE
# krignrrf5tcve<-function(ri){
#   krigcv<-c()
#   set.seed(11+20*ri[2])
#   dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),] #random order
#   krigcve<-c()
#   for (j in 0:9){
#     dim10<-dim(dirfr)[1]%/%10
#     dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),11:16]
#     dirfcv5<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigm(dirfcv5[,11:16],t,p,ri[1]*dirfcv5$rfvar5t,64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcve<-c(krigcve,NA)
#     }else{
#       krigcve<-c(krigcve,sum(abs(dirfcv1[,3]-predict(krigcvm,dirfcv1[,-3],type="SK")$mean))/dim10)
#     }
#     #}
#     krigcv<-c(krigcv,mean(krigcve, na.rm = T))
#   }
#   print(c(ri,krigcv))
#   return(c(ri,krigcv))
# }
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krignrrf5tcvs.txt")
# clusterExport(cl, list("krignrrf5tcve","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krignrrf5tcvs<-parApply(cl, rim, 1, krignrrf5tcve)
# )
# stopCluster(cl)
# 
# save.image("~/ECEdemo/directboots/3.RData")
# 
# knrrf5tcvsm<-data.frame(rep(NA,4))
# for(j in 1:8){
#   knrrf5tcvse<-c()
#   for(i in ((j-1)*4+1):(j*4)){
#     knrrf5tcvse<-c(knrrf5tcvse,mean(krignrrf5tcvs[,i][-c(1,2)]))
#   }
#   knrrf5tcvsm<-cbind(knrrf5tcvsm,knrrf5tcvse)
# }
# knrrf5tcvsm<-knrrf5tcvsm[,-1]
# colnames(knrrf5tcvsm)<-c(0.01,0.05,0.1,0.2,0.3,0.5,0.8,1.5)
# save.image("~/ECEdemo/directboots/3.RData")
# 
# # plot distribution of CVEs for ratio RF #
# pmcv<-data.frame(V1=c(rep("kriging",4),rep("noisy kriging (RF test)",4),rep("noisy kriging (0.01*RF test)",4),
#                       rep("noisy kriging (0.05*RF test)",4),rep("noisy kriging (0.1*RF test)",4),
#                       rep("noisy kriging (0.2*RF test)",4),rep("noisy kriging (0.3*RF test)",4),
#                       rep("noisy kriging (0.5*RF test)",4),rep("noisy kriging (0.8*RF test)",4),
#                       rep("noisy kriging (1.5*RF test)",4)),
#                  V2=c(krigcvsm,krignrf5tcvsm,knrrf5tcvsm[,1],
#                       knrrf5tcvsm[,2],knrrf5tcvsm[,3],
#                       knrrf5tcvsm[,4],knrrf5tcvsm[,5],
#                       knrrf5tcvsm[,6],knrrf5tcvsm[,7],knrrf5tcvsm[,8]))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "noisy kriging (RF test)","noisy kriging (0.01*RF test)", 
#                                     "noisy kriging (0.05*RF test)","noisy kriging (0.1*RF test)",
#                                     "noisy kriging (0.2*RF test)", "noisy kriging (0.3*RF test)",
#                                     "noisy kriging (0.5*RF test)","noisy kriging (0.8*RF test)",
#                                     "noisy kriging (1.5*RF test)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   coord_flip()+
#   labs(y="CVE (10-fold MAE)",x="")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"))
# 
# #* repeat seed 10-fold CVE(MAE) with var *#
# #Input: name of var & seed  Output: input and CVE
# krigntvrf5cve<-function(tsi){
#   krigcv<-c()
#   set.seed(11+20*as.numeric(tsi[2]))
#   dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),] #random order
#   for (j in 0:9){
#     dim10<-dim(dirfr)[1]%/%10
#     dirfcv1<-dirfr[(1+dim10*j):(dim10*(j+1)),11:16]
#     dirfcv5<-dirfr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigm(dirfcv5[,11:16],t,p,dirfcv5[,tsi[1]],64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcv<-c(krigcv,NA)
#     }else{
#       krigcv<-c(krigcv,sum(abs(dirfcv1[,3]-predict(krigcvm,dirfcv1[,-3],type="SK")$mean))/dim10)
#     }
#     #}
#   }
#   print(c(tsi,krigcv))
#   return(c(tsi,krigcv))
# }
# 
# tsim<-matrix(ncol=2)
# for(ts in c("btvar5t","svrvar5t","rfvar5t",
#             "tv8rfvar5","tv10rfvar5","tv12rfvar5",
#             "tv6rfvar5","tv14rfvar5","tv15rfvar5",
#             "tv155rfvar5")){
#   for(i in 1:100){
#     tsim<-rbind(tsim,c(ts,i))
#   }
# }
# tsim<-tsim[-1,]
# 
# tsim3<-matrix(ncol=2)
# for(ts in c("tv15rfvar5m")){
#   for(i in 1:100){
#     tsim3<-rbind(tsim3,c(ts,i))
#   }
# }
# tsim3<-tsim3[-1,]
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krigntvrf5cvs.txt")
# clusterExport(cl, list("krigntvrf5cve","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krigntvrf5cvs<-parApply(cl, tsim, 1, krigntvrf5cve)
# )
# stopCluster(cl)
# 
# save.image("~/ECEdemo/directboots/3.RData")
# 
# cl<-makeCluster(detectCores(),outfile="krigntvrf5cvs.txt")
# clusterExport(cl, list("krigntvrf5cve","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krigntvrf5cvs3<-parApply(cl, tsim3, 1, krigntvrf5cve)
# )
# stopCluster(cl)
# 
# save.image("~/ECEdemo/directboots/3.RData")
# 
# # kntvrf5cvsm<-data.frame(rep(NA,4))
# # for(j in 1:3){
# #   kntvrf5cvse<-c()
# #   for(i in ((j-1)*4+1):(j*4)){
# #     kntvrf5cvse<-c(kntvrf5cvse,mean(as.numeric(krigntvrf5cvs[,i][-c(1,2)])))
# #   }
# #   kntvrf5cvsm<-cbind(kntvrf5cvsm,kntvrf5cvse)
# # }
# # kntvrf5cvsm<-kntvrf5cvsm[,-1]
# kntvrf5cvsm<-as.data.frame(matrix(nrow = 100,ncol = 10))
# for(j in 1:10){
#   for(i in 1:100){
#     kntvrf5cvsm[i,j]<-mean(as.numeric(krigntvrf5cvs[-c(1,2),(j-1)*100+i]),na.rm=T)
#   }
# }
# colnames(kntvrf5cvsm)<-c("btvar5t","svrvar5t","rfvar5t",
#                          "tv8rfvar5","tv10rfvar5","tv12rfvar5",
#                          "tv6rfvar5","tv14rfvar5","tv15rfvar5","tv155rfvar5")
# 
# save.image("~/ECEdemo/directboots/3.RData")
# 
# cvetv15m<-c()
# for(i in 1:100){
#   cvetv15m<-c(cvetv15m,mean(as.numeric(krigntvrf5cvs3[-c(1,2),i]),na.rm=T))
# }
# 
kncvin2<-matrix(ncol = 3)
for(i in 1:100){
  for(j in 0:9){
    kncvin2<-rbind(kncvin2,c(i,j,"tv15rfvar5m"))
  }
}
kncvin2<-kncvin2[-1,]

#return all predictions
krigncve2<-function(ijv){
  set.seed(11+20*as.numeric(ijv[1]))
  dirfr<-dirfs[sample(dim(dirfs)[1],dim(dirfs)[1]),] #random order
  dim10<-dim(dirfr)[1]%/%10
  dirfcv1<-dirfr[(1+dim10*as.numeric(ijv[2])):(dim10*(as.numeric(ijv[2])+1)),11:16]
  dirfcv5<-dirfr[-((1+dim10*as.numeric(ijv[2])):(dim10*(as.numeric(ijv[2])+1))),]
  krigcvm<-try(krigm(dirfcv5[,11:16],t,p,dirfcv5[,ijv[3]],64))
  if ('try-error' %in% class(krigcvm)) {
    prete<-rep(NA,dim(dirfcv1)[1]);pretr<-rep(NA,dim(dirfcv5)[1])
    preteva<-rep(NA,dim(dirfcv1)[1]);pretrva<-rep(NA,dim(dirfcv5)[1])
  }else{
    pretr<-predict(krigcvm,dirfcv5[,c(11,12,14:16)],type="SK")$mean
    pretrva<-predict(krigcvm,dirfcv5[,c(11,12,14:16)],type="SK")$sd
    prete<-predict(krigcvm,dirfcv1[,-3],type="SK")$mean
    preteva<-predict(krigcvm,dirfcv1[,-3],type="SK")$sd
  }
  res<-list(data.frame(dirfcv1[,3],prete,preteva),data.frame(dirfcv5[,13],pretr,pretrva))
  fnm1<-paste("cv10n/cv10npre3_", ijv[1], "_", ijv[2], ".csv", sep = "")  #3包括parameters
  fnm2<-paste("cv10n/cv10nptr3_", ijv[1], "_", ijv[2], ".csv", sep = "")
  write.table(res[[1]], fnm1, append = TRUE, sep = ",")
  write.table(res[[2]], fnm2, append = TRUE, sep = ",")
  return(res)
}

cl<-makeCluster(detectCores(),outfile="krigntvrf5cvs.txt")
clusterExport(cl, list("krigm","t","p","dirfs"))
clusterEvalQ(cl,{library(DiceKriging)})

system.time(
  krigncvs22<-parApply(cl, kncvin2, 1, krigncve2)
)
stopCluster(cl)

save.image("3.RData")

cv10nmae<-c(); cv10nr2<-c()
for(i in 1:100){
  te<-c(); pte<-c()
  for(j in 1:10){
    te<-c(te,krigncvs22[[((i-1)*10+j)]][[1]][,1])
    pte<-c(pte,krigncvs22[[((i-1)*10+j)]][[1]][,2])
  }
  ptemae<-mean(abs(pte - te),na.rm=T)
  pter2<-round(1-sum((pte-te)^2,na.rm=T)/sum((te-mean(te,na.rm=T))^2,na.rm=T),3)
  cv10nmae<-c(cv10nmae,ptemae)
  cv10nr2<-c(cv10nr2,pter2)
}
cv10nr2s<-c()
for(i in 1:1000){
  te<-krigncvs22[[i]][[1]][,1]
  pte<-krigncvs22[[i]][[1]][,2]
  pter2<-round(1-sum((pte-te)^2,na.rm=T)/sum((te-mean(te,na.rm=T))^2,na.rm=T),3)
  cv10nr2s<-c(cv10nr2s,pter2)
}

# # plot distribution of CVEs for tvrfvar5 #
# pmcv<-data.frame(V1=c(rep("without",100),rep("nugget",100),rep("res. var (bootstrap)",100),
#                       rep("res. var (1000 test)",100),rep("res. var (800 test)",100),
#                       rep("res. var (600 test)",100),rep("res. var (400 test)",100),
#                       rep("res. var (200 test)",100),rep("res. var (100 test)",100),
#                       rep("res. var (50 test)",100)),
#                  V2=c(krigocvsm,krigcvsm,kntvrf5cvsm[,3],kntvrf5cvsm[,7],kntvrf5cvsm[,4],
#                       kntvrf5cvsm[,5],kntvrf5cvsm[,6],kntvrf5cvsm[,8],kntvrf5cvsm[,9],
#                       kntvrf5cvsm[,10]),
#                  V3=c(c(mean(krigocvsm),rep(NA,99)),c(mean(krigcvsm),rep(NA,99)),c(mean(kntvrf5cvsm[,3]),rep(NA,99)),
#                       c(mean(kntvrf5cvsm[,7]),rep(NA,99)),c(mean(kntvrf5cvsm[,4]),rep(NA,99)),
#                       c(mean(kntvrf5cvsm[,5]),rep(NA,99)),c(mean(kntvrf5cvsm[,6]),rep(NA,99)),
#                       c(mean(kntvrf5cvsm[,8]),rep(NA,99)),c(mean(kntvrf5cvsm[,9]),rep(NA,99)),
#                       c(mean(kntvrf5cvsm[,10]),rep(NA,99))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("without","nugget","res. var (bootstrap)",
#                                     "res. var (1000 test)","res. var (800 test)",
#                                     "res. var (600 test)", "res. var (400 test)",
#                                     "res. var (200 test)","res. var (100 test)",
#                                     "res. var (50 test)"))
# #导出数据
# pmcvout<-data.frame(krigocvsm,krigcvsm,kntvrf5cvsm[,7],kntvrf5cvsm[,4],
#                          kntvrf5cvsm[,5],kntvrf5cvsm[,6],kntvrf5cvsm[,8],kntvrf5cvsm[,9],
#                          kntvrf5cvsm[,10],cvetv15m)
# colnames(pmcvout)<-c("without var.","nugget var.",
#                      "609 train","809 train",
#                      "1009 train", "1209 train",
#                      "1409 train","1509 train",
#                      "1559 train","1509 train source mean")
# library(openxlsx)
# write.xlsx(pmcvout,"CVEs ECs.xlsx")
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="CVE (10-fold MAE)",x="")+
#   theme_bw(base_size = 27)+
#   theme(legend.position = "none",
#         panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         axis.text = element_text(face="bold", color="black", size=24, family="serif"), 
#         axis.title = element_text(face="bold",family="serif"),
#         axis.ticks.length=unit(-0.25, "cm"),
#         plot.margin = unit(c(0, 0.1, 0, -0.9), "cm"))
# 
# 
# maep<-data.frame(krigocvsm,krigcvsm,kntvrf5cvsm[,9])
# global_min <- min(unlist(maep))
# global_max <- max(unlist(maep))
# 
# # 定义全局归一化函数
# global_normalize <- function(x) {
#   (x - global_min) / (global_max - global_min)
# }
# 
# # 对数据框中的每个数值进行归一化
# maepn <- as.data.frame(lapply(maep, global_normalize))
# pmcv<-data.frame(V1=c(rep("without",100),rep("nugget",100),rep("resampling",100)),
#                  V2=c(maepn[,1],maepn[,2],maepn[,3]),
#                  V3=c(c(mean(maepn[,1]),rep(NA,99)),c(mean(maepn[,2]),rep(NA,99)),c(mean(maepn[,3]),rep(NA,99))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("without","nugget","resampling"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Scaled error",x="")+
#   theme_bw(base_size = 32)+
#   theme(legend.position = "none",
#         panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         panel.border = element_rect(color = "black", linewidth = 2),
#         axis.text = element_text(color="black", size=29, family="sans"),
#         axis.title.x = element_text(family="sans", margin = margin(t = 16)),
#         axis.ticks.length=unit(-0.25, "cm"),
#         plot.margin = unit(c(0, 0.4, 0.1, -1.3), "cm"))
# 
# 
# ##***********##
# 
# 

#* FigS4c *#
f3aind<-order(cv10nr2s-cv10or2s,decreasing = T)[1:20]
cv10nr2s[f3aind];cv10or2s[f3aind]
f3aselect<-f3aind[5]

library(ggh4x)
fn.plot.gpar3 <- function(x, y1, y2, lim, xlab, ylab, cr2) {    
  # 创建一个数据框，以便在 ggplot2 中使用    
  df <- data.frame(    
    x = x,    
    y = c(y1, y2),  
    z = factor(c(rep("tr", length(y1)), rep("te", length(y2))), levels = c("tr", "te"))  # 明确设置因子水平  
  )    
  
  # 使用 ggplot2 绘图    
  ggplot(df, aes(x = x, y = y, color = z)) + 
    geom_point(data = subset(df, z == "tr"), aes(color = z), size = 2.5) +
    geom_point(data = subset(df, z == "te"), aes(color = z), size = 4) +
    scale_color_manual(values = c(rgb(227,41,53,maxColorValue = 255), rgb(0,72,131,maxColorValue = 255)),  # 手动设置颜色值，与因子的水平顺序一致    
                       labels = c("tr", "te")) +  
    geom_abline(intercept = 0, slope = 1, color = "black", size = 1.2) +  # 添加对角线    
    theme_bw(base_size = 35)+
    theme(legend.position = "none",
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(),  # 移除默认边框
          axis.line = element_line(color = "black", linewidth = 0.8),  # 保留坐标轴线
          axis.text = element_text(color="black", size=27, family="sans"),
          axis.title = element_text(family="sans"),
          axis.title.x = element_text(margin = ggplot2::margin(t = -2), hjust = 1),
          axis.title.y = element_text(margin = ggplot2::margin(r = -4)),  # Y轴标题向右移动
          axis.ticks.length=unit(0.35, "cm"),
          axis.ticks = element_line(linewidth = 0.75),
          plot.margin = unit(c(0.3, 0.65, -0.1, 0), "cm"))+
    # 添加次刻度线（需要ggh4x包）
    scale_y_continuous(limits = lim, name = ylab,
                       breaks = c(0, 0.25, 0.5), labels = c("0", "0.25", "0.5"),
                       guide = guide_axis(minor.ticks = TRUE),minor_breaks = waiver()) +
    scale_x_continuous(limits = lim, name = xlab, 
                       breaks = c(0, 0.25, 0.5), labels = c("0", "0.25", "0.5"),
                       guide = guide_axis(minor.ticks = TRUE),minor_breaks = waiver()) +
    theme(ggh4x.axis.ticks.length.minor=rel(1/2))+
    annotate("text", x = 0.21, y = 0.57, label = bquote(paste("R"^2 == .(cr2))),
             color = rgb(0,72,131,maxColorValue = 255), size = 10, family = "sans")
}

jpeg("FigS4c 1.jpg", width = 3970, height = 2800, res = 600)
fn.plot.gpar3(x=c(krigocvs2[[346]][[2]][,1],krigocvs2[[346]][[1]][,1]),
              y1=krigocvs2[[346]][[2]][,2],y2=krigocvs2[[346]][[1]][,2],
              lim=c(0,0.6),xlab=expression(paste("Measured ECs", " (" * 10^-6 * "K·m", bold("/"), "V)")),
              ylab=bquote(atop("Predicted ECs", " (" * 10^-6 * "K·m" * bold("/") * "V)")),0.932)
dev.off()
jpeg("FigS4c 2.jpg", width = 3970, height = 2800, res = 600)
fn.plot.gpar3(x=c(krigncvs22[[346]][[2]][,1],krigncvs22[[346]][[1]][,1]),
              y1=krigncvs22[[346]][[2]][,2],y2=krigncvs22[[346]][[1]][,2],
              lim=c(0,0.6),xlab=expression(paste("Measured ECs", " (" * 10^-6 * "K·m", bold("/"), "V)")),
              ylab=bquote(atop("Predicted ECs", " (" * 10^-6 * "K·m" * bold("/") * "V)")),0.934)
dev.off()


# 
# ##** iterative opportunity cost **##
# 
# #* OC of kriging without noise *# 
# # Input: virtual data size, max iteration times, seed for kriging, seed for sampling data, type of utility function 
# # Output: opportunity cost of each iteration
# itoco<-function(vn, mt, sseed, dseed, uf){
#   fvs<-dirfs[1354,]  #max ECs in dirfs
#   dirfsl<-dirfs[-1354,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, dirfsl[sample(dim(dirfsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(dirfs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigmo(ftd[,11:16],t,p,sseed))
#     if ('try-error' %in% class(krigcvm)) {
#       set.seed(dseed+5)
#       newd<<-fvs[sample(dim(fvs)[1], 1),]
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(11,12,14,15,16)],type="SK")
#       if(uf=="pre"){
#         newd<<-fvs[which(kpre[["mean"]]==max(kpre[["mean"]],na.rm=T)),][1,]
#       }else if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb,na.rm=T)),][1,]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,13]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego,na.rm=T)),][1,]
#       }
#     }
#     fvs<-anti_join(fvs,newd)
#     ftd<-rbind(ftd,newd)
#     oce<<-dirfs[1354,]$ECs-newd$ECs
#     print(oce)    
#     oc<-c(oc,oce)
#     if(oce==0){
#       break()
#     }
#   }
#   gc()
#   return(oc)
# }
# 
# vim<-matrix(ncol=2)
# for(v in c(800,950,1100,1250,1400,1500)){
#   for(i in c(34,77,6,12)){
#     vim<-rbind(vim,c(v,i))
#   }
# }
# vim<-vim[-1,]
# 
# cl<-makeCluster(detectCores(),outfile="oegooc.txt")
# clusterExport(cl, list("itoco","krigmo","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocopar<-function(vi){
#   return(itoco(vi[1],500,64,vi[2],"ego"))
# }
# 
# system.time(
#   oegooc<-parApply(cl, vim, 1, itocopar)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itoego<-matrix(nrow = 4,ncol=6)
# for(j in 1:6){
#   for(i in 1:4){
#     itoego[i,j]<-length(oegooc[[(j-1)*4+i]])
#   }
# }
# colnames(itoego)<-c("800","950","1100","1250","1400","1500")
# 
# # 不同虚拟空间，用四种效能函数
# vuim<-matrix(ncol=3)
# for(vn in c(1126,1287,1448)){
#   for(uf in c("pre","ucb","ego")){
#     for(i in 1:100){
#       vuim<-rbind(vuim,c(vn,uf,5*i+7))
#     }
#   }
# }
# vuim<-vuim[-1,]
# 
# vuim2<-matrix(ncol=3)
# for(vn in c(1126,1287,1448)){
#   for(uf in c("pre","ucb","ego","sko")){
#     for(i in 1:100){
#       vuim2<-rbind(vuim2,c(vn,uf,5*i+7))
#     }
#   }
# }
# vuim2<-vuim2[-1,]
# 
# cl<-makeCluster(detectCores(),outfile="oufoc.txt")
# clusterExport(cl, list("itoco","krigmo","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocouf<-function(ui){
#   return(itoco(as.numeric(ui[1]),1000,64,as.numeric(ui[3]),ui[2]))
# }
# 
# system.time(
#   oufoc<-parApply(cl, vuim, 1, itocouf)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itouf<-matrix(nrow = 100,ncol=9)
# for(j in 1:9){
#   for(i in 1:100){
#     itouf[i,j]<-length(oufoc[[(j-1)*100+i]])
#   }
# }
# colnames(itouf)<-c("pre7","ucb7","ego7","pre8","ucb8","ego8","pre9","ucb9","ego9")
# 
# 
# #* OC of kriging with nugget *#
# # Input: virtual data size, max iteration times, seed for kriging, seed for sampling data, type of utility function 
# # Output: opportunity cost of each iteration
# itoc<-function(vn, mt, sseed, dseed, uf){
#   fvs<-dirfs[1354,]  #max ECs in dirfs
#   dirfsl<-dirfs[-1354,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, dirfsl[sample(dim(dirfsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(dirfs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,11:16],t,p,0,64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       krigcvm<-try(krigm(ftd[,11:16],t,p,0,567))
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(11,12,14,15,16)],type="SK")
#       if(uf=="pre"){
#         newd<<-fvs[which(kpre[["mean"]]==max(kpre[["mean"]],na.rm=T)),][1,]
#       }else if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb,na.rm=T)),][1,]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,13]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego,na.rm=T)),][1,]
#       }else if(uf=="sko"){
#         ze<-(kpre[["mean"]]-max(kpre[["mean"]]-krigcvm@covariance@nugget))/kpre[["sd"]]
#         kpresko<-(1-krigcvm@covariance@nugget/sqrt((krigcvm@covariance@nugget)^2+(kpre[["sd"]])^2))*
#           kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpresko==max(kpresko,na.rm=T)),][1,]
#       }
#       fvs<-anti_join(fvs,newd)
#       ftd<-rbind(ftd,newd)
#     }
#     oce<<-dirfs[1354,]$ECs-newd$ECs
#     print(oce)    
#     oc<-c(oc,oce)
#     if(oce==0){
#       break()
#     }
#   }
#   gc()
#   return(oc)
# }
# 
# 
# # EGO #
# # vn= 1500 #
# cl<-makeCluster(detectCores(),outfile="3egooc1500.txt")
# clusterExport(cl, list("itoc","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocego1500<-function(dseed){
#   return(itoc(1500,500,64,dseed,"ego"))
# }
# 
# system.time(
#   egooc1500<-parLapply(cl, c(34,77,6,12), itocego1500)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itego1500<-c()
# for (i in 1:length(egooc1500)){
#   itego1500<-c(itego1500, length(egooc1500[[i]]))
# }
# 
# # vn= 1400 #
# cl<-makeCluster(detectCores(),outfile="3egooc1400.txt")
# clusterExport(cl, list("itoc","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocego1400<-function(dseed){
#   return(itoc(1400,500,64,dseed,"ego"))
# }
# 
# system.time(
#   egooc1400<-parLapply(cl, c(34,77,6,12), itocego1400)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itego1400<-c()
# for (i in 1:length(egooc1400)){
#   itego1400<-c(itego1400, length(egooc1400[[i]]))
# }
# 
# # vn= 1250 #
# cl<-makeCluster(detectCores(),outfile="3egooc1250.txt")
# clusterExport(cl, list("itoc","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocego1250<-function(dseed){
#   return(itoc(1250,500,64,dseed,"ego"))
# }
# 
# system.time(
#   egooc1250<-parLapply(cl, c(34,77,6,12), itocego1250)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itego1250<-c()
# for (i in 1:length(egooc1250)){
#   itego1250<-c(itego1250, length(egooc1250[[i]]))
# }
# 
# # vn= 1100 #
# cl<-makeCluster(detectCores(),outfile="3egooc1100.txt")
# clusterExport(cl, list("itoc","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocego1100<-function(dseed){
#   return(itoc(1100,500,64,dseed,"ego"))
# }
# 
# system.time(
#   egooc1100<-parLapply(cl, c(34,77,6,12), itocego1100)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itego1100<-c()
# for (i in 1:length(egooc1100)){
#   itego1100<-c(itego1100, length(egooc1100[[i]]))
# }
# 
# # vn= 950 #
# cl<-makeCluster(detectCores(),outfile="3egooc950.txt")
# clusterExport(cl, list("itoc","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocego950<-function(dseed){
#   return(itoc(950,500,64,dseed,"ego"))
# }
# 
# system.time(
#   egooc950<-parLapply(cl, c(34,77,6,12), itocego950)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itego950<-c()
# for (i in 1:length(egooc950)){
#   itego950<-c(itego950, length(egooc950[[i]]))
# }
# 
# # vn= 800 #
# cl<-makeCluster(detectCores(),outfile="3egooc800.txt")
# clusterExport(cl, list("itoc","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocego800<-function(dseed){
#   return(itoc(800,500,64,dseed,"ego"))
# }
# 
# system.time(
#   egooc800<-parLapply(cl, c(34,77,6,12), itocego800)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itego800<-c()
# for (i in 1:length(egooc800)){
#   itego800<-c(itego800, length(egooc800[[i]]))
# }
# 
# # 不同虚拟空间，用四种效能函数
# cl<-makeCluster(detectCores(),outfile="ufoc.txt")
# clusterExport(cl, list("itoc","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocuf<-function(ui){
#   return(itoc(as.numeric(ui[1]),1000,64,as.numeric(ui[3]),ui[2]))
# }
# 
# system.time(
#   ufoc<-parApply(cl, vuim2, 1, itocuf)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# ituf<-matrix(nrow = 100,ncol=12)
# for(j in 1:12){
#   for(i in 1:100){
#     ituf[i,j]<-length(ufoc[[(j-1)*100+i]])
#   }
# }
# colnames(ituf)<-c("pre7","ucb7","ego7","sko7","pre8","ucb8","ego8","sko8",
#                   "pre9","ucb9","ego9","sko9")
# 
# 
# # egoocall<-matrix(nrow=10,ncol=length(krigegooc))
# # for(i in 1:length(krigegooc)){
# #   for(j in 1:length(krigegooc[[i]])){
# #     egoocall[j,i]<-krigegooc[[i]][j]
# #   }
# # }
# # egoocmean<-c()
# # for(i in 1:10){
# #   egoocmean<-c(egoocmean,mean(egoocall[i,],na.rm=T))
# # }
# # egoocsd<-c()
# # for(i in 1:10){
# #   egoocsd<-c(egoocsd,sd(egoocall[i,],na.rm=T))
# # }
# 
# 
# #* OC of kriging with noise *# 
# # Input: virtual data size, max iteration times, seed for kriging, seed for sampling data, type of utility function 
# # Output: opportunity cost of each iteration
# itocnt5a<-function(vn, mt, sseed, dseed, uf){
#   fvs<-dirfs[1354,]  #max ECs in dirfs
#   dirfsl<-dirfs[-1354,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, dirfsl[sample(dim(dirfsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(dirfs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,11:16],t,p,ftd$tbtvar5a,64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       krigcvm<-try(krigm(ftd[,11:16],t,p,0,567))
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(11,12,14,15,16)],type="SK")
#       if(uf=="pre"){
#         newd<<-fvs[which(kpre[["mean"]]==max(kpre[["mean"]],na.rm=T)),][1,]
#       }else if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb,na.rm=T)),][1,]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,13]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego,na.rm=T)),][1,]
#       }
#       fvs<-anti_join(fvs,newd)
#       ftd<-rbind(ftd,newd)
#     }
#     oce<<-dirfs[1354,]$ECs-newd$ECs
#     print(oce)
#     oc<-c(oc,oce)    
#     if(oce==0){
#       break()
#     }
#   }
#   gc()
#   return(oc)
# }
# 
# 
# # EGO #
# # vn=1500 #
# cl<-makeCluster(detectCores(),outfile="3nt5aegooc1500.txt")
# clusterExport(cl, list("itocnt5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnt5aego1500<-function(dseed){
#   return(itocnt5a(1500,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nt5aegooc1500<-parLapply(cl, c(34,77,6,12), itocnt5aego1500)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnt5aego1500<-c()
# for (i in 1:length(nt5aegooc1500)){
#   itnt5aego1500<-c(itnt5aego1500, length(nt5aegooc1500[[i]]))
# }
# 
# # vn=1400 #
# cl<-makeCluster(detectCores(),outfile="3nt5aegooc1400.txt")
# clusterExport(cl, list("itocnt5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnt5aego1400<-function(dseed){
#   return(itocnt5a(1400,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nt5aegooc1400<-parLapply(cl, c(34,77,6,12), itocnt5aego1400)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnt5aego1400<-c()
# for (i in 1:length(nt5aegooc1400)){
#   itnt5aego1400<-c(itnt5aego1400, length(nt5aegooc1400[[i]]))
# }
# 
# # vn= 1250 #
# cl<-makeCluster(detectCores(),outfile="3nt5aegooc1250.txt")
# clusterExport(cl, list("itocnt5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnt5aego1250<-function(dseed){
#   return(itocnt5a(1250,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nt5aegooc1250<-parLapply(cl, c(34,77,6,12), itocnt5aego1250)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnt5aego1250<-c()
# for (i in 1:length(nt5aegooc1250)){
#   itnt5aego1250<-c(itnt5aego1250, length(nt5aegooc1250[[i]]))
# }
# 
# # vn= 1100 #
# cl<-makeCluster(detectCores(),outfile="3nt5aegooc1100.txt")
# clusterExport(cl, list("itocnt5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnt5aego1100<-function(dseed){
#   return(itocnt5a(1100,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nt5aegooc1100<-parLapply(cl, c(34,77,6,12), itocnt5aego1100)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnt5aego1100<-c()
# for (i in 1:length(nt5aegooc1100)){
#   itnt5aego1100<-c(itnt5aego1100, length(nt5aegooc1100[[i]]))
# }
# 
# # vn= 950 #
# cl<-makeCluster(detectCores(),outfile="3nt5aegooc950.txt")
# clusterExport(cl, list("itocnt5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnt5aego950<-function(dseed){
#   return(itocnt5a(950,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nt5aegooc950<-parLapply(cl, c(34,77,6,12), itocnt5aego950)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnt5aego950<-c()
# for (i in 1:length(nt5aegooc950)){
#   itnt5aego950<-c(itnt5aego950, length(nt5aegooc950[[i]]))
# }
# 
# # vn= 800 #
# cl<-makeCluster(detectCores(),outfile="3nt5aegooc800.txt")
# clusterExport(cl, list("itocnt5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnt5aego800<-function(dseed){
#   return(itocnt5a(800,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nt5aegooc800<-parLapply(cl, c(34,77,6,12), itocnt5aego800)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnt5aego800<-c()
# for (i in 1:length(nt5aegooc800)){
#   itnt5aego800<-c(itnt5aego800, length(nt5aegooc800[[i]]))
# }
# 
# 
# #* iteration OC with btvar5t *#
# itocn5t<-function(vn, mt, sseed, dseed, uf){
#   fvs<-dirfs[1354,]  #max ECs in dirfs
#   dirfsl<-dirfs[-1354,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, dirfsl[sample(dim(dirfsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(dirfs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,11:16],t,p,ftd$btvar5t,64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       krigcvm<-try(krigm(ftd[,11:16],t,p,0,567))
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(11,12,14,15,16)],type="SK")
#       if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb)),][1,]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,13]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego)),][1,]
#       }
#       fvs<-anti_join(fvs,newd)
#       ftd<-rbind(ftd,newd)
#     }
#     oce<<-dirfs[1354,]$ECs-newd$ECs
#     print(oce)
#     oc<-c(oc,oce)    
#     if(oce==0){
#       break()
#     }
#   }
#   gc()
#   return(oc)
# }
# 
# # vn=1500 #
# cl<-makeCluster(detectCores(),outfile="3n5tegooc1500.txt")
# clusterExport(cl, list("itocn5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocn5tego1500<-function(dseed){
#   return(itocn5t(1500,500,64,dseed,"ego"))
# }
# 
# system.time(
#   n5tegooc1500<-parLapply(cl, c(34,77,6,12), itocn5tego1500)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itn5tego1500<-c()
# for (i in 1:length(n5tegooc1500)){
#   itn5tego1500<-c(itn5tego1500, length(n5tegooc1500[[i]]))
# }
# 
# # vn=1400 #
# cl<-makeCluster(detectCores(),outfile="3n5tegooc1400.txt")
# clusterExport(cl, list("itocn5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocn5tego1400<-function(dseed){
#   return(itocn5t(1400,500,64,dseed,"ego"))
# }
# 
# system.time(
#   n5tegooc1400<-parLapply(cl, c(34,77,6,12), itocn5tego1400)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itn5tego1400<-c()
# for (i in 1:length(n5tegooc1400)){
#   itn5tego1400<-c(itn5tego1400, length(n5tegooc1400[[i]]))
# }
# 
# # vn= 1250 #
# cl<-makeCluster(detectCores(),outfile="3n5tegooc1250.txt")
# clusterExport(cl, list("itocn5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocn5tego1250<-function(dseed){
#   return(itocn5t(1250,500,64,dseed,"ego"))
# }
# 
# system.time(
#   n5tegooc1250<-parLapply(cl, c(34,77,6,12), itocn5tego1250)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itn5tego1250<-c()
# for (i in 1:length(n5tegooc1250)){
#   itn5tego1250<-c(itn5tego1250, length(n5tegooc1250[[i]]))
# }
# 
# # vn= 1100 #
# cl<-makeCluster(detectCores(),outfile="3n5tegooc1100.txt")
# clusterExport(cl, list("itocn5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocn5tego1100<-function(dseed){
#   return(itocn5t(1100,500,64,dseed,"ego"))
# }
# 
# system.time(
#   n5tegooc1100<-parLapply(cl, c(34,77,6,12), itocn5tego1100)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itn5tego1100<-c()
# for (i in 1:length(n5tegooc1100)){
#   itn5tego1100<-c(itn5tego1100, length(n5tegooc1100[[i]]))
# }
# 
# # vn= 950 #
# cl<-makeCluster(detectCores(),outfile="3n5tegooc950.txt")
# clusterExport(cl, list("itocn5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocn5tego950<-function(dseed){
#   return(itocn5t(950,500,64,dseed,"ego"))
# }
# 
# system.time(
#   n5tegooc950<-parLapply(cl, c(34,77,6,12), itocn5tego950)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itn5tego950<-c()
# for (i in 1:length(n5tegooc950)){
#   itn5tego950<-c(itn5tego950, length(n5tegooc950[[i]]))
# }
# 
# # vn= 800 #
# cl<-makeCluster(detectCores(),outfile="3n5tegooc800.txt")
# clusterExport(cl, list("itocn5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocn5tego800<-function(dseed){
#   return(itocn5t(800,500,64,dseed,"ego"))
# }
# 
# system.time(
#   n5tegooc800<-parLapply(cl, c(34,77,6,12), itocn5tego800)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itn5tego800<-c()
# for (i in 1:length(n5tegooc800)){
#   itn5tego800<-c(itn5tego800, length(n5tegooc800[[i]]))
# }
# 
# 
# #* iteration OC with 0.2*btvar5a *#
# itoc2n5a<-function(vn, mt, sseed, dseed, uf){
#   fvs<-dirfs[1354,]  #max ECs in dirfs
#   dirfsl<-dirfs[-1354,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, dirfsl[sample(dim(dirfsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(dirfs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,11:16],t,p,ftd$btvar5a*0.2,64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       krigcvm<-try(krigm(ftd[,11:16],t,p,0,567))
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(11,12,14,15,16)],type="SK")
#       if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb)),]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,13]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego)),]
#       }
#       fvs<-anti_join(fvs,newd)
#       ftd<-rbind(ftd,newd)
#     }
#     oce<<-dirfs[1354,]$ECs-newd$ECs
#     print(oce)
#     oc<-c(oc,oce)    
#     if(oce==0){
#       break()
#     }
#   }
#   gc()
#   return(oc)
# }
# 
# # vn=1400 #
# cl<-makeCluster(detectCores(),outfile="32n5aegooc1400.txt")
# clusterExport(cl, list("itoc2n5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itoc2n5aego1400<-function(dseed){
#   return(itoc2n5a(1400,500,64,dseed,"ego"))
# }
# 
# system.time(
#   n5a2egooc1400<-parLapply(cl, c(34,77,6,12), itoc2n5aego1400)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# it2n5aego1400<-c()
# for (i in 1:length(n5a2egooc1400)){
#   it2n5aego1400<-c(it2n5aego1400, length(n5a2egooc1400[[i]]))
# }
# 
# # vn= 1250 #
# cl<-makeCluster(detectCores(),outfile="32n5aegooc1250.txt")
# clusterExport(cl, list("itoc2n5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itoc2n5aego1250<-function(dseed){
#   return(itoc2n5a(1250,500,64,dseed,"ego"))
# }
# 
# system.time(
#   n5a2egooc1250<-parLapply(cl, c(34,77,6,12), itoc2n5aego1250)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# it2n5aego1250<-c()
# for (i in 1:length(n5a2egooc1250)){
#   it2n5aego1250<-c(it2n5aego1250, length(n5a2egooc1250[[i]]))
# }
# 
# # vn= 1100 #
# cl<-makeCluster(detectCores(),outfile="32n5aegooc1100.txt")
# clusterExport(cl, list("itoc2n5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itoc2n5aego1100<-function(dseed){
#   return(itoc2n5a(1100,500,64,dseed,"ego"))
# }
# 
# system.time(
#   n5a2egooc1100<-parLapply(cl, c(34,77,6,12), itoc2n5aego1100)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# it2n5aego1100<-c()
# for (i in 1:length(n5a2egooc1100)){
#   it2n5aego1100<-c(it2n5aego1100, length(n5a2egooc1100[[i]]))
# }
# 
# # vn= 950 #
# cl<-makeCluster(detectCores(),outfile="32n5aegooc950.txt")
# clusterExport(cl, list("itoc2n5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itoc2n5aego950<-function(dseed){
#   return(itoc2n5a(950,500,64,dseed,"ego"))
# }
# 
# system.time(
#   n5a2egooc950<-parLapply(cl, c(34,77,6,12), itoc2n5aego950)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# it2n5aego950<-c()
# for (i in 1:length(n5a2egooc950)){
#   it2n5aego950<-c(it2n5aego950, length(n5a2egooc950[[i]]))
# }
# 
# # vn= 800 #
# cl<-makeCluster(detectCores(),outfile="32n5aegooc800.txt")
# clusterExport(cl, list("itoc2n5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itoc2n5aego800<-function(dseed){
#   return(itoc2n5a(800,500,64,dseed,"ego"))
# }
# 
# system.time(
#   n5a2egooc800<-parLapply(cl, c(34,77,6,12), itoc2n5aego800)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# it2n5aego800<-c()
# for (i in 1:length(n5a2egooc800)){
#   it2n5aego800<-c(it2n5aego800, length(n5a2egooc800[[i]]))
# }
# 
# # #* plot OC *#
# # pocx<-c(rep(1:10,4))
# # pocm<-c(ucbocmean,egoocmean,nucbocmean,negoocmean)
# # pocsd<-c(ucbocsd,egoocsd,nucbocsd,negoocsd)
# # pocd<-rep(c("kriging + UCB", "kriging + EGO", "noisy kriging + UCB", "noisy kriging + EGO"),each=10)
# # pocdat<-as.data.frame(cbind(pocx,pocm,pocsd,pocd))
# # pocdat$pocx<-as.numeric(pocdat$pocx)
# # pocdat$pocm<-as.numeric(pocdat$pocm)
# # pocdat$pocsd<-as.numeric(pocdat$pocsd)
# # pocdat$pocd<-ordered(pocdat$pocd,levels = c("kriging + UCB", "kriging + EGO", "noisy kriging + UCB", "noisy kriging + EGO"))
# # 
# # ggplot(data.frame(x=pocdat$pocx,y=pocdat$pocm),aes(x,y,color=pocdat$pocd))+
# #   geom_line(size=1.2)+
# #   geom_errorbar(aes(x=pocdat$pocx,ymin=pocdat$pocm-0.5*pocdat$pocsd, ymax=pocdat$pocm+0.5*pocdat$pocsd), size=1.2,width=0.2)+
# #   scale_color_manual(name = " ",values = c("red","grey","orange","purple"),
# #                      labels = c("kriging + UCB", "kriging + EGO", "noisy kriging + UCB", "noisy kriging + EGO")) +
# #   scale_x_continuous(breaks=c(1:10))+
# #   xlab("Iteration times")+ylab("Opportunity cost")+
# #   theme_bw(base_size = 30)+
# #   theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank(),
# #         axis.text=element_text(color="black",size=24),
# #         axis.title=element_text(face="bold"))
# 
# # plot Iteration times -- Virtual size #
# pittx<-c(rep(c(800,950,1100,1250,1400),8))
# pittm<-c(mean(itego800),mean(itego950),mean(itego1100),mean(itego1250),mean(itego1400),
#          mean(itn5aego800),mean(itn5aego950),mean(itn5aego1100),mean(itn5aego1250),mean(itn5aego1400),
#          mean(itn5tego800),mean(itn5tego950),mean(itn5tego1100),mean(itn5tego1250),mean(itn5tego1400),
#          mean(itnsv5aego800),mean(itnsv5aego950),mean(itnsv5aego1100),mean(itnsv5aego1250),mean(itnsv5aego1400),
#          mean(itnsv5tego800),mean(itnsv5tego950),mean(itnsv5tego1100),mean(itnsv5tego1250),mean(itnsv5tego1400),
#          mean(itnrf5aego800),mean(itnrf5aego950),mean(itnrf5aego1100),mean(itnrf5aego1250),mean(itnrf5aego1400),
#          mean(itnrf5tego800),mean(itnrf5tego950),mean(itnrf5tego1100),mean(itnrf5tego1250),mean(itnrf5tego1400),
#          mean(itnrf5aego800),mean(itnrfpego950),mean(itnrfpego1100),mean(itnrfpego1250),mean(itnrfpego1400))
# pittsd<-c(sd(itego800),sd(itego950),sd(itego1100),sd(itego1250),sd(itego1400),
#          sd(itn5aego800),sd(itn5aego950),sd(itn5aego1100),sd(itn5aego1250),sd(itn5aego1400),
#          sd(itn5tego800),sd(itn5tego950),sd(itn5tego1100),sd(itn5tego1250),sd(itn5tego1400),
#          sd(itnsv5aego800),sd(itnsv5aego950),sd(itnsv5aego1100),sd(itnsv5aego1250),sd(itnsv5aego1400),
#          sd(itnsv5tego800),sd(itnsv5tego950),sd(itnsv5tego1100),sd(itnsv5tego1250),sd(itnsv5tego1400),
#          sd(itnrf5aego800),sd(itnrf5aego950),sd(itnrf5aego1100),sd(itnrf5aego1250),sd(itnrf5aego1400),
#          sd(itnrf5tego800),sd(itnrf5tego950),sd(itnrf5tego1100),sd(itnrf5tego1250),sd(itnrf5tego1400),
#          sd(itnrf5aego800),sd(itnrfpego950),sd(itnrfpego1100),sd(itnrfpego1250),sd(itnrfpego1400))
# pittd<-rep(c("kriging + EGO", "noisy kriging (GB all) + EGO",
#              "noisy kriging (GB test) + EGO", "noisy kriging (SVR all) + EGO",
#              "noisy kriging (SVR test) + EGO","noisy kriging (RF all) + EGO",
#              "noisy kriging (RF test) + EGO","noisy kriging (RF tree) + EGO"),each=5)
# pittdat<-as.data.frame(cbind(pittx,pittm,pittsd,pittd))
# pittdat$pittx<-as.numeric(pittdat$pittx)
# pittdat$pittm<-as.numeric(pittdat$pittm)
# pittdat$pittsd<-as.numeric(pittdat$pittsd)
# pittdat$pittd<-ordered(pittdat$pittd,levels = c("kriging + EGO", "noisy kriging (GB all) + EGO",
#                                                 "noisy kriging (GB test) + EGO", "noisy kriging (SVR all) + EGO",
#                                                 "noisy kriging (SVR test) + EGO","noisy kriging (RF all) + EGO",
#                                                 "noisy kriging (RF test) + EGO","noisy kriging (RF tree) + EGO"))
# pittdat$palem<-rep(1,40)-pittdat$pittm/pittdat$pittx
# pittdat$palesd<-pittdat$pittsd/pittdat$pittx
# 
# ggplot(data.frame(x=pittdat$pittx,y=pittdat$pittm),aes(x,y,color=pittdat$pittd))+
#   geom_line(size=1.2)+
#   geom_errorbar(aes(x=pittdat$pittx,ymin=pittdat$pittm-0.5*pittdat$pittsd, ymax=pittdat$pittm+0.5*pittdat$pittsd), size=1.2,width=16)+
#   scale_color_manual(name = " ",values = c("black","grey","red","blue","green","yellow","orange","purple"),
#                      labels = c("kriging + EGO", "noisy kriging (GB all) + EGO",
#                                 "noisy kriging (GB test) + EGO", "noisy kriging (SVR all) + EGO",
#                                 "noisy kriging (SVR test) + EGO","noisy kriging (RF all) + EGO",
#                                 "noisy kriging (RF test) + EGO","noisy kriging (RF tree) + EGO")) +
#   scale_x_continuous(breaks=c(800,950,1100,1250,1400))+
#   xlab("Number of virtual data")+ylab("Iteration times")+
#   theme_bw(base_size = 30)+
#   theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank(),
#         axis.text=element_text(color="black",size=24),
#         axis.title=element_text(face="bold"))
# 
# ggplot(data.frame(x=pittdat$pittx,y=pittdat$palem),aes(x,y,color=pittdat$pittd))+
#   geom_line(size=1.2)+
#   geom_errorbar(aes(x=pittdat$pittx,ymin=pittdat$palem-0.5*pittdat$palesd, ymax=pittdat$palem+0.5*pittdat$palesd), size=1.2,width=16)+
#   scale_color_manual(name = " ",values = c("black","grey","red","blue","green","yellow","orange","purple"),
#                      labels = c("kriging + EGO", "noisy kriging (GB all) + EGO",
#                                 "noisy kriging (GB test) + EGO", "noisy kriging (SVR all) + EGO",
#                                 "noisy kriging (SVR test) + EGO","noisy kriging (RF all) + EGO",
#                                 "noisy kriging (RF test) + EGO","noisy kriging (RF tree) + EGO")) +
#   scale_x_continuous(breaks=c(800,950,1100,1250,1400))+
#   xlab("Number of virtual data")+ylab("AL efficiency")+
#   theme_bw(base_size = 30)+
#   theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank(),
#         axis.text=element_text(color="black",size=24),
#         axis.title=element_text(face="bold"))
# 
# #* iteration times with svrvar5a *#
# itocnsv5a<-function(vn, mt, sseed, dseed, uf){
#   fvs<-dirfs[1354,]  #max ECs in dirfs
#   dirfsl<-dirfs[-1354,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, dirfsl[sample(dim(dirfsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(dirfs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,11:16],t,p,ftd$svrvar5a,64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       krigcvm<-try(krigm(ftd[,11:16],t,p,0,567))
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(11,12,14,15,16)],type="SK")
#       if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb)),]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,13]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego)),]
#       }
#       fvs<-anti_join(fvs,newd)
#       ftd<-rbind(ftd,newd)
#     }
#     oce<<-dirfs[1354,]$ECs-newd$ECs
#     print(oce)
#     oc<-c(oc,oce)    
#     if(oce==0){
#       break()
#     }
#   }
#   gc()
#   return(oc)
# }
# 
# # vn= 1500 #
# cl<-makeCluster(detectCores(),outfile="nsv5aegooc1500.txt")
# clusterExport(cl, list("itocnsv5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnsv5aego1500<-function(dseed){
#   return(itocnsv5a(1500,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nsv5aegooc1500<-parLapply(cl, c(34,77,6,12), itocnsv5aego1500)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnsv5aego1500<-c()
# for (i in 1:length(nsv5aegooc1500)){
#   itnsv5aego1500<-c(itnsv5aego1500, length(nsv5aegooc1500[[i]]))
# }
# 
# # vn= 1400 #
# cl<-makeCluster(detectCores(),outfile="nsv5aegooc1400.txt")
# clusterExport(cl, list("itocnsv5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnsv5aego1400<-function(dseed){
#   return(itocnsv5a(1400,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nsv5aegooc1400<-parLapply(cl, c(34,77,6,12), itocnsv5aego1400)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnsv5aego1400<-c()
# for (i in 1:length(nsv5aegooc1400)){
#   itnsv5aego1400<-c(itnsv5aego1400, length(nsv5aegooc1400[[i]]))
# }
# 
# # vn= 1250 #
# cl<-makeCluster(detectCores(),outfile="nsv5aegooc1250.txt")
# clusterExport(cl, list("itocnsv5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnsv5aego1250<-function(dseed){
#   return(itocnsv5a(1250,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nsv5aegooc1250<-parLapply(cl, c(34,77,6,12), itocnsv5aego1250)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnsv5aego1250<-c()
# for (i in 1:length(nsv5aegooc1250)){
#   itnsv5aego1250<-c(itnsv5aego1250, length(nsv5aegooc1250[[i]]))
# }
# 
# # vn= 1100 #
# cl<-makeCluster(detectCores(),outfile="nsv5aegooc1100.txt")
# clusterExport(cl, list("itocnsv5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnsv5aego1100<-function(dseed){
#   return(itocnsv5a(1100,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nsv5aegooc1100<-parLapply(cl, c(34,77,6,12), itocnsv5aego1100)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnsv5aego1100<-c()
# for (i in 1:length(nsv5aegooc1100)){
#   itnsv5aego1100<-c(itnsv5aego1100, length(nsv5aegooc1100[[i]]))
# }
# 
# # vn= 950 #
# cl<-makeCluster(detectCores(),outfile="nsv5aegooc950.txt")
# clusterExport(cl, list("itocnsv5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnsv5aego950<-function(dseed){
#   return(itocnsv5a(950,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nsv5aegooc950<-parLapply(cl, c(34,77,6,12), itocnsv5aego950)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnsv5aego950<-c()
# for (i in 1:length(nsv5aegooc950)){
#   itnsv5aego950<-c(itnsv5aego950, length(nsv5aegooc950[[i]]))
# }
# 
# # vn= 800 #
# cl<-makeCluster(detectCores(),outfile="nsv5aegooc800.txt")
# clusterExport(cl, list("itocnsv5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnsv5aego800<-function(dseed){
#   return(itocnsv5a(800,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nsv5aegooc800<-parLapply(cl, c(34,77,6,12), itocnsv5aego800)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnsv5aego800<-c()
# for (i in 1:length(nsv5aegooc800)){
#   itnsv5aego800<-c(itnsv5aego800, length(nsv5aegooc800[[i]]))
# }
# 
# #* iteration times with svrvar5t *#
# itocnsv5t<-function(vn, mt, sseed, dseed, uf){
#   fvs<-dirfs[1354,]  #max ECs in dirfs
#   dirfsl<-dirfs[-1354,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, dirfsl[sample(dim(dirfsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(dirfs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,11:16],t,p,ftd$svrvar5t,64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       krigcvm<-try(krigm(ftd[,11:16],t,p,0,567))
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(11,12,14,15,16)],type="SK")
#       if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb)),][1,]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,13]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego)),][1,]
#       }
#       fvs<-anti_join(fvs,newd)
#       ftd<-rbind(ftd,newd)
#     }
#     oce<<-dirfs[1354,]$ECs-newd$ECs
#     print(oce)
#     oc<-c(oc,oce)    
#     if(oce==0){
#       break()
#     }
#   }
#   gc()
#   return(oc)
# }
# 
# # vn= 1500 #
# cl<-makeCluster(detectCores(),outfile="nsv5tegooc1500.txt")
# clusterExport(cl, list("itocnsv5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnsv5tego1500<-function(dseed){
#   return(itocnsv5t(1500,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nsv5tegooc1500<-parLapply(cl, c(34,77,6,12), itocnsv5tego1500)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnsv5tego1500<-c()
# for (i in 1:length(nsv5tegooc1500)){
#   itnsv5tego1500<-c(itnsv5tego1500, length(nsv5tegooc1500[[i]]))
# }
# 
# # vn= 1400 #
# cl<-makeCluster(detectCores(),outfile="nsv5tegooc1400.txt")
# clusterExport(cl, list("itocnsv5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnsv5tego1400<-function(dseed){
#   return(itocnsv5t(1400,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nsv5tegooc1400<-parLapply(cl, c(34,77,6,12), itocnsv5tego1400)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnsv5tego1400<-c()
# for (i in 1:length(nsv5tegooc1400)){
#   itnsv5tego1400<-c(itnsv5tego1400, length(nsv5tegooc1400[[i]]))
# }
# 
# # vn= 1250 #
# cl<-makeCluster(detectCores(),outfile="nsv5tegooc1250.txt")
# clusterExport(cl, list("itocnsv5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnsv5tego1250<-function(dseed){
#   return(itocnsv5t(1250,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nsv5tegooc1250<-parLapply(cl, c(34,77,6,12), itocnsv5tego1250)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnsv5tego1250<-c()
# for (i in 1:length(nsv5tegooc1250)){
#   itnsv5tego1250<-c(itnsv5tego1250, length(nsv5tegooc1250[[i]]))
# }
# 
# # vn= 1100 #
# cl<-makeCluster(detectCores(),outfile="nsv5tegooc1100.txt")
# clusterExport(cl, list("itocnsv5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnsv5tego1100<-function(dseed){
#   return(itocnsv5t(1100,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nsv5tegooc1100<-parLapply(cl, c(34,77,6,12), itocnsv5tego1100)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnsv5tego1100<-c()
# for (i in 1:length(nsv5tegooc1100)){
#   itnsv5tego1100<-c(itnsv5tego1100, length(nsv5tegooc1100[[i]]))
# }
# 
# # vn= 950 #
# cl<-makeCluster(detectCores(),outfile="nsv5tegooc950.txt")
# clusterExport(cl, list("itocnsv5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnsv5tego950<-function(dseed){
#   return(itocnsv5t(950,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nsv5tegooc950<-parLapply(cl, c(34,77,6,12), itocnsv5tego950)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnsv5tego950<-c()
# for (i in 1:length(nsv5tegooc950)){
#   itnsv5tego950<-c(itnsv5tego950, length(nsv5tegooc950[[i]]))
# }
# 
# # vn= 800 #
# cl<-makeCluster(detectCores(),outfile="nsv5tegooc800.txt")
# clusterExport(cl, list("itocnsv5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnsv5tego800<-function(dseed){
#   return(itocnsv5t(800,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nsv5tegooc800<-parLapply(cl, c(34,77,6,12), itocnsv5tego800)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnsv5tego800<-c()
# for (i in 1:length(nsv5tegooc800)){
#   itnsv5tego800<-c(itnsv5tego800, length(nsv5tegooc800[[i]]))
# }
# 
# #* iteration times with rfvar5a *#
# itocnrf5a<-function(vn, mt, sseed, dseed, uf){
#   fvs<-dirfs[1354,]  #max ECs in dirfs
#   dirfsl<-dirfs[-1354,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, dirfsl[sample(dim(dirfsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(dirfs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,11:16],t,p,ftd$rfvar5a,64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       krigcvm<-try(krigm(ftd[,11:16],t,p,0,567))
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(11,12,14,15,16)],type="SK")
#       if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb)),]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,13]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego)),]
#       }
#       fvs<-anti_join(fvs,newd)
#       ftd<-rbind(ftd,newd)
#     }
#     oce<<-dirfs[1354,]$ECs-newd$ECs
#     print(oce)
#     oc<-c(oc,oce)    
#     if(oce==0){
#       break()
#     }
#   }
#   gc()
#   return(oc)
# }
# 
# # vn= 1500 #
# cl<-makeCluster(detectCores(),outfile="nrf5aegooc1500.txt")
# clusterExport(cl, list("itocnrf5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrf5aego1500<-function(dseed){
#   return(itocnrf5a(1500,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrf5aegooc1500<-parLapply(cl, c(34,77,6,12), itocnrf5aego1500)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrf5aego1500<-c()
# for (i in 1:length(nrf5aegooc1500)){
#   itnrf5aego1500<-c(itnrf5aego1500, length(nrf5aegooc1500[[i]]))
# }
# 
# # vn= 1400 #
# cl<-makeCluster(detectCores(),outfile="nrf5aegooc1400.txt")
# clusterExport(cl, list("itocnrf5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrf5aego1400<-function(dseed){
#   return(itocnrf5a(1400,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrf5aegooc1400<-parLapply(cl, c(34,77,6,12), itocnrf5aego1400)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrf5aego1400<-c()
# for (i in 1:length(nrf5aegooc1400)){
#   itnrf5aego1400<-c(itnrf5aego1400, length(nrf5aegooc1400[[i]]))
# }
# 
# # vn= 1250 #
# cl<-makeCluster(detectCores(),outfile="nrf5aegooc1250.txt")
# clusterExport(cl, list("itocnrf5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrf5aego1250<-function(dseed){
#   return(itocnrf5a(1250,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrf5aegooc1250<-parLapply(cl, c(34,77,6,12), itocnrf5aego1250)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrf5aego1250<-c()
# for (i in 1:length(nrf5aegooc1250)){
#   itnrf5aego1250<-c(itnrf5aego1250, length(nrf5aegooc1250[[i]]))
# }
# 
# # vn= 1100 #
# cl<-makeCluster(detectCores(),outfile="nrf5aegooc1100.txt")
# clusterExport(cl, list("itocnrf5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrf5aego1100<-function(dseed){
#   return(itocnrf5a(1100,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrf5aegooc1100<-parLapply(cl, c(34,77,6,12), itocnrf5aego1100)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrf5aego1100<-c()
# for (i in 1:length(nrf5aegooc1100)){
#   itnrf5aego1100<-c(itnrf5aego1100, length(nrf5aegooc1100[[i]]))
# }
# 
# # vn= 950 #
# cl<-makeCluster(detectCores(),outfile="nrf5aegooc950.txt")
# clusterExport(cl, list("itocnrf5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrf5aego950<-function(dseed){
#   return(itocnrf5a(950,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrf5aegooc950<-parLapply(cl, c(34,77,6,12), itocnrf5aego950)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrf5aego950<-c()
# for (i in 1:length(nrf5aegooc950)){
#   itnrf5aego950<-c(itnrf5aego950, length(nrf5aegooc950[[i]]))
# }
# 
# # vn= 800 #
# cl<-makeCluster(detectCores(),outfile="nrf5aegooc800.txt")
# clusterExport(cl, list("itocnrf5a","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrf5aego800<-function(dseed){
#   return(itocnrf5a(800,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrf5aegooc800<-parLapply(cl, c(34,77,6,12), itocnrf5aego800)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrf5aego800<-c()
# for (i in 1:length(nrf5aegooc800)){
#   itnrf5aego800<-c(itnrf5aego800, length(nrf5aegooc800[[i]]))
# }
# 
# #* iteration times with rfvar5t *#
# itocnrf5t<-function(vn, mt, sseed, dseed, uf){
#   fvs<-dirfs[1354,]  #max ECs in dirfs
#   dirfsl<-dirfs[-1354,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, dirfsl[sample(dim(dirfsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(dirfs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,11:16],t,p,ftd$rfvar5t,64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       krigcvm<-try(krigm(ftd[,11:16],t,p,0,567))
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(11,12,14,15,16)],type="SK")
#       if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb)),][1,]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,13]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego)),][1,]
#       }
#       fvs<-anti_join(fvs,newd)
#       ftd<-rbind(ftd,newd)
#     }
#     oce<<-dirfs[1354,]$ECs-newd$ECs
#     print(oce)
#     oc<-c(oc,oce)    
#     if(oce==0){
#       break()
#     }
#   }
#   gc()
#   return(oc)
# }
# 
# # vn= 1500 #
# cl<-makeCluster(detectCores(),outfile="nrf5tegooc1500.txt")
# clusterExport(cl, list("itocnrf5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrf5tego1500<-function(dseed){
#   return(itocnrf5t(1500,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrf5tegooc1500<-parLapply(cl, c(34,77,6,12), itocnrf5tego1500)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrf5tego1500<-c()
# for (i in 1:length(nrf5tegooc1500)){
#   itnrf5tego1500<-c(itnrf5tego1500, length(nrf5tegooc1500[[i]]))
# }
# 
# # vn= 1400 #
# cl<-makeCluster(detectCores(),outfile="nrf5tegooc1400.txt")
# clusterExport(cl, list("itocnrf5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrf5tego1400<-function(dseed){
#   return(itocnrf5t(1400,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrf5tegooc1400<-parLapply(cl, c(34,77,6,12), itocnrf5tego1400)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrf5tego1400<-c()
# for (i in 1:length(nrf5tegooc1400)){
#   itnrf5tego1400<-c(itnrf5tego1400, length(nrf5tegooc1400[[i]]))
# }
# 
# # vn= 1250 #
# cl<-makeCluster(detectCores(),outfile="nrf5tegooc1250.txt")
# clusterExport(cl, list("itocnrf5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrf5tego1250<-function(dseed){
#   return(itocnrf5t(1250,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrf5tegooc1250<-parLapply(cl, c(34,77,6,12), itocnrf5tego1250)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrf5tego1250<-c()
# for (i in 1:length(nrf5tegooc1250)){
#   itnrf5tego1250<-c(itnrf5tego1250, length(nrf5tegooc1250[[i]]))
# }
# 
# # vn= 1100 #
# cl<-makeCluster(detectCores(),outfile="nrf5tegooc1100.txt")
# clusterExport(cl, list("itocnrf5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrf5tego1100<-function(dseed){
#   return(itocnrf5t(1100,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrf5tegooc1100<-parLapply(cl, c(34,77,6,12), itocnrf5tego1100)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrf5tego1100<-c()
# for (i in 1:length(nrf5tegooc1100)){
#   itnrf5tego1100<-c(itnrf5tego1100, length(nrf5tegooc1100[[i]]))
# }
# 
# # vn= 950 #
# cl<-makeCluster(detectCores(),outfile="nrf5tegooc950.txt")
# clusterExport(cl, list("itocnrf5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrf5tego950<-function(dseed){
#   return(itocnrf5t(950,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrf5tegooc950<-parLapply(cl, c(34,77,6,12), itocnrf5tego950)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrf5tego950<-c()
# for (i in 1:length(nrf5tegooc950)){
#   itnrf5tego950<-c(itnrf5tego950, length(nrf5tegooc950[[i]]))
# }
# 
# # vn= 800 #
# cl<-makeCluster(detectCores(),outfile="nrf5tegooc800.txt")
# clusterExport(cl, list("itocnrf5t","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrf5tego800<-function(dseed){
#   return(itocnrf5t(800,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrf5tegooc800<-parLapply(cl, c(34,77,6,12), itocnrf5tego800)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrf5tego800<-c()
# for (i in 1:length(nrf5tegooc800)){
#   itnrf5tego800<-c(itnrf5tego800, length(nrf5tegooc800[[i]]))
# }
# 
# #* iteration times with rfpvar *#
# itocnrfp<-function(vn, mt, sseed, dseed, uf){
#   fvs<-dirfs[1354,]  #max ECs in dirfs
#   dirfsl<-dirfs[-1354,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, dirfsl[sample(dim(dirfsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(dirfs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,11:16],t,p,ftd$rfpvar,64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       krigcvm<-try(krigm(ftd[,11:16],t,p,0,567))
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(11,12,14,15,16)],type="SK")
#       if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb)),]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,13]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego)),]
#       }
#       fvs<-anti_join(fvs,newd)
#       ftd<-rbind(ftd,newd)
#     }
#     oce<<-dirfs[1354,]$ECs-newd$ECs
#     print(oce)
#     oc<-c(oc,oce)    
#     if(oce==0){
#       break()
#     }
#   }
#   gc()
#   return(oc)
# }
# 
# # vn= 1500 #
# cl<-makeCluster(detectCores(),outfile="nrfpegooc1500.txt")
# clusterExport(cl, list("itocnrfp","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrfpego1500<-function(dseed){
#   return(itocnrfp(1500,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrfpegooc1500<-parLapply(cl, c(34,77,6,12), itocnrfpego1500)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrfpego1500<-c()
# for (i in 1:length(nrfpegooc1500)){
#   itnrfpego1500<-c(itnrfpego1500, length(nrfpegooc1500[[i]]))
# }
# 
# # vn= 1400 #
# cl<-makeCluster(detectCores(),outfile="nrfpegooc1400.txt")
# clusterExport(cl, list("itocnrfp","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrfpego1400<-function(dseed){
#   return(itocnrfp(1400,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrfpegooc1400<-parLapply(cl, c(34,77,6,12), itocnrfpego1400)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrfpego1400<-c()
# for (i in 1:length(nrfpegooc1400)){
#   itnrfpego1400<-c(itnrfpego1400, length(nrfpegooc1400[[i]]))
# }
# 
# # vn= 1250 #
# cl<-makeCluster(detectCores(),outfile="nrfpegooc1250.txt")
# clusterExport(cl, list("itocnrfp","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrfpego1250<-function(dseed){
#   return(itocnrfp(1250,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrfpegooc1250<-parLapply(cl, c(34,77,6,12), itocnrfpego1250)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrfpego1250<-c()
# for (i in 1:length(nrfpegooc1250)){
#   itnrfpego1250<-c(itnrfpego1250, length(nrfpegooc1250[[i]]))
# }
# 
# # vn= 1100 #
# cl<-makeCluster(detectCores(),outfile="nrfpegooc1100.txt")
# clusterExport(cl, list("itocnrfp","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrfpego1100<-function(dseed){
#   return(itocnrfp(1100,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrfpegooc1100<-parLapply(cl, c(34,77,6,12), itocnrfpego1100)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrfpego1100<-c()
# for (i in 1:length(nrfpegooc1100)){
#   itnrfpego1100<-c(itnrfpego1100, length(nrfpegooc1100[[i]]))
# }
# 
# # vn= 950 #
# cl<-makeCluster(detectCores(),outfile="nrfpegooc950.txt")
# clusterExport(cl, list("itocnrfp","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrfpego950<-function(dseed){
#   return(itocnrfp(950,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrfpegooc950<-parLapply(cl, c(34,77,6,12), itocnrfpego950)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrfpego950<-c()
# for (i in 1:length(nrfpegooc950)){
#   itnrfpego950<-c(itnrfpego950, length(nrfpegooc950[[i]]))
# }
# 
# # vn= 800 #
# cl<-makeCluster(detectCores(),outfile="nrfpegooc800.txt")
# clusterExport(cl, list("itocnrfp","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrfpego800<-function(dseed){
#   return(itocnrfp(800,500,64,dseed,"ego"))
# }
# 
# system.time(
#   nrfpegooc800<-parLapply(cl, c(34,77,6,12), itocnrfpego800)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnrfpego800<-c()
# for (i in 1:length(nrfpegooc800)){
#   itnrfpego800<-c(itnrfpego800, length(nrfpegooc800[[i]]))
# }
# 
# # plot Iteration times -- Virtual size for different bootstrap models #
# pittx<-c(rep(c(800,950,1100,1250,1400,1500),5))
# pittm<-c(mean(itoego[,1]),mean(itoego[,2]),mean(itoego[,3]),mean(itoego[,4]),mean(itoego[,5]),mean(itoego[,6]),
#          mean(itego800),mean(itego950),mean(itego1100),mean(itego1250),mean(itego1400),mean(itego1500),
#          mean(itn5tego800),mean(itn5tego950),mean(itn5tego1100),mean(itn5tego1250),mean(itn5tego1400),mean(itn5tego1500),
#          mean(itnsv5tego800),mean(itnsv5tego950),mean(itnsv5tego1100),mean(itnsv5tego1250),mean(itnsv5tego1400),mean(itnsv5tego1500),
#          mean(itnrf5tego800),mean(itnrf5tego950),mean(itnrf5tego1100),mean(itnrf5tego1250),mean(itnrf5tego1400),mean(itnrf5tego1500))
# pittsd<-c(sd(itoego[,1]),sd(itoego[,2]),sd(itoego[,3]),sd(itoego[,4]),sd(itoego[,5]),sd(itoego[,6]),
#           sd(itego800),sd(itego950),sd(itego1100),sd(itego1250),sd(itego1400),sd(itego1500),
#           sd(itn5tego800),sd(itn5tego950),sd(itn5tego1100),sd(itn5tego1250),sd(itn5tego1400),sd(itn5tego1500),
#           sd(itnsv5tego800),sd(itnsv5tego950),sd(itnsv5tego1100),sd(itnsv5tego1250),sd(itnsv5tego1400),sd(itnsv5tego1500),
#           sd(itnrf5tego800),sd(itnrf5tego950),sd(itnrf5tego1100),sd(itnrf5tego1250),sd(itnrf5tego1400),sd(itnrf5tego1500))
# pittd<-rep(c("kriging + EGO", "kriging with nugget + EGO", "noisy kriging (GB bootstrap) + EGO",
#              "noisy kriging (SVR bootstrap) + EGO","noisy kriging (RF bootstrap) + EGO"),each=6)
# pittdat<-as.data.frame(cbind(pittx,pittm,pittsd,pittd))
# pittdat$pittx<-as.numeric(pittdat$pittx)
# pittdat$pittm<-as.numeric(pittdat$pittm)
# pittdat$pittsd<-as.numeric(pittdat$pittsd)
# pittdat$pittd<-ordered(pittdat$pittd,levels = c("kriging + EGO", "kriging with nugget + EGO", "noisy kriging (GB bootstrap) + EGO",
#                                                 "noisy kriging (SVR bootstrap) + EGO","noisy kriging (RF bootstrap) + EGO"))
# pittdat$palem<-rep(1,30)-pittdat$pittm/pittdat$pittx
# pittdat$palesd<-pittdat$pittsd/pittdat$pittx
# 
# ggplot(data.frame(x=pittdat$pittx,y=pittdat$pittm),aes(x,y,color=pittdat$pittd))+
#   geom_line(size=1.2)+
#   geom_errorbar(aes(x=pittdat$pittx,ymin=pittdat$pittm-0.5*pittdat$pittsd, ymax=pittdat$pittm+0.5*pittdat$pittsd), size=1.2,width=16)+
#   scale_color_manual(name = " ",values = c("grey","blue","orange","purple"),
#                      labels = c("kriging + EGO", "noisy kriging (GB bootstrap) + EGO",
#                                 "noisy kriging (SVR bootstrap) + EGO","noisy kriging (RF bootstrap) + EGO")) +
#   scale_x_continuous(breaks=c(800,950,1100,1250,1400,1500))+
#   xlab("Number of virtual data")+ylab("Iteration times")+
#   theme_bw(base_size = 30)+
#   theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank(),
#         axis.text=element_text(color="black",size=24),
#         axis.title=element_text(face="bold"))
# 
# ggplot(data.frame(x=pittdat$pittx,y=pittdat$palem),aes(x,y,color=pittdat$pittd))+
#   geom_line(size=1.2)+
#   geom_errorbar(aes(x=pittdat$pittx,ymin=pittdat$palem-0.5*pittdat$palesd, ymax=pittdat$palem+0.5*pittdat$palesd), size=1.2,width=16)+
#   scale_color_manual(name = " ",values = c("grey","blue","orange","purple","red"),
#                      labels = c("kriging + EGO", "kriging with nugget + EGO", "noisy kriging (GB bootstrap) + EGO",
#                                 "noisy kriging (SVR bootstrap) + EGO","noisy kriging (RF bootstrap) + EGO")) +
#   scale_x_continuous(breaks=c(800,950,1100,1250,1400,1500))+
#   xlab("Number of virtual data")+ylab("AL efficiency")+
#   theme_bw(base_size = 30)+
#   theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank(),
#         axis.text=element_text(color="black",size=24),
#         axis.title=element_text(face="bold"))
# 
# 
# #* iteration times with ratio rfvar5a *#
# # Input: virtual data size, ratio for var, max iteration times, seed for kriging, seed for sampling data, type of utility function 
# # Output: vn, rat, opportunity cost of each iteration
# itocnrrf<-function(vn, rat, mt, sseed, dseed, uf){
#   fvs<-dirfs[1354,]  #max ECs in dirfs
#   dirfsl<-dirfs[-1354,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, dirfsl[sample(dim(dirfsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(dirfs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,11:16],t,p,rat*ftd$rfvar5a,64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       krigcvm<-try(krigm(ftd[,11:16],t,p,0,567))
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(11,12,14,15,16)],type="SK")
#       if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb)),]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,13]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego)),]
#       }
#       fvs<-anti_join(fvs,newd)
#       ftd<-rbind(ftd,newd)
#     }
#     oce<<-dirfs[1354,]$ECs-newd$ECs
#     print(oce)
#     oc<-c(oc,oce)    
#     if(oce==0){
#       break()
#     }
#   }
#   gc()
#   write.table(c(vn, rat, oc),"itocnrrf.csv",append = TRUE,sep = ",")
#   return(c(vn, rat, oc))
# }
# 
# vrim<-matrix(ncol=3)
# for(v in c(800,1100,1500)){
#   for(r in c(0.01,0.1,0.3,0.5,0.8,1.5,2,5)){
#     for(i in c(34,77,6,12)){
#       vrim<-rbind(vrim,c(v,r,i))
#     }
#   }
# }
# vrim<-vrim[-1,]
# 
# cl<-makeCluster(detectCores(),outfile="nrrfegooc.txt")
# clusterExport(cl, list("itocnrrf","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrrfpar<-function(vri){
#   return(itocnrrf(vri[1],vri[2],500,64,vri[3],"ego"))
# }
# 
# system.time(
#   nrrfegooc<-parApply(cl, vrim, 1, itocnrrfpar)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# # vn = 800 #
# itnrrfego800<-matrix(nrow = 4,ncol=8)
# for(j in 1:8){
#   for(i in 1:4){
#     itnrrfego800[i,j]<-length(nrrfegooc[[(j-1)*4+i]])-2
#   }
# }
# # plot distribution of itt for ratio RF
# pmcv<-data.frame(V1=c(rep("kriging",4),rep("noisy kriging (RF all)",4),rep("noisy kriging (0.01*RF all)",4),
#                       rep("noisy kriging (0.1*RF all)",4),rep("noisy kriging (0.3*RF all)",4),
#                       rep("noisy kriging (0.5*RF all)",4),rep("noisy kriging (0.8*RF all)",4),
#                       rep("noisy kriging (1.5*RF all)",4),rep("noisy kriging (2*RF all)",4),
#                       rep("noisy kriging (5*RF all)",4)),
#                  V2=c(itego800,itnrf5aego800,itnrrfego800[,1],
#                       itnrrfego800[,2],itnrrfego800[,3],
#                       itnrrfego800[,4],itnrrfego800[,5],
#                       itnrrfego800[,6],itnrrfego800[,7],itnrrfego800[,8]))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "noisy kriging (RF all)","noisy kriging (0.01*RF all)", 
#                                     "noisy kriging (0.1*RF all)","noisy kriging (0.3*RF all)",
#                                     "noisy kriging (0.5*RF all)","noisy kriging (0.8*RF all)",
#                                     "noisy kriging (1.5*RF all)","noisy kriging (2*RF all)",
#                                     "noisy kriging (5*RF all)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 800")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 18, hjust=0, vjust = -0.5))
# 
# # vn = 1100 #
# itnrrfego1100<-matrix(nrow = 4,ncol=8)
# for(j in 1:8){
#   for(i in 1:4){
#     itnrrfego1100[i,j]<-length(nrrfegooc[[(j-1)*4+i+32]])-2
#   }
# }
# # plot distribution of itt for ratio RF
# pmcv<-data.frame(V1=c(rep("kriging",4),rep("noisy kriging (RF all)",4),rep("noisy kriging (0.01*RF all)",4),
#                       rep("noisy kriging (0.1*RF all)",4),rep("noisy kriging (0.3*RF all)",4),
#                       rep("noisy kriging (0.5*RF all)",4),rep("noisy kriging (0.8*RF all)",4),
#                       rep("noisy kriging (1.5*RF all)",4),rep("noisy kriging (2*RF all)",4),
#                       rep("noisy kriging (5*RF all)",4)),
#                  V2=c(itego1100,itnrf5aego1100,itnrrfego1100[,1],
#                       itnrrfego1100[,2],itnrrfego1100[,3],
#                       itnrrfego1100[,4],itnrrfego1100[,5],
#                       itnrrfego1100[,6],itnrrfego1100[,7],itnrrfego1100[,8]))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "noisy kriging (RF all)","noisy kriging (0.01*RF all)", 
#                                     "noisy kriging (0.1*RF all)","noisy kriging (0.3*RF all)",
#                                     "noisy kriging (0.5*RF all)","noisy kriging (0.8*RF all)",
#                                     "noisy kriging (1.5*RF all)","noisy kriging (2*RF all)",
#                                     "noisy kriging (5*RF all)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 1100")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 18, hjust=0, vjust = -0.5))
# 
# # vn = 1500 #
# itnrrfego1500<-matrix(nrow = 4,ncol=8)
# for(j in 1:8){
#   for(i in 1:4){
#     itnrrfego1500[i,j]<-length(nrrfegooc[[(j-1)*4+i+64]])-2
#   }
# }
# # plot distribution of itt for ratio RF
# pmcv<-data.frame(V1=c(rep("kriging",4),rep("noisy kriging (RF all)",4),rep("noisy kriging (0.01*RF all)",4),
#                       rep("noisy kriging (0.1*RF all)",4),rep("noisy kriging (0.3*RF all)",4),
#                       rep("noisy kriging (0.5*RF all)",4),rep("noisy kriging (0.8*RF all)",4),
#                       rep("noisy kriging (1.5*RF all)",4),rep("noisy kriging (2*RF all)",4),
#                       rep("noisy kriging (5*RF all)",4)),
#                  V2=c(itego1500,itnrf5aego1500,itnrrfego1500[,1],
#                       itnrrfego1500[,2],itnrrfego1500[,3],
#                       itnrrfego1500[,4],itnrrfego1500[,5],
#                       itnrrfego1500[,6],itnrrfego1500[,7],itnrrfego1500[,8]))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "noisy kriging (RF all)","noisy kriging (0.01*RF all)", 
#                                     "noisy kriging (0.1*RF all)","noisy kriging (0.3*RF all)",
#                                     "noisy kriging (0.5*RF all)","noisy kriging (0.8*RF all)",
#                                     "noisy kriging (1.5*RF all)","noisy kriging (2*RF all)",
#                                     "noisy kriging (5*RF all)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 1500")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 18, hjust=0, vjust = -0.5))
# 
# 
# #* iteration times with ratio rfvar5t *#
# # Input: virtual data size, ratio for var, max iteration times, seed for kriging, seed for sampling data, type of utility function
# # Output: vn, rat, opportunity cost of each iteration
# itocnrrft<-function(vn, rat, mt, sseed, dseed, uf){
#   fvs<-dirfs[1354,]  #max ECs in dirfs
#   dirfsl<-dirfs[-1354,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, dirfsl[sample(dim(dirfsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(dirfs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,11:16],t,p,rat*ftd$rfvar5t,64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       krigcvm<-try(krigm(ftd[,11:16],t,p,0,567))
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(11,12,14,15,16)],type="SK")
#       if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb)),]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,13]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego)),][1,]
#       }
#       fvs<-anti_join(fvs,newd)
#       ftd<-rbind(ftd,newd)
#     }
#     oce<<-dirfs[1354,]$ECs-newd$ECs
#     print(oce)
#     oc<-c(oc,oce)
#     if(oce==0){
#       break()
#     }
#   }
#   gc()
#   write.table(c(vn, rat, oc),"itocnrrft.csv",append = TRUE,sep = ",")
#   return(c(vn, rat, oc))
# }
# 
# cl<-makeCluster(detectCores(),outfile="nrrftegooc.txt")
# clusterExport(cl, list("itocnrrft","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnrrftpar<-function(vri){
#   return(itocnrrft(vri[1],vri[2],500,64,vri[3],"ego"))
# }
# 
# system.time(
#   nrrftegooc<-parApply(cl, vrim, 1, itocnrrftpar)
# )
# stopCluster(cl)
# 
# save.image("~/ECEdemo/directboots/3.RData")
# 
# # vn = 800 #
# itnrrftego800<-matrix(nrow = 4,ncol=8)
# for(j in 1:8){
#   for(i in 1:4){
#     itnrrftego800[i,j]<-length(nrrftegooc[[(j-1)*4+i]])-2
#   }
# }
# # plot distribution of itt for ratio RF
# pmcv<-data.frame(V1=c(rep("kriging",4),rep("noisy kriging (RF test)",4),rep("noisy kriging (0.01*RF test)",4),
#                       rep("noisy kriging (0.1*RF test)",4),rep("noisy kriging (0.3*RF test)",4),
#                       rep("noisy kriging (0.5*RF test)",4),rep("noisy kriging (0.8*RF test)",4),
#                       rep("noisy kriging (1.5*RF test)",4),rep("noisy kriging (2*RF test)",4),
#                       rep("noisy kriging (5*RF test)",4)),
#                  V2=c(itego800,itnrf5tego800,itnrrftego800[,1],
#                       itnrrftego800[,2],itnrrftego800[,3],
#                       itnrrftego800[,4],itnrrftego800[,5],
#                       itnrrftego800[,6],itnrrftego800[,7],itnrrftego800[,8]))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "noisy kriging (RF test)","noisy kriging (0.01*RF test)", 
#                                     "noisy kriging (0.1*RF test)","noisy kriging (0.3*RF test)",
#                                     "noisy kriging (0.5*RF test)","noisy kriging (0.8*RF test)",
#                                     "noisy kriging (1.5*RF test)","noisy kriging (2*RF test)",
#                                     "noisy kriging (5*RF test)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 800")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 18, hjust=0, vjust = -0.5))
# 
# # vn = 1100 #
# itnrrftego1100<-matrix(nrow = 4,ncol=8)
# for(j in 1:8){
#   for(i in 1:4){
#     itnrrftego1100[i,j]<-length(nrrftegooc[[(j-1)*4+i+32]])-2
#   }
# }
# # plot distribution of itt for ratio RF
# pmcv<-data.frame(V1=c(rep("kriging",4),rep("noisy kriging (RF test)",4),rep("noisy kriging (0.01*RF test)",4),
#                       rep("noisy kriging (0.1*RF test)",4),rep("noisy kriging (0.3*RF test)",4),
#                       rep("noisy kriging (0.5*RF test)",4),rep("noisy kriging (0.8*RF test)",4),
#                       rep("noisy kriging (1.5*RF test)",4),rep("noisy kriging (2*RF test)",4),
#                       rep("noisy kriging (5*RF test)",4)),
#                  V2=c(itego1100,itnrf5tego1100,itnrrftego1100[,1],
#                       itnrrftego1100[,2],itnrrftego1100[,3],
#                       itnrrftego1100[,4],itnrrftego1100[,5],
#                       itnrrftego1100[,6],itnrrftego1100[,7],itnrrftego1100[,8]))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "noisy kriging (RF test)","noisy kriging (0.01*RF test)", 
#                                     "noisy kriging (0.1*RF test)","noisy kriging (0.3*RF test)",
#                                     "noisy kriging (0.5*RF test)","noisy kriging (0.8*RF test)",
#                                     "noisy kriging (1.5*RF test)","noisy kriging (2*RF test)",
#                                     "noisy kriging (5*RF test)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 1100")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 18, hjust=0, vjust = -0.5))
# 
# # vn = 1500 #
# itnrrftego1500<-matrix(nrow = 4,ncol=8)
# for(j in 1:8){
#   for(i in 1:4){
#     itnrrftego1500[i,j]<-length(nrrftegooc[[(j-1)*4+i+64]])-2
#   }
# }
# # plot distribution of itt for ratio RF
# pmcv<-data.frame(V1=c(rep("kriging",4),rep("noisy kriging (RF test)",4),rep("noisy kriging (0.01*RF test)",4),
#                       rep("noisy kriging (0.1*RF test)",4),rep("noisy kriging (0.3*RF test)",4),
#                       rep("noisy kriging (0.5*RF test)",4),rep("noisy kriging (0.8*RF test)",4),
#                       rep("noisy kriging (1.5*RF test)",4),rep("noisy kriging (2*RF test)",4),
#                       rep("noisy kriging (5*RF test)",4)),
#                  V2=c(itego1500,itnrf5tego1500,itnrrftego1500[,1],
#                       itnrrftego1500[,2],itnrrftego1500[,3],
#                       itnrrftego1500[,4],itnrrftego1500[,5],
#                       itnrrftego1500[,6],itnrrftego1500[,7],itnrrftego1500[,8]))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "noisy kriging (RF test)","noisy kriging (0.01*RF test)", 
#                                     "noisy kriging (0.1*RF test)","noisy kriging (0.3*RF test)",
#                                     "noisy kriging (0.5*RF test)","noisy kriging (0.8*RF test)",
#                                     "noisy kriging (1.5*RF test)","noisy kriging (2*RF test)",
#                                     "noisy kriging (5*RF test)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 1500")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 18, hjust=0, vjust = -0.5))
# 
# 
# #* iteration times with tvrfvar5 *#
# # Input: virtual data size, var column, max iteration times, seed for kriging, seed for sampling data, type of utility function
# # Output: vn, vc, opportunity cost of each iteration
# itocntv<-function(vn, vc, mt, sseed, dseed, uf){
#   fvs<-dirfs[1354,]  #max ECs in dirfs
#   dirfsl<-dirfs[-1354,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, dirfsl[sample(dim(dirfsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(dirfs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,11:16],t,p,ftd[,vc],64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       krigcvm<-try(krigm(ftd[,11:16],t,p,0,567))
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(11,12,14,15,16)],type="SK")
#       if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb)),][1,]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,13]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego)),][1,]
#       }
#       fvs<-anti_join(fvs,newd)
#       ftd<-rbind(ftd,newd)
#     }
#     oce<<-dirfs[1354,]$ECs-newd$ECs
#     print(oce)
#     oc<-c(oc,oce)
#     if(oce==0){
#       break()
#     }
#   }
#   gc()
#   write.table(c(vn, vc, oc),"itocntv.csv",append = TRUE,sep = ",")
#   return(c(vn, vc, oc))
# }
# 
# vvim<-matrix(ncol=3)
# for(v in c(800,950,1100,1250,1400,1500)){
#   for(vc in c("tv8rfvar5","tv10rfvar5","tv12rfvar5")){
#     for(i in c(34,77,6,12)){
#       vvim<-rbind(vvim,c(v,vc,i))
#     }
#   }
# }
# vvim<-vvim[-1,]
# 
# cl<-makeCluster(detectCores(),outfile="ntvrfegooc.txt")
# clusterExport(cl, list("itocntv","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocntvrf<-function(vvi){
#   return(itocntv(as.numeric(vvi[1]),vvi[2],500,64,as.numeric(vvi[3]),"ego"))
# }
# 
# system.time(
#   ntvrfegooc<-parApply(cl, vvim, 1, itocntvrf)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# vvim2<-matrix(ncol=3)
# for(v in c(800,950,1100,1250,1400,1500)){
#   for(vc in c("tv6rfvar5","tv14rfvar5","tv15rfvar5")){
#     for(i in c(34,77,6,12)){
#       vvim2<-rbind(vvim2,c(v,vc,i))
#     }
#   }
# }
# vvim2<-vvim2[-1,]
# 
# cl<-makeCluster(detectCores(),outfile="ntvrfegooc.txt")
# clusterExport(cl, list("itocntv","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocntvrf<-function(vvi){
#   return(itocntv(as.numeric(vvi[1]),vvi[2],500,64,as.numeric(vvi[3]),"ego"))
# }
# 
# system.time(
#   ntvrfegooc_2<-parApply(cl, vvim2, 1, itocntvrf)
# )
# stopCluster(cl)
# 
# save.image("~/ECEdemo/directboots/3.RData")
# 
# vvim3<-matrix(ncol=3)
# for(v in c(800,950,1100,1250,1400,1500)){
#   for(vc in c("tv155rfvar5")){
#     for(i in c(34,77,6,12)){
#       vvim3<-rbind(vvim3,c(v,vc,i))
#     }
#   }
# }
# vvim3<-vvim3[-1,]
# 
# cl<-makeCluster(detectCores(),outfile="ntvrfegooc.txt")
# clusterExport(cl, list("itocntv","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocntvrf<-function(vvi){
#   return(itocntv(as.numeric(vvi[1]),vvi[2],500,64,as.numeric(vvi[3]),"ego"))
# }
# 
# system.time(
#   ntvrfegooc_3<-parApply(cl, vvim3, 1, itocntvrf)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# # vn = 800 #
# itntvrfego800<-matrix(nrow = 4,ncol=3)
# for(j in 1:3){
#   for(i in 1:4){
#     itntvrfego800[i,j]<-length(ntvrfegooc[[(j-1)*4+i]])-2
#   }
# }
# itntvrfego800_2<-matrix(nrow = 4,ncol=3)
# for(j in 1:3){
#   for(i in 1:4){
#     itntvrfego800_2[i,j]<-length(ntvrfegooc_2[[(j-1)*4+i]])-2
#   }
# }
# itntv155rfego800<-c()
# for(i in 1:4){
#   itntv155rfego800<-c(itntv155rfego800,length(ntvrfegooc_3[[i]])-2)
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",4),rep("kriging with nugget",4),rep("noisy kriging (RF bootstrap)",4),
#                       rep("noisy kriging (1000 test RF)",4),rep("noisy kriging (800 test RF)",4),
#                       rep("noisy kriging (600 test RF)",4),rep("noisy kriging (400 test RF)",4),
#                       rep("noisy kriging (200 test RF)",4),rep("noisy kriging (100 test RF)",4),
#                       rep("noisy kriging (50 test RF)",4)),
#                  V2=c(itoego[,1],itego800,itnrf5tego800,itntvrfego800_2[,1],itntvrfego800[,1],
#                       itntvrfego800[,2],itntvrfego800[,3],itntvrfego800_2[,2],itntvrfego800_2[,3],
#                       itntv155rfego800),
#                  V3=c(c(mean(itoego[,1]),rep(NA,3)),c(mean(itego800),rep(NA,3)),c(mean(itnrf5tego800),rep(NA,3)),
#                       c(mean(itntvrfego800_2[,1]),rep(NA,3)),c(mean(itntvrfego800[,1]),rep(NA,3)),
#                       c(mean(itntvrfego800[,2]),rep(NA,3)),c(mean(itntvrfego800[,3]),rep(NA,3)),
#                       c(mean(itntvrfego800_2[,2]),rep(NA,3)),c(mean(itntvrfego800_2[,3]),rep(NA,3)),
#                       c(mean(itntv155rfego800),rep(NA,3))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (1000 test RF)","noisy kriging (800 test RF)",
#                                     "noisy kriging (600 test RF)","noisy kriging (400 test RF)",
#                                     "noisy kriging (200 test RF)","noisy kriging (100 test RF)",
#                                     "noisy kriging (50 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 800")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# # vn = 1100 #
# itntvrfego1100<-matrix(nrow = 4,ncol=3)
# for(j in 1:3){
#   for(i in 1:4){
#     itntvrfego1100[i,j]<-length(ntvrfegooc[[(j-1)*4+i+24]])-2
#   }
# }
# itntvrfego1100_2<-matrix(nrow = 4,ncol=3)
# for(j in 1:3){
#   for(i in 1:4){
#     itntvrfego1100_2[i,j]<-length(ntvrfegooc_2[[(j-1)*4+i+24]])-2
#   }
# }
# itntv155rfego1100<-c()
# for(i in 1:4){
#   itntv155rfego1100<-c(itntv155rfego1100,length(ntvrfegooc_3[[i+8]])-2)
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",4),rep("kriging with nugget",4),rep("noisy kriging (RF bootstrap)",4),
#                       rep("noisy kriging (1000 test RF)",4),rep("noisy kriging (800 test RF)",4),
#                       rep("noisy kriging (600 test RF)",4),rep("noisy kriging (400 test RF)",4),
#                       rep("noisy kriging (200 test RF)",4),rep("noisy kriging (100 test RF)",4),
#                       rep("noisy kriging (50 test RF)",4)),
#                  V2=c(itoego[,3],itego1100,itnrf5tego1100,itntvrfego1100_2[,1],itntvrfego1100[,1],
#                       itntvrfego1100[,2],itntvrfego1100[,3],itntvrfego1100_2[,2],itntvrfego1100_2[,3],
#                       itntv155rfego1100),
#                  V3=c(c(mean(itoego[,3]),rep(NA,3)),c(mean(itego1100),rep(NA,3)),c(mean(itnrf5tego1100),rep(NA,3)),
#                       c(mean(itntvrfego1100_2[,1]),rep(NA,3)),c(mean(itntvrfego1100[,1]),rep(NA,3)),
#                       c(mean(itntvrfego1100[,2]),rep(NA,3)),c(mean(itntvrfego1100[,3]),rep(NA,3)),
#                       c(mean(itntvrfego1100_2[,2]),rep(NA,3)),c(mean(itntvrfego1100_2[,3]),rep(NA,3)),
#                       c(mean(itntv155rfego1100),rep(NA,3))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (1000 test RF)","noisy kriging (800 test RF)",
#                                     "noisy kriging (600 test RF)","noisy kriging (400 test RF)",
#                                     "noisy kriging (200 test RF)","noisy kriging (100 test RF)",
#                                     "noisy kriging (50 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 1100")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# # vn = 1500 #
# itntvrfego1500<-matrix(nrow = 4,ncol=3)
# for(j in 1:3){
#   for(i in 1:4){
#     itntvrfego1500[i,j]<-length(ntvrfegooc[[(j-1)*4+i+60]])-2
#   }
# }
# itntvrfego1500_2<-matrix(nrow = 4,ncol=3)
# for(j in 1:3){
#   for(i in 1:4){
#     itntvrfego1500_2[i,j]<-length(ntvrfegooc_2[[(j-1)*4+i+60]])-2
#   }
# }
# itntv155rfego1500<-c()
# for(i in 1:4){
#   itntv155rfego1500<-c(itntv155rfego1500,length(ntvrfegooc_3[[i+20]])-2)
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",4),rep("kriging with nugget",4),rep("noisy kriging (RF bootstrap)",4),
#                       rep("noisy kriging (1000 test RF)",4),rep("noisy kriging (800 test RF)",4),
#                       rep("noisy kriging (600 test RF)",4),rep("noisy kriging (400 test RF)",4),
#                       rep("noisy kriging (200 test RF)",4),rep("noisy kriging (100 test RF)",4),
#                       rep("noisy kriging (50 test RF)",4)),
#                  V2=c(itoego[,6],itego1500,itnrf5tego1500,itntvrfego1500_2[,1],itntvrfego1500[,1],
#                       itntvrfego1500[,2],itntvrfego1500[,3],itntvrfego1500_2[,2],itntvrfego1500_2[,3],
#                       itntv155rfego1500),
#                  V3=c(c(mean(itoego[,6]),rep(NA,3)),c(mean(itego1500),rep(NA,3)),c(mean(itnrf5tego1500),rep(NA,3)),
#                       c(mean(itntvrfego1500_2[,1]),rep(NA,3)),c(mean(itntvrfego1500[,1]),rep(NA,3)),
#                       c(mean(itntvrfego1500[,2]),rep(NA,3)),c(mean(itntvrfego1500[,3]),rep(NA,3)),
#                       c(mean(itntvrfego1500_2[,2]),rep(NA,3)),c(mean(itntvrfego1500_2[,3]),rep(NA,3)),
#                       c(mean(itntv155rfego1500),rep(NA,3))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (1000 test RF)","noisy kriging (800 test RF)",
#                                     "noisy kriging (600 test RF)","noisy kriging (400 test RF)",
#                                     "noisy kriging (200 test RF)","noisy kriging (100 test RF)",
#                                     "noisy kriging (50 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 1500")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# 
# # 不同虚拟空间，用四种效能函数
# # Input: virtual data size, max iteration times, seed for kriging, seed for sampling data, type of utility function, type of var
# # Output: vn, vt, opportunity cost of each iteration
# itocn<-function(vn, mt, sseed, dseed, uf, vt){
#   fvs<-dirfs[1354,]  #max ECs in dirfs
#   dirfsl<-dirfs[-1354,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, dirfsl[sample(dim(dirfsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(dirfs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,11:16],t,p,ftd[,vt],64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       krigcvm<-try(krigm(ftd[,11:16],t,p,0,567))
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(11,12,14,15,16)],type="SK")
#       if(uf=="pre"){
#         newd<<-fvs[which(kpre[["mean"]]==max(kpre[["mean"]],na.rm=T)),][1,]
#       }else if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb,na.rm=T)),][1,]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,13]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego,na.rm=T)),][1,]
#       }
#       fvs<-anti_join(fvs,newd)
#       ftd<-rbind(ftd,newd)
#     }
#     oce<<-dirfs[1354,]$ECs-newd$ECs
#     print(oce)
#     oc<-c(oc,oce)
#     if(oce==0){
#       break()
#     }
#   }
#   gc()
#   write.table(c(vn, vt, oc),"itocn.csv",append = TRUE,sep = ",")
#   return(c(vn, vt, oc))
# }
# 
# cl<-makeCluster(detectCores(),outfile="nufoc.txt")
# clusterExport(cl, list("itocn","krigm","t","p","dirfs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnuf<-function(ui){
#   return(itocn(as.numeric(ui[1]),1000,64,as.numeric(ui[3]),ui[2],"tv15rfvar5"))
# }
# 
# system.time(
#   nufoc<-parApply(cl, vuim, 1, itocnuf)
# )
# stopCluster(cl)
# 
# save.image("3.RData")
# 
# itnuf<-matrix(nrow = 100,ncol=9)
# for(j in 1:9){
#   for(i in 1:100){
#     itnuf[i,j]<-length(nufoc[[(j-1)*100+i]])-2
#   }
# }
# colnames(itnuf)<-c("pre7","ucb7","ego7","pre8","ucb8","ego8","pre9","ucb9","ego9")
# # 
# # plot distribution of itt uf
# #70% vs
# pmcv<-data.frame(V1=c(rep("kriging + Pre",100),rep("kriging + UCB",100),rep("kriging + EGO",100),
#                       rep("kriging with nugget + Pre",100),rep("kriging with nugget + UCB",100),
#                       rep("kriging with nugget + EGO",100),rep("kriging with nugget + SKO",100),
#                       rep("noisy kriging (100 test RF) + Pre",100),rep("noisy kriging (100 test RF) + UCB",100),
#                       rep("noisy kriging (100 test RF) + EGO",100)),
#                  V2=c(itouf[,1],itouf[,2],itouf[,3],ituf[,1],ituf[,2],ituf[,3],ituf[,4],
#                       itnuf[,1],itnuf[,2],itnuf[,3]),
#                  V3=c(c(mean(itouf[,1]),rep(NA,99)),c(mean(itouf[,2]),rep(NA,99)),c(mean(itouf[,3]),rep(NA,99)),
#                       c(mean(ituf[,1]),rep(NA,99)),c(mean(ituf[,2]),rep(NA,99)),c(mean(ituf[,3]),rep(NA,99)),c(mean(ituf[,4]),rep(NA,99)),
#                       c(mean(itnuf[,1]),rep(NA,99)),c(mean(itnuf[,2]),rep(NA,99)),c(mean(itnuf[,3]),rep(NA,99))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging + Pre", "kriging + UCB", "kriging + EGO",
#                                     "kriging with nugget + Pre","kriging with nugget + UCB",
#                                     "kriging with nugget + EGO","kriging with nugget + SKO",
#                                     "noisy kriging (100 test RF) + Pre","noisy kriging (100 test RF) + UCB",
#                                     "noisy kriging (100 test RF) + EGO"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("EC strength multi-source 70% virtual space")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# #80% vs
# pmcv<-data.frame(V1=c(rep("kriging + Pre",100),rep("kriging + UCB",100),rep("kriging + EGO",100),
#                       rep("kriging with nugget + Pre",100),rep("kriging with nugget + UCB",100),
#                       rep("kriging with nugget + EGO",100),rep("kriging with nugget + SKO",100),
#                       rep("noisy kriging (100 test RF) + Pre",100),rep("noisy kriging (100 test RF) + UCB",100),
#                       rep("noisy kriging (100 test RF) + EGO",100)),
#                  V2=c(itouf[,4],itouf[,5],itouf[,6],ituf[,5],ituf[,6],ituf[,7],ituf[,8],
#                       itnuf[,4],itnuf[,5],itnuf[,6]),
#                  V3=c(c(mean(itouf[,4]),rep(NA,99)),c(mean(itouf[,5]),rep(NA,99)),c(mean(itouf[,6]),rep(NA,99)),
#                       c(mean(ituf[,5]),rep(NA,99)),c(mean(ituf[,6]),rep(NA,99)),c(mean(ituf[,7]),rep(NA,99)),c(mean(ituf[,8]),rep(NA,99)),
#                       c(mean(itnuf[,4]),rep(NA,99)),c(mean(itnuf[,5]),rep(NA,99)),c(mean(itnuf[,6]),rep(NA,99))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging + Pre", "kriging + UCB", "kriging + EGO",
#                                     "kriging with nugget + Pre","kriging with nugget + UCB",
#                                     "kriging with nugget + EGO","kriging with nugget + SKO",
#                                     "noisy kriging (100 test RF) + Pre","noisy kriging (100 test RF) + UCB",
#                                     "noisy kriging (100 test RF) + EGO"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("EC strength multi-source 80% virtual space")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# #90% vs
# pmcv<-data.frame(V1=c(rep("kriging + Pre",100),rep("kriging + UCB",100),rep("kriging + EGO",100),
#                       rep("kriging with nugget + Pre",100),rep("kriging with nugget + UCB",100),
#                       rep("kriging with nugget + EGO",100),rep("kriging with nugget + SKO",100),
#                       rep("noisy kriging (100 test RF) + Pre",100),rep("noisy kriging (100 test RF) + UCB",100),
#                       rep("noisy kriging (100 test RF) + EGO",100)),
#                  V2=c(itouf[,7],itouf[,8],itouf[,9],ituf[,9],ituf[,10],ituf[,11],ituf[,12],
#                       itnuf[,7],itnuf[,8],itnuf[,9]),
#                  V3=c(c(mean(itouf[,7]),rep(NA,99)),c(mean(itouf[,8]),rep(NA,99)),c(mean(itouf[,9]),rep(NA,99)),
#                       c(mean(ituf[,9]),rep(NA,99)),c(mean(ituf[,10]),rep(NA,99)),c(mean(ituf[,11]),rep(NA,99)),c(mean(ituf[,12]),rep(NA,99)),
#                       c(mean(itnuf[,7]),rep(NA,99)),c(mean(itnuf[,8]),rep(NA,99)),c(mean(itnuf[,9]),rep(NA,99))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging + Pre", "kriging + UCB", "kriging + EGO",
#                                     "kriging with nugget + Pre","kriging with nugget + UCB",
#                                     "kriging with nugget + EGO","kriging with nugget + SKO",
#                                     "noisy kriging (100 test RF) + Pre","noisy kriging (100 test RF) + UCB",
#                                     "noisy kriging (100 test RF) + EGO"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("EC strength multi-source 90% virtual space")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# ##************##
# 
# 
# 
# ##** Predicting in virtual space **##
# btovs1<-read.csv("~/ECEdemo/directboots/BTO-VS 2.csv")[,2:8]
# btovs1<-cbind(btovs1,read.csv("~/ECEdemo/directboots/BTO-VS 2.csv")[,c("NCT", "z")])
# btovs1<-rename(btovs1,zeff=z)
# btovs1[,"E"]<-rep(20,dim(btovs1)[1])
# btovs1[,"T_K"]<-rep(293,dim(btovs1)[1])
# btovs1[,"sTmt"]<-rep(0.277778,dim(btovs1)[1])
# btovs1<-btovs1[,c(1:7,10,11,8,9,12)]
# gc()
# #* prediction models *#
# preo<-krigmo(dirfs[,11:16],t,p,64)
# prenu<-krigm(dirfs[,11:16],t,p,0,64)
# preno<-krigm(dirfs[,11:16],t,p,dirfs[,"tv15rfvar5"],64)
prenosa<-krigm(dirfs[,11:16],t,p,dirfs[,"tv15rfvar5m"],64)
# 
# # Input: the number(/10000) of btovs   Output: number, mean and sigma2 of predictions
# prevs<-function(n){
#   gc()
#   preovs<-predict(preo, btovs1[((n-1)*10000+1):(n*10000), c(8:12)], type = "SK")
#   prenuvs<-predict(prenu, btovs1[((n-1)*10000+1):(n*10000), c(8:12)], type = "SK")
#   prenovs<-predict(preno, btovs1[((n-1)*10000+1):(n*10000), c(8:12)], type = "SK")
#   gc()
#   prevsp<-data.frame(ind=c(((n-1)*10000+1):(n*10000)), om=preovs[["mean"]], os=preovs[["sd"]],
#                      num=prenuvs[["mean"]], nus=prenuvs[["sd"]],
#                      nom=prenovs[["mean"]], nos=prenovs[["sd"]])
#   gc()
#   print(n)
#   return(prevsp)
# }
# 
# #cluster initialization
# cores<-detectCores()
# cl<-makeCluster(cores,outfile="prevs.txt")
# clusterExport(cl, list("btovs1","preo","prenu","preno"))
# clusterEvalQ(cl,{library(MuFiCokriging)})
# 
# system.time(
#   prevsall<-parLapply(cl, c(1:27), prevs)
# )
# 
# stopCluster(cl)
# 
# prevsallc<-prevsall[[1]]
# for(i in 2:length(prevsall)){
#   prevsallc<-rbind(prevsallc,prevsall[[i]])
# }
# rm(prevsall)
# gc()
# preovs_la<-predict(preo, btovs1[270001:273897, c(8:12)], type = "SK")
# prenuvs_la<-predict(prenu, btovs1[270001:273897, c(8:12)], type = "SK")
# prenovs_la<-predict(preno, btovs1[270001:273897, c(8:12)], type = "SK")
# gc()
# prevsallc<-rbind(prevsallc, data.frame(ind=c(270001:273897), om=preovs_la[["mean"]], os=preovs_la[["sd"]],
#                                        num=prenuvs_la[["mean"]], nus=prenuvs_la[["sd"]],
#                                        nom=prenovs_la[["mean"]], nos=prenovs_la[["sd"]]))
# gc()
# 
# prevsallc[,"div"]<-abs(prevsallc$om-prevsallc$num)+abs(prevsallc$num-prevsallc$nom)+
#   abs(prevsallc$nom-prevsallc$om)
# divind<-order(prevsallc$div, decreasing = T)
# gc()
# 
# vso<-cbind(btovs1[divind[1:20],], prevsallc[divind[1:20],-1])
# 
# rm(prevsallc)
# rm(preovs_la)
# rm(prenuvs_la)
# rm(prenovs_la)
# rm(btovs1)
# gc()
# 
# save.image("3.RData")
# 
# 
# #predict some selected samples
# btovss<-read.csv("selected VS.csv")[,2:8]
# btovss<-cbind(btovss,read.csv("selected VS.csv")[,c("NCT", "z")])
# btovss<-rename(btovss,zeff=z)
# btovss[,"E"]<-rep(20,dim(btovss)[1])
# btovss[,"T_K"]<-rep(293,dim(btovss)[1])
# btovss[,"sTmt"]<-rep(0.277778,dim(btovss)[1])
# btovss<-btovss[,c(1:7,10,11,8,9,12)]
# 
# preovss<-predict(preo, btovss[, c(8:12)], type = "SK")
# prenuvss<-predict(prenu, btovss[, c(8:12)], type = "SK")
# prenovss<-predict(preno, btovss[, c(8:12)], type = "SK")
# prevssp<-data.frame(ind=c(1:12), om=preovss[["mean"]], os=preovss[["sd"]],
#                    num=prenuvss[["mean"]], nus=prenuvss[["sd"]],
#                    nom=prenovss[["mean"]], nos=prenovss[["sd"]])
# prevssp$div<-abs(prevssp$om-prevssp$num)+abs(prevssp$num-prevssp$nom)+
#   abs(prevssp$nom-prevssp$om)
# 
# 
# #predict some random selected samples
# btovsrs<-read.csv("random selected VS +.csv")[,c(1:7,11)]
# btovsrs<-cbind(btovsrs,read.csv("random selected VS +.csv")[,c("NCT","z")])
# btovsrs<-rename(btovsrs,zeff=z)
# btovsrs[,"E"]<-rep(20,dim(btovsrs)[1])
# btovsrs[,"T_K"]<-rep(293,dim(btovsrs)[1])
# btovsrs[,"sTmt"]<-rep(0.277778,dim(btovsrs)[1])
# btovsrs[10,"sTmt"]<-0.222222
# btovsrs<-btovsrs[,c(1:8,11,12,9,10,13)]
# 
# library(Rtsne)
# intsnet<-as.matrix(rbind(dirfs[,c(11,12,14:16)],btovsrs[,c(8:12)]))
# #normalize
# for(i in 1:5){
#   intsnet[,i]<-(intsnet[,i]-min(intsnet[,i])) / (max(intsnet[,i])-min(intsnet[,i]))
# }
# intsnet<-normalize_input(intsnet)
# set.seed(3)
# intsne<-Rtsne(intsnet)
# plot(x=intsne$Y[,1],y=intsne$Y[,2],
#      pch= c(rep(1,length(dirfs[,1])),rep(17,length(btovsrs[,1]))),
#      col= c(rep("red",length(dirfs[,1])),rep("black",length(btovsrs[,1]))), cex = 1.2,
#      #xlim = c(-20,28), ylim = c(-25,30),
#      xlab="tSNE_1",ylab="tSNE_2",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# 
# preovsrs<-predict(preo, btovsrs[, c(9:13)], type = "SK")
# prenuvsrs<-predict(prenu, btovsrs[, c(9:13)], type = "SK")
# prenovsrs<-predict(preno, btovsrs[, c(9:13)], type = "SK")
# prenosavsrs<-predict(prenosa, btovsrs[, c(9:13)], type = "SK")
# prevsrsp<-data.frame(btovsrs, om=preovsrs[["mean"]], os=preovsrs[["sd"]],
#                      num=prenuvsrs[["mean"]], nus=prenuvsrs[["sd"]],
#                      nom=prenovsrs[["mean"]], nos=prenovsrs[["sd"]],
#                      nosam=prenosavsrs[["mean"]], nosas=prenosavsrs[["sd"]])
# write.table(prevsrsp,"exp compare.csv",append = T,sep = ",")
# 
# library(readxl)
# expcomp <- read_excel("experiments compare.xlsx", sheet = "ECs")
# expcomp<-as.matrix(expcomp)
# plot(rep(expcomp[,4],3),y=c(expcomp[,1],expcomp[,2],expcomp[,3]),
#      pch=c(rep(15,4),rep(16,4),rep(17,4)),
#      col= c(rep("grey",4),rep("red",4),rep("blue",4)), cex = 1.4,
#      xlim = c(0.075,0.22), ylim = c(0.075,0.22),
#      xlab="Measured ECs (1e-6K▪m/V)", ylab="Predicted ECs (1e-6K▪m/V)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2)



refds<-c()
for(i in 1:28){
  dirfsp<-dirfs[which(dirfs$reference_id==i),]
  refds<-c(refds,rep(dim(dirfsp)[1],dim(dirfsp)[1]))
}
refval<-c()
for(i in 1:28){
  dirfsp<-dirfs[which(dirfs$reference_id==i),13]
  refval<-c(refval,rep(mean(dirfsp),length(dirfsp)))
}
tv15rfvar5sd<-c()
for(i in 1:28){
  tv15rfvar5p<-dirfs[which(dirfs$reference_id==i),]$tv15rfvar5
  tv15rfvar5sd<-c(tv15rfvar5sd,rep(sd(tv15rfvar5p),length(tv15rfvar5p)))
}
ttt<-unique(cbind(dirfs[,c(19,36)],refds,refval,tv15rfvar5sd))
library(openxlsx)
write.xlsx(ttt,"Fig S3 data.xlsx")