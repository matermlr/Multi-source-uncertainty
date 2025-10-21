#相比于3版 补充更多论文数据，目标d33*

setwd("~/ECEdemo/directboots/otherd")
load("~/ECEdemo/directboots/otherd/estr4.RData")
# setwd("F:/机器学习/Multi-fidelity/ECE demo/directboots/otherd")

# library(readxl)
library(dplyr)
# library(corrgram)      #for PCC
library(MuFiCokriging)
library(parallel)
library(gbm)           #for gradient boosting
library(e1071)         #for SVR.rbf
library(randomForest)
# library(ggplot2)
# library(plotly)        #for 3D visualization

# ##** input data and preprocessing **##
# 
# #* input yuan's data *#
# estr<-read.csv("Electrostrain.csv", header=T)
# estrs<-cbind(estr[,c(2:9,1)],estr[,c("NCT","p", "tA.B", "AV", "ENMB","D","NTO")])
# estrs$y<-(estrs$y/2)*1e4   #transfer stran to d33*
# estrs<-cbind(refID=rep(1,81),estrs)
# 
# #* input references' data *#
# estrr<-read_excel("Electrostrain bipolar.xlsx")[82:180,]
# 
# #* rearrange data *#
# estrr<-estrr[,c(1,6:13,17)]
# colnames(estrr)[2:9]<-c("ba","ca","sr","cd","ti","zr","sn","hf")
# estrr$`d33*_pm/V`<-round(estrr$`d33*_pm/V`,1)
# 
# #* unique *#
# estrr<-anti_join(estrr,estrr[duplicated(estrr[,2:9]),])
# 
# #* calculate features *#
# #run ceramic-fea.R
# estrrf<-fn.data.features(estrr)
# 
# #* data for model *#
# estrrs<-cbind(estrrf[,c(1:10)],estrrf[,c("NCT","p", "tA.B", "AV", "ENMB","D","NTO")])
# colnames(estrrs)<-colnames(estrs)
# estrs<-rbind(estrs,estrrs)
# estrs<-estrs[,c(2:17,1)]
# 
# # rm.NA #
# for(i in 1:dim(estrs)[2]){
#   estrs<-subset(estrs,!is.na(estrs[,i]))
# }
# 
# # unique #
# estrs<-anti_join(estrs,estrs[duplicated(estrs[,-c(9,17)]),])
# 
# ##*************##
# 
# 
# 
# ##** bootstrap **##
# testsamp<-matrix(nrow=500,ncol=164)
# for(i in 1:500){
#   set.seed(i)
#   testsamp[i,]<-sample(dim(estrs)[1], dim(estrs)[1], replace = TRUE)
# }
# duplicated(testsamp)
#   
# #* GB *#
# # tune #
# #Input: parameters vector  Output: parameters, cve, r2
# gbpd<-function(paras){
#   gbr2<-c()
#   gbcv<-c()
#   for (i in 3:7){
#     set.seed(11+20*i)
#     gbdt<-try(gbm(y~., data = estrs[,9:16], n.trees = paras[1],
#                   interaction.depth = paras[2], shrinkage = paras[3], cv.folds = 10))
#     if ('try-error' %in% class(gbdt)) {
#       gbr2<-c(gbr2,NA)
#     }else{
#       gbr2<-c(gbr2,1-sum((estrs[,9]-as.numeric(gbdt$fit))^2)/sum((estrs[,9]-mean(unlist(estrs[,9])))^2))
#     }
#     estrr<-estrs[sample(dim(estrs)[1],dim(estrs)[1]),9:16] #random order
#     gbcve<-c()
#     for (j in 0:9){
#       dim10<-dim(estrr)[1]%/%10
#       estrcv1<-estrr[(1+dim10*j):(dim10*(j+1)),]
#       estrcv2<-estrr[-((1+dim10*j):(dim10*(j+1))),]
#       gbcvm<-try(gbm(y~., data = estrcv2, n.trees = paras[1],
#                      interaction.depth = paras[2], shrinkage = paras[3], cv.folds = 10))
#       if ('try-error' %in% class(gbcvm)) {
#         gbcve<-c(gbcve,NA)
#       }else{
#         gbcve<-c(gbcve,sum((estrcv1[,1]-predict(gbcvm,estrcv1))^2)/dim10)
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
# for(nt in c(100,200,500,1000,2000)){
#   for(id in c(2,4,8,16)){
#     for(sh in c(0.01,0.1))
#     gbin<-rbind(gbin,c(nt,id,sh))
#   }
# }
# gbin<-gbin[-1,]
# 
# gbin2<-matrix(ncol=3)
# for(nt in c(1000,2000,3000,5000)){
#   for(id in c(4,6,8)){
#     for(sh in c(0.001,0.005,0.01))
#       gbin2<-rbind(gbin2,c(nt,id,sh))
#   }
# }
# gbin2<-gbin2[-1,]
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="gbdtune.txt")
# clusterExport(cl, list("gbpd","estrs"))
# clusterEvalQ(cl,{library(gbm)})
# 
# system.time(
#   gbdtune<-parApply(cl, gbin2, 1, gbpd)  #gbtune is a matrix
# )
# 
# stopCluster(cl)
# 
# save.image("estr4.RData")
# 
# gbdbp<-gbdtune[,which.min(gbdtune[4,])]
# 
# save.image("estr4.RData")
# 
# # Input: seed(bootstrap number)  Output: GB predictions
# gbbt<-function(B){
#   set.seed(B)
#   estrbt<-estrs[sample(dim(estrs)[1], dim(estrs)[1], replace = TRUE),9:16]
#   estrbt<-estrbt[!duplicated(estrbt),]
#   estrbttest<-anti_join(estrs,estrbt)
#   btgbm<-gbm(y~., data = estrbt, n.trees = 3000,
#              interaction.depth = 4, shrinkage = 0.005, cv.folds = 10,
#              n.minobsinnode=2)
#   prebttest<-predict(btgbm,estrbttest)
#   pretestna<-c()
#   for(i in 1:dim(estrs)[1]){
#     ind<-which(estrbttest$NCT==estrs$NCT[i] & estrbttest$p==estrs$p[i] &
#                  estrbttest$tA.B==estrs$tA.B[i] & estrbttest$AV==estrs$AV[i] &
#                  estrbttest$ENMB==estrs$ENMB[i] & estrbttest$D==estrs$D[i] & 
#                  estrbttest$NTO==estrs$NTO[i])
#     if(length(ind)==0){
#       pretestna<-c(pretestna,NA)
#     }else{
#       pretestna<-c(pretestna,prebttest[ind])
#     }
#   }
#   gc()
#   write.table(data.frame(pretestna,predict(btgbm,estrs)),"gbbt.csv",append=T,sep = ",")
#   return(data.frame(pretestna,predict(btgbm,estrs)))
# }
# 
# # B=500 #
# cl<-makeCluster(detectCores(),outfile="gbbt5.txt")
# clusterExport(cl, list("gbbt","estrs"))
# clusterEvalQ(cl,{library(gbm);library(dplyr)})
# 
# system.time(
#   gbbtpre5<-parLapply(cl, 1:500, gbbt)
# )
# stopCluster(cl)
# 
# save.image("estr4.RData")
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
# estrs<-cbind(estrs,btvar5t)
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
#     svrm<-try(svm(y~.,data=estrs[,9:16],type="eps-regression",kernel="radial",
#                   cost=cg[1],gamma=cg[2]))
#     if ('try-error' %in% class(svrm)) {
#       svrr2<-c(svrr2,NA)
#     }else{
#       svrr2<-c(svrr2,1-sum((predict(svrm,estrs)-estrs[,9])^2)/sum((estrs[,9]-mean(unlist(estrs[,9])))^2))
#     }
#     estrr<-estrs[sample(dim(estrs)[1],dim(estrs)[1]),9:16] #random order
#     svrcve<-c()
#     for (j in 0:9){
#       dim10<-dim(estrr)[1]%/%10
#       estrcv1<-estrr[(1+dim10*j):(dim10*(j+1)),]
#       estrcv2<-estrr[-((1+dim10*j):(dim10*(j+1))),]
#       svrcvm<-try(svm(y~.,data=estrcv2,type="eps-regression",kernel="radial",
#                       cost=cg[1],gamma=cg[2]))
#       if ('try-error' %in% class(svrcvm)) {
#         svrcve<-c(svrcve,NA)
#       }else{
#         svrcve<-c(svrcve,sum((estrcv1[,1]-predict(svrcvm,estrcv1))^2)/dim10)
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
# for(c in c(0.1,1,10,50,100)){
#   for(g in c(0.1,0.5,1,10,20)){
#     svrin<-rbind(svrin,c(c,g))
#   }
# }
# svrin<-svrin[-1,]
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="svrdtune.txt")
# clusterExport(cl, list("svrpd","estrs"))
# clusterEvalQ(cl,{library(e1071)})
# 
# system.time(
#   svrdtune<-parApply(cl, svrin, 1, svrpd)
# )
# 
# stopCluster(cl)
# save.image("estr4.RData")
# svrdbp<-svrdtune[,which.min(svrdtune[3,])]
# save.image("estr4.RData")
# 
# # Input: seed(bootstrap number)  Output: SVR.r predictions
# svrbt<-function(B){
#   set.seed(B)
#   estrbt<-estrs[sample(dim(estrs)[1], dim(estrs)[1], replace = TRUE),9:16]
#   estrbt<-estrbt[!duplicated(estrbt),]
#   estrbttest<-anti_join(estrs[,9:16],estrbt)
#   btsvrm<-svm(y~.,data=estrbt,type="eps-regression",kernel="radial",
#               cost=10,gamma=1)
#   prebttest<-predict(btsvrm,estrbttest)
#   pretestna<-c()
#   for(i in 1:dim(estrs)[1]){
#     ind<-which(estrbttest$NCT==estrs$NCT[i] & estrbttest$p==estrs$p[i] &
#                  estrbttest$tA.B==estrs$tA.B[i] & estrbttest$AV==estrs$AV[i] &
#                  estrbttest$ENMB==estrs$ENMB[i] & estrbttest$D==estrs$D[i] &
#                  estrbttest$NTO==estrs$NTO[i])
#     if(length(ind)==0){
#       pretestna<-c(pretestna,NA)
#     }else{
#       pretestna<-c(pretestna,prebttest[ind])
#     }
#   }
#   gc()
#   write.table(data.frame(pretestna,predict(btsvrm,estrs)),"svrbt.csv",append=T,sep = ",")
#   return(data.frame(pretestna,predict(btsvrm,estrs)))
# }
# 
# # B=500 #
# cl<-makeCluster(detectCores(),outfile="svrbt5.txt")
# clusterExport(cl, list("svrbt","estrs"))
# clusterEvalQ(cl,{library(e1071);library(dplyr)})
# 
# system.time(
#   svrbtpre5<-parLapply(cl, 1:500, svrbt)
# )
# stopCluster(cl)
# 
# save.image("estr4.RData")
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
# estrs<-cbind(estrs,svrvar5t)
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
#     rfm<-try(randomForest(y~.,data=estrs[,9:16],ntree=nm[1],mtry=nm[2]))
#     if ('try-error' %in% class(rfm)) {
#       rfr2<-c(rfr2,NA)
#     }else{
#       rfr2<-c(rfr2,1-sum((rfm$y-rfm$predicted)^2)/sum((estrs[,9]-mean(unlist(estrs[,9])))^2))
#     }
#     estrr<-estrs[sample(dim(estrs)[1],dim(estrs)[1]),9:16] #random order
#     rfcve<-c()
#     for (j in 0:9){
#       dim10<-dim(estrr)[1]%/%10
#       estrcv1<-estrr[(1+dim10*j):(dim10*(j+1)),]
#       estrcv2<-estrr[-((1+dim10*j):(dim10*(j+1))),]
#       rfcvm<-try(randomForest(y~.,data=estrcv2,ntree=nm[1],mtry=nm[2]))
#       if ('try-error' %in% class(rfcvm)) {
#         rfcve<-c(rfcve,NA)
#       }else{
#         rfcve<-c(rfcve,sum((estrcv1[,1]-predict(rfcvm,estrcv1))^2)/dim10)
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
# rfin2<-matrix(ncol=2)
# for(nt in c(1900,2000,2100,2500)){
#   for(mt in c(1,2,3)){
#     rfin2<-rbind(rfin2,c(nt,mt))
#   }
# }
# rfin2<-rfin2[-1,]
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="rfdtune.txt")
# clusterExport(cl, list("rfpd","estrs"))
# clusterEvalQ(cl,{library(randomForest)})
# 
# system.time(
#   rfdtune<-parApply(cl, rfin2, 1, rfpd)
# )
# 
# stopCluster(cl)
# 
# save.image("estr4.RData")
# 
# rfdbp<-rfdtune[,which.min(rfdtune[3,])]
# save.image("estr4.RData")
# 
# # Input: seed(bootstrap number)  Output: rf.r predictions
# rfbt<-function(B){
#   set.seed(B)
#   estrbt<-estrs[sample(dim(estrs)[1], dim(estrs)[1], replace = TRUE),9:16]
#   estrbt<-estrbt[!duplicated(estrbt),]
#   estrbttest<-anti_join(estrs[,9:16],estrbt)
#   btrfm<-randomForest(y~.,data=estrbt,ntree=2000,mtry=2)
#   prebttest<-predict(btrfm,estrbttest)
#   pretestna<-c()
#   for(i in 1:dim(estrs)[1]){
#     ind<-which(estrbttest$NCT==estrs$NCT[i] & estrbttest$p==estrs$p[i] &
#                  estrbttest$tA.B==estrs$tA.B[i] & estrbttest$AV==estrs$AV[i] &
#                  estrbttest$ENMB==estrs$ENMB[i] & estrbttest$D==estrs$D[i] & 
#                  estrbttest$NTO==estrs$NTO[i])
#     if(length(ind)==0){
#       pretestna<-c(pretestna,NA)
#     }else{
#       pretestna<-c(pretestna,prebttest[ind])
#     }
#   }
#   gc()
#   write.table(data.frame(pretestna,predict(btrfm,estrs)),"rfbt.csv",append=T,sep = ",")
#   return(data.frame(pretestna,predict(btrfm,estrs)))
# }
# 
# # B=500 #
# cl<-makeCluster(detectCores(),outfile="rfbt5.txt")
# clusterExport(cl, list("rfbt","estrs"))
# clusterEvalQ(cl,{library(randomForest);library(dplyr)})
# 
# system.time(
#   rfbtpre5<-parLapply(cl, 1:500, rfbt)
# )
# stopCluster(cl)
# 
# save.image("estr4.RData")
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
# estrs<-cbind(estrs,rfvar5t)
# 
# # plot distribution of var estimated by different models #
# pmvar<-data.frame(V1=c(rep("GB bootstrap",81),rep("SVR bootstrap",81),
#                        rep("RF bootstrap",81),rep("RF tree",81)),
#                   V2=c(btvar5a,svrvar5a,rfvar5a,rfpvar))
# pmvar$V1<-ordered(pmvar$V1,levels = c("GB bootstrap","SVR bootstrap",
#                                       "RF bootstrap","RF tree"))
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
# 
# #* test method with RF *#
# # Input: seed(bootstrap number) and sample train size  Output: rf.r predictions
# rftv<-function(Bts){
#   set.seed(Bts[1])
#   estrtv<-estrs[sample(dim(estrs)[1], Bts[2], replace = F),9:16]
#   estrtvtest<-anti_join(estrs[,9:16],estrtv)
#   tvrfm<-randomForest(y~.,data=estrtv,ntree=2000,mtry=2)
#   pretvtest<-predict(tvrfm,estrtvtest)
#   pretestna<-c()
#   for(i in 1:dim(estrs)[1]){
#     ind<-which(estrtvtest$NCT==estrs$NCT[i] & estrtvtest$p==estrs$p[i] &
#                  estrtvtest$tA.B==estrs$tA.B[i] & estrtvtest$AV==estrs$AV[i] &
#                  estrtvtest$ENMB==estrs$ENMB[i] & estrtvtest$D==estrs$D[i] & 
#                  estrtvtest$NTO==estrs$NTO[i])
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
# for(ts in c(4,14,24,54,84,114,144,154,160)){
#   for(B in 1:500){
#     Btsin<-rbind(Btsin,c(B,ts))
#   }
# }
# Btsin<-Btsin[-1,]
# # 
# # Btsin2<-matrix(ncol=2)
# # for(ts in c(123,124)){
# #   for(B in 1:500){
# #     Btsin2<-rbind(Btsin2,c(B,ts))
# #   }
# # }
# # Btsin2<-Btsin2[-1,]
# 
# cl<-makeCluster(detectCores(),outfile="rftv5.txt")
# clusterExport(cl, list("rftv","estrs"))
# clusterEvalQ(cl,{library(randomForest);library(dplyr)})
# 
# system.time(
#   rftvpre5<-parApply(cl, Btsin, 1, rftv)
# )
# stopCluster(cl)
# 
# save.image("estr4.RData")
# 
# tv04rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv04rfvar5<-c(tv04rfvar5,var(rftvpre5[i,1:500],na.rm=T))
# }
# tv1rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv1rfvar5<-c(tv1rfvar5,var(rftvpre5[i,501:1000],na.rm=T))
# }
# tv2rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv2rfvar5<-c(tv2rfvar5,var(rftvpre5[i,1001:1500],na.rm=T))
# }
# tv5rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv5rfvar5<-c(tv5rfvar5,var(rftvpre5[i,1501:2000],na.rm=T))
# }
# tv8rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv8rfvar5<-c(tv8rfvar5,var(rftvpre5[i,2001:2500],na.rm=T))
# }
# tv11rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv11rfvar5<-c(tv11rfvar5,var(rftvpre5[i,2501:3000],na.rm=T))
# }
# tv14rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv14rfvar5<-c(tv14rfvar5,var(rftvpre5[i,3001:3500],na.rm=T))
# }
# tv15rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv15rfvar5<-c(tv15rfvar5,var(rftvpre5[i,3501:4000],na.rm=T))
# }
# tv160rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv160rfvar5<-c(tv160rfvar5,var(rftvpre5[i,4001:4500],na.rm=T))
# }
# 
# estrs<-cbind(estrs,tv04rfvar5,tv1rfvar5,tv2rfvar5,tv5rfvar5,tv8rfvar5,tv11rfvar5,tv14rfvar5,
#              tv15rfvar5,tv160rfvar5)
# 
# 
# #kriging estimated nugget
# estrkrig<-krigm(estrs[,9:16],t,p,0,64)
# knug<-estrkrig@covariance@nugget
# 
# # plot distribution of var estimated by different models #
# pmvar<-data.frame(V1=c(#rep("GB bootstrap",164),rep("SVR bootstrap",164),
#   rep("bootstrap",164),rep("4 train",164),
#   rep("14 train",164),rep("24 train",164),
#   rep("54 train",164),rep("84 train",164),
#   rep("114 train",164),rep("144 train",164),
#   rep("154 train",164),rep("160 train",164),"nugget"),
#   V2=c(#btvar5t,svrvar5t,
#     rfvar5t,tv04rfvar5,tv1rfvar5,tv2rfvar5,tv5rfvar5,
#     tv8rfvar5,tv11rfvar5,tv14rfvar5,tv15rfvar5,tv160rfvar5,knug))
# pmvar$V1<-ordered(pmvar$V1,levels = c(#"GB bootstrap","SVR bootstrap",
#   "bootstrap","4 train",
#   "14 train","24 train",
#   "54 train","84 train",
#   "114 train","144 train",
#   "154 train","160 train","nugget"))
# p<-ggplot(pmvar, aes(x=V1, y=V2)) +
#   stat_boxplot(geom="errorbar",width=0.3,size=1,position=position_dodge(0.6),color="grey")+
#   geom_boxplot(aes(fill=V1),size=1,alpha=0.5,position=position_dodge(0.8),width=0.8,
#                color="grey",outlier.colour = "grey")+
#   #ylim(0,8000)+
#   labs(y="Estimated variances",x="")+
#   theme_bw(base_size = 27)+
#   theme(legend.position = "none",
#         panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         axis.text = element_text(face="bold", color="black", size=23, family="serif"),
#         axis.text.x = element_text(angle = 25,hjust = 1),
#         axis.title = element_text(face="bold",family="serif"),
#         axis.ticks.length=unit(-0.25, "cm"),
#         plot.margin = unit(c(0.4, 0.1, -1.2, 0), "cm"))
# p
# ggsave("var distribution m v3.png", plot = p, dpi = 600, width = 8, height = 6, units = "in")
# 
# # plot var vs ref #
# ggplot(estrs, aes(x=as.ordered(refID), y=rfvar5t)) +
#   stat_boxplot(geom="errorbar",width=0.3,size=1,position=position_dodge(0.6),color="grey")+
#   geom_boxplot(aes(fill=as.ordered(refID)),size=1,alpha=0.5,position=position_dodge(0.8),width=0.8,
#                color="grey",outlier.colour = "grey")+
#   #ylim(0,0.00069)+
#   labs(y="Estimated variances",x="Reference ID")+
#   theme_bw(base_size = 27)+
#   theme(legend.position = "none",
#         panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         axis.text = element_text(face="bold", color="black", size=18, family="serif"),
#         axis.title = element_text(face="bold",family="serif"),
#         axis.ticks.length=unit(-0.25, "cm"),
#         plot.margin = unit(c(0.3, 0.1, -0.1, 0), "cm"))
# 
# # plot var vs data size of ref #
# refds<-c()
# for(i in 1:25){
#   estrsp<-estrs[which(estrs$refID==i),]
#   refds<-c(refds,rep(dim(estrsp)[1],dim(estrsp)[1]))
# }
# estrs<-cbind(estrs,refds)
# ggplot(estrs, aes(x=refds, y=rfvar5t)) +
#   stat_boxplot(aes(group=refds),geom="errorbar",width=1.6,size=1.2,position=position_dodge(0.6),color="grey")+
#   geom_boxplot(aes(group=refds,fill=as.ordered(refds)),size=0.5,alpha=0.5,position=position_dodge(0.8),width=1.2,
#                color="grey",outlier.colour = "grey")+
#   #ylim(0,3200)+
#   labs(y="Estimated variances",x="Data size of reference")+
#   theme_bw(base_size = 27)+
#   theme(legend.position = "none",
#         panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         axis.text = element_text(face="bold", color="black", size=18, family="serif"),
#         axis.title = element_text(face="bold",family="serif"),
#         axis.ticks.length=unit(-0.25, "cm"),
#         plot.margin = unit(c(0.3, 0.1, -0.1, 0), "cm"))
# 
# 
# #* consistent var for each source #*
# btvar5tm<-c()
# for(i in 1:15){
#   btvar5tp<-estrs[which(estrs$refID==i),]$btvar5t
#   btvar5tm<-c(btvar5tm,rep(mean(btvar5tp),length(btvar5tp)))
# }
# 
# svrvar5tm<-c()
# for(i in 1:15){
#   svrvar5tp<-estrs[which(estrs$refID==i),]$svrvar5t
#   svrvar5tm<-c(svrvar5tm,rep(mean(svrvar5tp),length(svrvar5tp)))
# }
# 
# rfvar5tm<-c()
# for(i in 1:15){
#   rfvar5tp<-estrs[which(estrs$refID==i),]$rfvar5t
#   rfvar5tm<-c(rfvar5tm,rep(mean(rfvar5tp),length(rfvar5tp)))
# }
# 
# tv4rfvar5m<-c()
# for(i in 1:15){
#   tv4rfvar5p<-estrs[which(estrs$refID==i),]$tv4rfvar5
#   tv4rfvar5m<-c(tv4rfvar5m,rep(mean(tv4rfvar5p),length(tv4rfvar5p)))
# }
# 
# estrs<-cbind(estrs,btvar5tm,svrvar5tm,rfvar5tm,tv4rfvar5m)
# 
# tv1rfvar5m<-c()
# for(i in 1:25){
#   tv1rfvar5p<-estrs[which(estrs$refID==i),]$tv1rfvar5
#   tv1rfvar5m<-c(tv1rfvar5m,rep(mean(tv1rfvar5p),length(tv1rfvar5p)))
# }
# 
# tv2rfvar5m<-c()
# for(i in 1:15){
#   tv2rfvar5p<-estrs[which(estrs$refID==i),]$tv2rfvar5
#   tv2rfvar5m<-c(tv2rfvar5m,rep(mean(tv2rfvar5p),length(tv2rfvar5p)))
# }
# 
# tv6rfvar5m<-c()
# for(i in 1:15){
#   tv6rfvar5p<-estrs[which(estrs$refID==i),]$tv6rfvar5
#   tv6rfvar5m<-c(tv6rfvar5m,rep(mean(tv6rfvar5p),length(tv6rfvar5p)))
# }
# 
# tv8rfvar5m<-c()
# for(i in 1:15){
#   tv8rfvar5p<-estrs[which(estrs$refID==i),]$tv8rfvar5
#   tv8rfvar5m<-c(tv8rfvar5m,rep(mean(tv8rfvar5p),length(tv8rfvar5p)))
# }
# 
# tv10rfvar5m<-c()
# for(i in 1:15){
#   tv10rfvar5p<-estrs[which(estrs$refID==i),]$tv10rfvar5
#   tv10rfvar5m<-c(tv10rfvar5m,rep(mean(tv10rfvar5p),length(tv10rfvar5p)))
# }
# 
# tv11rfvar5m<-c()
# for(i in 1:15){
#   tv11rfvar5p<-estrs[which(estrs$refID==i),]$tv11rfvar5
#   tv11rfvar5m<-c(tv11rfvar5m,rep(mean(tv11rfvar5p),length(tv11rfvar5p)))
# }
# 
# tv121rfvar5m<-c()
# for(i in 1:15){
#   tv121rfvar5p<-estrs[which(estrs$refID==i),]$tv121rfvar5
#   tv121rfvar5m<-c(tv121rfvar5m,rep(mean(tv121rfvar5p),length(tv121rfvar5p)))
# }
# 
# tv124rfvar5m<-c()
# for(i in 1:15){
#   tv124rfvar5p<-estrs[which(estrs$refID==i),]$tv124rfvar5
#   tv124rfvar5m<-c(tv124rfvar5m,rep(mean(tv124rfvar5p),length(tv124rfvar5p)))
# }
# 
# estrs<-cbind(estrs,tv1rfvar5m)
# 
# ##*********##
# 
# 
# ##** kriging model without noise **##
# # Kriging model #
# # Input: data (y at col 1), scale of θ, p, seed for kriging
# krigmo<-function(dat,t,p,sseed){
#   for(n in c(0,1e-20,1e-10)){
#     set.seed(sseed)
#     cvhkm<-try(km(formula = ~1,design = dat[,-1],
#                response = dat[,1],covtype = "powexp",
#                control = list(trace=F),
#                #lower = c(t,1e-10),
#                nugget = n))
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
#   estrr<-estrs[sample(dim(estrs)[1],dim(estrs)[1]),9:16] #random order
#   for (j in 0:9){
#     dim10<-dim(estrr)[1]%/%10
#     estrcv1<-estrr[(1+dim10*j):(dim10*(j+1)),]
#     estrcv2<-estrr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigmo(estrcv2,t,p,64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcv<-c(krigcv,NA)
#     }else{
#       krigcv<-c(krigcv,sum(abs(estrcv1[,1]-predict(krigcvm,estrcv1[,-1],type="SK")$mean))/dim10)
#     }
#     #}
#   }
#   print(c(i,krigcv))
#   return(c(i,krigcv))
# }
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krigocvs.txt")
# clusterExport(cl, list("krigocve","krigmo","t","p","estrs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krigocvs<-parLapply(cl, 1:100, krigocve)
# )
# stopCluster(cl)
# 
# save.image("estr4.RData")
# 
# krigocvsm<-c()
# for(i in 1:length(krigocvs)){
#   krigocvsm<-c(krigocvsm,mean(krigocvs[[i]][-1]))
# }
# 
#* repeat seed 10-fold CVE(MAE) *#
krigocve2<-function(ij){    #3:j为折数0:9，放在cvin里
  set.seed(11+20*ij[1])
  estrr<-estrs[sample(dim(estrs)[1],dim(estrs)[1]),9:16] #random order
  dim10<-dim(estrr)[1]%/%10
  estrcv1<-estrr[(1+dim10*ij[2]):(dim10*(ij[2]+1)),]
  estrcv2<-estrr[-((1+dim10*ij[2]):(dim10*(ij[2]+1))),]
  krigcvm<-try(krigmo(estrcv2,t,p,64))   #sseed可替换为8*k
  if ('try-error' %in% class(krigcvm)) {
    prete<-rep(NA,dim(estrcv1)[1]);pretr<-rep(NA,dim(estrcv2)[1])
    preteva<-rep(NA,dim(estrcv1)[1]);pretrva<-rep(NA,dim(estrcv2)[1])
  }else{
    pretr<-predict(krigcvm,estrcv2[,-1],type="SK")$mean
    pretrva<-predict(krigcvm,estrcv2[,-1],type="SK")$sd
    prete<-predict(krigcvm,estrcv1[,-1],type="SK")$mean
    preteva<-predict(krigcvm,estrcv1[,-1],type="SK")$sd
  }
  res<-list(data.frame(estrcv1[,1],prete,preteva),data.frame(estrcv2[,1],pretr,pretrva))
  fnm1<-paste("cv10o/cv10pre4_", ij[1], "_", ij[2], ".csv", sep = "")  #3包括parameters
  fnm2<-paste("cv10o/cv10ptr4_", ij[1], "_", ij[2], ".csv", sep = "")
  write.table(res[[1]], fnm1, append = TRUE, sep = ",")
  write.table(res[[2]], fnm2, append = TRUE, sep = ",")
  return(res)
}

#cluster initialization
cl<-makeCluster(detectCores(),outfile="krigocvs.txt")
clusterExport(cl, list("krigmo","t","p","estrs"))
clusterEvalQ(cl,{library(DiceKriging)})

system.time(
  krigocvs2<-parApply(cl, kcvin, 1, krigocve2)
)
stopCluster(cl)

save.image("estr4.RData")

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
# ##** kriging model with nugget **##
# #* one model for test *#
# #随机选16个作测试集，diagonal plot
# set.seed(7)
# test.estrs<-estrs[sample(1:dim(estrs)[1],16),]
# tt.estrs<-anti_join(estrs, test.estrs)
# test.y<-test.estrs[,9]
# 
# # Kriging model #
# # Input: data (y at col 1), scale of θ, p, noise (#33 or nugget)
# krigm<-function(dat,t,p,n,sseed){
#   set.seed(sseed)
#   if(length(n)==1){
#     cvhkm<-km(formula = ~1,design = as.data.frame(dat[,-1]),
#               response = dat[,1],covtype = "powexp",
#               control = list(trace=F),
#               #lower = c(t,1e-10),
#               nugget.estim = T)
#   }else{
#     cvhkm<-km(formula = ~1,design = as.data.frame(dat[,-1]),
#               response = dat[,1],covtype = "powexp",
#               control = list(trace=F),
#               #lower = c(t,1e-10),
#               noise.var = n)
#   }
#   return(cvhkm)
# }
# 
# tkrig<-krigm(tt.estrs[,9:16],t,p,0,567)
# tprekrig<-predict(tkrig,test.estrs[,10:16],type="SK")$mean
# plot(x=test.y,y=tprekrig,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0.01,0.145), ylim = c(0.01,0.145),
#      xlab="Measured electrostrain (%)",ylab="Predicted electrostrain (%)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="Kriging on the test set",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# tkrigr2<-1-sum((test.y-tprekrig)^2)/sum((test.y-mean(test.y))^2)
# 
# #* repeat seed 10-fold CVE(MAE) *#
# krigcve2<-function(ij){    #3:j为折数0:9，放在cvin里
#   set.seed(11+20*ij[1])
#   estrr<-estrs[sample(dim(estrs)[1],dim(estrs)[1]),9:16] #random order
#   dim10<-dim(estrr)[1]%/%10
#   estrcv1<-estrr[(1+dim10*ij[2]):(dim10*(ij[2]+1)),]
#   estrcv2<-estrr[-((1+dim10*ij[2]):(dim10*(ij[2]+1))),]
#   krigcvm<-try(krigm(estrcv2,t,p,0,64))   #sseed可替换为8*k
#   if ('try-error' %in% class(krigcvm)) {
#     prete<-rep(NA,dim(estrcv1)[1]);pretr<-rep(NA,dim(estrcv2)[1])
#     preteva<-rep(NA,dim(estrcv1)[1]);pretrva<-rep(NA,dim(estrcv2)[1])
#   }else{
#     pretr<-predict(krigcvm,estrcv2[,-1],type="SK")$mean
#     pretrva<-predict(krigcvm,estrcv2[,-1],type="SK")$sd
#     prete<-predict(krigcvm,estrcv1[,-1],type="SK")$mean
#     preteva<-predict(krigcvm,estrcv1[,-1],type="SK")$sd
#   }
#   res<-list(data.frame(estrcv1[,1],prete,preteva),data.frame(estrcv2[,1],pretr,pretrva))
#   fnm1<-paste("cv10/cv10pre4_", ij[1], "_", ij[2], ".csv", sep = "")  #3包括parameters
#   fnm2<-paste("cv10/cv10ptr4_", ij[1], "_", ij[2], ".csv", sep = "")
#   write.table(res[[1]], fnm1, append = TRUE, sep = ",")
#   write.table(res[[2]], fnm2, append = TRUE, sep = ",")
#   return(res)
# }
# 
# kcvin<-matrix(ncol = 2)
# for(i in 1:100){
#   for(j in 0:9){
#     kcvin<-rbind(kcvin,c(i,j))
#   }
# }
# kcvin<-kcvin[-1,]
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krigcvs.txt")
# clusterExport(cl, list("krigm","t","p","estrs"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krigcvs2<-parApply(cl, kcvin, 1, krigcve2)
# )
# stopCluster(cl)
# 
# save.image("estr4.RData")
# 
# # krigcvsm<-c()
# # for(i in 1:length(krigcvs)){
# #   krigcvsm<-c(krigcvsm,mean(krigcvs[[i]][-1]))
# # }
# 
# cv10mae<-c(); cv10r2<-c()
# for(i in 1:100){
#   te<-c(); pte<-c()
#   for(j in 1:10){
#     te<-c(te,krigcvs2[[((i-1)*10+j)]][[1]][,1])
#     pte<-c(pte,krigcvs2[[((i-1)*10+j)]][[1]][,2])
#   }
#   ptemae<-mean(abs(pte - te),na.rm=T)
#   pter2<-round(1-sum((pte-te)^2,na.rm=T)/sum((te-mean(te,na.rm=T))^2,na.rm=T),3)
#   cv10mae<-c(cv10mae,ptemae)
#   cv10r2<-c(cv10r2,pter2)
# }
# cv10r2s<-c()
# for(i in 1:1000){
#   te<-krigcvs2[[i]][[1]][,1]
#   pte<-krigcvs2[[i]][[1]][,2]
#   pter2<-round(1-sum((pte-te)^2,na.rm=T)/sum((te-mean(te,na.rm=T))^2,na.rm=T),3)
#   cv10r2s<-c(cv10r2s,pter2)
# }
# 
# ##***********##
# 
# 
# 
# ##** kriging model with noise **##
# #* one model for test *#
# tnkgb<-krigm(tt.estrs[,9:16],t,p,tt.estrs$btvar5a,567)
# tprenkgb<-predict(tnkgb,test.estrs[,10:16],type="SK")$mean
# plot(x=test.y,y=tprenkgb,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0.01,0.145), ylim = c(0.01,0.145),
#      xlab="Measured electrostrain (%)",ylab="Predicted electrostrain (%)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="Noisy riging (GB bootstrap) on the test set",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# tnkgbr2<-1-sum((test.y-tprenkgb)^2)/sum((test.y-mean(test.y))^2)
# 
# tnksvr<-krigm(tt.estrs[,9:16],t,p,tt.estrs$svrvar5a,567)
# tprenksvr<-predict(tnksvr,test.estrs[,10:16],type="SK")$mean
# plot(x=test.y,y=tprenksvr,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0.01,0.145), ylim = c(0.01,0.145),
#      xlab="Measured electrostrain (%)",ylab="Predicted electrostrain (%)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="Noisy riging (SVR bootstrap) on the test set",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# tnksvrr2<-1-sum((test.y-tprenksvr)^2)/sum((test.y-mean(test.y))^2)
# 
# tnkrf<-krigm(tt.estrs[,9:16],t,p,tt.estrs$rfvar5a,567)
# tprenkrf<-predict(tnkrf,test.estrs[,10:16],type="SK")$mean
# plot(x=test.y,y=tprenkrf,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0.01,0.145), ylim = c(0.01,0.145),
#      xlab="Measured electrostrain (%)",ylab="Predicted electrostrain (%)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="Noisy riging (RF bootstrap) on the test set",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# tnkrfr2<-1-sum((test.y-tprenkrf)^2)/sum((test.y-mean(test.y))^2)
# 
# tnkrfp<-krigm(tt.estrs[,9:16],t,p,tt.estrs$rfpvar5,567)
# tprenkrfp<-predict(tnkrfp,test.estrs[,10:16],type="SK")$mean
# plot(x=test.y,y=tprenkrfp,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0.01,0.145), ylim = c(0.01,0.145),
#      xlab="Measured electrostrain (%)",ylab="Predicted electrostrain (%)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="Noisy riging (rfp bootstrap) on the test set",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# tnkrfpr2<-1-sum((test.y-tprenkrfp)^2)/sum((test.y-mean(test.y))^2)
# 
# #* var models' performances *#
# #GB
# tgb<-gbm(y~., data = tt.estrs[,9:16], n.trees = 100,
#          interaction.depth = 2, shrinkage = 0.1, cv.folds = 10)
# tpregb<-predict(tgb,test.estrs[,9:16])
# plot(x=test.y,y=tpregb,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0.01,0.145), ylim = c(0.01,0.145),
#      xlab="Measured electrostrain (%)",ylab="Predicted electrostrain (%)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="GB on the test set",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# tgbr2<-1-sum((test.y-tpregb)^2)/sum((test.y-mean(test.y))^2)
# 
# #SVR
# tsv<-svm(y~.,data=tt.estrs[,9:16],type="eps-regression",kernel="radial",
#          cost=1,gamma=1)
# tpresv<-predict(tsv,test.estrs[,9:16])
# plot(x=test.y,y=tpresv,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0.01,0.145), ylim = c(0.01,0.145),
#      xlab="Measured electrostrain (%)",ylab="Predicted electrostrain (%)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="SVR on the test set",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# tsvr2<-1-sum((test.y-tpresv)^2)/sum((test.y-mean(test.y))^2)
# 
# #RF
# trf<-randomForest(y~.,data=tt.estrs[,9:16],ntree=6000,mtry=1)
# tprerf<-predict(trf,test.estrs[,9:16])
# plot(x=test.y,y=tprerf,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
#      col= rgb(0, 139, 69, maxColorValue = 255), cex = 1.2,
#      xlim = c(0.01,0.145), ylim = c(0.01,0.145),
#      xlab="Measured electrostrain (%)",ylab="Predicted electrostrain (%)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),
#      main="RF on the test set",cex.main=1.5,font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2,col=rgb(0, 134, 139, maxColorValue = 255))
# trfr2<-1-sum((test.y-tprerf)^2)/sum((test.y-mean(test.y))^2)
# 
# 
# #* repeat seed 10-fold CVE(MAE) with var *#
# # Input: cv data order seed and var character  Output: iv and CVE
# krigncve<-function(iv){
#   krigcv<-c()
#   set.seed(11+20*as.numeric(iv[1]))
#   estrr<-estrs[sample(dim(estrs)[1],dim(estrs)[1]),] #random order
#   for (j in 0:9){
#     dim10<-dim(estrr)[1]%/%10
#     estrcv1<-estrr[(1+dim10*j):(dim10*(j+1)),9:16]
#     estrcv5<-estrr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigm(estrcv5[,9:16],t,p,estrcv5[,iv[2]],64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcv<-c(krigcv,NA)
#     }else{
#       krigcv<-c(krigcv,sum(abs(estrcv1[,1]-predict(krigcvm,estrcv1[,-1],type="SK")$mean))/dim10)
#     }
#     #}
#   }
#   print(c(iv,krigcv))
#   write.table(c(iv,krigcv),"krigncve.csv",append=T,sep = ",")
#   return(c(iv,krigcv))
# }

krigncve2<-function(ijv){
  set.seed(11+20*as.numeric(ijv[1]))
  estrr<-estrs[sample(dim(estrs)[1],dim(estrs)[1]),] #random order
  dim10<-dim(estrr)[1]%/%10
  estrcv1<-estrr[(1+dim10*as.numeric(ijv[2])):(dim10*(as.numeric(ijv[2])+1)),9:16]
  estrcv5<-estrr[-((1+dim10*as.numeric(ijv[2])):(dim10*(as.numeric(ijv[2])+1))),]
  krigcvm<-try(krigm(estrcv5[,9:16],t,p,estrcv5[,ijv[3]],64))
  if ('try-error' %in% class(krigcvm)) {
    prete<-rep(NA,dim(estrcv1)[1]);pretr<-rep(NA,dim(estrcv5)[1])
    preteva<-rep(NA,dim(estrcv1)[1]);pretrva<-rep(NA,dim(estrcv5)[1])
  }else{
    pretr<-predict(krigcvm,estrcv5[,10:16],type="SK")$mean
    pretrva<-predict(krigcvm,estrcv5[,10:16],type="SK")$sd
    prete<-predict(krigcvm,estrcv1[,-1],type="SK")$mean
    preteva<-predict(krigcvm,estrcv1[,-1],type="SK")$sd
  }
  res<-list(data.frame(estrcv1[,1],prete,preteva),data.frame(estrcv5[,9],pretr,pretrva))
  fnm1<-paste("cv10n/cv10npre4_", ijv[1], "_", ijv[2], ".csv", sep = "")  #3包括parameters
  fnm2<-paste("cv10n/cv10nptr4_", ijv[1], "_", ijv[2], ".csv", sep = "")
  write.table(res[[1]], fnm1, append = TRUE, sep = ",")
  write.table(res[[2]], fnm2, append = TRUE, sep = ",")
  return(res)
}
# 
# kncvin<-matrix(ncol = 2)
# for(v in c("btvar5t","svrvar5t",
#            "rfvar5t",
#            "tv04rfvar5","tv1rfvar5","tv2rfvar5","tv5rfvar5","tv8rfvar5",
#            "tv11rfvar5","tv14rfvar5","tv15rfvar5","tv160rfvar5")){
#   for(i in 1:100){
#     kncvin<-rbind(kncvin,c(i,v))
#   }
# }
# kncvin<-kncvin[-1,]
# 
#  
# kncvin3<-matrix(ncol = 2)
# for(i in 1:100){
#   kncvin3<-rbind(kncvin3,c(i,"tv1rfvar5m"))
# }
# kncvin3<-kncvin3[-1,]

kncvin3<-matrix(ncol = 3)
for(i in 1:100){
  for(j in 0:9){
    kncvin3<-rbind(kncvin3,c(i,j,"tv1rfvar5m"))
  }
}
kncvin3<-kncvin3[-1,]

cl<-makeCluster(detectCores(),outfile="krigncvs.txt")
clusterExport(cl, list("krigncve","krigm","t","p","estrs"))
clusterEvalQ(cl,{library(DiceKriging)})

system.time(
  krigncvs32<-parApply(cl, kncvin3, 1, krigncve2)
)
stopCluster(cl)

save.image("estr4.RData")
# 
# krigncvsm<-as.data.frame(matrix(nrow = 100,ncol = 12))
# for(j in 1:12){
#   for(i in 1:100){
#     krigncvsm[i,j]<-mean(as.numeric(krigncvs[-c(1,2),(j-1)*100+i]),na.rm=T)
#   }
# }
# 
# colnames(krigncvsm)<-c("btvar5t","svrvar5t","rfvar5t",
#                        "tv04rfvar5","tv1rfvar5","tv2rfvar5","tv5rfvar5","tv8rfvar5",
#                        "tv11rfvar5","tv14rfvar5","tv15rfvar5","tv160rfvar5"#,
#                        #"btvar5tm","svrvar5tm","rfvar5tm","tv4rfvar5m",
#                        #"tv1rfvar5m","tv2rfvar5m","tv6rfvar5m","tv8rfvar5m","tv10rfvar5m",
#                        #"tv11rfvar5m","tv121rfvar5m","tv124rfvar5m"
#                        )
# 
# cvetv1m<-c()
# for(i in 1:100){
#   cvetv1m<-c(cvetv1m,mean(as.numeric(krigncvs3[-c(1,2),i]),na.rm=T))
# }
# 
# save.image("estr4.RData")

cv10nmae<-c(); cv10nr2<-c()
for(i in 1:100){
  te<-c(); pte<-c()
  for(j in 1:10){
    te<-c(te,krigncvs32[[((i-1)*10+j)]][[1]][,1])
    pte<-c(pte,krigncvs32[[((i-1)*10+j)]][[1]][,2])
  }
  ptemae<-mean(abs(pte - te),na.rm=T)
  pter2<-round(1-sum((pte-te)^2,na.rm=T)/sum((te-mean(te,na.rm=T))^2,na.rm=T),3)
  cv10nmae<-c(cv10nmae,ptemae)
  cv10nr2<-c(cv10nr2,pter2)
}
cv10nr2s<-c()
for(i in 1:1000){
  te<-krigncvs32[[i]][[1]][,1]
  pte<-krigncvs32[[i]][[1]][,2]
  pter2<-round(1-sum((pte-te)^2,na.rm=T)/sum((te-mean(te,na.rm=T))^2,na.rm=T),3)
  cv10nr2s<-c(cv10nr2s,pter2)
}


#* Fig3a *#
f3ascreen<-which(cv10nr2s>0.85)
f3aind<-order(cv10nr2s[f3ascreen]-cv10or2s[f3ascreen],decreasing = T)[1:10]
cv10nr2s[f3ascreen][f3aind];cv10or2s[f3ascreen][f3aind]
f3aselect<-f3ascreen[10]

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
                       breaks = c(0, 500, 1000), labels = c("0", "500", "1000"),
                       guide = guide_axis(minor.ticks = TRUE),minor_breaks = waiver()) +
    scale_x_continuous(limits = lim, name = xlab, 
                       breaks = c(0, 500, 1000), labels = c("0", "500", "1000"),
                       guide = guide_axis(minor.ticks = TRUE),minor_breaks = waiver()) +
    theme(ggh4x.axis.ticks.length.minor=rel(1/2))+
    annotate("text", x = 400, y = 1100, label = bquote(paste("R"^2 == .(cr2))),
             color = rgb(0,72,131,maxColorValue = 255), size = 10, family = "sans")
}

jpeg("Fig3a 1.jpg", width = 4015, height = 2800, res = 600)
fn.plot.gpar3(x=c(krigocvs2[[191]][[2]][,1],krigocvs2[[191]][[1]][,1]),
              y1=krigocvs2[[191]][[2]][,2],y2=krigocvs2[[191]][[1]][,2],
              lim=c(0,1155),xlab=expression(paste("Measured ", d[33], "*", " (" * pm * bold("/") * V * ")")),
              ylab=bquote(atop("Predicted d"["33"]~"*", " (" * pm * bold("/") * V * ")")),0.636)
dev.off()
jpeg("Fig3a 2.jpg", width = 4015, height = 2800, res = 600)
fn.plot.gpar3(x=c(krigncvs32[[191]][[2]][,1],krigncvs32[[191]][[1]][,1]),
              y1=krigncvs32[[191]][[2]][,2],y2=krigncvs32[[191]][[1]][,2],
              lim=c(0,1155),xlab=expression(paste("Measured ", d[33], "*", " (" * pm * bold("/") * V * ")")),
              ylab=bquote(atop("Predicted d"["33"]~"*", " (" * pm * bold("/") * V * ")")),0.879)
dev.off()

# 
# # plot distribution of CVEs for different models #
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (GB bootstrap)",10),
#                       rep("noisy kriging (SVR bootstrap)",10),rep("noisy kriging (RF bootstrap)",10)),
#                  V2=c(krigocvsm, krigcvsm, krigncvsm$btvar5t, 
#                       krigncvsm$svrvar5t,krigncvsm$rfvar5t),
#                  V3=c(c(mean(krigocvsm),rep(NA,9)),c(mean(krigcvsm),rep(NA,9)),c(mean(krigncvsm$btvar5t),rep(NA,9)),
#                       c(mean(krigncvsm$svrvar5t),rep(NA,9)),c(mean(krigncvsm$rfvar5t),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (GB bootstrap)", 
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
# # #5 10 30 50 70 75
# pmcv<-data.frame(V1=c(rep("without",100), rep("nugget",100),rep("res. var (bootstrap)",100),
#                       rep("res. var (4 train)",100),rep("res. var (14 train)",100),
#                       rep("res. var (24 train)",100),rep("res. var (54 train)",100),
#                       rep("res. var (84 train)",100),rep("res. var (114 train)",100),
#                       rep("res. var (144 train)",100),
#                       rep("res. var (154 train)",100),rep("res. var (160 train)",100)),
#                  V2=c(krigocvsm, krigcvsm, krigncvsm$rfvar5t,
#                       krigncvsm$tv04rfvar5,krigncvsm$tv1rfvar5,
#                       krigncvsm$tv2rfvar5, krigncvsm$tv5rfvar5,
#                       krigncvsm$tv8rfvar5,krigncvsm$tv11rfvar5,
#                       krigncvsm$tv14rfvar5,
#                       krigncvsm$tv15rfvar5, krigncvsm$tv160rfvar5),
#                  V3=c(c(mean(krigocvsm),rep(NA,99)),c(mean(krigcvsm),rep(NA,99)),c(mean(krigncvsm$rfvar5t),rep(NA,99)),
#                       c(mean(krigncvsm$tv04rfvar5),rep(NA,99)),c(mean(krigncvsm$tv1rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv2rfvar5),rep(NA,99)),c(mean(krigncvsm$tv5rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv8rfvar5),rep(NA,99)),c(mean(krigncvsm$tv11rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv14rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv15rfvar5),rep(NA,99)),c(mean(krigncvsm$tv160rfvar5),rep(NA,99))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("without", "nugget", "res. var (bootstrap)",
#                                     "res. var (4 train)","res. var (14 train)",
#                                     "res. var (24 train)", "res. var (54 train)",
#                                     "res. var (84 train)", "res. var (114 train)",
#                                     "res. var (144 train)", "res. var (154 train)",
#                                     "res. var (160 train)"))
# # #导出数据
# pmcvout<-data.frame(krigocvsm,krigcvsm,krigncvsm$tv15rfvar5,krigncvsm$tv14rfvar5,
#                     krigncvsm$tv11rfvar5,krigncvsm$tv8rfvar5,krigncvsm$tv5rfvar5,
#                     krigncvsm$tv2rfvar5,krigncvsm$tv1rfvar5,krigncvsm$tv04rfvar5,
#                     cvetv1m)
# colnames(pmcvout)<-c("without var.","nugget var.","154 train","144 train",
#                      "114 train", "84 train","54 train",
#                      "24 train","14 train","4 train",
#                      "14 train source mean")
# library(openxlsx)
# write.xlsx(pmcvout,"CVEs enerd m.xlsx")
# p<-ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="MAE (10-fold cross validation)",x="")+
#   theme_bw(base_size = 27)+
#   theme(legend.position = "none",
#         panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         axis.text = element_text(face="bold", color="black", size=24, family="serif"), 
#         axis.title = element_text(face="bold",family="serif"),
#         axis.ticks.length=unit(-0.25, "cm"),
#         plot.margin = unit(c(0, 0.1, 0, -0.9), "cm"))
# p
# ggsave("compare cv tv m 100 v3.png", plot = p, dpi = 600, width = 8, height = 6, units = "in")
# 
# # plot distribution of CVEs for models with consistent var for each source #
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (GB bootstrap mean)",10),
#                       rep("noisy kriging (SVR bootstrap mean)",10),rep("noisy kriging (RF bootstrap mean)",10)),
#                  V2=c(krigocvsm, krigcvsm, krigncvsm$btvar5tm, 
#                       krigncvsm$svrvar5tm,krigncvsm$rfvar5tm),
#                  V3=c(c(mean(krigocvsm),rep(NA,9)),c(mean(krigcvsm),rep(NA,9)),c(mean(krigncvsm$btvar5tm),rep(NA,9)),
#                       c(mean(krigncvsm$svrvar5tm),rep(NA,9)),c(mean(krigncvsm$rfvar5tm),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (GB bootstrap mean)", 
#                                     "noisy kriging (SVR bootstrap mean)","noisy kriging (RF bootstrap mean)"))
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
# pmcv<-data.frame(V1=c(rep("kriging",10), rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap mean)",10),
#                       rep("noisy kriging (110 RF test mean)",10),rep("noisy kriging (100 RF test mean)",10),
#                       rep("noisy kriging (80 RF test mean)",10),rep("noisy kriging (60 RF test mean)",10),
#                       rep("noisy kriging (40 RF test mean)",10),
#                       rep("noisy kriging (10 RF test mean)",10),rep("noisy kriging (5 RF test mean)",10),
#                       rep("noisy kriging (2 RF test mean)",10)),
#                  V2=c(krigocvsm, krigcvsm, krigncvsm$rfvar5tm,
#                       krigncvsm$tv1rfvar5m,krigncvsm$tv2rfvar5m,
#                       krigncvsm$tv4rfvar5m, krigncvsm$tv6rfvar5m,
#                       krigncvsm$tv8rfvar5m,
#                       krigncvsm$tv11rfvar5m, krigncvsm$tv121rfvar5m,
#                       krigncvsm$tv124rfvar5m),
#                  V3=c(c(mean(krigocvsm),rep(NA,9)),c(mean(krigcvsm),rep(NA,9)),c(mean(krigncvsm$rfvar5tm),rep(NA,9)),
#                       c(mean(krigncvsm$tv1rfvar5m),rep(NA,9)),c(mean(krigncvsm$tv2rfvar5m),rep(NA,9)),
#                       c(mean(krigncvsm$tv4rfvar5m),rep(NA,9)),c(mean(krigncvsm$tv6rfvar5m),rep(NA,9)),
#                       c(mean(krigncvsm$tv8rfvar5m),rep(NA,9)),
#                       c(mean(krigncvsm$tv11rfvar5m),rep(NA,9)),c(mean(krigncvsm$tv121rfvar5m),rep(NA,9)),
#                       c(mean(krigncvsm$tv124rfvar5m),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap mean)",
#                                     "noisy kriging (110 RF test mean)", "noisy kriging (100 RF test mean)", 
#                                     "noisy kriging (80 RF test mean)", "noisy kriging (60 RF test mean)", 
#                                     "noisy kriging (40 RF test mean)", 
#                                     "noisy kriging (10 RF test mean)", "noisy kriging (5 RF test mean)",
#                                     "noisy kriging (2 RF test mean)"))
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
# 
# ##***********##
# 
# 
# 
# ##** iterative opportunity cost **##
# #* OC of kriging without noise *# 
# # Input: virtual data size, max iteration times, seed for kriging, seed for sampling data, type of utility function 
# # Output: opportunity cost of each iteration
# itoco<-function(vn, mt, sseed, dseed, uf){
#   fvs<-estrs[72,]  #max y in estrs
#   estrsl<-estrs[-72,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, estrsl[sample(dim(estrsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(estrs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigmo(ftd[,9:16],t,p,64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       set.seed(dseed+5)
#       newd<<-fvs[sample(dim(fvs)[1], 1),]
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(10:16)],type="SK")
#       if(uf=="pre"){
#         newd<<-fvs[which(kpre[["mean"]]==max(kpre[["mean"]],na.rm=T)),][1,]
#       }else if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb,na.rm=T)),][1,]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,9]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego,na.rm=T)),][1,]
#       }
#     }      
#     fvs<-anti_join(fvs,newd)
#     ftd<-rbind(ftd,newd)
#     oce<<-estrs[72,]$y-newd$y
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
# for(v in c(80,100,120,140,160)){
#   for(i in c(34,77,6,12,55,47,60,10,111,555)){
#     vim<-rbind(vim,c(v,i))
#   }
# }
# vim<-vim[-1,]
# 
# cl<-makeCluster(detectCores(),outfile="oegooc.txt")
# clusterExport(cl, list("itoco","krigmo","t","p","estrs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocopar<-function(vi){
#   return(itoco(vi[1],120,64,vi[2],"ego"))
# }
# 
# system.time(
#   oegooc<-parApply(cl, vim, 1, itocopar)
# )
# stopCluster(cl)
# 
# save.image("estr4.RData")
# 
# itoego<-matrix(nrow = 10,ncol=5)
# for(j in 1:5){
#   for(i in 1:10){
#     itoego[i,j]<-length(oegooc[[(j-1)*10+i]])
#   }
# }
# colnames(itoego)<-c("60","75","90","105","120")
# 
# cl<-makeCluster(detectCores(),outfile="oucboc.txt")
# clusterExport(cl, list("itoco","krigmo","t","p","estrs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocoucbpar<-function(vi){
#   return(itoco(vi[1],120,64,vi[2],"ucb"))
# }
# 
# system.time(
#   oucboc<-parApply(cl, vim, 1, itocoucbpar)
# )
# stopCluster(cl)
# 
# save.image("estr4.RData")
# 
# itoucb<-matrix(nrow = 10,ncol=5)
# for(j in 1:5){
#   for(i in 1:10){
#     itoucb[i,j]<-length(oucboc[[(j-1)*10+i]])
#   }
# }
# colnames(itoucb)<-c("60","75","90","105","120")
# 
# # 固定80%虚拟空间，用四种效能函数
# # 不同虚拟空间，用四种效能函数
# vuim<-matrix(ncol=3)
# for(vn in c(115,131,148)){
#   for(uf in c("pre","ucb","ego")){
#     for(i in 1:100){
#       vuim<-rbind(vuim,c(vn,uf,5*i+7))
#     }
#   }
# }
# vuim<-vuim[-1,]
# 
# vuim2<-matrix(ncol=3)
# for(vn in c(115,131,148)){
#   for(uf in c("pre","ucb","ego","sko")){
#     for(i in 1:100){
#       vuim2<-rbind(vuim2,c(vn,uf,5*i+7))
#     }
#   }
# }
# vuim2<-vuim2[-1,]
# 
# cl<-makeCluster(detectCores(),outfile="oufoc.txt")
# clusterExport(cl, list("itoco","krigmo","t","p","estrs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocouf<-function(ui){
#   return(itoco(as.numeric(ui[1]),131,64,as.numeric(ui[3]),ui[2]))
# }
# 
# system.time(
#   oufoc<-parApply(cl, vuim, 1, itocouf)
# )
# stopCluster(cl)
# 
# save.image("estr4.RData")
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
#   fvs<-estrs[72,]  #max y in estrs
#   estrsl<-estrs[-72,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, estrsl[sample(dim(estrsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(estrs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,9:16],t,p,0,64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       set.seed(dseed+5)
#       newd<<-fvs[sample(dim(fvs)[1], 1),]
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(10:16)],type="SK")
#       if(uf=="pre"){
#         newd<<-fvs[which(kpre[["mean"]]==max(kpre[["mean"]],na.rm=T)),][1,]
#       }else if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb,na.rm=T)),][1,]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,9]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego,na.rm=T)),][1,]
#       }else if(uf=="sko"){
#         ze<-(kpre[["mean"]]-max(kpre[["mean"]]-krigcvm@covariance@nugget))/kpre[["sd"]]
#         kpresko<-(1-krigcvm@covariance@nugget/sqrt((krigcvm@covariance@nugget)^2+(kpre[["sd"]])^2))*
#           kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpresko==max(kpresko,na.rm=T)),][1,]
#       }
#     }      
#     fvs<-anti_join(fvs,newd)
#     ftd<-rbind(ftd,newd)
#     oce<<-estrs[72,]$y-newd$y
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
# cl<-makeCluster(detectCores(),outfile="oegooc.txt")
# clusterExport(cl, list("itoc","krigm","t","p","estrs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocpar<-function(vi){
#   return(itoc(vi[1],120,64,vi[2],"ego"))
# }
# 
# system.time(
#   egooc<-parApply(cl, vim, 1, itocpar)
# )
# stopCluster(cl)
# 
# save.image("estr4.RData")
# 
# itego<-matrix(nrow = 10,ncol=5)
# for(j in 1:5){
#   for(i in 1:10){
#     itego[i,j]<-length(egooc[[(j-1)*10+i]])
#   }
# }
# colnames(itego)<-c("60","75","90","105","120")
# 
# cl<-makeCluster(detectCores(),outfile="oucboc.txt")
# clusterExport(cl, list("itoc","krigm","t","p","estrs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocucbpar<-function(vi){
#   return(itoc(vi[1],120,64,vi[2],"ucb"))
# }
# 
# system.time(
#   ucboc<-parApply(cl, vim, 1, itocucbpar)
# )
# stopCluster(cl)
# 
# save.image("estr4.RData")
# 
# itucb<-matrix(nrow = 10,ncol=5)
# for(j in 1:5){
#   for(i in 1:10){
#     itucb[i,j]<-length(ucboc[[(j-1)*10+i]])
#   }
# }
# colnames(itucb)<-c("60","75","90","105","120")
# 
# # 不同虚拟空间，用四种效能函数
# cl<-makeCluster(detectCores(),outfile="ufoc.txt")
# clusterExport(cl, list("itoc","krigm","t","p","estrs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocuf<-function(ui){
#   return(itoc(as.numeric(ui[1]),131,64,as.numeric(ui[3]),ui[2]))
# }
# 
# system.time(
#   ufoc<-parApply(cl, vuim2, 1, itocuf)
# )
# stopCluster(cl)
# 
# save.image("estr4.RData")
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
# # Input: virtual data size, max iteration times, seed for kriging, seed for sampling data, type of utility function, type of var
# # Output: vt and opportunity cost of each iteration
# itocn<-function(vn, mt, sseed, dseed, uf, vt){
#   fvs<-estrs[72,]  #max y in estrs
#   estrsl<-estrs[-72,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, estrsl[sample(dim(estrsl)[1], vn-1, replace = F),])
#   ftd<-anti_join(estrs,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,9:16],t,p,ftd[,vt],64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       set.seed(dseed+5)
#       newd<<-fvs[sample(dim(fvs)[1], 1),]
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(10:16)],type="SK")
#       if(uf=="pre"){
#         newd<<-fvs[which(kpre[["mean"]]==max(kpre[["mean"]],na.rm=T)),][1,]
#       }else if(uf=="ucb"){
#         kpreucb<-kpre[["mean"]]+kpre[["sd"]]
#         newd<<-fvs[which(kpreucb==max(kpreucb,na.rm=T)),][1,]
#       }else if(uf=="ego"){
#         ze<-(kpre[["mean"]]-max(ftd[,9]))/kpre[["sd"]]
#         kpreego<-kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpreego==max(kpreego,na.rm=T)),][1,]
#       }
#     }
#     fvs<-anti_join(fvs,newd)
#     ftd<-rbind(ftd,newd)
#     oce<<-estrs[72,]$y-newd$y
#     print(oce)
#     oc<-c(oc,oce)    
#     if(oce==0){
#       break()
#     }
#   }
#   gc()
#   return(c(vn,vt,oc))
# }
# 
# vvim<-matrix(ncol=3)
# for(v in c(80,100,120,140,160)){
#   for(vc in c(#"btvar5t","svrvar5t",
#               "rfvar5t",
#               "tv04rfvar5","tv1rfvar5","tv2rfvar5","tv5rfvar5","tv8rfvar5",
#               "tv11rfvar5","tv14rfvar5","tv15rfvar5","tv160rfvar5")){
#     for(i in c(34,77,6,12,55,47,60,10,111,555)){
#       vvim<-rbind(vvim,c(v,vc,i))
#     }
#   }
# }
# vvim<-vvim[-1,]
# 
# vvim2<-matrix(ncol=3)
# for(v in c(80,100,120,140,160)){
#   for(vc in c("btvar5t","svrvar5t")){
#     for(i in c(34,77,6,12,55,47,60,10,111,555)){
#       vvim2<-rbind(vvim2,c(v,vc,i))
#     }
#   }
# }
# vvim2<-vvim2[-1,]
# 
# cl<-makeCluster(detectCores(),outfile="negooc4.txt")
# clusterExport(cl, list("itocn","krigm","t","p","estrs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnpar<-function(vvi){
#   return(itocn(as.numeric(vvi[1]),120,64,as.numeric(vvi[3]),"ego",vvi[2]))
# }
# 
# system.time(
#   negooc2<-parApply(cl, vvim2, 1, itocnpar)
# )
# stopCluster(cl)
# 
# save.image("estr4.RData")
# 
# # vn = 80 #
# itnego80<-matrix(nrow = 10,ncol=10)
# for(j in 1:10){
#   for(i in 1:10){
#     itnego80[i,j]<-length(negooc[[(j-1)*10+i]])-2
#   }
# }
# for(j in 1:2){
#   itnego80c<-c()
#   for(i in 1:10){
#     itnego80c<-c(itnego80c,length(negooc2[[(j-1)*10+i]])-2)
#   }
#   itnego80<-cbind(itnego80,itnego80c)
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap)",10),
#                       rep("noisy kriging (150 test RF)",10),
#                       rep("noisy kriging (140 test RF)",10),rep("noisy kriging (110 test RF)",10),
#                       rep("noisy kriging (80 test RF)",10),rep("noisy kriging (50 test RF)",10),
#                       rep("noisy kriging (20 test RF)",10),rep("noisy kriging (10 test RF)",10),
#                       rep("noisy kriging (4 test RF)",10)),
#                  V2=c(itoego[,1],itego[,1],itnego80[,1],itnego80[,3],
#                       itnego80[,4],itnego80[,5],itnego80[,6],itnego80[,7],
#                       itnego80[,8],itnego80[,9],itnego80[,10]),
#                  V3=c(c(mean(itoego[,1]),rep(NA,9)),c(mean(itego[,1]),rep(NA,9)),c(mean(itnego80[,1]),rep(NA,9)),
#                       c(mean(itnego80[,3]),rep(NA,9)),
#                       c(mean(itnego80[,4]),rep(NA,9)),c(mean(itnego80[,5]),rep(NA,9)),
#                       c(mean(itnego80[,6]),rep(NA,9)),c(mean(itnego80[,7]),rep(NA,9)),
#                       c(mean(itnego80[,8]),rep(NA,9)),c(mean(itnego80[,9]),rep(NA,9)),
#                       c(mean(itnego80[,10]),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (150 test RF)",
#                                     "noisy kriging (140 test RF)","noisy kriging (110 test RF)",
#                                     "noisy kriging (80 test RF)","noisy kriging (50 test RF)",
#                                     "noisy kriging (20 test RF)","noisy kriging (10 test RF)",
#                                     "noisy kriging (4 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 80")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# # vn = 100 #
# itnego100<-matrix(nrow = 10,ncol=10)
# for(j in 1:10){
#   for(i in 1:10){
#     itnego100[i,j]<-length(negooc[[(j-1)*10+i+100]])-2
#   }
# }
# for(j in 1:2){
#   itnego100c<-c()
#   for(i in 1:10){
#     itnego100c<-c(itnego100c,length(negooc2[[(j-1)*10+i+20]])-2)
#   }
#   itnego100<-cbind(itnego100,itnego100c)
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap)",10),
#                       rep("noisy kriging (150 test RF)",10),
#                       rep("noisy kriging (140 test RF)",10),rep("noisy kriging (110 test RF)",10),
#                       rep("noisy kriging (80 test RF)",10),rep("noisy kriging (50 test RF)",10),
#                       rep("noisy kriging (20 test RF)",10),rep("noisy kriging (10 test RF)",10),
#                       rep("noisy kriging (4 test RF)",10)),
#                  V2=c(itoego[,2],itego[,2],itnego100[,1],itnego100[,3],
#                       itnego100[,4],itnego100[,5],itnego100[,6],itnego100[,7],
#                       itnego100[,8],itnego100[,9],itnego100[,10]),
#                  V3=c(c(mean(itoego[,2]),rep(NA,9)),c(mean(itego[,2]),rep(NA,9)),c(mean(itnego100[,1]),rep(NA,9)),
#                       c(mean(itnego100[,3]),rep(NA,9)),
#                       c(mean(itnego100[,4]),rep(NA,9)),c(mean(itnego100[,5]),rep(NA,9)),
#                       c(mean(itnego100[,6]),rep(NA,9)),c(mean(itnego100[,7]),rep(NA,9)),
#                       c(mean(itnego100[,8]),rep(NA,9)),c(mean(itnego100[,9]),rep(NA,9)),
#                       c(mean(itnego100[,10]),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (150 test RF)",
#                                     "noisy kriging (140 test RF)","noisy kriging (110 test RF)",
#                                     "noisy kriging (80 test RF)","noisy kriging (50 test RF)",
#                                     "noisy kriging (20 test RF)","noisy kriging (10 test RF)",
#                                     "noisy kriging (4 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 100")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# # vn = 120 #
# itnego120<-matrix(nrow = 10,ncol=10)
# for(j in 1:10){
#   for(i in 1:10){
#     itnego120[i,j]<-length(negooc[[(j-1)*10+i+200]])-2
#   }
# }
# for(j in 1:2){
#   itnego120c<-c()
#   for(i in 1:10){
#     itnego120c<-c(itnego120c,length(negooc2[[(j-1)*10+i+40]])-2)
#   }
#   itnego120<-cbind(itnego120,itnego120c)
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap)",10),
#                       rep("noisy kriging (150 test RF)",10),
#                       rep("noisy kriging (140 test RF)",10),rep("noisy kriging (110 test RF)",10),
#                       rep("noisy kriging (80 test RF)",10),rep("noisy kriging (50 test RF)",10),
#                       rep("noisy kriging (20 test RF)",10),rep("noisy kriging (10 test RF)",10),
#                       rep("noisy kriging (4 test RF)",10)),
#                  V2=c(itoego[,3],itego[,3],itnego120[,1],itnego120[,3],
#                       itnego120[,4],itnego120[,5],itnego120[,6],itnego120[,7],
#                       itnego120[,8],itnego120[,9],itnego120[,10]),
#                  V3=c(c(mean(itoego[,3]),rep(NA,9)),c(mean(itego[,3]),rep(NA,9)),c(mean(itnego120[,1]),rep(NA,9)),
#                       c(mean(itnego120[,3]),rep(NA,9)),
#                       c(mean(itnego120[,4]),rep(NA,9)),c(mean(itnego120[,5]),rep(NA,9)),
#                       c(mean(itnego120[,6]),rep(NA,9)),c(mean(itnego120[,7]),rep(NA,9)),
#                       c(mean(itnego120[,8]),rep(NA,9)),c(mean(itnego120[,9]),rep(NA,9)),
#                       c(mean(itnego120[,10]),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (150 test RF)",
#                                     "noisy kriging (140 test RF)","noisy kriging (110 test RF)",
#                                     "noisy kriging (80 test RF)","noisy kriging (50 test RF)",
#                                     "noisy kriging (20 test RF)","noisy kriging (10 test RF)",
#                                     "noisy kriging (4 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 120")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# # vn = 140 #
# itnego140<-matrix(nrow = 10,ncol=10)
# for(j in 1:10){
#   for(i in 1:10){
#     itnego140[i,j]<-length(negooc[[(j-1)*10+i+300]])-2
#   }
# }
# for(j in 1:2){
#   itnego140c<-c()
#   for(i in 1:10){
#     itnego140c<-c(itnego140c,length(negooc2[[(j-1)*10+i+60]])-2)
#   }
#   itnego140<-cbind(itnego140,itnego140c)
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap)",10),
#                       rep("noisy kriging (150 test RF)",10),
#                       rep("noisy kriging (140 test RF)",10),rep("noisy kriging (110 test RF)",10),
#                       rep("noisy kriging (80 test RF)",10),rep("noisy kriging (50 test RF)",10),
#                       rep("noisy kriging (20 test RF)",10),rep("noisy kriging (10 test RF)",10),
#                       rep("noisy kriging (4 test RF)",10)),
#                  V2=c(itoego[,4],itego[,4],itnego140[,1],itnego140[,3],
#                       itnego140[,4],itnego140[,5],itnego140[,6],itnego140[,7],
#                       itnego140[,8],itnego140[,9],itnego140[,10]),
#                  V3=c(c(mean(itoego[,4]),rep(NA,9)),c(mean(itego[,4]),rep(NA,9)),c(mean(itnego140[,1]),rep(NA,9)),
#                       c(mean(itnego140[,3]),rep(NA,9)),
#                       c(mean(itnego140[,4]),rep(NA,9)),c(mean(itnego140[,5]),rep(NA,9)),
#                       c(mean(itnego140[,6]),rep(NA,9)),c(mean(itnego140[,7]),rep(NA,9)),
#                       c(mean(itnego140[,8]),rep(NA,9)),c(mean(itnego140[,9]),rep(NA,9)),
#                       c(mean(itnego140[,10]),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (150 test RF)",
#                                     "noisy kriging (140 test RF)","noisy kriging (110 test RF)",
#                                     "noisy kriging (80 test RF)","noisy kriging (50 test RF)",
#                                     "noisy kriging (20 test RF)","noisy kriging (10 test RF)",
#                                     "noisy kriging (4 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 140")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# # vn = 160 #
# itnego160<-matrix(nrow = 10,ncol=10)
# for(j in 1:10){
#   for(i in 1:10){
#     itnego160[i,j]<-length(negooc[[(j-1)*10+i+400]])-2
#   }
# }
# for(j in 1:2){
#   itnego160c<-c()
#   for(i in 1:10){
#     itnego160c<-c(itnego160c,length(negooc2[[(j-1)*10+i+80]])-2)
#   }
#   itnego160<-cbind(itnego160,itnego160c)
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap)",10),
#                       rep("noisy kriging (150 test RF)",10),
#                       rep("noisy kriging (140 test RF)",10),rep("noisy kriging (110 test RF)",10),
#                       rep("noisy kriging (80 test RF)",10),rep("noisy kriging (50 test RF)",10),
#                       rep("noisy kriging (20 test RF)",10),rep("noisy kriging (10 test RF)",10),
#                       rep("noisy kriging (4 test RF)",10)),
#                  V2=c(itoego[,5],itego[,5],itnego160[,1],itnego160[,3],
#                       itnego160[,4],itnego160[,5],itnego160[,6],itnego160[,7],
#                       itnego160[,8],itnego160[,9],itnego160[,10]),
#                  V3=c(c(mean(itoego[,5]),rep(NA,9)),c(mean(itego[,5]),rep(NA,9)),c(mean(itnego160[,1]),rep(NA,9)),
#                       c(mean(itnego160[,3]),rep(NA,9)),
#                       c(mean(itnego160[,4]),rep(NA,9)),c(mean(itnego160[,5]),rep(NA,9)),
#                       c(mean(itnego160[,6]),rep(NA,9)),c(mean(itnego160[,7]),rep(NA,9)),
#                       c(mean(itnego160[,8]),rep(NA,9)),c(mean(itnego160[,9]),rep(NA,9)),
#                       c(mean(itnego160[,10]),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (150 test RF)",
#                                     "noisy kriging (140 test RF)","noisy kriging (110 test RF)",
#                                     "noisy kriging (80 test RF)","noisy kriging (50 test RF)",
#                                     "noisy kriging (20 test RF)","noisy kriging (10 test RF)",
#                                     "noisy kriging (4 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 160")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# # plot Iteration times -- Virtual size for different bootstrap models #
# pittx<-c(rep(c(80,100,120,140,160),5))
# pittm<-c(mean(itoego[,1]),mean(itoego[,2]),mean(itoego[,3]),mean(itoego[,4]),mean(itoego[,5]),
#          mean(itego[,1]),mean(itego[,2]),mean(itego[,3]),mean(itego[,4]),mean(itego[,5]),
#          mean(itnego80[,11]),mean(itnego100[,11]),mean(itnego120[,11]),mean(itnego140[,11]),mean(itnego160[,11]),
#          mean(itnego80[,12]),mean(itnego100[,12]),mean(itnego120[,12]),mean(itnego140[,12]),mean(itnego160[,12]),
#          mean(itnego80[,1]),mean(itnego100[,1]),mean(itnego120[,1]),mean(itnego140[,1]),mean(itnego160[,1]))
# pittsd<-c(sd(itoego[,1]),sd(itoego[,2]),sd(itoego[,3]),sd(itoego[,4]),sd(itoego[,5]),
#          sd(itego[,1]),sd(itego[,2]),sd(itego[,3]),sd(itego[,4]),sd(itego[,5]),
#          sd(itnego80[,11]),sd(itnego100[,11]),sd(itnego120[,11]),sd(itnego140[,11]),sd(itnego160[,11]),
#          sd(itnego80[,12]),sd(itnego100[,12]),sd(itnego120[,12]),sd(itnego140[,12]),sd(itnego160[,12]),
#          sd(itnego80[,1]),sd(itnego100[,1]),sd(itnego120[,1]),sd(itnego140[,1]),sd(itnego160[,1]))
# pittd<-rep(c("kriging + EGO", "kriging with nugget + EGO", "noisy kriging (GB bootstrap) + EGO",
#              "noisy kriging (SVR bootstrap) + EGO","noisy kriging (RF bootstrap) + EGO"),each=5)
# pittdat<-as.data.frame(cbind(pittx,pittm,pittsd,pittd))
# pittdat$pittx<-as.numeric(pittdat$pittx)
# pittdat$pittm<-as.numeric(pittdat$pittm)
# pittdat$pittsd<-as.numeric(pittdat$pittsd)
# pittdat$pittd<-ordered(pittdat$pittd,levels = c("kriging + EGO", "kriging with nugget + EGO", "noisy kriging (GB bootstrap) + EGO",
#                                                 "noisy kriging (SVR bootstrap) + EGO","noisy kriging (RF bootstrap) + EGO"))
# pittdat$palem<-rep(1,25)-pittdat$pittm/pittdat$pittx
# pittdat$palesd<-pittdat$pittsd/pittdat$pittx
# 
# ggplot(data.frame(x=pittdat$pittx,y=pittdat$pittm),aes(x,y,color=pittdat$pittd))+
#   geom_line(size=1.2)+
#   geom_errorbar(aes(x=pittdat$pittx,ymin=pittdat$pittm-0.5*pittdat$pittsd, ymax=pittdat$pittm+0.5*pittdat$pittsd), size=1.2,width=16)+
#   scale_color_manual(name = " ",values = c("grey","red","blue","orange","purple"),
#                      labels = c("kriging + EGO", "noisy kriging (GB bootstrap) + EGO",
#                                 "noisy kriging (SVR bootstrap) + EGO","noisy kriging (RF bootstrap) + EGO",
#                                 "noisy kriging (RF tree) + EGO")) +
#   scale_x_continuous(breaks=c(80,100,120,140,160))+
#   xlab("Number of virtual data")+ylab("Iteration times")+
#   theme_bw(base_size = 30)+
#   theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank(),
#         axis.text=element_text(color="black",size=24),
#         axis.title=element_text(face="bold"))
# 
# ggplot(data.frame(x=pittdat$pittx,y=pittdat$palem),aes(x,y,color=pittdat$pittd))+
#   geom_line(size=1.2)+
#   geom_errorbar(aes(x=pittdat$pittx,ymin=pittdat$palem-0.5*pittdat$palesd, ymax=pittdat$palem+0.5*pittdat$palesd), size=1.2,width=1.6)+
#   scale_color_manual(name = " ",values = c("grey","blue","orange","purple","red"),
#                      labels = c("kriging + EGO", "kriging with nugget + EGO", "noisy kriging (GB bootstrap) + EGO",
#                                 "noisy kriging (SVR bootstrap) + EGO","noisy kriging (RF bootstrap) + EGO")) +
#   scale_x_continuous(breaks=c(80,100,120,140,160))+
#   xlab("Number of virtual data")+ylab("AL efficiency")+
#   theme_bw(base_size = 30)+
#   theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank(),
#         axis.text=element_text(color="black",size=24),
#         axis.title=element_text(face="bold"))
# 
# #* UCB OC of kriging with noise *#
# cl<-makeCluster(detectCores(),outfile="nucboc3.txt")
# clusterExport(cl, list("itocn","krigm","t","p","estrs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnucbpar<-function(vvi){
#   return(itocn(as.numeric(vvi[1]),120,64,as.numeric(vvi[3]),"ucb",vvi[2]))
# }
# 
# system.time(
#   nucboc<-parApply(cl, vvim, 1, itocnucbpar)
# )
# stopCluster(cl)
# 
# save.image("estr4.RData")
# 
# # vn = 60 #
# itnucb60<-matrix(nrow = 10,ncol=13)
# for(j in 1:13){
#   for(i in 1:10){
#     itnucb60[i,j]<-length(nucboc[[(j-1)*10+i]])-2
#   }
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap)",10),
#                       rep("noisy kriging (110 test RF)",10),rep("noisy kriging (100 test RF)",10),
#                       rep("noisy kriging (80 test RF)",10),rep("noisy kriging (60 test RF)",10),
#                       rep("noisy kriging (40 test RF)",10),rep("noisy kriging (20 test RF)",10),
#                       rep("noisy kriging (10 test RF)",10),rep("noisy kriging (5 test RF)",10),
#                       rep("noisy kriging (3 test RF)",10),rep("noisy kriging (2 test RF)",10)),
#                  V2=c(itoucb[,1],itucb[,1],itnucb60[,3],itnucb60[,4],itnucb60[,5],
#                       itnucb60[,6],itnucb60[,7],itnucb60[,8],itnucb60[,9],
#                       itnucb60[,10],itnucb60[,11],itnucb60[,12],itnucb60[,13]),
#                  V3=c(c(mean(itoucb[,1]),rep(NA,9)),c(mean(itucb[,1]),rep(NA,9)),c(mean(itnucb60[,3]),rep(NA,9)),
#                       c(mean(itnucb60[,4]),rep(NA,9)),c(mean(itnucb60[,5]),rep(NA,9)),
#                       c(mean(itnucb60[,6]),rep(NA,9)),c(mean(itnucb60[,7]),rep(NA,9)),
#                       c(mean(itnucb60[,8]),rep(NA,9)),c(mean(itnucb60[,9]),rep(NA,9)),
#                       c(mean(itnucb60[,10]),rep(NA,9)),c(mean(itnucb60[,11]),rep(NA,9)),
#                       c(mean(itnucb60[,12]),rep(NA,9)),c(mean(itnucb60[,13]),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (110 test RF)","noisy kriging (100 test RF)",
#                                     "noisy kriging (80 test RF)","noisy kriging (60 test RF)",
#                                     "noisy kriging (40 test RF)","noisy kriging (20 test RF)",
#                                     "noisy kriging (10 test RF)","noisy kriging (5 test RF)",
#                                     "noisy kriging (3 test RF)","noisy kriging (2 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 60")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# # vn = 75 #
# itnucb75<-matrix(nrow = 10,ncol=13)
# for(j in 1:13){
#   for(i in 1:10){
#     itnucb75[i,j]<-length(nucboc[[(j-1)*10+i+130]])-2
#   }
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap)",10),
#                       rep("noisy kriging (110 test RF)",10),rep("noisy kriging (100 test RF)",10),
#                       rep("noisy kriging (80 test RF)",10),rep("noisy kriging (60 test RF)",10),
#                       rep("noisy kriging (40 test RF)",10),rep("noisy kriging (20 test RF)",10),
#                       rep("noisy kriging (10 test RF)",10),rep("noisy kriging (5 test RF)",10),
#                       rep("noisy kriging (3 test RF)",10),rep("noisy kriging (2 test RF)",10)),
#                  V2=c(itoucb[,2],itucb[,2],itnucb75[,3],itnucb75[,4],itnucb75[,5],
#                       itnucb75[,6],itnucb75[,7],itnucb75[,8],itnucb75[,9],
#                       itnucb75[,10],itnucb75[,11],itnucb75[,12],itnucb75[,13]),
#                  V3=c(c(mean(itoucb[,2]),rep(NA,9)),c(mean(itucb[,2]),rep(NA,9)),c(mean(itnucb75[,3]),rep(NA,9)),
#                       c(mean(itnucb75[,4]),rep(NA,9)),c(mean(itnucb75[,5]),rep(NA,9)),
#                       c(mean(itnucb75[,6]),rep(NA,9)),c(mean(itnucb75[,7]),rep(NA,9)),
#                       c(mean(itnucb75[,8]),rep(NA,9)),c(mean(itnucb75[,9]),rep(NA,9)),
#                       c(mean(itnucb75[,10]),rep(NA,9)),c(mean(itnucb75[,11]),rep(NA,9)),
#                       c(mean(itnucb75[,12]),rep(NA,9)),c(mean(itnucb75[,13]),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (110 test RF)","noisy kriging (100 test RF)",
#                                     "noisy kriging (80 test RF)","noisy kriging (60 test RF)",
#                                     "noisy kriging (40 test RF)","noisy kriging (20 test RF)",
#                                     "noisy kriging (10 test RF)","noisy kriging (5 test RF)",
#                                     "noisy kriging (3 test RF)","noisy kriging (2 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 75")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# # vn = 90 #
# itnucb90<-matrix(nrow = 10,ncol=13)
# for(j in 1:13){
#   for(i in 1:10){
#     itnucb90[i,j]<-length(nucboc[[(j-1)*10+i+260]])-2
#   }
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap)",10),
#                       rep("noisy kriging (110 test RF)",10),rep("noisy kriging (100 test RF)",10),
#                       rep("noisy kriging (80 test RF)",10),rep("noisy kriging (60 test RF)",10),
#                       rep("noisy kriging (40 test RF)",10),rep("noisy kriging (20 test RF)",10),
#                       rep("noisy kriging (10 test RF)",10),rep("noisy kriging (5 test RF)",10),
#                       rep("noisy kriging (3 test RF)",10),rep("noisy kriging (2 test RF)",10)),
#                  V2=c(itoucb[,3],itucb[,3],itnucb90[,3],itnucb90[,4],itnucb90[,5],
#                       itnucb90[,6],itnucb90[,7],itnucb90[,8],itnucb90[,9],
#                       itnucb90[,10],itnucb90[,11],itnucb90[,12],itnucb90[,13]),
#                  V3=c(c(mean(itoucb[,3]),rep(NA,9)),c(mean(itucb[,3]),rep(NA,9)),c(mean(itnucb90[,3]),rep(NA,9)),
#                       c(mean(itnucb90[,4]),rep(NA,9)),c(mean(itnucb90[,5]),rep(NA,9)),
#                       c(mean(itnucb90[,6]),rep(NA,9)),c(mean(itnucb90[,7]),rep(NA,9)),
#                       c(mean(itnucb90[,8]),rep(NA,9)),c(mean(itnucb90[,9]),rep(NA,9)),
#                       c(mean(itnucb90[,10]),rep(NA,9)),c(mean(itnucb90[,11]),rep(NA,9)),
#                       c(mean(itnucb90[,12]),rep(NA,9)),c(mean(itnucb90[,13]),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (110 test RF)","noisy kriging (100 test RF)",
#                                     "noisy kriging (80 test RF)","noisy kriging (60 test RF)",
#                                     "noisy kriging (40 test RF)","noisy kriging (20 test RF)",
#                                     "noisy kriging (10 test RF)","noisy kriging (5 test RF)",
#                                     "noisy kriging (3 test RF)","noisy kriging (2 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 90")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# # vn = 105 #
# itnucb105<-matrix(nrow = 10,ncol=13)
# for(j in 1:13){
#   for(i in 1:10){
#     itnucb105[i,j]<-length(nucboc[[(j-1)*10+i+390]])-2
#   }
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap)",10),
#                       rep("noisy kriging (110 test RF)",10),rep("noisy kriging (100 test RF)",10),
#                       rep("noisy kriging (80 test RF)",10),rep("noisy kriging (60 test RF)",10),
#                       rep("noisy kriging (40 test RF)",10),rep("noisy kriging (20 test RF)",10),
#                       rep("noisy kriging (10 test RF)",10),rep("noisy kriging (5 test RF)",10),
#                       rep("noisy kriging (3 test RF)",10),rep("noisy kriging (2 test RF)",10)),
#                  V2=c(itoucb[,4],itucb[,4],itnucb105[,3],itnucb105[,4],itnucb105[,5],
#                       itnucb105[,6],itnucb105[,7],itnucb105[,8],itnucb105[,9],
#                       itnucb105[,10],itnucb105[,11],itnucb105[,12],itnucb105[,13]),
#                  V3=c(c(mean(itoucb[,4]),rep(NA,9)),c(mean(itucb[,4]),rep(NA,9)),c(mean(itnucb105[,3]),rep(NA,9)),
#                       c(mean(itnucb105[,4]),rep(NA,9)),c(mean(itnucb105[,5]),rep(NA,9)),
#                       c(mean(itnucb105[,6]),rep(NA,9)),c(mean(itnucb105[,7]),rep(NA,9)),
#                       c(mean(itnucb105[,8]),rep(NA,9)),c(mean(itnucb105[,9]),rep(NA,9)),
#                       c(mean(itnucb105[,10]),rep(NA,9)),c(mean(itnucb105[,11]),rep(NA,9)),
#                       c(mean(itnucb105[,12]),rep(NA,9)),c(mean(itnucb105[,13]),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (110 test RF)","noisy kriging (100 test RF)",
#                                     "noisy kriging (80 test RF)","noisy kriging (60 test RF)",
#                                     "noisy kriging (40 test RF)","noisy kriging (20 test RF)",
#                                     "noisy kriging (10 test RF)","noisy kriging (5 test RF)",
#                                     "noisy kriging (3 test RF)","noisy kriging (2 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 105")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# # vn = 120 #
# itnucb120<-matrix(nrow = 10,ncol=13)
# for(j in 1:13){
#   for(i in 1:10){
#     itnucb120[i,j]<-length(nucboc[[(j-1)*10+i+520]])-2
#   }
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap)",10),
#                       rep("noisy kriging (110 test RF)",10),rep("noisy kriging (100 test RF)",10),
#                       rep("noisy kriging (80 test RF)",10),rep("noisy kriging (60 test RF)",10),
#                       rep("noisy kriging (40 test RF)",10),rep("noisy kriging (20 test RF)",10),
#                       rep("noisy kriging (10 test RF)",10),rep("noisy kriging (5 test RF)",10),
#                       rep("noisy kriging (3 test RF)",10),rep("noisy kriging (2 test RF)",10)),
#                  V2=c(itoucb[,5],itucb[,5],itnucb120[,3],itnucb120[,4],itnucb120[,5],
#                       itnucb120[,6],itnucb120[,7],itnucb120[,8],itnucb120[,9],
#                       itnucb120[,10],itnucb120[,11],itnucb120[,12],itnucb120[,13]),
#                  V3=c(c(mean(itoucb[,5]),rep(NA,9)),c(mean(itucb[,5]),rep(NA,9)),c(mean(itnucb120[,3]),rep(NA,9)),
#                       c(mean(itnucb120[,4]),rep(NA,9)),c(mean(itnucb120[,5]),rep(NA,9)),
#                       c(mean(itnucb120[,6]),rep(NA,9)),c(mean(itnucb120[,7]),rep(NA,9)),
#                       c(mean(itnucb120[,8]),rep(NA,9)),c(mean(itnucb120[,9]),rep(NA,9)),
#                       c(mean(itnucb120[,10]),rep(NA,9)),c(mean(itnucb120[,11]),rep(NA,9)),
#                       c(mean(itnucb120[,12]),rep(NA,9)),c(mean(itnucb120[,13]),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (110 test RF)","noisy kriging (100 test RF)",
#                                     "noisy kriging (80 test RF)","noisy kriging (60 test RF)",
#                                     "noisy kriging (40 test RF)","noisy kriging (20 test RF)",
#                                     "noisy kriging (10 test RF)","noisy kriging (5 test RF)",
#                                     "noisy kriging (3 test RF)","noisy kriging (2 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 120")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# # plot Iteration times -- Virtual size for different bootstrap models #
# pittx<-c(rep(c(60,75,90,105,120),5))
# pittm<-c(mean(itoucb[,1]),mean(itoucb[,2]),mean(itoucb[,3]),mean(itoucb[,4]),mean(itoucb[,5]),
#          mean(itucb[,1]),mean(itucb[,2]),mean(itucb[,3]),mean(itucb[,4]),mean(itucb[,5]),
#          mean(itnucb60[,1]),mean(itnucb75[,1]),mean(itnucb90[,1]),mean(itnucb105[,1]),mean(itnucb120[,1]),
#          mean(itnucb60[,2]),mean(itnucb75[,2]),mean(itnucb90[,2]),mean(itnucb105[,2]),mean(itnucb120[,2]),
#          mean(itnucb60[,3]),mean(itnucb75[,3]),mean(itnucb90[,3]),mean(itnucb105[,3]),mean(itnucb120[,3]))
# pittsd<-c(sd(itoucb[,1]),sd(itoucb[,2]),sd(itoucb[,3]),sd(itoucb[,4]),sd(itoucb[,5]),
#           sd(itucb[,1]),sd(itucb[,2]),sd(itucb[,3]),sd(itucb[,4]),sd(itucb[,5]),
#           sd(itnucb60[,1]),sd(itnucb75[,1]),sd(itnucb90[,1]),sd(itnucb105[,1]),sd(itnucb120[,1]),
#           sd(itnucb60[,2]),sd(itnucb75[,2]),sd(itnucb90[,2]),sd(itnucb105[,2]),sd(itnucb120[,2]),
#           sd(itnucb60[,3]),sd(itnucb75[,3]),sd(itnucb90[,3]),sd(itnucb105[,3]),sd(itnucb120[,3]))
# pittd<-rep(c("kriging + UCB", "kriging with nugget + UCB", "noisy kriging (GB bootstrap) + UCB",
#              "noisy kriging (SVR bootstrap) + UCB","noisy kriging (RF bootstrap) + UCB"),each=5)
# pittdat<-as.data.frame(cbind(pittx,pittm,pittsd,pittd))
# pittdat$pittx<-as.numeric(pittdat$pittx)
# pittdat$pittm<-as.numeric(pittdat$pittm)
# pittdat$pittsd<-as.numeric(pittdat$pittsd)
# pittdat$pittd<-ordered(pittdat$pittd,levels = c("kriging + UCB", "kriging with nugget + UCB", "noisy kriging (GB bootstrap) + UCB",
#                                                 "noisy kriging (SVR bootstrap) + UCB","noisy kriging (RF bootstrap) + UCB"))
# pittdat$palem<-rep(1,25)-pittdat$pittm/pittdat$pittx
# pittdat$palesd<-pittdat$pittsd/pittdat$pittx
# 
# ggplot(data.frame(x=pittdat$pittx,y=pittdat$pittm),aes(x,y,color=pittdat$pittd))+
#   geom_line(size=1.2)+
#   geom_errorbar(aes(x=pittdat$pittx,ymin=pittdat$pittm-0.5*pittdat$pittsd, ymax=pittdat$pittm+0.5*pittdat$pittsd), size=1.2,width=16)+
#   scale_color_manual(name = " ",values = c("grey","red","blue","orange","purple"),
#                      labels = c("kriging + UCB", "noisy kriging (GB bootstrap) + UCB",
#                                 "noisy kriging (SVR bootstrap) + UCB","noisy kriging (RF bootstrap) + UCB",
#                                 "noisy kriging (RF tree) + UCB")) +
#   scale_x_continuous(breaks=c(60,75,90,105,120))+
#   xlab("Number of virtual data")+ylab("Iteration times")+
#   theme_bw(base_size = 30)+
#   theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank(),
#         axis.text=element_text(color="black",size=24),
#         axis.title=element_text(face="bold"))
# 
# ggplot(data.frame(x=pittdat$pittx,y=pittdat$palem),aes(x,y,color=pittdat$pittd))+
#   geom_line(size=1.2)+
#   geom_errorbar(aes(x=pittdat$pittx,ymin=pittdat$palem-0.5*pittdat$palesd, ymax=pittdat$palem+0.5*pittdat$palesd), size=1.2,width=1.6)+
#   scale_color_manual(name = " ",values = c("grey","blue","orange","purple","red"),
#                      labels = c("kriging + UCB", "kriging with nugget + UCB", "noisy kriging (GB bootstrap) + UCB",
#                                 "noisy kriging (SVR bootstrap) + UCB","noisy kriging (RF bootstrap) + UCB")) +
#   scale_x_continuous(breaks=c(60,75,90,105,120))+
#   xlab("Number of virtual data")+ylab("AL efficiency")+
#   theme_bw(base_size = 30)+
#   theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank(),
#         axis.text=element_text(color="black",size=24),
#         axis.title=element_text(face="bold"))
# 
# # 不同虚拟空间，用四种效能函数
# cl<-makeCluster(detectCores(),outfile="nufoc.txt")
# clusterExport(cl, list("itocn","krigm","t","p","estrs"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnuf<-function(ui){
#   return(itocn(as.numeric(ui[1]),131,64,as.numeric(ui[3]),ui[2],"tv1rfvar5"))
# }
# 
# system.time(
#   nufoc<-parApply(cl, vuim, 1, itocnuf)
# )
# stopCluster(cl)
# 
# save.image("estr4.RData")
# 
# itnuf<-matrix(nrow = 100,ncol=9)
# for(j in 1:9){
#   for(i in 1:100){
#     itnuf[i,j]<-length(nufoc[[(j-1)*100+i]])-2
#   }
# }
# colnames(itnuf)<-c("pre7","ucb7","ego7","pre8","ucb8","ego8","pre9","ucb9","ego9")
# 
# save.image("estr4.RData")
# 
# # plot distribution of itt uf
# #70% vs
# pmcv<-data.frame(V1=c(rep("kriging + Pre",100),rep("kriging + UCB",100),rep("kriging + EGO",100),
#                       rep("kriging with nugget + Pre",100),rep("kriging with nugget + UCB",100),
#                       rep("kriging with nugget + EGO",100),rep("kriging with nugget + SKO",100),
#                       rep("noisy kriging (150 test RF) + Pre",100),rep("noisy kriging (150 test RF) + UCB",100),
#                       rep("noisy kriging (150 test RF) + EGO",100)),
#                  V2=c(itouf[,1],itouf[,2],itouf[,3],ituf[,1],ituf[,2],ituf[,3],ituf[,4],
#                       itnuf[,1],itnuf[,2],itnuf[,3]),
#                  V3=c(c(mean(itouf[,1]),rep(NA,99)),c(mean(itouf[,2]),rep(NA,99)),c(mean(itouf[,3]),rep(NA,99)),
#                       c(mean(ituf[,1]),rep(NA,99)),c(mean(ituf[,2]),rep(NA,99)),c(mean(ituf[,3]),rep(NA,99)),c(mean(ituf[,4]),rep(NA,99)),
#                       c(mean(itnuf[,1]),rep(NA,99)),c(mean(itnuf[,2]),rep(NA,99)),c(mean(itnuf[,3]),rep(NA,99))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging + Pre", "kriging + UCB", "kriging + EGO",
#                                     "kriging with nugget + Pre","kriging with nugget + UCB",
#                                     "kriging with nugget + EGO","kriging with nugget + SKO",
#                                     "noisy kriging (150 test RF) + Pre","noisy kriging (150 test RF) + UCB",
#                                     "noisy kriging (150 test RF) + EGO"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Electrostrain multi-source 70% virtual space")+
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
#                       rep("noisy kriging (150 test RF) + Pre",100),rep("noisy kriging (150 test RF) + UCB",100),
#                       rep("noisy kriging (150 test RF) + EGO",100)),
#                  V2=c(itouf[,4],itouf[,5],itouf[,6],ituf[,5],ituf[,6],ituf[,7],ituf[,8],
#                       itnuf[,4],itnuf[,5],itnuf[,6]),
#                  V3=c(c(mean(itouf[,4]),rep(NA,99)),c(mean(itouf[,5]),rep(NA,99)),c(mean(itouf[,6]),rep(NA,99)),
#                       c(mean(ituf[,5]),rep(NA,99)),c(mean(ituf[,6]),rep(NA,99)),c(mean(ituf[,7]),rep(NA,99)),c(mean(ituf[,8]),rep(NA,99)),
#                       c(mean(itnuf[,4]),rep(NA,99)),c(mean(itnuf[,5]),rep(NA,99)),c(mean(itnuf[,6]),rep(NA,99))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging + Pre", "kriging + UCB", "kriging + EGO",
#                                     "kriging with nugget + Pre","kriging with nugget + UCB",
#                                     "kriging with nugget + EGO","kriging with nugget + SKO",
#                                     "noisy kriging (150 test RF) + Pre","noisy kriging (150 test RF) + UCB",
#                                     "noisy kriging (150 test RF) + EGO"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Electrostrain multi-source 80% virtual space")+
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
#                       rep("noisy kriging (150 test RF) + Pre",100),rep("noisy kriging (150 test RF) + UCB",100),
#                       rep("noisy kriging (150 test RF) + EGO",100)),
#                  V2=c(itouf[,7],itouf[,8],itouf[,9],ituf[,9],ituf[,10],ituf[,11],ituf[,12],
#                       itnuf[,7],itnuf[,8],itnuf[,9]),
#                  V3=c(c(mean(itouf[,7]),rep(NA,99)),c(mean(itouf[,8]),rep(NA,99)),c(mean(itouf[,9]),rep(NA,99)),
#                       c(mean(ituf[,9]),rep(NA,99)),c(mean(ituf[,10]),rep(NA,99)),c(mean(ituf[,11]),rep(NA,99)),c(mean(ituf[,12]),rep(NA,99)),
#                       c(mean(itnuf[,7]),rep(NA,99)),c(mean(itnuf[,8]),rep(NA,99)),c(mean(itnuf[,9]),rep(NA,99))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging + Pre", "kriging + UCB", "kriging + EGO",
#                                     "kriging with nugget + Pre","kriging with nugget + UCB",
#                                     "kriging with nugget + EGO","kriging with nugget + SKO",
#                                     "noisy kriging (150 test RF) + Pre","noisy kriging (150 test RF) + UCB",
#                                     "noisy kriging (150 test RF) + EGO"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Electrostrain multi-source 90% virtual space")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# ##************##



# ##** Predicting in virtual space **##
# btovs1<-read.csv("~/ECEdemo/directboots/BTO-VS 2.csv")[,2:8]
# btovs1<-cbind(btovs1,read.csv("~/ECEdemo/directboots/BTO-VS 2.csv")[,c("NCT","p", "tA.B", "AV", "ENMB","D","NTO")])
# #将虚拟空间描述符值对应为训练集中百分数含量计算出值
# btovs1$NCT<-btovs1$NCT*100
# btovs1$NTO<-btovs1$NTO*100
# btovs1$p<-btovs1$p*10000
# btovs1$tA.B<-btovs1$tA.B*10000
# gc()
# #* prediction models *#
# preo<-krigmo(estrs[,9:16],t,p,64)
# prenu<-krigm(estrs[,9:16],t,p,0,64)
# preno<-krigm(estrs[,9:16],t,p,estrs[,"tv1rfvar5"],64)
# prenosa<-krigm(estrs[,9:16],t,p,estrs[,"tv1rfvar5m"],64)
# 
# # Input: the number(/10000) of btovs   Output: number, mean and sigma2 of predictions
# prevs<-function(n){
#   gc()
#   preovs<-predict(preo, btovs1[((n-1)*10000+1):(n*10000), c(8:14)], type = "SK")
#   prenuvs<-predict(prenu, btovs1[((n-1)*10000+1):(n*10000), c(8:14)], type = "SK")
#   prenovs<-predict(preno, btovs1[((n-1)*10000+1):(n*10000), c(8:14)], type = "SK")
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
# preovs_la<-predict(preo, btovs1[270001:273897, c(8:14)], type = "SK")
# prenuvs_la<-predict(prenu, btovs1[270001:273897, c(8:14)], type = "SK")
# prenovs_la<-predict(preno, btovs1[270001:273897, c(8:14)], type = "SK")
# gc()
# prevsallc<-rbind(prevsallc, data.frame(ind=c(270001:273897), om=preovs_la[["mean"]], os=preovs_la[["sd"]],
#                                   num=prenuvs_la[["mean"]], nus=prenuvs_la[["sd"]],
#                                   nom=prenovs_la[["mean"]], nos=prenovs_la[["sd"]]))
# gc()
# 
# prevsallc[,"div"]<-abs(prevsallc$om-prevsallc$num)+abs(prevsallc$num-prevsallc$nom)+
#                    abs(prevsallc$nom-prevsallc$om)
# divind<-order(prevsallc$div, decreasing = T)
# gc()
# 
# vso<-cbind(btovs1[divind[1:20],], prevsallc[divind[1:20],-1])
# 
# rm(prevsallc)
# rm(preovs_la)
# rm(prenuvs_la)
# rm(prenovs_la)
# gc()
# 
# save.image("estr4.RData")
# 
# 
# #predict some selected samples
# btovss<-read.csv("selected VS.csv")[,2:8]
# btovss<-cbind(btovss,read.csv("selected VS.csv")[,c("NCT","p", "tA.B", "AV", "ENMB","D","NTO")])
# #将虚拟空间描述符值对应为训练集中百分数含量计算出值
# btovss$NCT<-btovss$NCT*100
# btovss$NTO<-btovss$NTO*100
# btovss$p<-btovss$p*10000
# btovss$tA.B<-btovss$tA.B*10000
# 
# preovss<-predict(preo, btovss[, c(8:14)], type = "SK")
# prenuvss<-predict(prenu, btovss[, c(8:14)], type = "SK")
# prenovss<-predict(preno, btovss[, c(8:14)], type = "SK")
# prevssp<-data.frame(ind=c(1:12), om=preovss[["mean"]], os=preovss[["sd"]],
#                     num=prenuvss[["mean"]], nus=prenuvss[["sd"]],
#                     nom=prenovss[["mean"]], nos=prenovss[["sd"]])
# prevssp$div<-abs(prevssp$om-prevssp$num)+abs(prevssp$num-prevssp$nom)+
#   abs(prevssp$nom-prevssp$om)
# 
# 
# #predict some random selected samples
# btovsrs<-read.csv("random selected VS.csv")[,2:8]
# btovsrs<-cbind(btovsrs,read.csv("random selected VS.csv")[,c("NCT","p", "tA.B", "AV", "ENMB","D","NTO")])
# #将虚拟空间描述符值对应为训练集中百分数含量计算出值
# btovsrs$NCT<-btovsrs$NCT*100
# btovsrs$NTO<-btovsrs$NTO*100
# btovsrs$p<-btovsrs$p*10000
# btovsrs$tA.B<-btovsrs$tA.B*10000
# 
# library(Rtsne)
# intsnet<-as.matrix(rbind(estrs[,c(10:16)],btovsrs[,c(8:14)]))
# #normalize
# for(i in 1:7){
#   intsnet[,i]<-(intsnet[,i]-min(intsnet[,i])) / (max(intsnet[,i])-min(intsnet[,i]))
# }
# intsnet<-normalize_input(intsnet)
# set.seed(3)
# intsne<-Rtsne(intsnet)
# plot(x=intsne$Y[,1],y=intsne$Y[,2],
#      pch= c(rep(1,length(estrs[,13])),rep(17,length(btovsrs[,13]))),
#      col= c(rep("red",length(estrs[,13])),rep("black",length(btovsrs[,13]))), cex = 1.2,
#      #xlim = c(-20,28), ylim = c(-25,30),
#      xlab="tSNE_1",ylab="tSNE_2",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# 
# preovsrs<-predict(preo, btovsrs[, c(8:14)], type = "SK")
# prenuvsrs<-predict(prenu, btovsrs[, c(8:14)], type = "SK")
# prenovsrs<-predict(preno, btovsrs[, c(8:14)], type = "SK")
# prevsrsp<-data.frame(ind=c(1:4), om=preovsrs[["mean"]], os=preovsrs[["sd"]],
#                     num=prenuvsrs[["mean"]], nus=prenuvsrs[["sd"]],
#                     nom=prenovsrs[["mean"]], nos=prenovsrs[["sd"]])
# prevsrsp$div<-abs(prevsrsp$om-prevsrsp$num)+abs(prevsrsp$num-prevsrsp$nom)+
#   abs(prevsrsp$nom-prevsrsp$om)
# 
# library(readxl)
# expcomp <- read_excel("experiments compare.xlsx", sheet = "estrain")
# expcomp<-as.matrix(expcomp)
# plot(rep(expcomp[,4],3),y=c(expcomp[,1],expcomp[,2],expcomp[,3]),
#        pch=c(rep(15,4),rep(16,4),rep(17,4)),
#        col= c(rep("grey",4),rep("red",4),rep("blue",4)), cex = 1.4,
#        xlim = c(100,800), ylim = c(100,800),
#        xlab="Measured d33* (pm/V)", ylab="Predicted d33* (pm/V)",
#        cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2)
# 
#   
#   
# ##** Adding round 1 data and predicting round 2 **##
# # data #
# library(readxl)
# expiter1 <- read_excel("Exp iter1.xlsx", sheet = "Sheet1")[-6,2:9]
# expiter1$cd<-rep(0,dim(expiter1)[1]); expiter1<-expiter1[,c(1:3,9,4:8)]
# colnames(expiter1)<-colnames(estrs)[1:9]
# expiter1<-fn.data.features(expiter1)
# expiter1<-cbind(expiter1[,c(1:9)],expiter1[,c("NCT","p", "tA.B", "AV", "ENMB","D","NTO")])
# expiter1$refID<-rep(1,dim(expiter1)[1])
# estrsi1<-rbind(estrs[,1:17],expiter1)
# 
# # estimate variances #
# #kriging estimated nugget
# estri1krig<-krigm(estrsi1[,9:16],t,p,0,64)
# ki1nug<-estri1krig@covariance@nugget
# 
# rftvi<-function(Bts){
#   set.seed(Bts[1])
#   estrtv<-estrsi1[sample(dim(estrsi1)[1], Bts[2], replace = F),9:16]
#   estrtvtest<-anti_join(estrsi1[,9:16],estrtv)
#   tvrfm<-randomForest(y~.,data=estrtv,ntree=2000,mtry=2)
#   pretvtest<-predict(tvrfm,estrtvtest)
#   pretestna<-c()
#   for(i in 1:dim(estrsi1)[1]){
#     ind<-which(estrtvtest$NCT==estrsi1$NCT[i] & estrtvtest$p==estrsi1$p[i] &
#                  estrtvtest$tA.B==estrsi1$tA.B[i] & estrtvtest$AV==estrsi1$AV[i] &
#                  estrtvtest$ENMB==estrsi1$ENMB[i] & estrtvtest$D==estrsi1$D[i] & 
#                  estrtvtest$NTO==estrsi1$NTO[i])
#     if(length(ind)==0){
#       pretestna<-c(pretestna,NA)
#     }else{
#       pretestna<-c(pretestna,pretvtest[ind])
#     }
#   }
#   gc()
#   return(pretestna)
# }
# # B=500 #
# Btsin3<-matrix(ncol=2)
# for(ts in c(14)){
#   for(B in 1:500){
#     Btsin3<-rbind(Btsin3,c(B,ts))
#   }
# }
# Btsin3<-Btsin3[-1,]
# 
# cl<-makeCluster(detectCores(),outfile="rftv5.txt")
# clusterExport(cl, list("estrsi1"))
# clusterEvalQ(cl,{library(randomForest);library(dplyr)})
# 
# system.time(
#   rftvi1pre5<-parApply(cl, Btsin3, 1, rftvi)
# )
# stopCluster(cl)
# 
# save.image("estr4.RData")
# 
# tvi11rfvar5<-c()
# for(i in 1:dim(rftvi1pre5)[1]){
#   tvi11rfvar5<-c(tvi11rfvar5,var(rftvi1pre5[i,1:500],na.rm=T))
# }
# 
# estrsi1<-cbind(estrsi1,tvi11rfvar5)
# 
# #* prediction models *#
# prei1o<-krigmo(estrsi1[,9:16],t,p,64)
# prei1nu<-krigm(estrsi1[,9:16],t,p,0,64)
# prei1no<-krigm(estrsi1[,9:16],t,p,estrsi1[,"tvi11rfvar5"],64)
# 
# preovsri1s<-predict(prei1o, btovsrs[, c(8:14)], type = "SK")
# prenuvsri1s<-predict(prei1nu, btovsrs[, c(8:14)], type = "SK")
# prenovsri1s<-predict(prei1no, btovsrs[, c(8:14)], type = "SK")
# prevsri1sp<-data.frame(ind=c(1:4), om=preovsri1s[["mean"]], os=preovsri1s[["sd"]],
#                      num=prenuvsri1s[["mean"]], nus=prenuvsri1s[["sd"]],
#                      nom=prenovsri1s[["mean"]], nos=prenovsri1s[["sd"]])
# prevsri1sp$div<-abs(prevsri1sp$om-prevsri1sp$num)+abs(prevsri1sp$num-prevsri1sp$nom)+
#   abs(prevsri1sp$nom-prevsri1sp$om)
# 
# expcompi1 <- read_excel("experiments compare.xlsx", sheet = "estrain")[6:9,]
# expcompi1<-as.matrix(expcompi1)
# plot(rep(expcompi1[,4],3),y=c(expcompi1[,1],expcompi1[,2],expcompi1[,3]),
#      pch=c(rep(15,4),rep(16,4),rep(17,4)),
#      col= c(rep("grey",4),rep("red",4),rep("blue",4)), cex = 1.4,
#      xlim = c(100,800), ylim = c(100,800),
#      xlab="Measured d33* (pm/V)", ylab="Predicted d33* (pm/V)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2)


# #* random Selected VS + *#
# library(openxlsx)
# btovsrpls<-read.xlsx("random selected VS +.xlsx")[,2:12]
# btovsrplsf<-cbind(btovsrpls[,1:3],rep(0,dim(btovsrpls)[1]),btovsrpls[4:7])
# colnames(btovsrplsf)<-c("ba","ca","sr","cd","ti","zr","sn","hf")
# btovsrplsf<-fn.data.features(btovsrplsf)
# btovsrpls<-cbind(btovsrpls,btovsrplsf[,c("tA.B","TA.B","EN","enp","ENMB",
#                                          "D","BD","AV","p","z","NCT","NTO")])
# write.table(btovsrpls,"random selected VS +.csv",sep = ",")
# 
# btovsrpls<-cbind(btovsrpls[,1:8],btovsrpls[,c("NCT","p", "tA.B", "AV", "ENMB","D","NTO")])
# #将虚拟空间描述符值对应为训练集中百分数含量计算出值
# btovsrpls$NCT<-btovsrpls$NCT*100
# btovsrpls$NTO<-btovsrpls$NTO*100
# btovsrpls$p<-btovsrpls$p*10000
# btovsrpls$tA.B<-btovsrpls$tA.B*10000
# 
# preovsrs<-predict(preo, btovsrpls[, c(9:15)], type = "SK")
# prenuvsrs<-predict(prenu, btovsrpls[, c(9:15)], type = "SK")
# prenovsrs<-predict(preno, btovsrpls[, c(9:15)], type = "SK")
# prenosavsrs<-predict(prenosa, btovsrpls[, c(9:15)], type = "SK")
# prevsrsp<-data.frame(btovsrpls, om=preovsrs[["mean"]], os=preovsrs[["sd"]],
#                      num=prenuvsrs[["mean"]], nus=prenuvsrs[["sd"]],
#                      nom=prenovsrs[["mean"]], nos=prenovsrs[["sd"]],
#                      nosam=prenosavsrs[["mean"]], nosas=prenosavsrs[["sd"]])
# write.table(prevsrsp,"exp compare.csv",append = T,sep = ",")



##** Fig 3b Predicting in sparse virtual space **##
btovs1<-read.csv("BTO-VS sparse.csv")[,2:8]
btovs1<-cbind(btovs1,read.csv("BTO-VS sparse.csv")[,c("NCT","p", "tA.B", "AV", "ENMB","D","NTO")])
#将虚拟空间描述符值对应为训练集中百分数含量计算出值
btovs1$NCT<-btovs1$NCT*100
btovs1$NTO<-btovs1$NTO*100
btovs1$p<-btovs1$p*10000
btovs1$tA.B<-btovs1$tA.B*10000
gc()

library(Rtsne)
vsstsned<-as.matrix(btovs1[,c(8:14)])
#normalize
for(i in 1:7){
  vsstsned[,i]<-(vsstsned[,i]-min(vsstsned[,i])) / (max(vsstsned[,i])-min(vsstsned[,i]))
}
vsstsned<-normalize_input(vsstsned)
set.seed(9)
vsstsne<-Rtsne(vsstsned)

vsspo<-predict(preo, btovs1[, c(8:14)], type = "SK")[["mean"]]
vsspnosa<-predict(prenosa, btovs1[, c(8:14)], type = "SK")[["mean"]]

jpeg("Fig3b 1.jpg", width = 3212, height = 2800, res = 600)
ggplot(data.frame(x=vsstsne$Y[,1],y=vsstsne$Y[,2],z=vsspo), 
       aes(x = x, y = y, color = z)) +    
  geom_point(size = 4) +
  scale_color_gradientn(colors = c("#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"),
                        limits = c(110, 810))+
  scale_x_continuous(name = "t-SNE 1",limits = c(-80,63)) +    
  scale_y_continuous(name = "t-SNE 2") + 
  theme_bw(base_size = 35)+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),  # 移除默认边框
        axis.line = element_line(color = "black", linewidth = 0.8),  # 保留坐标轴线
        axis.text = element_text(color="black", size=27, family="sans"),
        axis.title = element_text(family="sans"),
        axis.title.y = element_text(margin = ggplot2::margin(r = -4)),  # Y轴标题向右移动
        axis.title.x = element_text(margin = ggplot2::margin(t = 0)),
        axis.ticks.length=unit(0.35, "cm"),
        axis.ticks = element_line(linewidth = 0.75),
        legend.title = element_blank(), 
        legend.text = element_text(family="sans",size=27),
        legend.position = c(0.17,0.17),legend.background = element_blank(),
        plot.margin = unit(c(0.3, 0, 0, 0), "cm"))
dev.off()

jpeg("Fig3b 2.jpg", width = 3212, height = 2800, res = 600)
ggplot(data.frame(x=vsstsne$Y[,1],y=vsstsne$Y[,2],z=vsspnosa), 
       aes(x = x, y = y, color = z)) +    
  geom_point(size = 4) +
  scale_color_gradientn(colors = c("#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"),
                        limits = c(110, 810))+
  scale_x_continuous(name = "t-SNE 1",limits = c(-80,63)) +    
  scale_y_continuous(name = "t-SNE 2") + 
  theme_bw(base_size = 35)+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),  # 移除默认边框
        axis.line = element_line(color = "black", linewidth = 0.8),  # 保留坐标轴线
        axis.text = element_text(color="black", size=27, family="sans"),
        axis.title = element_text(family="sans"),
        axis.title.y = element_text(margin = ggplot2::margin(r = -4)),  # Y轴标题向右移动
        axis.title.x = element_text(margin = ggplot2::margin(t = 0)),
        axis.ticks.length=unit(0.35, "cm"),
        axis.ticks = element_line(linewidth = 0.75),
        legend.title = element_blank(), 
        legend.text = element_text(family="sans",size=27),
        legend.position = c(0.17,0.17),legend.background = element_blank(),
        plot.margin = unit(c(0.3, 0, 0, 0), "cm"))
dev.off()
######################


refval<-c()
for(i in 1:25){
  estrsp<-estrs[which(estrs$refID==i),9]
  refval<-c(refval,rep(mean(estrsp),length(estrsp)))
}
tv1rfvar5sd<-c()
for(i in 1:25){
  tv1rfvar5p<-estrs[which(estrs$refID==i),]$tv1rfvar5
  tv1rfvar5sd<-c(tv1rfvar5sd,rep(sd(tv1rfvar5p),length(tv1rfvar5p)))
}
ttt<-unique(cbind(estrs[,c(17,31,28)],refval,tv1rfvar5sd))
library(openxlsx)
write.xlsx(ttt,"Fig S1 data.xlsx")