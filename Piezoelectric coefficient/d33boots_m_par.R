#多源d33(Berlincourt)数据

setwd("~/ECEdemo/directboots/d33d")
load("~/ECEdemo/directboots/d33d/d33m.RData")
# setwd("F:/机器学习/Multi-fidelity/ECE demo/directboots/d33d")

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
# #* input data *#
# d33r<-read_excel("d33 data for BaTiO3.xlsx")
# 
# #* rearrange data *#
# d33r<-d33r[,c(1,3:11)]
# colnames(d33r)<-c("refID","ba","ca","sr","cd","ti","zr","sn","hf","y")
# d33r$y<-round(d33r$y,1)
# 
# #* unique *#
# d33r<-anti_join(d33r,d33r[duplicated(d33r[,2:9]),])
# 
# #* calculate features *#
# #run ceramic-fea.R
# d33rf<-fn.data.features(d33r)
# 
# #* data for model *#
# d33rs<-cbind(d33rf[,c(1:10)],d33rf[,c("NCT", "tA.B", "z", "enp")])
# d33s<-d33rs[,c(2:14,1)]
# 
# # rm.NA #
# for(i in 1:dim(d33s)[2]){
#   d33s<-subset(d33s,!is.na(d33s[,i]))
# }
# 
# # unique #
# d33s<-anti_join(d33s,d33s[duplicated(d33s[,-c(9,14)]),])
# 
# ##*************##
# 
# 
# 
# ##** bootstrap **##
# testsamp<-matrix(nrow=500,ncol=227)
# for(i in 1:500){
#   set.seed(i)
#   testsamp[i,]<-sample(dim(d33s)[1], dim(d33s)[1], replace = TRUE)
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
#     gbdt<-try(gbm(y~., data = d33s[,9:13], n.trees = paras[1],
#                   interaction.depth = paras[2], shrinkage = paras[3], cv.folds = 10))
#     if ('try-error' %in% class(gbdt)) {
#       gbr2<-c(gbr2,NA)
#     }else{
#       gbr2<-c(gbr2,1-sum((d33s[,9]-as.numeric(gbdt$fit))^2)/sum((d33s[,9]-mean(unlist(d33s[,9])))^2))
#     }
#     d33r<-d33s[sample(dim(d33s)[1],dim(d33s)[1]),9:13] #random order
#     gbcve<-c()
#     for (j in 0:9){
#       dim10<-dim(d33r)[1]%/%10
#       d33cv1<-d33r[(1+dim10*j):(dim10*(j+1)),]
#       d33cv2<-d33r[-((1+dim10*j):(dim10*(j+1))),]
#       gbcvm<-try(gbm(y~., data = d33cv2, n.trees = paras[1],
#                      interaction.depth = paras[2], shrinkage = paras[3], cv.folds = 10))
#       if ('try-error' %in% class(gbcvm)) {
#         gbcve<-c(gbcve,NA)
#       }else{
#         gbcve<-c(gbcve,sum((d33cv1[,1]-predict(gbcvm,d33cv1))^2)/dim10)
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
# for(nt in c(500,750,1000)){
#   for(id in c(4,6,8)){
#     for(sh in c(0.001,0.005,0.01,0.05))
#       gbin2<-rbind(gbin2,c(nt,id,sh))
#   }
# }
# gbin2<-gbin2[-1,]
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="gbdtune.txt")
# clusterExport(cl, list("gbpd","d33s"))
# clusterEvalQ(cl,{library(gbm)})
# 
# system.time(
#   gbdtune<-parApply(cl, gbin2, 1, gbpd)  #gbtune is a matrix
# )
# 
# stopCluster(cl)
# 
# save.image("d33m.RData")
# 
# gbdbp<-gbdtune[,which.min(gbdtune[4,])]
# 
# save.image("d33m.RData")
# 
# # Input: seed(bootstrap number)  Output: GB predictions
# gbbt<-function(B){
#   set.seed(B)
#   d33bt<-d33s[sample(dim(d33s)[1], dim(d33s)[1], replace = TRUE),9:13]
#   d33bt<-d33bt[!duplicated(d33bt),]
#   d33bttest<-anti_join(d33s,d33bt)
#   btgbm<-gbm(y~., data = d33bt, n.trees = 500,
#              interaction.depth = 4, shrinkage = 0.01, cv.folds = 10,
#              n.minobsinnode=2)
#   prebttest<-predict(btgbm,d33bttest)
#   pretestna<-c()
#   for(i in 1:dim(d33s)[1]){
#     ind<-which(d33bttest$NCT==d33s$NCT[i] & d33bttest$tA.B==d33s$tA.B[i] &
#                  d33bttest$z==d33s$z[i] & d33bttest$enp==d33s$enp[i])
#     if(length(ind)==0){
#       pretestna<-c(pretestna,NA)
#     }else{
#       pretestna<-c(pretestna,prebttest[ind])
#     }
#   }
#   gc()
#   write.table(data.frame(pretestna,predict(btgbm,d33s)),"gbbt.csv",append=T,sep = ",")
#   return(data.frame(pretestna,predict(btgbm,d33s)))
# }
# 
# # B=500 #
# cl<-makeCluster(detectCores(),outfile="gbbt5.txt")
# clusterExport(cl, list("gbbt","d33s"))
# clusterEvalQ(cl,{library(gbm);library(dplyr)})
# 
# system.time(
#   gbbtpre5<-parLapply(cl, 1:500, gbbt)
# )
# stopCluster(cl)
# 
# save.image("d33m.RData")
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
# d33s<-cbind(d33s,btvar5t)
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
#     svrm<-try(svm(y~.,data=d33s[,9:13],type="eps-regression",kernel="radial",
#                   cost=cg[1],gamma=cg[2]))
#     if ('try-error' %in% class(svrm)) {
#       svrr2<-c(svrr2,NA)
#     }else{
#       svrr2<-c(svrr2,1-sum((predict(svrm,d33s)-d33s[,9])^2)/sum((d33s[,9]-mean(unlist(d33s[,9])))^2))
#     }
#     d33r<-d33s[sample(dim(d33s)[1],dim(d33s)[1]),9:13] #random order
#     svrcve<-c()
#     for (j in 0:9){
#       dim10<-dim(d33r)[1]%/%10
#       d33cv1<-d33r[(1+dim10*j):(dim10*(j+1)),]
#       d33cv2<-d33r[-((1+dim10*j):(dim10*(j+1))),]
#       svrcvm<-try(svm(y~.,data=d33cv2,type="eps-regression",kernel="radial",
#                       cost=cg[1],gamma=cg[2]))
#       if ('try-error' %in% class(svrcvm)) {
#         svrcve<-c(svrcve,NA)
#       }else{
#         svrcve<-c(svrcve,sum((d33cv1[,1]-predict(svrcvm,d33cv1))^2)/dim10)
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
# svrin2<-matrix(ncol=2)
# for(c in c(1,10,15,20)){
#   for(g in c(1e-4,0.001,0.01,0.1,0.5)){
#     svrin2<-rbind(svrin2,c(c,g))
#   }
# }
# svrin2<-svrin2[-1,]
# 
# svrin3<-matrix(ncol=2)
# for(c in c(5,6,7,8,9,10,11,12,13,14)){
#   for(g in c(0.05,0.08,0.1,0.12,0.15,0.2,0.3)){
#     svrin3<-rbind(svrin3,c(c,g))
#   }
# }
# svrin3<-svrin3[-1,]
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="svrdtune.txt")
# clusterExport(cl, list("svrpd","d33s"))
# clusterEvalQ(cl,{library(e1071)})
# 
# system.time(
#   svrdtune<-parApply(cl, svrin3, 1, svrpd)
# )
# 
# stopCluster(cl)
# save.image("d33m.RData")
# svrdbp<-svrdtune[,which.min(svrdtune[3,])]
# save.image("d33m.RData")
# 
# # Input: seed(bootstrap number)  Output: SVR.r predictions
# svrbt<-function(B){
#   set.seed(B)
#   d33bt<-d33s[sample(dim(d33s)[1], dim(d33s)[1], replace = TRUE),9:13]
#   d33bt<-d33bt[!duplicated(d33bt),]
#   d33bttest<-anti_join(d33s[,9:13],d33bt)
#   btsvrm<-svm(y~.,data=d33bt,type="eps-regression",kernel="radial",
#               cost=5,gamma=0.15)
#   prebttest<-predict(btsvrm,d33bttest)
#   pretestna<-c()
#   for(i in 1:dim(d33s)[1]){
#     ind<-which(d33bttest$NCT==d33s$NCT[i] & d33bttest$tA.B==d33s$tA.B[i] &
#                  d33bttest$z==d33s$z[i] & d33bttest$enp==d33s$enp[i])
#     if(length(ind)==0){
#       pretestna<-c(pretestna,NA)
#     }else{
#       pretestna<-c(pretestna,prebttest[ind])
#     }
#   }
#   gc()
#   write.table(data.frame(pretestna),"svrbt.csv",append=T,sep = ",")
#   return(data.frame(pretestna))
# }
# 
# # B=500 #
# cl<-makeCluster(detectCores(),outfile="svrbt5.txt")
# clusterExport(cl, list("svrbt","d33s"))
# clusterEvalQ(cl,{library(e1071);library(dplyr)})
# 
# system.time(
#   svrbtpre5<-parLapply(cl, 1:500, svrbt)
# )
# stopCluster(cl)
# 
# save.image("d33m.RData")
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
# d33s<-cbind(d33s,svrvar5t)
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
#     rfm<-try(randomForest(y~.,data=d33s[,9:13],ntree=nm[1],mtry=nm[2]))
#     if ('try-error' %in% class(rfm)) {
#       rfr2<-c(rfr2,NA)
#     }else{
#       rfr2<-c(rfr2,1-sum((rfm$y-rfm$predicted)^2)/sum((d33s[,9]-mean(unlist(d33s[,9])))^2))
#     }
#     d33r<-d33s[sample(dim(d33s)[1],dim(d33s)[1]),9:13] #random order
#     rfcve<-c()
#     for (j in 0:9){
#       dim10<-dim(d33r)[1]%/%10
#       d33cv1<-d33r[(1+dim10*j):(dim10*(j+1)),]
#       d33cv2<-d33r[-((1+dim10*j):(dim10*(j+1))),]
#       rfcvm<-try(randomForest(y~.,data=d33cv2,ntree=nm[1],mtry=nm[2]))
#       if ('try-error' %in% class(rfcvm)) {
#         rfcve<-c(rfcve,NA)
#       }else{
#         rfcve<-c(rfcve,sum((d33cv1[,1]-predict(rfcvm,d33cv1))^2)/dim10)
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
# for(nt in c(1100,1200,1300,1400,1500)){
#   for(mt in c(3,4,5)){
#     rfin2<-rbind(rfin2,c(nt,mt))
#   }
# }
# rfin2<-rfin2[-1,]
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="rfdtune.txt")
# clusterExport(cl, list("rfpd","d33s"))
# clusterEvalQ(cl,{library(randomForest)})
# 
# system.time(
#   rfdtune<-parApply(cl, rfin2, 1, rfpd)
# )
# 
# stopCluster(cl)
# 
# save.image("d33m.RData")
# 
# rfdbp<-rfdtune[,which.min(rfdtune[3,])]
# save.image("d33m.RData")
# 
# # Input: seed(bootstrap number)  Output: rf.r predictions
# rfbt<-function(B){
#   set.seed(B)
#   d33bt<-d33s[sample(dim(d33s)[1], dim(d33s)[1], replace = TRUE),9:13]
#   d33bt<-d33bt[!duplicated(d33bt),]
#   d33bttest<-anti_join(d33s[,9:13],d33bt)
#   btrfm<-randomForest(y~.,data=d33bt,ntree=1300,mtry=4)
#   prebttest<-predict(btrfm,d33bttest)
#   pretestna<-c()
#   for(i in 1:dim(d33s)[1]){
#     ind<-which(d33bttest$NCT==d33s$NCT[i] & d33bttest$tA.B==d33s$tA.B[i] &
#                  d33bttest$z==d33s$z[i] & d33bttest$enp==d33s$enp[i])
#     if(length(ind)==0){
#       pretestna<-c(pretestna,NA)
#     }else{
#       pretestna<-c(pretestna,prebttest[ind])
#     }
#   }
#   gc()
#   write.table(data.frame(pretestna,predict(btrfm,d33s)),"rfbt.csv",append=T,sep = ",")
#   return(data.frame(pretestna,predict(btrfm,d33s)))
# }
# 
# # B=500 #
# cl<-makeCluster(detectCores(),outfile="rfbt5.txt")
# clusterExport(cl, list("rfbt","d33s"))
# clusterEvalQ(cl,{library(randomForest);library(dplyr)})
# 
# system.time(
#   rfbtpre5<-parLapply(cl, 1:500, rfbt)
# )
# stopCluster(cl)
# 
# save.image("d33m.RData")
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
# d33s<-cbind(d33s,rfvar5t)
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
#   d33tv<-d33s[sample(dim(d33s)[1], Bts[2], replace = F),9:13]
#   d33tvtest<-anti_join(d33s[,9:13],d33tv)
#   tvrfm<-randomForest(y~.,data=d33tv,ntree=1300,mtry=4)
#   pretvtest<-predict(tvrfm,d33tvtest)
#   pretestna<-c()
#   for(i in 1:dim(d33s)[1]){
#     ind<-which(d33tvtest$NCT==d33s$NCT[i] & d33tvtest$tA.B==d33s$tA.B[i] &
#                  d33tvtest$z==d33s$z[i] & d33tvtest$enp==d33s$enp[i])
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
# for(ts in c(3,7,17,57,97,137,177,217,222,225)){
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
# clusterExport(cl, list("rftv","d33s"))
# clusterEvalQ(cl,{library(randomForest);library(dplyr)})
# 
# system.time(
#   rftvpre5<-parApply(cl, Btsin, 1, rftv)
# )
# stopCluster(cl)
# 
# save.image("d33m.RData")
# 
# tv03rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv03rfvar5<-c(tv03rfvar5,var(rftvpre5[i,1:500],na.rm=T))
# }
# tv07rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv07rfvar5<-c(tv07rfvar5,var(rftvpre5[i,501:1000],na.rm=T))
# }
# tv1rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv1rfvar5<-c(tv1rfvar5,var(rftvpre5[i,1001:1500],na.rm=T))
# }
# tv5rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv5rfvar5<-c(tv5rfvar5,var(rftvpre5[i,1501:2000],na.rm=T))
# }
# tv9rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv9rfvar5<-c(tv9rfvar5,var(rftvpre5[i,2001:2500],na.rm=T))
# }
# tv13rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv13rfvar5<-c(tv13rfvar5,var(rftvpre5[i,2501:3000],na.rm=T))
# }
# tv17rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv17rfvar5<-c(tv17rfvar5,var(rftvpre5[i,3001:3500],na.rm=T))
# }
# tv21rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv21rfvar5<-c(tv21rfvar5,var(rftvpre5[i,3501:4000],na.rm=T))
# }
# tv222rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv222rfvar5<-c(tv222rfvar5,var(rftvpre5[i,4001:4500],na.rm=T))
# }
# tv225rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv225rfvar5<-c(tv225rfvar5,var(rftvpre5[i,4501:5000],na.rm=T))
# }
# 
# d33s<-cbind(d33s,tv03rfvar5,tv07rfvar5,tv1rfvar5,tv5rfvar5,tv9rfvar5,tv13rfvar5,tv17rfvar5,
#             tv21rfvar5,tv222rfvar5,tv225rfvar5)
# 
# 
# #kriging estimated nugget
# d33krig<-krigm(d33s[,9:13],t,p,0,64)
# knug<-d33krig@covariance@nugget
# 
# # plot distribution of var estimated by different models #
# pmvar<-data.frame(V1=c(#rep("GB bootstrap",227),rep("SVR bootstrap",227),
#                        rep("RF bootstrap",227),rep("224 RF test",227),
#                        rep("220 RF test",227),rep("210 RF test",227),
#                        rep("170 RF test",227),rep("130 RF test",227),
#                        rep("90 RF test",227),rep("50 RF test",227),
#                        rep("10 RF test",227),rep("5 RF test",227),
#                        "estimated nugget"),
#                   V2=c(#btvar5t,svrvar5t,
#                        rfvar5t,tv03rfvar5,tv07rfvar5,tv1rfvar5,tv5rfvar5,
#                        tv9rfvar5,tv13rfvar5,tv17rfvar5,tv21rfvar5,tv222rfvar5,knug))
# pmvar$V1<-ordered(pmvar$V1,levels = c(#"GB bootstrap","SVR bootstrap",
#                                       "RF bootstrap","224 RF test",
#                                       "220 RF test","210 RF test",
#                                       "170 RF test","130 RF test",
#                                       "90 RF test","50 RF test",
#                                       "10 RF test","5 RF test",
#                                       "estimated nugget"))
# ggplot(pmvar, aes(x=V1, y=V2)) +
#   stat_boxplot(geom="errorbar",width=0.3,size=1,position=position_dodge(0.6),color="grey")+
#   geom_boxplot(aes(fill=V1),size=1,alpha=0.5,position=position_dodge(0.8),width=0.8,
#                color="grey",outlier.colour = "grey")+
#   ylim(0,8500)+
#   labs(y="Estimated variances",x="")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.text.x = element_text(angle = 25,hjust = 1),
#         axis.title=element_text(face="bold"))
# 
# # plot var vs ref #
# ggplot(d33s, aes(x=as.ordered(refID), y=rfvar5t)) +
#   stat_boxplot(geom="errorbar",width=0.3,size=1,position=position_dodge(0.6),color="grey")+
#   geom_boxplot(aes(fill=as.ordered(refID)),size=1,alpha=0.5,position=position_dodge(0.8),width=0.8,
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
# for(i in 1:29){
#   d33sp<-d33s[which(d33s$refID==i),]
#   refds<-c(refds,rep(dim(d33sp)[1],dim(d33sp)[1]))
# }
# d33s<-cbind(d33s,refds)
# ggplot(d33s, aes(x=refds, y=rfvar5t)) +
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
# # 
# # 
# # #* consistent var for each source #*
# # btvar5tm<-c()
# # for(i in 1:15){
# #   btvar5tp<-d33s[which(d33s$refID==i),]$btvar5t
# #   btvar5tm<-c(btvar5tm,rep(mean(btvar5tp),length(btvar5tp)))
# # }
# # 
# # svrvar5tm<-c()
# # for(i in 1:15){
# #   svrvar5tp<-d33s[which(d33s$refID==i),]$svrvar5t
# #   svrvar5tm<-c(svrvar5tm,rep(mean(svrvar5tp),length(svrvar5tp)))
# # }
# # 
# # rfvar5tm<-c()
# # for(i in 1:15){
# #   rfvar5tp<-d33s[which(d33s$refID==i),]$rfvar5t
# #   rfvar5tm<-c(rfvar5tm,rep(mean(rfvar5tp),length(rfvar5tp)))
# # }
# # 
# # tv4rfvar5m<-c()
# # for(i in 1:15){
# #   tv4rfvar5p<-d33s[which(d33s$refID==i),]$tv4rfvar5
# #   tv4rfvar5m<-c(tv4rfvar5m,rep(mean(tv4rfvar5p),length(tv4rfvar5p)))
# # }
# # 
# # d33s<-cbind(d33s,btvar5tm,svrvar5tm,rfvar5tm,tv4rfvar5m)
# # 
tv13rfvar5m<-c()
for(i in 1:29){
  tv13rfvar5p<-d33s[which(d33s$refID==i),]$tv13rfvar5
  tv13rfvar5m<-c(tv13rfvar5m,rep(mean(tv13rfvar5p),length(tv13rfvar5p)))
}

d33s<-cbind(d33s,tv13rfvar5m)

tv03rfvar5m<-c()
for(i in 1:29){
  tv03rfvar5p<-d33s[which(d33s$refID==i),]$tv03rfvar5
  tv03rfvar5m<-c(tv03rfvar5m,rep(mean(tv03rfvar5p),length(tv03rfvar5p)))
}

d33s<-cbind(d33s,tv03rfvar5m)
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
#   d33ro<-d33s[sample(dim(d33s)[1],dim(d33s)[1]),9:13] #random order
#   for (j in 0:9){
#     dim10<-dim(d33ro)[1]%/%10
#     d33cv1<-d33ro[(1+dim10*j):(dim10*(j+1)),]
#     d33cv2<-d33ro[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigmo(d33cv2,t,p,64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcv<-c(krigcv,NA)
#     }else{
#       krigcv<-c(krigcv,sum(abs(d33cv1[,1]-predict(krigcvm,d33cv1[,-1],type="SK")$mean))/dim10)
#     }
#     #}
#   }
#   print(c(i,krigcv))
#   return(c(i,krigcv))
# }
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krigocvs.txt")
# clusterExport(cl, list("krigocve","krigmo","t","p","d33s"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krigocvs<-parLapply(cl, 1:100, krigocve)
# )
# stopCluster(cl)
# 
# save.image("d33m.RData")
# 
# krigocvsm<-c()
# for(i in 1:length(krigocvs)){
#   krigocvsm<-c(krigocvsm,mean(krigocvs[[i]][-1]))
# }
# 
# ##***********##
# 
# 
# ##** kriging model with nugget **##
# #* one model for test *#
# #随机选16个作测试集，diagonal plot
# set.seed(7)
# test.d33s<-d33s[sample(1:dim(d33s)[1],16),]
# tt.d33s<-anti_join(d33s, test.d33s)
# test.y<-test.d33s[,9]
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
# tkrig<-krigm(tt.d33s[,9:13],t,p,0,567)
# tprekrig<-predict(tkrig,test.d33s[,10:13],type="SK")$mean
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
# krigcve<-function(i){
#   krigcv<-c()
#   set.seed(11+20*i)
#   d33ro<-d33s[sample(dim(d33s)[1],dim(d33s)[1]),9:13] #random order
#   for (j in 0:9){
#     dim10<-dim(d33ro)[1]%/%10
#     d33cv1<-d33ro[(1+dim10*j):(dim10*(j+1)),]
#     d33cv2<-d33ro[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#       krigcvm<-try(krigm(d33cv2,t,p,0,64))   #sseed可替换为8*k
#       if ('try-error' %in% class(krigcvm)) {
#         krigcv<-c(krigcv,NA)
#       }else{
#         krigcv<-c(krigcv,sum(abs(d33cv1[,1]-predict(krigcvm,d33cv1[,-1],type="SK")$mean))/dim10)
#       }
#     #}
#   }
#   print(c(i,krigcv))
#   return(c(i,krigcv))
# }
# 
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krigcvs.txt")
# clusterExport(cl, list("krigcve","krigm","t","p","d33s"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krigcvs<-parLapply(cl, 1:100, krigcve)
# )
# stopCluster(cl)
# 
# save.image("d33m.RData")
# 
# krigcvsm<-c()
# for(i in 1:length(krigcvs)){
#   krigcvsm<-c(krigcvsm,mean(krigcvs[[i]][-1]))
# }
# 
# ##***********##
# 
# 
# 
# ##** kriging model with noise **##
# #* one model for test *#
# tnkgb<-krigm(tt.d33s[,9:13],t,p,tt.d33s$btvar5a,567)
# tprenkgb<-predict(tnkgb,test.d33s[,10:13],type="SK")$mean
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
# tnksvr<-krigm(tt.d33s[,9:13],t,p,tt.d33s$svrvar5a,567)
# tprenksvr<-predict(tnksvr,test.d33s[,10:13],type="SK")$mean
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
# tnkrf<-krigm(tt.d33s[,9:13],t,p,tt.d33s$rfvar5a,567)
# tprenkrf<-predict(tnkrf,test.d33s[,10:13],type="SK")$mean
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
# tnkrfp<-krigm(tt.d33s[,9:13],t,p,tt.d33s$rfpvar5,567)
# tprenkrfp<-predict(tnkrfp,test.d33s[,10:13],type="SK")$mean
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
# tgb<-gbm(y~., data = tt.d33s[,9:13], n.trees = 100,
#          interaction.depth = 2, shrinkage = 0.1, cv.folds = 10)
# tpregb<-predict(tgb,test.d33s[,9:13])
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
# tsv<-svm(y~.,data=tt.d33s[,9:13],type="eps-regression",kernel="radial",
#          cost=1,gamma=1)
# tpresv<-predict(tsv,test.d33s[,9:13])
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
# trf<-randomForest(y~.,data=tt.d33s[,9:13],ntree=6000,mtry=1)
# tprerf<-predict(trf,test.d33s[,9:13])
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
#   d33ro<-d33s[sample(dim(d33s)[1],dim(d33s)[1]),] #random order
#   for (j in 0:9){
#     dim10<-dim(d33ro)[1]%/%10
#     d33cv1<-d33ro[(1+dim10*j):(dim10*(j+1)),9:13]
#     d33cv5<-d33ro[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigm(d33cv5[,9:13],t,p,d33cv5[,iv[2]],64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcv<-c(krigcv,NA)
#     }else{
#       krigcv<-c(krigcv,sum(abs(d33cv1[,1]-predict(krigcvm,d33cv1[,-1],type="SK")$mean))/dim10)
#     }
#     #}
#   }
#   print(c(iv,krigcv))
#   write.table(c(iv,krigcv),"krigncve.csv",append=T,sep = ",")
#   return(c(iv,krigcv))
# }
# 
# kncvin<-matrix(ncol = 2)
# for(v in c(#"btvar5t","svrvar5t",
#            "rfvar5t",
#            "tv03rfvar5","tv07rfvar5","tv1rfvar5","tv5rfvar5","tv9rfvar5",
#            "tv13rfvar5","tv17rfvar5","tv21rfvar5","tv222rfvar5")){
#   for(i in 1:100){
#     kncvin<-rbind(kncvin,c(i,v))
#   }
# }
# kncvin<-kncvin[-1,]
# 
kncvin3<-matrix(ncol = 2)
for(i in 1:100){
  kncvin3<-rbind(kncvin3,c(i,"tv13rfvar5m"))
}
kncvin3<-kncvin3[-1,]
# 
# cl<-makeCluster(detectCores(),outfile="krigncvs.txt")
# clusterExport(cl, list("krigncve","krigm","t","p","d33s"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krigncvs<-parApply(cl, kncvin, 1, krigncve)
# )
# stopCluster(cl)
# 
# save.image("d33m.RData")

cl<-makeCluster(detectCores(),outfile="krigncvs.txt")
clusterExport(cl, list("krigncve","krigm","t","p","d33s"))
clusterEvalQ(cl,{library(DiceKriging)})

system.time(
  krigncvs3<-parApply(cl, kncvin3, 1, krigncve)
)
stopCluster(cl)

save.image("d33m.RData")
# 
# krigncvsm<-as.data.frame(matrix(nrow = 100,ncol = 10))
# for(j in 1:10){
#   for(i in 1:100){
#     krigncvsm[i,j]<-mean(as.numeric(krigncvs[-c(1,2),(j-1)*100+i]),na.rm=T)
#   }
# }
# colnames(krigncvsm)<-c(#"btvar5t","svrvar5t",
#                        "rfvar5t",
#                        "tv03rfvar5","tv07rfvar5","tv1rfvar5","tv5rfvar5","tv9rfvar5",
#                        "tv13rfvar5","tv17rfvar5","tv21rfvar5","tv222rfvar5")
# 
# save.image("d33m.RData")

cvetv13m<-c()
for(i in 1:100){
  cvetv13m<-c(cvetv13m,mean(as.numeric(krigncvs3[-c(1,2),i]),na.rm=T))
}
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
# #5 10 30 50 70 75
# pmcv<-data.frame(V1=c(rep("kriging",100), rep("kriging with nugget",100),rep("noisy kriging (RF bootstrap)",100),
#                       rep("noisy kriging (224 RF test)",100),
#                       rep("noisy kriging (220 RF test)",100),rep("noisy kriging (210 RF test)",100),
#                       rep("noisy kriging (170 RF test)",100),rep("noisy kriging (130 RF test)",100),
#                       rep("noisy kriging (90 RF test)",100),rep("noisy kriging (50 RF test)",100),
#                       rep("noisy kriging (10 RF test)",100),rep("noisy kriging (5 RF test)",100)),
#                  V2=c(krigocvsm, krigcvsm, krigncvsm$rfvar5t,
#                       krigncvsm$tv03rfvar5,
#                       krigncvsm$tv07rfvar5,krigncvsm$tv1rfvar5,
#                       krigncvsm$tv5rfvar5,krigncvsm$tv9rfvar5,
#                       krigncvsm$tv13rfvar5,krigncvsm$tv17rfvar5,
#                       krigncvsm$tv21rfvar5, krigncvsm$tv222rfvar5),
#                  V3=c(c(mean(krigocvsm),rep(NA,99)),c(mean(krigcvsm),rep(NA,99)),c(mean(krigncvsm$rfvar5t),rep(NA,99)),
#                       c(mean(krigncvsm$tv03rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv07rfvar5),rep(NA,99)),c(mean(krigncvsm$tv1rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv5rfvar5),rep(NA,99)),c(mean(krigncvsm$tv9rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv13rfvar5),rep(NA,99)),c(mean(krigncvsm$tv17rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv21rfvar5),rep(NA,99)),c(mean(krigncvsm$tv222rfvar5),rep(NA,99))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (224 RF test)",
#                                     "noisy kriging (220 RF test)", "noisy kriging (210 RF test)",
#                                     "noisy kriging (170 RF test)", "noisy kriging (130 RF test)",
#                                     "noisy kriging (90 RF test)", "noisy kriging (50 RF test)",
#                                     "noisy kriging (10 RF test)", "noisy kriging (5 RF test)"))
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
# pmcv<-data.frame(V1=c(rep("without",100), rep("nugget",100),rep("res. var (bootstrap)",100),
#                       rep("res. var (3 train)",100),
#                       rep("res. var (7 train)",100),rep("res. var (17 train)",100),
#                       rep("res. var (57 train)",100),rep("res. var (97 train)",100),
#                       rep("res. var (137 train)",100),rep("res. var (177 train)",100),
#                       rep("res. var (217 train)",100),rep("res. var (222 train)",100)),
#                  V2=c(krigocvsm, krigcvsm, krigncvsm$rfvar5t,
#                       krigncvsm$tv03rfvar5,
#                       krigncvsm$tv07rfvar5,krigncvsm$tv1rfvar5,
#                       krigncvsm$tv5rfvar5,krigncvsm$tv9rfvar5,
#                       krigncvsm$tv13rfvar5,krigncvsm$tv17rfvar5,
#                       krigncvsm$tv21rfvar5, krigncvsm$tv222rfvar5),
#                  V3=c(c(mean(krigocvsm),rep(NA,99)),c(mean(krigcvsm),rep(NA,99)),c(mean(krigncvsm$rfvar5t),rep(NA,99)),
#                       c(mean(krigncvsm$tv03rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv07rfvar5),rep(NA,99)),c(mean(krigncvsm$tv1rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv5rfvar5),rep(NA,99)),c(mean(krigncvsm$tv9rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv13rfvar5),rep(NA,99)),c(mean(krigncvsm$tv17rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv21rfvar5),rep(NA,99)),c(mean(krigncvsm$tv222rfvar5),rep(NA,99))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("without", "nugget", "res. var (bootstrap)",
#                                     "res. var (3 train)",
#                                     "res. var (7 train)", "res. var (17 train)",
#                                     "res. var (57 train)", "res. var (97 train)",
#                                     "res. var (137 train)", "res. var (177 train)",
#                                     "res. var (217 train)", "res. var (222 train)"))
#导出数据
pmcvout<-data.frame(krigocvsm,krigcvsm,krigncvsm$tv21rfvar5,krigncvsm$tv17rfvar5,
                    krigncvsm$tv13rfvar5,krigncvsm$tv9rfvar5,krigncvsm$tv5rfvar5,
                    krigncvsm$tv1rfvar5,cvetv13m)
colnames(pmcvout)<-c("without var.","nugget var.","217 train","177 train",
                     "137 train", "97 train","57 train",
                     "17 train","137 train source mean")
library(openxlsx)
write.xlsx(pmcvout,"CVEs d33 m.xlsx")
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
# # # plot distribution of CVEs for models with consistent var for each source #
# # pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (GB bootstrap mean)",10),
# #                       rep("noisy kriging (SVR bootstrap mean)",10),rep("noisy kriging (RF bootstrap mean)",10)),
# #                  V2=c(krigocvsm, krigcvsm, krigncvsm$btvar5tm, 
# #                       krigncvsm$svrvar5tm,krigncvsm$rfvar5tm),
# #                  V3=c(c(mean(krigocvsm),rep(NA,9)),c(mean(krigcvsm),rep(NA,9)),c(mean(krigncvsm$btvar5tm),rep(NA,9)),
# #                       c(mean(krigncvsm$svrvar5tm),rep(NA,9)),c(mean(krigncvsm$rfvar5tm),rep(NA,9))))
# # pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (GB bootstrap mean)", 
# #                                     "noisy kriging (SVR bootstrap mean)","noisy kriging (RF bootstrap mean)"))
# # ggplot(pmcv, aes(x=V1, y=V2)) +
# #   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
# #   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
# #   coord_flip()+
# #   labs(y="CVE (10-fold MAE)",x="")+
# #   theme_bw(base_size = 20)+
# #   theme(legend.position = "none",panel.grid.major = element_blank(),
# #         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
# #         axis.title=element_text(face="bold"))
# # 
# # pmcv<-data.frame(V1=c(rep("kriging",10), rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap mean)",10),
# #                       rep("noisy kriging (110 RF test mean)",10),rep("noisy kriging (100 RF test mean)",10),
# #                       rep("noisy kriging (80 RF test mean)",10),rep("noisy kriging (60 RF test mean)",10),
# #                       rep("noisy kriging (40 RF test mean)",10),
# #                       rep("noisy kriging (10 RF test mean)",10),rep("noisy kriging (5 RF test mean)",10),
# #                       rep("noisy kriging (2 RF test mean)",10)),
# #                  V2=c(krigocvsm, krigcvsm, krigncvsm$rfvar5tm, 
# #                       krigncvsm$tv1rfvar5m,krigncvsm$tv2rfvar5m,
# #                       krigncvsm$tv4rfvar5m, krigncvsm$tv6rfvar5m, 
# #                       krigncvsm$tv8rfvar5m, 
# #                       krigncvsm$tv11rfvar5m, krigncvsm$tv121rfvar5m,
# #                       krigncvsm$tv124rfvar5m),
# #                  V3=c(c(mean(krigocvsm),rep(NA,9)),c(mean(krigcvsm),rep(NA,9)),c(mean(krigncvsm$rfvar5tm),rep(NA,9)),
# #                       c(mean(krigncvsm$tv1rfvar5m),rep(NA,9)),c(mean(krigncvsm$tv2rfvar5m),rep(NA,9)),
# #                       c(mean(krigncvsm$tv4rfvar5m),rep(NA,9)),c(mean(krigncvsm$tv6rfvar5m),rep(NA,9)),
# #                       c(mean(krigncvsm$tv8rfvar5m),rep(NA,9)),
# #                       c(mean(krigncvsm$tv11rfvar5m),rep(NA,9)),c(mean(krigncvsm$tv121rfvar5m),rep(NA,9)),
# #                       c(mean(krigncvsm$tv124rfvar5m),rep(NA,9))))
# # pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap mean)",
# #                                     "noisy kriging (110 RF test mean)", "noisy kriging (100 RF test mean)", 
# #                                     "noisy kriging (80 RF test mean)", "noisy kriging (60 RF test mean)", 
# #                                     "noisy kriging (40 RF test mean)", 
# #                                     "noisy kriging (10 RF test mean)", "noisy kriging (5 RF test mean)",
# #                                     "noisy kriging (2 RF test mean)"))
# # ggplot(pmcv, aes(x=V1, y=V2)) +
# #   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
# #   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
# #   coord_flip()+
# #   labs(y="CVE (10-fold MAE)",x="")+
# #   theme_bw(base_size = 20)+
# #   theme(legend.position = "none",panel.grid.major = element_blank(),
# #         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
# #         axis.title=element_text(face="bold"))
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
#   fvs<-d33s[19,]  #max y in d33s
#   d33sl<-d33s[-19,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, d33sl[sample(dim(d33sl)[1], vn-1, replace = F),])
#   ftd<-anti_join(d33s,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigmo(ftd[,9:13],t,p,64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       set.seed(dseed+5)
#       newd<<-fvs[sample(dim(fvs)[1], 1),]
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(10:13)],type="SK")
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
#     oce<<-d33s[19,]$y-newd$y
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
# for(v in c(110,140,170,200,220)){
#   for(i in c(34,77,6,12,55,47,60,10,111,555)){
#     vim<-rbind(vim,c(v,i))
#   }
# }
# vim<-vim[-1,]
# 
# cl<-makeCluster(detectCores(),outfile="oegooc.txt")
# clusterExport(cl, list("itoco","krigmo","t","p","d33s"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocopar<-function(vi){
#   return(itoco(vi[1],200,64,vi[2],"ego"))
# }
# 
# system.time(
#   oegooc<-parApply(cl, vim, 1, itocopar)
# )
# stopCluster(cl)
# 
# save.image("d33m.RData")
# 
# itoego<-matrix(nrow = 10,ncol=5)
# for(j in 1:5){
#   for(i in 1:10){
#     itoego[i,j]<-length(oegooc[[(j-1)*10+i]])
#   }
# }
# colnames(itoego)<-c("110","140","170","200","220")
# 
# # 不同虚拟空间，用四种效能函数
# vuim<-matrix(ncol=3)
# for(vn in c(159,182,204)){
#   for(uf in c("pre","ucb","ego")){
#     for(i in 1:100){
#       vuim<-rbind(vuim,c(vn,uf,5*i+7))
#     }
#   }
# }
# vuim<-vuim[-1,]
# 
# vuim2<-matrix(ncol=3)
# for(vn in c(159,182,204)){
#   for(uf in c("pre","ucb","ego","sko")){
#     for(i in 1:100){
#       vuim2<-rbind(vuim2,c(vn,uf,5*i+7))
#     }
#   }
# }
# vuim2<-vuim2[-1,]
# 
# cl<-makeCluster(detectCores(),outfile="oufoc.txt")
# clusterExport(cl, list("itoco","krigmo","t","p","d33s"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocouf<-function(ui){
#   return(itoco(as.numeric(ui[1]),204,64,as.numeric(ui[3]),ui[2]))
# }
# 
# system.time(
#   oufoc<-parApply(cl, vuim, 1, itocouf)
# )
# stopCluster(cl)
# 
# save.image("d33m.RData")
# 
# itouf<-matrix(nrow = 100,ncol=9)
# for(j in 1:9){
#   for(i in 1:100){
#     itouf[i,j]<-length(oufoc[[(j-1)*100+i]])
#   }
# }
# colnames(itouf)<-c("pre7","ucb7","ego7","pre8","ucb8","ego8","pre9","ucb9","ego9")
# 
# #* OC of kriging with nugget *# 
# # Input: virtual data size, max iteration times, seed for kriging, seed for sampling data, type of utility function 
# # Output: opportunity cost of each iteration
# itoc<-function(vn, mt, sseed, dseed, uf){
#   fvs<-d33s[19,]  #max y in d33s
#   d33sl<-d33s[-19,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, d33sl[sample(dim(d33sl)[1], vn-1, replace = F),])
#   ftd<-anti_join(d33s,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,9:13],t,p,0,64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       set.seed(dseed+5)
#       newd<<-fvs[sample(dim(fvs)[1], 1),]
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(10:13)],type="SK")
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
#                  kpre[["sd"]]*(dnorm(ze)+ze*pnorm(ze))
#         newd<<-fvs[which(kpresko==max(kpresko,na.rm=T)),][1,]
#       }
#     }      
#     fvs<-anti_join(fvs,newd)
#     ftd<-rbind(ftd,newd)
#     oce<<-d33s[19,]$y-newd$y
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
# clusterExport(cl, list("itoc","krigm","t","p","d33s"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocpar<-function(vi){
#   return(itoc(vi[1],200,64,vi[2],"ego"))
# }
# 
# system.time(
#   egooc<-parApply(cl, vim, 1, itocpar)
# )
# stopCluster(cl)
# 
# save.image("d33m.RData")
# 
# itego<-matrix(nrow = 10,ncol=5)
# for(j in 1:5){
#   for(i in 1:10){
#     itego[i,j]<-length(egooc[[(j-1)*10+i]])
#   }
# }
# colnames(itego)<-c("110","140","170","200","220")
# 
# # 不同虚拟空间，用四种效能函数
# cl<-makeCluster(detectCores(),outfile="ufoc.txt")
# clusterExport(cl, list("itoc","krigm","t","p","d33s"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocuf<-function(ui){
#   return(itoc(as.numeric(ui[1]),204,64,as.numeric(ui[3]),ui[2]))
# }
# 
# system.time(
#   ufoc<-parApply(cl, vuim2, 1, itocuf)
# )
# stopCluster(cl)
# 
# save.image("d33m.RData")
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
#   fvs<-d33s[19,]  #max y in d33s
#   d33sl<-d33s[-19,]
#   set.seed(dseed)
#   fvs<-rbind(fvs, d33sl[sample(dim(d33sl)[1], vn-1, replace = F),])
#   ftd<-anti_join(d33s,fvs)
#   oc<-c()
#   for(i in 1:mt){
#     krigcvm<-try(krigm(ftd[,9:13],t,p,ftd[,vt],64))   #sseed可替换
#     if ('try-error' %in% class(krigcvm)) {
#       set.seed(dseed+5)
#       newd<<-fvs[sample(dim(fvs)[1], 1),]
#     }else{
#       kpre<-predict(krigcvm,fvs[,c(10:13)],type="SK")
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
#     oce<<-d33s[19,]$y-newd$y
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
# for(v in c(110,140,170,200,220)){
#   for(vc in c("btvar5t","svrvar5t","rfvar5t",
#               "tv03rfvar5","tv07rfvar5","tv1rfvar5","tv5rfvar5","tv9rfvar5","tv13rfvar5",
#               "tv17rfvar5","tv21rfvar5","tv222rfvar5")){
#     for(i in c(34,77,6,12,55,47,60,10,111,555)){
#       vvim<-rbind(vvim,c(v,vc,i))
#     }
#   }
# }
# vvim<-vvim[-1,]
# 
# cl<-makeCluster(detectCores(),outfile="negooc3.txt")
# clusterExport(cl, list("itocn","krigm","t","p","d33s"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnpar<-function(vvi){
#   return(itocn(as.numeric(vvi[1]),220,64,as.numeric(vvi[3]),"ego",vvi[2]))
# }
# 
# system.time(
#   negooc<-parApply(cl, vvim, 1, itocnpar)
# )
# stopCluster(cl)
# 
# save.image("d33m.RData")
# 
# # vn = 110 #
# itnego110<-matrix(nrow = 10,ncol=12)
# for(j in 1:12){
#   for(i in 1:10){
#     itnego110[i,j]<-length(negooc[[(j-1)*10+i]])-2
#   }
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap)",10),
#                       rep("noisy kriging (224 test RF)",10),rep("noisy kriging (220 test RF)",10),
#                       rep("noisy kriging (210 test RF)",10),rep("noisy kriging (170 test RF)",10),
#                       rep("noisy kriging (130 test RF)",10),rep("noisy kriging (90 test RF)",10),
#                       rep("noisy kriging (50 test RF)",10),rep("noisy kriging (10 test RF)",10),
#                       rep("noisy kriging (5 test RF)",10)),
#                  V2=c(itoego[,1],itego[,1],itnego110[,3],itnego110[,4],itnego110[,5],
#                       itnego110[,6],itnego110[,7],itnego110[,8],itnego110[,9],
#                       itnego110[,10],itnego110[,11],itnego110[,12]),
#                  V3=c(c(mean(itoego[,1]),rep(NA,9)),c(mean(itego[,1]),rep(NA,9)),c(mean(itnego110[,3]),rep(NA,9)),
#                       c(mean(itnego110[,4]),rep(NA,9)),c(mean(itnego110[,5]),rep(NA,9)),
#                       c(mean(itnego110[,6]),rep(NA,9)),c(mean(itnego110[,7]),rep(NA,9)),
#                       c(mean(itnego110[,8]),rep(NA,9)),c(mean(itnego110[,9]),rep(NA,9)),
#                       c(mean(itnego110[,10]),rep(NA,9)),c(mean(itnego110[,11]),rep(NA,9)),
#                       c(mean(itnego110[,12]),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (224 test RF)","noisy kriging (220 test RF)",
#                                     "noisy kriging (210 test RF)","noisy kriging (170 test RF)",
#                                     "noisy kriging (130 test RF)","noisy kriging (90 test RF)",
#                                     "noisy kriging (50 test RF)","noisy kriging (10 test RF)",
#                                     "noisy kriging (5 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 110")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# # vn = 140 #
# itnego140<-matrix(nrow = 10,ncol=12)
# for(j in 1:12){
#   for(i in 1:10){
#     itnego140[i,j]<-length(negooc[[(j-1)*10+i+120]])-2
#   }
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap)",10),
#                       rep("noisy kriging (224 test RF)",10),rep("noisy kriging (220 test RF)",10),
#                       rep("noisy kriging (210 test RF)",10),rep("noisy kriging (170 test RF)",10),
#                       rep("noisy kriging (130 test RF)",10),rep("noisy kriging (90 test RF)",10),
#                       rep("noisy kriging (50 test RF)",10),rep("noisy kriging (10 test RF)",10),
#                       rep("noisy kriging (5 test RF)",10)),
#                  V2=c(itoego[,2],itego[,2],itnego140[,3],itnego140[,4],itnego140[,5],
#                       itnego140[,6],itnego140[,7],itnego140[,8],itnego140[,9],
#                       itnego140[,10],itnego140[,11],itnego140[,12]),
#                  V3=c(c(mean(itoego[,2]),rep(NA,9)),c(mean(itego[,2]),rep(NA,9)),c(mean(itnego140[,3]),rep(NA,9)),
#                       c(mean(itnego140[,4]),rep(NA,9)),c(mean(itnego140[,5]),rep(NA,9)),
#                       c(mean(itnego140[,6]),rep(NA,9)),c(mean(itnego140[,7]),rep(NA,9)),
#                       c(mean(itnego140[,8]),rep(NA,9)),c(mean(itnego140[,9]),rep(NA,9)),
#                       c(mean(itnego140[,10]),rep(NA,9)),c(mean(itnego140[,11]),rep(NA,9)),
#                       c(mean(itnego140[,12]),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (224 test RF)","noisy kriging (220 test RF)",
#                                     "noisy kriging (210 test RF)","noisy kriging (170 test RF)",
#                                     "noisy kriging (130 test RF)","noisy kriging (90 test RF)",
#                                     "noisy kriging (50 test RF)","noisy kriging (10 test RF)",
#                                     "noisy kriging (5 test RF)"))
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
# # vn = 170 #
# itnego170<-matrix(nrow = 10,ncol=12)
# for(j in 1:12){
#   for(i in 1:10){
#     itnego170[i,j]<-length(negooc[[(j-1)*10+i+240]])-2
#   }
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap)",10),
#                       rep("noisy kriging (224 test RF)",10),rep("noisy kriging (220 test RF)",10),
#                       rep("noisy kriging (210 test RF)",10),rep("noisy kriging (170 test RF)",10),
#                       rep("noisy kriging (130 test RF)",10),rep("noisy kriging (90 test RF)",10),
#                       rep("noisy kriging (50 test RF)",10),rep("noisy kriging (10 test RF)",10),
#                       rep("noisy kriging (5 test RF)",10)),
#                  V2=c(itoego[,3],itego[,3],itnego170[,3],itnego170[,4],itnego170[,5],
#                       itnego170[,6],itnego170[,7],itnego170[,8],itnego170[,9],
#                       itnego170[,10],itnego170[,11],itnego170[,12]),
#                  V3=c(c(mean(itoego[,3]),rep(NA,9)),c(mean(itego[,3]),rep(NA,9)),c(mean(itnego170[,3]),rep(NA,9)),
#                       c(mean(itnego170[,4]),rep(NA,9)),c(mean(itnego170[,5]),rep(NA,9)),
#                       c(mean(itnego170[,6]),rep(NA,9)),c(mean(itnego170[,7]),rep(NA,9)),
#                       c(mean(itnego170[,8]),rep(NA,9)),c(mean(itnego170[,9]),rep(NA,9)),
#                       c(mean(itnego170[,10]),rep(NA,9)),c(mean(itnego170[,11]),rep(NA,9)),
#                       c(mean(itnego170[,12]),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (224 test RF)","noisy kriging (220 test RF)",
#                                     "noisy kriging (210 test RF)","noisy kriging (170 test RF)",
#                                     "noisy kriging (130 test RF)","noisy kriging (90 test RF)",
#                                     "noisy kriging (50 test RF)","noisy kriging (10 test RF)",
#                                     "noisy kriging (5 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 170")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# # vn = 200 #
# itnego200<-matrix(nrow = 10,ncol=12)
# for(j in 1:12){
#   for(i in 1:10){
#     itnego200[i,j]<-length(negooc[[(j-1)*10+i+360]])-2
#   }
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap)",10),
#                       rep("noisy kriging (224 test RF)",10),rep("noisy kriging (220 test RF)",10),
#                       rep("noisy kriging (210 test RF)",10),rep("noisy kriging (170 test RF)",10),
#                       rep("noisy kriging (130 test RF)",10),rep("noisy kriging (90 test RF)",10),
#                       rep("noisy kriging (50 test RF)",10),rep("noisy kriging (10 test RF)",10),
#                       rep("noisy kriging (5 test RF)",10)),
#                  V2=c(itoego[,4],itego[,4],itnego200[,3],itnego200[,4],itnego200[,5],
#                       itnego200[,6],itnego200[,7],itnego200[,8],itnego200[,9],
#                       itnego200[,10],itnego200[,11],itnego200[,12]),
#                  V3=c(c(mean(itoego[,4]),rep(NA,9)),c(mean(itego[,4]),rep(NA,9)),c(mean(itnego200[,3]),rep(NA,9)),
#                       c(mean(itnego200[,4]),rep(NA,9)),c(mean(itnego200[,5]),rep(NA,9)),
#                       c(mean(itnego200[,6]),rep(NA,9)),c(mean(itnego200[,7]),rep(NA,9)),
#                       c(mean(itnego200[,8]),rep(NA,9)),c(mean(itnego200[,9]),rep(NA,9)),
#                       c(mean(itnego200[,10]),rep(NA,9)),c(mean(itnego200[,11]),rep(NA,9)),
#                       c(mean(itnego200[,12]),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (224 test RF)","noisy kriging (220 test RF)",
#                                     "noisy kriging (210 test RF)","noisy kriging (170 test RF)",
#                                     "noisy kriging (130 test RF)","noisy kriging (90 test RF)",
#                                     "noisy kriging (50 test RF)","noisy kriging (10 test RF)",
#                                     "noisy kriging (5 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 200")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# # vn = 220 #
# itnego220<-matrix(nrow = 10,ncol=12)
# for(j in 1:12){
#   for(i in 1:10){
#     itnego220[i,j]<-length(negooc[[(j-1)*10+i+480]])-2
#   }
# }
# # plot distribution of itt tvrfvar5
# pmcv<-data.frame(V1=c(rep("kriging",10),rep("kriging with nugget",10),rep("noisy kriging (RF bootstrap)",10),
#                       rep("noisy kriging (224 test RF)",10),rep("noisy kriging (220 test RF)",10),
#                       rep("noisy kriging (210 test RF)",10),rep("noisy kriging (170 test RF)",10),
#                       rep("noisy kriging (130 test RF)",10),rep("noisy kriging (90 test RF)",10),
#                       rep("noisy kriging (50 test RF)",10),rep("noisy kriging (10 test RF)",10),
#                       rep("noisy kriging (5 test RF)",10)),
#                  V2=c(itoego[,5],itego[,5],itnego220[,3],itnego220[,4],itnego220[,5],
#                       itnego220[,6],itnego220[,7],itnego220[,8],itnego220[,9],
#                       itnego220[,10],itnego220[,11],itnego220[,12]),
#                  V3=c(c(mean(itoego[,5]),rep(NA,9)),c(mean(itego[,5]),rep(NA,9)),c(mean(itnego220[,3]),rep(NA,9)),
#                       c(mean(itnego220[,4]),rep(NA,9)),c(mean(itnego220[,5]),rep(NA,9)),
#                       c(mean(itnego220[,6]),rep(NA,9)),c(mean(itnego220[,7]),rep(NA,9)),
#                       c(mean(itnego220[,8]),rep(NA,9)),c(mean(itnego220[,9]),rep(NA,9)),
#                       c(mean(itnego220[,10]),rep(NA,9)),c(mean(itnego220[,11]),rep(NA,9)),
#                       c(mean(itnego220[,12]),rep(NA,9))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging", "kriging with nugget", "noisy kriging (RF bootstrap)",
#                                     "noisy kriging (224 test RF)","noisy kriging (220 test RF)",
#                                     "noisy kriging (210 test RF)","noisy kriging (170 test RF)",
#                                     "noisy kriging (130 test RF)","noisy kriging (90 test RF)",
#                                     "noisy kriging (50 test RF)","noisy kriging (10 test RF)",
#                                     "noisy kriging (5 test RF)"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("Virtual size = 220")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# 
# # plot Iteration times -- Virtual size for different bootstrap models #
# pittx<-c(rep(c(110,140,170,200,220),5))
# pittm<-c(mean(itoego[,1]),mean(itoego[,2]),mean(itoego[,3]),mean(itoego[,4]),mean(itoego[,5]),
#          mean(itego[,1]),mean(itego[,2]),mean(itego[,3]),mean(itego[,4]),mean(itego[,5]),
#          mean(itnego110[,1]),mean(itnego140[,1]),mean(itnego170[,1]),mean(itnego200[,1]),mean(itnego220[,1]),
#          mean(itnego110[,2]),mean(itnego140[,2]),mean(itnego170[,2]),mean(itnego200[,2]),mean(itnego220[,2]),
#          mean(itnego110[,3]),mean(itnego140[,3]),mean(itnego170[,3]),mean(itnego200[,3]),mean(itnego220[,3]))
# pittsd<-c(sd(itoego[,1]),sd(itoego[,2]),sd(itoego[,3]),sd(itoego[,4]),sd(itoego[,5]),
#          sd(itego[,1]),sd(itego[,2]),sd(itego[,3]),sd(itego[,4]),sd(itego[,5]),
#          sd(itnego110[,1]),sd(itnego140[,1]),sd(itnego170[,1]),sd(itnego200[,1]),sd(itnego220[,1]),
#          sd(itnego110[,2]),sd(itnego140[,2]),sd(itnego170[,2]),sd(itnego200[,2]),sd(itnego220[,2]),
#          sd(itnego110[,3]),sd(itnego140[,3]),sd(itnego170[,3]),sd(itnego200[,3]),sd(itnego220[,3]))
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
#   scale_x_continuous(breaks=c(110,140,170,200,220))+
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
#   scale_x_continuous(breaks=c(110,140,170,200,220))+
#   xlab("Number of virtual data")+ylab("AL efficiency")+
#   theme_bw(base_size = 30)+
#   theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank(),
#         axis.text=element_text(color="black",size=24),
#         axis.title=element_text(face="bold"))
# 
# 
# # 固定80%虚拟空间，用四种效能函数
# cl<-makeCluster(detectCores(),outfile="nufoc.txt")
# clusterExport(cl, list("itocn","krigm","t","p","d33s"))
# clusterEvalQ(cl,{library(DiceKriging);library(dplyr)})
# 
# itocnuf<-function(ui){
#   return(itocn(as.numeric(ui[1]),204,64,as.numeric(ui[3]),ui[2],"tv03rfvar5"))
# }
# 
# system.time(
#   nufoc<-parApply(cl, vuim, 1, itocnuf)
# )
# stopCluster(cl)
# 
# save.image("d33m.RData")
# 
# itnuf<-matrix(nrow = 100,ncol=9)
# for(j in 1:9){
#   for(i in 1:100){
#     itnuf[i,j]<-length(nufoc[[(j-1)*100+i]])-2
#   }
# }
# colnames(itnuf)<-c("pre7","ucb7","ego7","pre8","ucb8","ego8","pre9","ucb9","ego9")
# 
# save.image("d33m.RData")
# # 
# # plot distribution of itt uf
# #70% vs
# pmcv<-data.frame(V1=c(rep("kriging + Pre",100),rep("kriging + UCB",100),rep("kriging + EGO",100),
#                       rep("kriging with nugget + Pre",100),rep("kriging with nugget + UCB",100),
#                       rep("kriging with nugget + EGO",100),rep("kriging with nugget + SKO",100),
#                       rep("noisy kriging (224 test RF) + Pre",100),rep("noisy kriging (224 test RF) + UCB",100),
#                       rep("noisy kriging (224 test RF) + EGO",100)),
#                  V2=c(itouf[,1],itouf[,2],itouf[,3],ituf[,1],ituf[,2],ituf[,3],ituf[,4],
#                       itnuf[,1],itnuf[,2],itnuf[,3]),
#                  V3=c(c(mean(itouf[,1]),rep(NA,99)),c(mean(itouf[,2]),rep(NA,99)),c(mean(itouf[,3]),rep(NA,99)),
#                       c(mean(ituf[,1]),rep(NA,99)),c(mean(ituf[,2]),rep(NA,99)),c(mean(ituf[,3]),rep(NA,99)),c(mean(ituf[,4]),rep(NA,99)),
#                       c(mean(itnuf[,1]),rep(NA,99)),c(mean(itnuf[,2]),rep(NA,99)),c(mean(itnuf[,3]),rep(NA,99))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging + Pre", "kriging + UCB", "kriging + EGO",
#                                     "kriging with nugget + Pre","kriging with nugget + UCB",
#                                     "kriging with nugget + EGO","kriging with nugget + SKO",
#                                     "noisy kriging (224 test RF) + Pre","noisy kriging (224 test RF) + UCB",
#                                     "noisy kriging (224 test RF) + EGO"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("d33 multi-source 70% virtual space")+
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
#                       rep("noisy kriging (224 test RF) + Pre",100),rep("noisy kriging (224 test RF) + UCB",100),
#                       rep("noisy kriging (224 test RF) + EGO",100)),
#                  V2=c(itouf[,4],itouf[,5],itouf[,6],ituf[,5],ituf[,6],ituf[,7],ituf[,8],
#                       itnuf[,4],itnuf[,5],itnuf[,6]),
#                  V3=c(c(mean(itouf[,4]),rep(NA,99)),c(mean(itouf[,5]),rep(NA,99)),c(mean(itouf[,6]),rep(NA,99)),
#                       c(mean(ituf[,5]),rep(NA,99)),c(mean(ituf[,6]),rep(NA,99)),c(mean(ituf[,7]),rep(NA,99)),c(mean(ituf[,8]),rep(NA,99)),
#                       c(mean(itnuf[,4]),rep(NA,99)),c(mean(itnuf[,5]),rep(NA,99)),c(mean(itnuf[,6]),rep(NA,99))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging + Pre", "kriging + UCB", "kriging + EGO",
#                                     "kriging with nugget + Pre","kriging with nugget + UCB",
#                                     "kriging with nugget + EGO","kriging with nugget + SKO",
#                                     "noisy kriging (224 test RF) + Pre","noisy kriging (224 test RF) + UCB",
#                                     "noisy kriging (224 test RF) + EGO"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("d33 multi-source 80% virtual space")+
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
#                       rep("noisy kriging (224 test RF) + Pre",100),rep("noisy kriging (224 test RF) + UCB",100),
#                       rep("noisy kriging (224 test RF) + EGO",100)),
#                  V2=c(itouf[,7],itouf[,8],itouf[,9],ituf[,9],ituf[,10],ituf[,11],ituf[,12],
#                       itnuf[,7],itnuf[,8],itnuf[,9]),
#                  V3=c(c(mean(itouf[,7]),rep(NA,99)),c(mean(itouf[,8]),rep(NA,99)),c(mean(itouf[,9]),rep(NA,99)),
#                       c(mean(ituf[,9]),rep(NA,99)),c(mean(ituf[,10]),rep(NA,99)),c(mean(ituf[,11]),rep(NA,99)),c(mean(ituf[,12]),rep(NA,99)),
#                       c(mean(itnuf[,7]),rep(NA,99)),c(mean(itnuf[,8]),rep(NA,99)),c(mean(itnuf[,9]),rep(NA,99))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("kriging + Pre", "kriging + UCB", "kriging + EGO",
#                                     "kriging with nugget + Pre","kriging with nugget + UCB",
#                                     "kriging with nugget + EGO","kriging with nugget + SKO",
#                                     "noisy kriging (224 test RF) + Pre","noisy kriging (224 test RF) + UCB",
#                                     "noisy kriging (224 test RF) + EGO"))
# ggplot(pmcv, aes(x=V1, y=V2)) +
#   geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
#   geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
#   coord_flip()+
#   labs(y="Iteration times",x="")+
#   ggtitle("d33 multi-source 90% virtual space")+
#   theme_bw(base_size = 20)+
#   theme(legend.position = "none",panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
#         axis.title=element_text(face="bold"),
#         plot.title = element_text(size = 20, hjust=0, vjust = -0.5))
# ##************##
# 
# 
# 
# ##** Predicting in virtual space **##
# btovs1<-read.csv("BTO-VS.csv")[,2:8]
# btovs1<-cbind(btovs1,read.csv("BTO-VS.csv")[,c("NCT", "tA.B", "z", "enp")])
# gc()
# #* prediction models *#
# preo<-krigmo(d33s[,9:13],t,p,64)
# prenu<-krigm(d33s[,9:13],t,p,0,64)
# preno<-krigm(d33s[,9:13],t,p,d33s[,"tv03rfvar5"],64)
prenosa<-krigm(d33s[,9:13],t,p,d33s[,"tv03rfvar5m"],64)
# 
# # Input: the number(/10000) of btovs   Output: number, mean and sigma2 of predictions
# prevs<-function(n){
#   gc()
#   preovs<-predict(preo, btovs1[((n-1)*10000+1):(n*10000), c(8:11)], type = "SK")
#   prenuvs<-predict(prenu, btovs1[((n-1)*10000+1):(n*10000), c(8:11)], type = "SK")
#   prenovs<-predict(preno, btovs1[((n-1)*10000+1):(n*10000), c(8:11)], type = "SK")
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
#   prevsall<-parLapply(cl, c(1:192), prevs)
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
# preovs_la<-predict(preo, btovs1[1920001:1926974, c(8:11)], type = "SK")
# prenuvs_la<-predict(prenu, btovs1[1920001:1926974, c(8:11)], type = "SK")
# prenovs_la<-predict(preno, btovs1[1920001:1926974, c(8:11)], type = "SK")
# gc()
# prevsallc<-rbind(prevsallc, data.frame(ind=c(1920001:1926974), om=preovs_la[["mean"]], os=preovs_la[["sd"]],
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
# save.image("d33m.RData")
# 
# 
# 
# #predict some selected samples
# btovss<-read.csv("selected VS.csv")[,2:8]
# btovss<-cbind(btovss,read.csv("selected VS.csv")[,c("NCT", "tA.B", "z", "enp")])
# 
# preovss<-predict(preo, btovss[, c(8:11)], type = "SK")
# prenuvss<-predict(prenu, btovss[, c(8:11)], type = "SK")
# prenovss<-predict(preno, btovss[, c(8:11)], type = "SK")
# prevssp<-data.frame(ind=c(1:12), om=preovss[["mean"]], os=preovss[["sd"]],
#                     num=prenuvss[["mean"]], nus=prenuvss[["sd"]],
#                     nom=prenovss[["mean"]], nos=prenovss[["sd"]])
# prevssp$div<-abs(prevssp$om-prevssp$num)+abs(prevssp$num-prevssp$nom)+
#   abs(prevssp$nom-prevssp$om)
# 
# 
# #predict some random selected samples
btovsrs<-read.csv("random selected VS +.csv")[,c(1:7,9)]
btovsrs<-cbind(btovsrs,read.csv("random selected VS +.csv")[,c("NCT", "tA.B", "z", "enp")])
# 
# # library(Rtsne)
# intsnet<-as.matrix(rbind(d33s[,c(10:13)],btovsrs[,c(8:11)]))
# #normalize
# for(i in 1:4){
#   intsnet[,i]<-(intsnet[,i]-min(intsnet[,i])) / (max(intsnet[,i])-min(intsnet[,i]))
# }
# intsnet<-normalize_input(intsnet)
# set.seed(3)
# intsne<-Rtsne(intsnet)
# plot(x=intsne$Y[,1],y=intsne$Y[,2],
#      pch= c(rep(1,length(d33s[,1])),rep(17,length(btovsrs[,1]))),
#      col= c(rep("red",length(d33s[,1])),rep("black",length(btovsrs[,1]))), cex = 1.2,
#      #xlim = c(-20,28), ylim = c(-25,30),
#      xlab="tSNE_1",ylab="tSNE_2",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# 
preovsrs<-predict(preo, btovsrs[, c(9:12)], type = "SK")
prenuvsrs<-predict(prenu, btovsrs[, c(9:12)], type = "SK")
prenovsrs<-predict(preno, btovsrs[, c(9:12)], type = "SK")
prenosavsrs<-predict(prenosa, btovsrs[, c(9:12)], type = "SK")
prevsrsp<-data.frame(btovsrs, om=preovsrs[["mean"]], os=preovsrs[["sd"]],
                     num=prenuvsrs[["mean"]], nus=prenuvsrs[["sd"]],
                     nom=prenovsrs[["mean"]], nos=prenovsrs[["sd"]],
                     nosam=prenosavsrs[["mean"]], nosas=prenosavsrs[["sd"]])
write.table(prevsrsp,"exp compare.csv",append = T,sep = ",")
# prevsrsp$div<-abs(prevsrsp$om-prevsrsp$num)+abs(prevsrsp$num-prevsrsp$nom)+
#   abs(prevsrsp$nom-prevsrsp$om)
# 
library(readxl)
expcomp <- read_excel("experiments compare.xlsx", sheet = "d33")
expcomp<-as.matrix(expcomp)
# plot(rep(expcomp[,4],3),y=c(expcomp[,1],expcomp[,2],expcomp[,3]),
#      pch=c(rep(15,4),rep(16,4),rep(17,4)),
#      col= c(rep("grey",4),rep("red",4),rep("blue",4)), cex = 1.4,
#      xlim = c(10,290), ylim = c(10,290),
#      xlab="Measured d33 (pC/N)", ylab="Predicted d33 (pC/N)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2)
# 
# ggplot(data.frame(x=rep(expcomp[,4],3),y=c(expcomp[,1],expcomp[,2],expcomp[,3]),
#                   z=factor(c(rep("without", 4), rep("nugget", 4), rep("rep. var", 4)), 
#                            levels = c("without", "nugget","rep. var"))), 
#        aes(x = x, y = y, color = z, shape = z)) +    
#   geom_point(size = 4) +
#   scale_color_manual(values = c("grey","red", "blue"),  # 手动设置颜色值，与因子的水平顺序一致    
#                      labels = c("without", "nugget","rep. var")) +  
#   scale_shape_manual(values = c(15, 16, 17),     
#                      labels = c("without", "nugget","rep. var")) + 
#   scale_x_continuous(name = "Measured d33 (pC/N)") +    
#   scale_y_continuous(name = "Predicted d33 (pC/N)") +
#   geom_abline(intercept = 0, slope = 1, color = "black", size = 1.2) +  # 添加对角线
#   theme_bw(base_size = 27) +  
#   theme(panel.grid.major = element_blank(),  
#         panel.grid.minor = element_blank(),  
#         axis.text = element_text(face = "bold", color = "black", size = 23, family = "serif"),  
#         axis.title = element_text(face = "bold", family = "serif"),  
#         axis.ticks.length = unit(-0.25, "cm"), 
#         legend.title = element_blank(), 
#         legend.text = element_text(face="bold",family="serif",size=20),
#         legend.position = c(0.18,0.89),legend.background = element_blank(),
#         plot.margin = unit(c(0, 0.1, 0, 0), "cm"))

library(ggh4x)
jpeg("expc d33 v2.jpg", width = 3300, height = 3000, res = 600)
ggplot(data.frame(x=rep(expcomp[,4],2),y=c(expcomp[,1],c(21,91,expcomp[3:4,3])),
                  z=factor(c(rep("without", 4), rep("source-aware variance", 4)),
                           levels = c("without", "source-aware variance"))),
       aes(x = x, y = y, color = z, shape = z)) +
  geom_point(size = 4) +
  scale_color_manual(values = c("#3480B8","orange"),  # 手动设置颜色值，与因子的水平顺序一致
                     labels = c("without", "source-aware variance")) +
  scale_shape_manual(values = c(15, 17),
                     labels = c("without", "source-aware variance")) +
  scale_x_continuous(name = "Measured d33 (pC/N)") +
  geom_abline(intercept = 0, slope = 1, color = "black", size = 1, linetype="dotted") +  # 添加对角线
  theme_bw(base_size = 22.5)+
  theme(legend.title = element_blank(),
        legend.position = c(0.36,0.9),legend.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),  # 移除默认边框
        axis.line = element_line(color = "black", linewidth = 0.8),  # 保留坐标轴线
        axis.text = element_text(color="black", size=17.5, family="sans"),
        axis.title = element_text(family="sans"),
        axis.ticks.length=unit(0.22, "cm"),
        axis.ticks = element_line(linewidth = 0.75),
        plot.margin = unit(c(0.3, 0.1, -0.1, 0), "cm"))+
  # 添加次刻度线（需要ggh4x包）
  scale_y_continuous(name = "Predicted d33 (pC/N)",guide = ggh4x::guide_axis_minor(),
                     minor_breaks = waiver()) +
  theme(ggh4x.axis.ticks.length.minor=rel(1/2))
dev.off()