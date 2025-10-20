#多源d33(Berlincourt)数据
#与第一版区别：发现输入数据中的异常值并纠正

setwd("~/ECEdemo/directboots/d33d")
load("~/ECEdemo/directboots/d33d/d33m2.RData")
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
# t<-1e-10; p<-1e-10

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
# #* 检查成分加和是否不等于1 *#
# a_not_equal_1 <- which(rowSums(d33r[, 2:5]) != 1)
# b_not_equal_1 <- which(rowSums(d33r[, 6:9]) != 1)
# d33r[b_not_equal_1, ]
# d33r[135,6]<-0.925
# d33r<-d33r[-c(250:254),]
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
# for(nt in c(500,800,1000,1200)){
#   for(mt in c(3,4,5,6,8)){
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
# save.image("d33m2.RData")
# 
# rfdbp<-rfdtune[,which.min(rfdtune[3,])]
# save.image("d33m2.RData")
# 
# # Input: seed(bootstrap number)  Output: rf.r predictions
# rfbt<-function(B){
#   set.seed(B)
#   d33bt<-d33s[sample(dim(d33s)[1], dim(d33s)[1], replace = TRUE),9:13]
#   d33bt<-d33bt[!duplicated(d33bt),]
#   d33bttest<-anti_join(d33s[,9:13],d33bt)
#   btrfm<-randomForest(y~.,data=d33bt,ntree=1000,mtry=4)
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
# save.image("d33m2.RData")
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
# 
# 
# #* test method with RF *#
# # Input: seed(bootstrap number) and sample train size  Output: rf.r predictions
# rftv<-function(Bts){
#   set.seed(Bts[1])
#   d33tv<-d33s[sample(dim(d33s)[1], Bts[2], replace = F),9:13]
#   d33tvtest<-anti_join(d33s[,9:13],d33tv)
#   tvrfm<-randomForest(y~.,data=d33tv,ntree=1000,mtry=4)
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
# for(ts in c(7,12,52,92,132,172,212,217)){
#   for(B in 1:500){
#     Btsin<-rbind(Btsin,c(B,ts))
#   }
# }
# Btsin<-Btsin[-1,]
# 
# Btsin2<-matrix(ncol=2)
# for(ts in c(123,124)){
#   for(B in 1:500){
#     Btsin2<-rbind(Btsin2,c(B,ts))
#   }
# }
# Btsin2<-Btsin2[-1,]
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
# save.image("d33m2.RData")
# 
# tv07rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv07rfvar5<-c(tv07rfvar5,var(rftvpre5[i,1:500],na.rm=T))
# }
# tv1rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv1rfvar5<-c(tv1rfvar5,var(rftvpre5[i,501:1000],na.rm=T))
# }
# tv5rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv5rfvar5<-c(tv5rfvar5,var(rftvpre5[i,1001:1500],na.rm=T))
# }
# tv9rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv9rfvar5<-c(tv9rfvar5,var(rftvpre5[i,1501:2000],na.rm=T))
# }
# tv13rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv13rfvar5<-c(tv13rfvar5,var(rftvpre5[i,2001:2500],na.rm=T))
# }
# tv17rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv17rfvar5<-c(tv17rfvar5,var(rftvpre5[i,2501:3000],na.rm=T))
# }
# tv21rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv21rfvar5<-c(tv21rfvar5,var(rftvpre5[i,3001:3500],na.rm=T))
# }
# tv217rfvar5<-c()
# for(i in 1:dim(rftvpre5)[1]){
#   tv217rfvar5<-c(tv217rfvar5,var(rftvpre5[i,3501:4000],na.rm=T))
# }
# d33s<-cbind(d33s,tv07rfvar5,tv1rfvar5,tv5rfvar5,tv9rfvar5,tv13rfvar5,tv17rfvar5,
#             tv21rfvar5,tv217rfvar5)
# 
# 
# #kriging estimated nugget
# d33krig<-krigm(d33s[,9:13],t,p,0,64)
# knug<-d33krig@covariance@nugget
# save.image("d33m2.RData")
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
#                        rfvar5t,tv05rfvar5,tv10rfvar5,tv1rfvar5,tv5rfvar5,
#                        tv9rfvar5,tv13rfvar5,tv17rfvar5,tv21rfvar5,tv22rfvar5,knug))
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
#导出数据
pmvarout<-data.frame(tv07rfvar5,tv1rfvar5,tv5rfvar5,
                     tv9rfvar5,tv13rfvar5,tv17rfvar5,tv21rfvar5,tv217rfvar5,knug)
colnames(pmvarout)<-c("7 train","12 train","52 train",
                      "92 train","132 train","172 train","212 train","217 train","nugget")
library(openxlsx)
write.xlsx(pmvarout,"Vars d33 m2.xlsx")


#* consistent var for each source #*
tv17rfvar5m<-c()
for(i in 1:29){
  tv17rfvar5p<-d33s[which(d33s$refID==i),]$tv17rfvar5
  tv17rfvar5m<-c(tv17rfvar5m,rep(mean(tv17rfvar5p),length(tv17rfvar5p)))
}

d33s<-cbind(d33s,tv17rfvar5m)
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
# save.image("d33m2.RData")
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
  d33r<-d33s[sample(dim(d33s)[1],dim(d33s)[1]),9:13] #random order
  dim10<-dim(d33r)[1]%/%10
  d33cv1<-d33r[(1+dim10*ij[2]):(dim10*(ij[2]+1)),]
  d33cv2<-d33r[-((1+dim10*ij[2]):(dim10*(ij[2]+1))),]
  krigcvm<-try(krigmo(d33cv2,t,p,64))   #sseed可替换为8*k
  if ('try-error' %in% class(krigcvm)) {
    prete<-rep(NA,dim(d33cv1)[1]);pretr<-rep(NA,dim(d33cv2)[1])
    preteva<-rep(NA,dim(d33cv1)[1]);pretrva<-rep(NA,dim(d33cv2)[1])
  }else{
    pretr<-predict(krigcvm,d33cv2[,-1],type="SK")$mean
    pretrva<-predict(krigcvm,d33cv2[,-1],type="SK")$sd
    prete<-predict(krigcvm,d33cv1[,-1],type="SK")$mean
    preteva<-predict(krigcvm,d33cv1[,-1],type="SK")$sd
  }
  res<-list(data.frame(d33cv1[,1],prete,preteva),data.frame(d33cv2[,1],pretr,pretrva))
  fnm1<-paste("cv10o/cv10pre4_", ij[1], "_", ij[2], ".csv", sep = "")  #3包括parameters
  fnm2<-paste("cv10o/cv10ptr4_", ij[1], "_", ij[2], ".csv", sep = "")
  write.table(res[[1]], fnm1, append = TRUE, sep = ",")
  write.table(res[[2]], fnm2, append = TRUE, sep = ",")
  return(res)
}

#cluster initialization
cl<-makeCluster(detectCores(),outfile="krigocvs.txt")
clusterExport(cl, list("krigmo","t","p","d33s"))
clusterEvalQ(cl,{library(DiceKriging)})

system.time(
  krigocvs2<-parApply(cl, kcvin, 1, krigocve2)
)
stopCluster(cl)

save.image("d33m2.RData")

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
# save.image("d33m2.RData")
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
# 
# #* var models' performances *#
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
#* repeat seed 10-fold CVE(MAE) with var *#
# Input: cv data order seed and var character  Output: iv and CVE
krigncve<-function(iv){
  krigcv<-c()
  set.seed(11+20*as.numeric(iv[1]))
  d33ro<-d33s[sample(dim(d33s)[1],dim(d33s)[1]),] #random order
  for (j in 0:9){
    dim10<-dim(d33ro)[1]%/%10
    d33cv1<-d33ro[(1+dim10*j):(dim10*(j+1)),9:13]
    d33cv5<-d33ro[-((1+dim10*j):(dim10*(j+1))),]
    #for(k in 7:9){
    krigcvm<-try(krigm(d33cv5[,9:13],t,p,d33cv5[,iv[2]],64))   #sseed可替换为8*k
    if ('try-error' %in% class(krigcvm)) {
      krigcv<-c(krigcv,NA)
    }else{
      krigcv<-c(krigcv,sum(abs(d33cv1[,1]-predict(krigcvm,d33cv1[,-1],type="SK")$mean))/dim10)
    }
    #}
  }
  print(c(iv,krigcv))
  write.table(c(iv,krigcv),"krigncve.csv",append=T,sep = ",")
  return(c(iv,krigcv))
}

kncvin<-matrix(ncol = 2)
for(v in c("tv07rfvar5","tv1rfvar5","tv5rfvar5","tv9rfvar5",
           "tv13rfvar5","tv17rfvar5","tv21rfvar5","tv217rfvar5")){
  for(i in 1:100){
    kncvin<-rbind(kncvin,c(i,v))
  }
}
kncvin<-kncvin[-1,]

kncvin3<-matrix(ncol = 2)
for(i in 1:100){
  kncvin3<-rbind(kncvin3,c(i,"tv17rfvar5m"))
}
kncvin3<-kncvin3[-1,]

cl<-makeCluster(detectCores(),outfile="krigncvs.txt")
clusterExport(cl, list("krigncve","krigm","t","p","d33s"))
clusterEvalQ(cl,{library(DiceKriging)})

system.time(
  krigncvs<-parApply(cl, kncvin, 1, krigncve)
)
stopCluster(cl)

save.image("d33m2.RData")

cl<-makeCluster(detectCores(),outfile="krigncvs.txt")
clusterExport(cl, list("krigncve","krigm","t","p","d33s"))
clusterEvalQ(cl,{library(DiceKriging)})

system.time(
  krigncvs3<-parApply(cl, kncvin3, 1, krigncve)
)
stopCluster(cl)

save.image("d33m2.RData")

krigncvsm<-as.data.frame(matrix(nrow = 100,ncol = 8))
for(j in 1:8){
  for(i in 1:100){
    krigncvsm[i,j]<-mean(as.numeric(krigncvs[-c(1,2),(j-1)*100+i]),na.rm=T)
  }
}
colnames(krigncvsm)<-c("tv07rfvar5","tv1rfvar5","tv5rfvar5","tv9rfvar5",
                       "tv13rfvar5","tv17rfvar5","tv21rfvar5","tv217rfvar5")
# 
# save.image("d33m2.RData")
# 
cvetv17m<-c()
for(i in 1:100){
  cvetv17m<-c(cvetv17m,mean(as.numeric(krigncvs3[-c(1,2),i]),na.rm=T))
}

kncvin2<-matrix(ncol = 3)
for(i in 1:100){
  for(j in 0:9){
    kncvin2<-rbind(kncvin2,c(i,j,"tv17rfvar5m"))
  }
}
kncvin2<-kncvin2[-1,]

#return all predictions
krigncve2<-function(ijv){
  set.seed(11+20*as.numeric(ijv[1]))
  d33r<-d33s[sample(dim(d33s)[1],dim(d33s)[1]),] #random order
  dim10<-dim(d33r)[1]%/%10
  d33cv1<-d33r[(1+dim10*as.numeric(ijv[2])):(dim10*(as.numeric(ijv[2])+1)),9:13]
  d33cv5<-d33r[-((1+dim10*as.numeric(ijv[2])):(dim10*(as.numeric(ijv[2])+1))),]
  krigcvm<-try(krigm(d33cv5[,9:13],t,p,d33cv5[,ijv[3]],64))
  if ('try-error' %in% class(krigcvm)) {
    prete<-rep(NA,dim(d33cv1)[1]);pretr<-rep(NA,dim(d33cv5)[1])
    preteva<-rep(NA,dim(d33cv1)[1]);pretrva<-rep(NA,dim(d33cv5)[1])
  }else{
    pretr<-predict(krigcvm,d33cv5[,10:13],type="SK")$mean
    pretrva<-predict(krigcvm,d33cv5[,10:13],type="SK")$sd
    prete<-predict(krigcvm,d33cv1[,-1],type="SK")$mean
    preteva<-predict(krigcvm,d33cv1[,-1],type="SK")$sd
  }
  res<-list(data.frame(d33cv1[,1],prete,preteva),data.frame(d33cv5[,9],pretr,pretrva))
  fnm1<-paste("cv10n/cv10npre2_", ijv[1], "_", ijv[2], ".csv", sep = "")  #3包括parameters
  fnm2<-paste("cv10n/cv10nptr2_", ijv[1], "_", ijv[2], ".csv", sep = "")
  write.table(res[[1]], fnm1, append = TRUE, sep = ",")
  write.table(res[[2]], fnm2, append = TRUE, sep = ",")
  return(res)
}

cl<-makeCluster(detectCores(),outfile="krigncvs.txt")
clusterExport(cl, list("krigncve","krigm","t","p","d33s"))
clusterEvalQ(cl,{library(DiceKriging)})

system.time(
  krigncvs22<-parApply(cl, kncvin2, 1, krigncve2)
)
stopCluster(cl)

save.image("d33m2.RData")

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
#                       krigncvsm$tv05rfvar5,
#                       krigncvsm$tv10rfvar5,krigncvsm$tv1rfvar5,
#                       krigncvsm$tv5rfvar5,krigncvsm$tv9rfvar5,
#                       krigncvsm$tv13rfvar5,krigncvsm$tv17rfvar5,
#                       krigncvsm$tv21rfvar5, krigncvsm$tv22rfvar5),
#                  V3=c(c(mean(krigocvsm),rep(NA,99)),c(mean(krigcvsm),rep(NA,99)),c(mean(krigncvsm$rfvar5t),rep(NA,99)),
#                       c(mean(krigncvsm$tv05rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv10rfvar5),rep(NA,99)),c(mean(krigncvsm$tv1rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv5rfvar5),rep(NA,99)),c(mean(krigncvsm$tv9rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv13rfvar5),rep(NA,99)),c(mean(krigncvsm$tv17rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv21rfvar5),rep(NA,99)),c(mean(krigncvsm$tv22rfvar5),rep(NA,99))))
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
#                       krigncvsm$tv05rfvar5,
#                       krigncvsm$tv10rfvar5,krigncvsm$tv1rfvar5,
#                       krigncvsm$tv5rfvar5,krigncvsm$tv9rfvar5,
#                       krigncvsm$tv13rfvar5,krigncvsm$tv17rfvar5,
#                       krigncvsm$tv21rfvar5, krigncvsm$tv22rfvar5),
#                  V3=c(c(mean(krigocvsm),rep(NA,99)),c(mean(krigcvsm),rep(NA,99)),c(mean(krigncvsm$rfvar5t),rep(NA,99)),
#                       c(mean(krigncvsm$tv05rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv10rfvar5),rep(NA,99)),c(mean(krigncvsm$tv1rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv5rfvar5),rep(NA,99)),c(mean(krigncvsm$tv9rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv13rfvar5),rep(NA,99)),c(mean(krigncvsm$tv17rfvar5),rep(NA,99)),
#                       c(mean(krigncvsm$tv21rfvar5),rep(NA,99)),c(mean(krigncvsm$tv22rfvar5),rep(NA,99))))
# pmcv$V1<-ordered(pmcv$V1,levels = c("without", "nugget", "res. var (bootstrap)",
#                                     "res. var (3 train)",
#                                     "res. var (7 train)", "res. var (17 train)",
#                                     "res. var (57 train)", "res. var (97 train)",
#                                     "res. var (137 train)", "res. var (177 train)",
#                                     "res. var (217 train)", "res. var (222 train)"))
#导出数据
pmcvout<-data.frame(krigocvsm,krigcvsm,krigncvsm$tv217rfvar5,krigncvsm$tv21rfvar5,krigncvsm$tv17rfvar5,
                    krigncvsm$tv13rfvar5,krigncvsm$tv9rfvar5,krigncvsm$tv5rfvar5,
                    krigncvsm$tv1rfvar5,krigncvsm$tv07rfvar,cvetv17m)
colnames(pmcvout)<-c("without var.","nugget var.","217 train","212 train","172 train",
                     "132 train", "92 train","52 train",
                     "12 train","7 train","177 train source mean")
library(openxlsx)
write.xlsx(pmcvout,"CVEs d33 m2.xlsx")
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
# 
# # plot var vs ref #
# # 自定义颜色(nature11色)
custom_colors <- c("#fa8878", "#ffbe7a", "#3480b8", "#add3e2", "#8dcec8",
                   "#c2bdde", "#e7dbd3", "#f79059", "#82afda", "#9bbf8a")
library(ggh4x)
jpeg("var vs ref v5.jpg", width = 7900, height = 3000, res = 600)
ggplot(d33s, aes(x=as.ordered(refID), y=rfvar5t)) +
  stat_boxplot(geom="errorbar",width=0.5,size=1,position=position_dodge(0.6),color="grey")+
  geom_boxplot(aes(fill=as.ordered(refID)),size=1,alpha=0.85,position=position_dodge(0.8),width=0.8,
               color="grey",outlier.shape = NA)+
  geom_point(aes(color=as.ordered(refID)),position = position_jitter(width = 0.16),
             size=1.7) +
  #ylim(0,0.00069)+
  labs(y="Estimated\nuncertainties",x="Reference ID")+
  # 添加自定义颜色
  scale_fill_manual(values = rep(custom_colors, length.out = length(unique(d33s$refID)))) +
  scale_color_manual(values = rep(custom_colors, length.out = length(unique(d33s$refID)))) +
  theme_bw(base_size = 35)+
  theme(legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),  # 移除默认边框
        axis.line = element_line(color = "black", linewidth = 0.8),  # 保留坐标轴线
        axis.text = element_text(color="black", size=27, family="sans"),
        axis.title = element_text(family="sans"),
        axis.ticks.length=unit(0.22, "cm"),
        axis.ticks = element_line(linewidth = 0.75),
        plot.margin = unit(c(0.3, 0.1, -0.1, 0), "cm"))+
  # 添加次刻度线（需要ggh4x包）
  scale_y_continuous(guide = ggh4x::guide_axis_minor(),
                     minor_breaks = waiver()) +
  theme(ggh4x.axis.ticks.length.minor=rel(1/2))
dev.off()
# 
# # plot var vs data size of ref #
# refds<-c()
# for(i in 1:29){
#   d33sp<-d33s[which(d33s$refID==i),]
#   refds<-c(refds,rep(dim(d33sp)[1],dim(d33sp)[1]))
# }
# d33s<-cbind(d33s,refds)
# library(ggbreak)
# jpeg("var vs refds v4.jpg", width = 3050, height = 3090, res = 600)
# ggplot(d33s, aes(x=refds, y=rfvar5t)) +
#   stat_boxplot(aes(group=refds),geom="errorbar",width=1.2,size=0.7,position=position_identity(),color="grey")+
#   geom_boxplot(aes(group=refds,fill=as.ordered(refds)),size=0.7,alpha=0.8,position=position_identity(),width=1.2,
#                color="grey",outlier.shape = NA)+
#   geom_point(aes(group=refds,color=as.ordered(refds)),position = position_jitter(width = 0.3),
#              size=1.2) +
#   #ylim(0,3200)+
#   labs(y="Estimated variances",x="Data size of reference")+
#   # 添加自定义颜色
#   scale_fill_manual(values = rep(custom_colors, length.out = length(unique(d33s$refds)))) +
#   scale_color_manual(values = rep(custom_colors, length.out = length(unique(d33s$refds)))) +
#   theme_bw(base_size = 22.5)+
#   theme(legend.position = "none",
#         panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         panel.border = element_blank(),  # 移除默认边框
#         axis.line = element_line(color = "black", linewidth = 0.8),  # 保留坐标轴线
#         axis.text = element_text(color="black", size=17.5, family="sans"),
#         axis.title = element_text(family="sans"),
#         axis.title.x = element_text(margin = margin(t = -4)),  # X轴标题向上移动
#         axis.title.y = element_text(margin = margin(r = -4)),  # Y轴标题向右移动
#         axis.ticks.length=unit(0.22, "cm"),
#         axis.ticks = element_line(linewidth = 0.75),
#         plot.margin = unit(c(0.3, 0.1, -0.1, 0), "cm"))+
#   # 添加次刻度线（需要ggh4x包）
#   scale_x_continuous(guide = ggh4x::guide_axis_minor(),breaks = c(0, 10, 20, 30, 100), #根据数据调整坐标
#                      minor_breaks = waiver()) +
#   scale_y_continuous(guide = ggh4x::guide_axis_minor(),
#                      minor_breaks = waiver()) +
#   theme(ggh4x.axis.ticks.length.minor=rel(1/2))+
#   scale_x_break(c(32, 98), scales = 0.34)+  # 添加断点（需要ggbreak包），需根据数据范围调整scale（基本上是右侧数值长度比左侧）
#   theme(axis.line.x.top = element_blank(),
#         axis.ticks.x.top = element_blank(),
#         axis.text.x.top = element_blank())
# dev.off()
# ##***********##
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
# preno<-krigm(d33s[,9:13],t,p,d33s[,"tv17rfvar5"],64)
# prenosa<-krigm(d33s[,9:13],t,p,d33s[,"tv17rfvar5m"],64)
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
# save.image("d33m2.RData")
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
#predict some random selected samples
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
write.table(prevsrsp,"exp compare2.csv",append = T,sep = ",")
# # prevsrsp$div<-abs(prevsrsp$om-prevsrsp$num)+abs(prevsrsp$num-prevsrsp$nom)+
# #   abs(prevsrsp$nom-prevsrsp$om)
# # 
# library(readxl)
# expcomp <- read_excel("experiments compare.xlsx", sheet = "d33")
# expcomp<-as.matrix(expcomp)
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
# 
# library(ggh4x)
# jpeg("expc d33 v2.jpg", width = 3300, height = 3000, res = 600)
# ggplot(data.frame(x=rep(expcomp[,4],2),y=c(expcomp[,1],c(21,91,expcomp[3:4,3])),
#                   z=factor(c(rep("without", 4), rep("source-aware variance", 4)),
#                            levels = c("without", "source-aware variance"))),
#        aes(x = x, y = y, color = z, shape = z)) +
#   geom_point(size = 4) +
#   scale_color_manual(values = c("#3480B8","orange"),  # 手动设置颜色值，与因子的水平顺序一致
#                      labels = c("without", "source-aware variance")) +
#   scale_shape_manual(values = c(15, 17),
#                      labels = c("without", "source-aware variance")) +
#   scale_x_continuous(name = "Measured d33 (pC/N)") +
#   geom_abline(intercept = 0, slope = 1, color = "black", size = 1, linetype="dotted") +  # 添加对角线
#   theme_bw(base_size = 22.5)+
#   theme(legend.title = element_blank(),
#         legend.position = c(0.36,0.9),legend.background = element_blank(),
#         panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         panel.border = element_blank(),  # 移除默认边框
#         axis.line = element_line(color = "black", linewidth = 0.8),  # 保留坐标轴线
#         axis.text = element_text(color="black", size=17.5, family="sans"),
#         axis.title = element_text(family="sans"),
#         axis.ticks.length=unit(0.22, "cm"),
#         axis.ticks = element_line(linewidth = 0.75),
#         plot.margin = unit(c(0.3, 0.1, -0.1, 0), "cm"))+
#   # 添加次刻度线（需要ggh4x包）
#   scale_y_continuous(name = "Predicted d33 (pC/N)",guide = ggh4x::guide_axis_minor(),
#                      minor_breaks = waiver()) +
#   theme(ggh4x.axis.ticks.length.minor=rel(1/2))
# dev.off()