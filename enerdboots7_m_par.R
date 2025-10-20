#相比于enerdboots6_m.R，修改删除数据中的异常值，设置RF参数corr.bias=F

setwd("~/ECEdemo/directboots/enerd")
load("~/ECEdemo/directboots/enerd/enerd7m.RData")
# setwd("F:/机器学习/Multi-fidelity/ECE demo/directboots/enerd")

library(readxl)
library(dplyr)
library(MuFiCokriging)
library(parallel)
library(randomForest)

# ##** input data and preprocessing **##
# 
# #* input data *#
# enerd<-read_excel("enerd of BaTiO3 4.0.xlsx")[2:14]
# 
# #* rearrange data *#
# enerd<-enerd[-c(251:253,281:283),]   #删除异价掺杂数据
# enerd<-enerd[,-c(2,11,12)]
# colnames(enerd)<-c("refID","ba","ca","sr","cd","ti","zr","sn","hf","y")
# 
# 
# #* Handle outliers *#
# for (i in 1:dim(enerd)[1]){
#   if(sum(enerd[i,2:9])!=200){
#     print("Data exception: Line")
#     print(i)
#   }
# }
# enerd[216,3]<-14
# 
# # 绘制频数分布直方图
# hist(enerd$y,
#      breaks = seq(0, 100, by = 10),    # 设置区间为0-10, 10-20, ..., 90-100
#      col = "lightblue",                # 填充颜色
#      border = "black",                 # 边框颜色
#      axes = TRUE)                      # 显示坐标轴
# 
# 
# #* calculate features *#
# #run ceramic-fea.R
# enerd<-fn.data.features(enerd)
# 
# #* data for model *#
# enerds<-cbind(enerd[,c(1:10)],enerd[,c("TA.B", "BD", "EN")])
# enerds<-enerds[,c(2:13,1)]
# 
# # rm.NA #
# for(i in 1:dim(enerds)[2]){
#   enerds<-subset(enerds,!is.na(enerds[,i]))
# }
# 
# #* unique *#
# enerds<-anti_join(enerds,enerds[duplicated(enerds[,-c(9,13)]),])
# 
# ##*************##
# 
# 
# 
# ##** bootstrap **##
# testsamp<-matrix(nrow=500,ncol=267)
# for(i in 1:500){
#   set.seed(i)
#   testsamp[i,]<-sample(dim(enerds)[1], dim(enerds)[1], replace = TRUE)
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
#     rfm<-try(randomForest(y~.,data=enerds[,9:12],ntree=nm[1],mtry=nm[2]))
#     if ('try-error' %in% class(rfm)) {
#       rfr2<-c(rfr2,NA)
#     }else{
#       rfr2<-c(rfr2,1-sum((rfm$y-rfm$predicted)^2)/sum((enerds[,9]-mean(unlist(enerds[,9])))^2))
#     }
#     enerdr<-enerds[sample(dim(enerds)[1],dim(enerds)[1]),9:12] #random order
#     rfcve<-c()
#     for (j in 0:9){
#       dim10<-dim(enerdr)[1]%/%10
#       enerdcv1<-enerdr[(1+dim10*j):(dim10*(j+1)),]
#       enerdcv2<-enerdr[-((1+dim10*j):(dim10*(j+1))),]
#       rfcvm<-try(randomForest(y~.,data=enerdcv2,ntree=nm[1],mtry=nm[2]))
#       if ('try-error' %in% class(rfcvm)) {
#         rfcve<-c(rfcve,NA)
#       }else{
#         rfcve<-c(rfcve,sum((enerdcv1[,1]-predict(rfcvm,enerdcv1))^2)/dim10)
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
# # rfin2<-matrix(ncol=2)
# # for(nt in c(200,400,500,700,1000)){
# #   for(mt in c(1,2,3)){
# #     rfin2<-rbind(rfin2,c(nt,mt))
# #   }
# # }
# # rfin2<-rfin2[-1,]
# 
# #cluster initialization
# cores<-detectCores()
# cl<-makeCluster(detectCores(),outfile="rfdtune.txt")
# clusterExport(cl, list("rfpd","enerds"))
# clusterEvalQ(cl,{library(randomForest)})
# 
# system.time(
#   rfdtune<-parApply(cl, rfin, 1, rfpd)
# )
# 
# stopCluster(cl)
# 
# save.image("enerd7m.RData")
# 
# rfdbp<-rfdtune[,which.min(rfdtune[3,])]
# save.image("enerd7m.RData")
# 
#
# #* test method with RF *#
# Input: seed(bootstrap number) and sample train size  Output: predictions
rftv<-function(Bts){
  set.seed(Bts[1])
  enerdtv<-enerds[sample(dim(enerds)[1], Bts[2], replace = F),9:12]
  enerdtvtest<-anti_join(enerds[,9:12],enerdtv)
  tvrfm<-randomForest(y~.,data=enerdtv,ntree=500,mtry=1,corr.bias=T,replace=F)
  pretvtest<-predict(tvrfm,enerdtvtest)
  pretestna<-c()
  for(i in 1:dim(enerds)[1]){
    ind<-which(enerdtvtest$TA.B==enerds$TA.B[i] &
                 enerdtvtest$BD==enerds$BD[i] & enerdtvtest$EN==enerds$EN[i])
    if(length(ind)==0){
      pretestna<-c(pretestna,NA)
    }else{
      pretestna<-c(pretestna,pretvtest[ind])
    }
  }
  gc()
  return(pretestna)
}

# B=500 #
Btsin<-matrix(ncol=2)
for(ts in c(16,46,76,106,136,166,196,226,256)){
  for(B in 1:500){
    Btsin<-rbind(Btsin,c(B,ts))
  }
}
Btsin<-Btsin[-1,]

cl<-makeCluster(detectCores(),outfile="rftv5.txt")
clusterExport(cl, list("rftv","enerds"))
clusterEvalQ(cl,{library(randomForest);library(dplyr)})

system.time(
  rftvpre5<-parApply(cl, Btsin, 1, rftv)
)
stopCluster(cl)

save.image("enerd7m.RData")

tv1rfvar5<-c()
for(i in 1:dim(rftvpre5)[1]){
  tv1rfvar5<-c(tv1rfvar5,var(rftvpre5[i,1:500],na.rm=T))
}
tv4rfvar5<-c()
for(i in 1:dim(rftvpre5)[1]){
  tv4rfvar5<-c(tv4rfvar5,var(rftvpre5[i,501:1000],na.rm=T))
}
tv7rfvar5<-c()
for(i in 1:dim(rftvpre5)[1]){
  tv7rfvar5<-c(tv7rfvar5,var(rftvpre5[i,1001:1500],na.rm=T))
}
tv10rfvar5<-c()
for(i in 1:dim(rftvpre5)[1]){
  tv10rfvar5<-c(tv10rfvar5,var(rftvpre5[i,1501:2000],na.rm=T))
}
tv13rfvar5<-c()
for(i in 1:dim(rftvpre5)[1]){
  tv13rfvar5<-c(tv13rfvar5,var(rftvpre5[i,2001:2500],na.rm=T))
}
tv16rfvar5<-c()
for(i in 1:dim(rftvpre5)[1]){
  tv16rfvar5<-c(tv16rfvar5,var(rftvpre5[i,2501:3000],na.rm=T))
}
tv19rfvar5<-c()
for(i in 1:dim(rftvpre5)[1]){
  tv19rfvar5<-c(tv19rfvar5,var(rftvpre5[i,3001:3500],na.rm=T))
}
tv22rfvar5<-c()
for(i in 1:dim(rftvpre5)[1]){
  tv22rfvar5<-c(tv22rfvar5,var(rftvpre5[i,3501:4000],na.rm=T))
}
tv25rfvar5<-c()
for(i in 1:dim(rftvpre5)[1]){
  tv25rfvar5<-c(tv25rfvar5,var(rftvpre5[i,4001:4500],na.rm=T))
}

enerds<-cbind(enerds,tv1rfvar5,tv4rfvar5,tv7rfvar5,tv10rfvar5,tv13rfvar5,
              tv16rfvar5,tv19rfvar5,tv22rfvar5,tv25rfvar5)

save.image("enerd7m.RData")
# 
tv7rfvar5m<-c()
for(i in 1:30){
  tv7rfvar5p<-enerds[which(enerds$refID==i),]$tv7rfvar5
  tv7rfvar5m<-c(tv7rfvar5m,rep(mean(tv7rfvar5p),length(tv7rfvar5p)))
}
enerds<-cbind(enerds,tv7rfvar5m)
tv7rfvar5sd<-c()
for(i in 1:30){
  tv7rfvar5p<-enerds[which(enerds$refID==i),]$tv7rfvar5
  tv7rfvar5sd<-c(tv7rfvar5sd,rep(sd(tv7rfvar5p),length(tv7rfvar5p)))
}

Ba0.9=na.omit(rftvpre5[1,1001:1500])
Ba0.65=na.omit(rftvpre5[4,1001:1500])
Ba0.65<-c(Ba0.65,rep(NA,14))
library(openxlsx)
write.xlsx(data.frame(Ba0.9,Ba0.65),"Fig 2a data2.xlsx")
#
#
# #kriging estimated nugget
# enerdkrig<-krigm(enerds[,9:12],t,p,0,64)
# knug<-enerdkrig@covariance@nugget
# 
# save.image("enerd7m.RData")
# 
# # plot distribution of var estimated by different models #
pmvar<-data.frame(V1=c(rep("250 RF test",266),rep("220 RF test",266),
                       rep("190 RF test",266),rep("160 RF test",266),
                       rep("130 RF test",266),rep("100 RF test",266),
                       rep("70 RF test",266),rep("40 RF test",266),
                       rep("10 RF test",266),"estimated nugget"),
                  V2=c(tv1rfvar5,tv4rfvar5,tv7rfvar5,tv10rfvar5,
                       tv13rfvar5,tv16rfvar5,tv19rfvar5,tv22rfvar5,
                       tv25rfvar5,knug))
pmvar$V1<-ordered(pmvar$V1,levels = c("250 RF test","220 RF test",
                                      "190 RF test","160 RF test",
                                      "130 RF test","100 RF test",
                                      "70 RF test","40 RF test",
                                      "10 RF test","estimated nugget"))
#导出数据
pmvarout<-data.frame(tv4rfvar5,tv7rfvar5,tv10rfvar5,tv13rfvar5,
                     tv16rfvar5,tv19rfvar5,tv22rfvar5,tv25rfvar5,knug)
colnames(pmvarout)<-c("46 train","76 train","106 train","136 train",
                      "166 train","196 train","226 train","256 train","nugget")
library(openxlsx)
write.xlsx(pmvarout,"Vars enerd m 7.xlsx")
ggplot(pmvar, aes(x=V1, y=V2)) +
  stat_boxplot(geom="errorbar",width=0.3,size=1,position=position_dodge(0.6),color="grey")+
  geom_boxplot(aes(fill=V1),size=1,alpha=0.5,position=position_dodge(0.8),width=0.8,
               color="grey",outlier.colour = "grey")+
  # ylim(0,125)+
  labs(y="Estimated variances",x="")+
  theme_bw(base_size = 20)+
  theme(legend.position = "none",panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),axis.text=element_text(color="black",size=16),
        axis.text.x = element_text(angle = 25,hjust = 1),
        axis.title=element_text(face="bold"))
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
#   enerdr<-enerds[sample(dim(enerds)[1],dim(enerds)[1]),9:12] #random order
#   for (j in 0:9){
#     dim10<-dim(enerdr)[1]%/%10
#     enerdcv1<-enerdr[(1+dim10*j):(dim10*(j+1)),]
#     enerdcv2<-enerdr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigmo(enerdcv2,t,p,64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcv<-c(krigcv,NA)
#     }else{
#       krigcv<-c(krigcv,sum(abs(enerdcv1[,1]-predict(krigcvm,enerdcv1[,-1],type="SK")$mean))/dim10)
#     }
#     #}
#   }
#   print(c(i,krigcv))
#   return(c(i,krigcv))
# }
#
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krigocvs.txt")
# clusterExport(cl, list("krigocve","krigmo","t","p","enerds"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krigocvs<-parLapply(cl, 1:100, krigocve)
# )
# stopCluster(cl)
# 
# save.image("enerd7m.RData")
# 
# krigocvsm<-c()
# for(i in 1:length(krigocvs)){
#   krigocvsm<-c(krigocvsm,mean(krigocvs[[i]][-1]))
# }
#
#返回所有数据
krigocve2<-function(ij){    #3:j为折数0:9，放在cvin里
  set.seed(11+20*ij[1])
  enerdr<-enerds[sample(dim(enerds)[1],dim(enerds)[1]),9:12] #random order
  dim10<-dim(enerdr)[1]%/%10
  enerdcv1<-enerdr[(1+dim10*ij[2]):(dim10*(ij[2]+1)),]
  enerdcv2<-enerdr[-((1+dim10*ij[2]):(dim10*(ij[2]+1))),]
  krigcvm<-try(krigmo(enerdcv2,t,p,64))   #sseed可替换为8*k
  if ('try-error' %in% class(krigcvm)) {
    prete<-rep(NA,dim(enerdcv1)[1]);pretr<-rep(NA,dim(enerdcv2)[1])
    preteva<-rep(NA,dim(enerdcv1)[1]);pretrva<-rep(NA,dim(enerdcv2)[1])
  }else{
    pretr<-predict(krigcvm,enerdcv2[,-1],type="SK")$mean
    pretrva<-predict(krigcvm,enerdcv2[,-1],type="SK")$sd
    prete<-predict(krigcvm,enerdcv1[,-1],type="SK")$mean
    preteva<-predict(krigcvm,enerdcv1[,-1],type="SK")$sd
  }
  res<-list(data.frame(enerdcv1[,1],prete,preteva),data.frame(enerdcv2[,1],pretr,pretrva))
  fnm1<-paste("cv10o/cv10pre7_", ij[1], "_", ij[2], ".csv", sep = "")  #3包括parameters
  fnm2<-paste("cv10o/cv10ptr7_", ij[1], "_", ij[2], ".csv", sep = "")
  write.table(res[[1]], fnm1, append = TRUE, sep = ",")
  write.table(res[[2]], fnm2, append = TRUE, sep = ",")
  return(res)
}

kcvin<-matrix(ncol = 2)
for(i in 1:100){
  for(j in 0:9){
    kcvin<-rbind(kcvin,c(i,j))
  }
}
kcvin<-kcvin[-1,]

#cluster initialization
cl<-makeCluster(detectCores(),outfile="krigcvs.txt")
clusterExport(cl, list("krigmo","t","p","enerds"))
clusterEvalQ(cl,{library(DiceKriging)})

system.time(
  krigocvs2<-parApply(cl, kcvin, 1, krigocve2)
)
stopCluster(cl)

save.image("enerd7m.RData")

# krigcvsm<-c()
# for(i in 1:length(krigcvs)){
#   krigcvsm<-c(krigcvsm,mean(krigcvs[[i]][-1]))
# }

cv10mae<-c(); cv10r2<-c()
for(i in 1:100){
  te<-c(); pte<-c()
  for(j in 1:10){
    te<-c(te,krigocvs2[[((i-1)*10+j)]][[1]][,1])
    pte<-c(pte,krigocvs2[[((i-1)*10+j)]][[1]][,2])
  }
  ptemae<-mean(abs(pte - te),na.rm=T)
  pter2<-round(1-sum((pte-te)^2,na.rm=T)/sum((te-mean(te,na.rm=T))^2,na.rm=T),3)
  cv10mae<-c(cv10mae,ptemae)
  cv10r2<-c(cv10r2,pter2)
}
cv10r2s<-c()
for(i in 1:1000){
  te<-krigocvs2[[i]][[1]][,1]
  pte<-krigocvs2[[i]][[1]][,2]
  pter2<-round(1-sum((pte-te)^2,na.rm=T)/sum((te-mean(te,na.rm=T))^2,na.rm=T),3)
  cv10r2s<-c(cv10r2s,pter2)
}
# ##***********##
#
#
# ##** kriging model with nugget **##
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
# #* repeat seed 10-fold CVE(MAE) *#
# krigcve<-function(i){
#   krigcv<-c()
#   set.seed(11+20*i)
#   enerdr<-enerds[sample(dim(enerds)[1],dim(enerds)[1]),9:12] #random order
#   for (j in 0:9){
#     dim10<-dim(enerdr)[1]%/%10
#     enerdcv1<-enerdr[(1+dim10*j):(dim10*(j+1)),]
#     enerdcv2<-enerdr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#       krigcvm<-try(krigm(enerdcv2,t,p,0,64))   #sseed可替换为8*k
#       if ('try-error' %in% class(krigcvm)) {
#         krigcv<-c(krigcv,NA)
#       }else{
#         krigcv<-c(krigcv,sum(abs(enerdcv1[,1]-predict(krigcvm,enerdcv1[,-1],type="SK")$mean))/dim10)
#       }
#     #}
#   }
#   print(c(i,krigcv))
#   return(c(i,krigcv))
# }
#
# #cluster initialization
# cl<-makeCluster(detectCores(),outfile="krigcvs.txt")
# clusterExport(cl, list("krigcve","krigm","t","p","enerds"))
# clusterEvalQ(cl,{library(DiceKriging)})
# 
# system.time(
#   krigcvs<-parLapply(cl, 1:100, krigcve)
# )
# stopCluster(cl)
# 
# save.image("enerd7m.RData")
# 
# krigcvsm<-c()
# for(i in 1:length(krigcvs)){
#   krigcvsm<-c(krigcvsm,mean(krigcvs[[i]][-1],na.rm=T))
# }
#
# ##***********##
# 
# 
# 
# ##** kriging model with noise **##
# #* repeat seed 10-fold CVE(MAE) with var *#
# # Input: cv data order seed and var character  Output: iv and CVE
# krigncve<-function(iv){
#   krigcv<-c()
#   set.seed(11+20*as.numeric(iv[1]))
#   enerdr<-enerds[sample(dim(enerds)[1],dim(enerds)[1]),] #random order
#   for (j in 0:9){
#     dim10<-dim(enerdr)[1]%/%10
#     enerdcv1<-enerdr[(1+dim10*j):(dim10*(j+1)),9:12]
#     enerdcv5<-enerdr[-((1+dim10*j):(dim10*(j+1))),]
#     #for(k in 7:9){
#     krigcvm<-try(krigm(enerdcv5[,9:12],t,p,enerdcv5[,iv[2]],64))   #sseed可替换为8*k
#     if ('try-error' %in% class(krigcvm)) {
#       krigcv<-c(krigcv,NA)
#     }else{
#       krigcv<-c(krigcv,sum(abs(enerdcv1[,1]-predict(krigcvm,enerdcv1[,-1],type="SK")$mean))/dim10)
#     }
#     #}
#   }
#   print(c(iv,krigcv))
#   write.table(c(iv,krigcv),"krigncve.csv",append=T,sep = ",")
#   return(c(iv,krigcv))
# }
# 

#返回所有预测结果
krigncve2<-function(ijv){
  set.seed(11+20*as.numeric(ijv[1]))
  enerdr<-enerds[sample(dim(enerds)[1],dim(enerds)[1]),] #random order
  dim10<-dim(enerdr)[1]%/%10
  enerdcv1<-enerdr[(1+dim10*as.numeric(ijv[2])):(dim10*(as.numeric(ijv[2])+1)),9:12]
  enerdcv5<-enerdr[-((1+dim10*as.numeric(ijv[2])):(dim10*(as.numeric(ijv[2])+1))),]
  krigcvm<-try(krigm(enerdcv5[,9:12],t,p,enerdcv5[,ijv[3]],64))
  if ('try-error' %in% class(krigcvm)) {
    prete<-rep(NA,dim(enerdcv1)[1]);pretr<-rep(NA,dim(enerdcv5)[1])
    preteva<-rep(NA,dim(enerdcv1)[1]);pretrva<-rep(NA,dim(enerdcv5)[1])
  }else{
    pretr<-predict(krigcvm,enerdcv5[,10:12],type="SK")$mean
    pretrva<-predict(krigcvm,enerdcv5[,10:12],type="SK")$sd
    prete<-predict(krigcvm,enerdcv1[,-1],type="SK")$mean
    preteva<-predict(krigcvm,enerdcv1[,-1],type="SK")$sd
  }
  res<-list(data.frame(enerdcv1[,1],prete,preteva),data.frame(enerdcv5[,9],pretr,pretrva))
  fnm1<-paste("cv10n/cv10npre7_", ijv[1], "_", ijv[2], ".csv", sep = "")  #3包括parameters
  fnm2<-paste("cv10n/cv10nptr7_", ijv[1], "_", ijv[2], ".csv", sep = "")
  write.table(res[[1]], fnm1, append = TRUE, sep = ",")
  write.table(res[[2]], fnm2, append = TRUE, sep = ",")
  return(res)
}


kncvin<-matrix(ncol = 2)
for(v in c("tv1rfvar5","tv4rfvar5","tv7rfvar5","tv10rfvar5",
           "tv13rfvar5","tv16rfvar5","tv19rfvar5","tv22rfvar5","tv25rfvar5")){
  for(i in 1:100){
    kncvin<-rbind(kncvin,c(i,v))
  }
}
kncvin<-kncvin[-1,]

kncvin3<-matrix(ncol = 2)
for(i in 1:100){
  kncvin3<-rbind(kncvin3,c(i,"tv7rfvar5m"))
}
kncvin3<-kncvin3[-1,]

kncvin2<-matrix(ncol = 3)
for(i in 1:100){
  for(j in 0:9){
    kncvin2<-rbind(kncvin2,c(i,j,"tv7rfvar5"))
  }
}
kncvin2<-kncvin2[-1,]

cl<-makeCluster(detectCores(),outfile="krigncvs.txt")
clusterExport(cl, list("krigncve","krigm","t","p","enerds"))
clusterEvalQ(cl,{library(DiceKriging)})

system.time(
  krigncvs22<-parApply(cl, kncvin2, 1, krigncve2)
)
stopCluster(cl)

save.image("enerd7m.RData")

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

#* Fig3a *#
f3ascreen<-which(cv10nr2s>0.7)
f3aind<-order(cv10nr2s[f3ascreen]-cv10r2s[f3ascreen],decreasing = T)[1:10]
cv10nr2s[f3ascreen][f3aind];cv10r2s[f3ascreen][f3aind]
f3aselect<-f3ascreen[2]

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
          axis.title.x = element_text(margin = ggplot2::margin(t = -4), hjust = 1),
          axis.title.y = element_text(margin = ggplot2::margin(r = -4)),  # Y轴标题向右移动
          axis.ticks.length=unit(0.35, "cm"),
          axis.ticks = element_line(linewidth = 0.75),
          plot.margin = unit(c(0.3, 0.65, -0.1, 0), "cm"))+
    # 添加次刻度线（需要ggh4x包）
    scale_y_continuous(limits = lim, name = ylab,
                       breaks = c(0, 45, 90), labels = c("0", "45", "90"),
                       guide = guide_axis(minor.ticks = TRUE),minor_breaks = waiver()) +
    scale_x_continuous(limits = lim, name = xlab, 
                       breaks = c(0, 45, 90), labels = c("0", "45", "90"),
                       guide = guide_axis(minor.ticks = TRUE),minor_breaks = waiver()) +
    theme(ggh4x.axis.ticks.length.minor=rel(1/2))+
    annotate("text", x = 22, y = 88, label = bquote(paste("R"^2 == .(cr2))),
             color = rgb(0,72,131,maxColorValue = 255), size = 10, family = "sans")
}

jpeg("Fig3a 1.jpg", width = 3720, height = 2800, res = 600)
fn.plot.gpar3(x=c(krigocvs2[[692]][[2]][,1],krigocvs2[[692]][[1]][,1]),
              y1=krigocvs2[[692]][[2]][,2],y2=krigocvs2[[692]][[1]][,2],
              lim=c(0,92), xlab=expression(paste("Measured U"["re"], " (" * mJ * bold("/") * cm^3 * ")")),
              ylab=bquote(atop("Predicted U"["re"], "(" * mJ * bold("/") * cm^3 * ")")),0.618)
dev.off()
jpeg("Fig3a 2.jpg", width = 3720, height = 2800, res = 600)
fn.plot.gpar3(x=c(krigncvs22[[692]][[2]][,1],krigncvs22[[692]][[1]][,1]),
              y1=krigncvs22[[692]][[2]][,2],y2=krigncvs22[[692]][[1]][,2],
              lim=c(0,92),xlab=expression(paste("Measured U"["re"], " (" * mJ * bold("/") * cm^3 * ")")),
              ylab=bquote(atop("Predicted U"["re"], "(" * mJ * bold("/") * cm^3 * ")")),0.728)
dev.off()


cl<-makeCluster(detectCores(),outfile="krigncvs.txt")
clusterExport(cl, list("krigncve","krigm","t","p","enerds"))
clusterEvalQ(cl,{library(DiceKriging)})

system.time(
  krigncvs<-parApply(cl, kncvin, 1, krigncve)
)
stopCluster(cl)

cl<-makeCluster(detectCores(),outfile="krigncvs.txt")
clusterExport(cl, list("krigncve","krigm","t","p","enerds"))
clusterEvalQ(cl,{library(DiceKriging)})

system.time(
  krigncvs3<-parApply(cl, kncvin3, 1, krigncve)
)
stopCluster(cl)

save.image("enerd7m.RData")
 
krigncvsm<-as.data.frame(matrix(nrow = 100,ncol = 9))
for(i in 1:100){
  krigncvsm[i,1]<-mean(as.numeric(krigncvs2[-c(1,2),i]),na.rm=T)
}
for(j in 2:9){
  for(i in 1:100){
    krigncvsm[i,j]<-mean(as.numeric(krigncvs[-c(1,2),(j-1)*100+i]),na.rm=T)
  }
}
colnames(krigncvsm)<-c("tv1rfvar5","tv4rfvar5","tv7rfvar5","tv10rfvar5",
                       "tv13rfvar5","tv16rfvar5","tv19rfvar5","tv22rfvar5","tv25rfvar5")
save.image("enerd7m.RData")

cvetv7m<-c()
for(i in 1:100){
  cvetv7m<-c(cvetv7m,mean(as.numeric(krigncvs3[-c(1,2),i]),na.rm=T))
}

pmcv<-data.frame(V1=c(rep("without",100), rep("nugget",100),
                      rep("res. var (250 test)",100),rep("res. var (220 test)",100),
                      rep("res. var (190 test)",100),rep("res. var (160 test)",100),
                      rep("res. var (130 test)",100),rep("res. var (100 test)",100),
                      rep("res. var (70 test)",100),rep("res. var (40 test)",100),
                      rep("res. var (10 test)",100)),
                 V2=c(krigocvsm, krigcvsm,
                      krigncvsm$tv1rfvar5,krigncvsm$tv4rfvar5,
                      krigncvsm$tv7rfvar5,krigncvsm$tv10rfvar5,
                      krigncvsm$tv13rfvar5, krigncvsm$tv16rfvar5,
                      krigncvsm$tv19rfvar5, krigncvsm$tv22rfvar5,
                      krigncvsm$tv25rfvar5),
                 V3=c(c(mean(krigocvsm),rep(NA,99)),c(mean(krigcvsm),rep(NA,99)),
                      c(mean(krigncvsm$tv1rfvar5),rep(NA,99)),c(mean(krigncvsm$tv4rfvar5),rep(NA,99)),
                      c(mean(krigncvsm$tv7rfvar5),rep(NA,99)),c(mean(krigncvsm$tv10rfvar5),rep(NA,99)),
                      c(mean(krigncvsm$tv13rfvar5),rep(NA,99)),c(mean(krigncvsm$tv16rfvar5),rep(NA,99)),
                      c(mean(krigncvsm$tv19rfvar5),rep(NA,99)),c(mean(krigncvsm$tv22rfvar5),rep(NA,99)),
                      c(mean(krigncvsm$tv25rfvar5),rep(NA,99))))
pmcv$V1<-ordered(pmcv$V1,levels = c("without", "nugget",
                                    "res. var (250 test)","res. var (220 test)",
                                    "res. var (190 test)", "res. var (160 test)",
                                    "res. var (130 test)", "res. var (100 test)",
                                    "res. var (70 test)", "res. var (40 test)",
                                    "res. var (10 test)"))
#导出数据
pmcvout<-data.frame(krigocvsm,krigcvsm,krigncvsm$tv25rfvar5,krigncvsm$tv22rfvar5,
                    krigncvsm$tv19rfvar5,krigncvsm$tv16rfvar5,krigncvsm$tv13rfvar5,
                    krigncvsm$tv10rfvar5,krigncvsm$tv7rfvar5,krigncvsm$tv4rfvar5,
                    cvetv7m)
colnames(pmcvout)<-c("without var.","nugget var.","256 train","226 train",
                     "196 train", "166 train","136 train",
                     "106 train","76 train","46 train",
                     "76 train source mean")
library(openxlsx)
write.xlsx(pmcvout,"CVEs enerd 7m.xlsx")
ggplot(pmcv, aes(x=V1, y=V2)) +
  geom_violin(aes(linetype=NA,fill=V1),alpha=0.5,position=position_dodge(0.8),width=1)+
  geom_point(aes(x=V1,y=V3),shape=17,size=3,color=rgb(0.4,0.4,0.4))+
  coord_flip()+
  labs(y="CVE (10-fold MAE)",x="")+
  theme_bw(base_size = 27)+
  theme(legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(face="bold", color="black", size=24, family="serif"),
        axis.title = element_text(face="bold",family="serif"),
        axis.ticks.length=unit(-0.25, "cm"),
        plot.margin = unit(c(0, 0.1, 0, -0.9), "cm"))
# 
# 
# # # plot var estimated by different models #
# # plot(x=btvar5,y=btvar,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
# #      col= rgb(59, 0, 159, maxColorValue = 255), cex = 1.2,
# #      xlim = c(0,0.006), ylim = c(0,0.006),
# #      xlab="Variance by GB (B=500)",ylab="Variance by GB with compositions (B=500)",
# #      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0), font.axis=2)
# # axis(1,lwd.ticks=2)
# # axis(2,lwd.ticks=2)
# # box(lwd=2)
# # abline(0,1,lwd=2,col=rgb(0, 34, 139, maxColorValue = 255))
# #
# # plot(x=btvar5,y=svrvar5,pch=21,bg=rgb(240, 255, 240, maxColorValue = 255),
# #      col= rgb(59, 0, 159, maxColorValue = 255), cex = 1.2,
# #      xlim = c(0,0.006), ylim = c(0,0.006),
# #      xlab="Variance by GB (B=500)",ylab="Variance by SVR (B=500)",
# #      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0), font.axis=2)
# # axis(1,lwd.ticks=2)
# # axis(2,lwd.ticks=2)
# # box(lwd=2)
# # abline(0,1,lwd=2,col=rgb(0, 34, 139, maxColorValue = 255))
# 
# ##***********##


# plot var vs ref #
# 自定义颜色(nature11色)
custom_colors <- c(rgb(144,201,231,maxColorValue = 255), rgb(33,158,188,maxColorValue = 255),
                   rgb(19,103,131,maxColorValue = 255), rgb(2,48,74,maxColorValue = 255),
                   rgb(254,183,5,maxColorValue = 255), rgb(255,158,2,maxColorValue = 255),
                   rgb(250,134,0,maxColorValue = 255))
library(ggh4x)
jpeg("var vs ref v5.jpg", width = 7900, height = 3000, res = 600)
ggplot(enerds, aes(x=as.ordered(refID), y=tv7rfvar5)) +
  stat_boxplot(geom="errorbar",width=0.5,size=1,position=position_dodge(0.6),color="grey")+
  geom_boxplot(aes(fill=as.ordered(refID)),size=1,alpha=0.75,position=position_dodge(0.8),width=0.8,
               color="grey",outlier.shape = NA)+
  geom_point(aes(color=as.ordered(refID)),position = position_jitter(width = 0.16),
             size=1.7) +
  # ylim(0,45)+
  labs(y="Estimated\nuncertainties",x="Reference ID")+
  # 添加自定义颜色
  scale_fill_manual(values = rep(custom_colors, length.out = length(unique(enerds$refID)))) +
  scale_color_manual(values = rep(custom_colors, length.out = length(unique(enerds$refID)))) +
  theme_bw(base_size = 35)+
  theme(legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),  # 移除默认边框
        axis.line = element_line(color = "black", linewidth = 0.8),  # 保留坐标轴线
        axis.text = element_text(color="black", family="sans",size=27),
        axis.text.x = element_text(angle = 90,hjust = 1.2,vjust=0.5,size=27),
        axis.title = element_text(family="sans"),
        axis.ticks.length=unit(0.35, "cm"),
        axis.ticks = element_line(linewidth = 0.75),
        plot.margin = unit(c(0.3, 0.1, -0.1, 0), "cm"))+
  # 添加次刻度线（需要ggh4x包）
  scale_y_continuous(guide = ggh4x::guide_axis_minor(),
                     minor_breaks = waiver()) +
  theme(ggh4x.axis.ticks.length.minor=rel(1/2))
dev.off()

# plot var vs data size of ref #
refds<-c()
for(i in 1:30){
  enerdsp<-enerds[which(enerds$refID==i),]
  refds<-c(refds,rep(dim(enerdsp)[1],dim(enerdsp)[1]))
}
enerds<-cbind(enerds,refds)
library(ggbreak)
jpeg("var vs refds v5.jpg", width = 3850, height = 3090, res = 600)
ggplot(enerds, aes(x=refds, y=tv7rfvar5)) +
  stat_boxplot(aes(group=refds),geom="errorbar",width=0.8,size=1,position=position_dodge(0.6),color="grey")+
  geom_boxplot(aes(group=refds,fill=as.ordered(refds)),size=1,alpha=0.85,position=position_dodge(0.9),width=0.8,
               color="grey",outlier.shape = NA)+
  geom_point(aes(group=refds,color=as.ordered(refds)),position = position_jitter(width = 0.15),
             size=1.7) +
  labs(y="Estimated\nuncertainties",x="Data size of reference")+
  # 添加自定义颜色
  scale_fill_manual(values = rep(custom_colors, length.out = length(unique(enerds$refds)))) +
  scale_color_manual(values = rep(custom_colors, length.out = length(unique(enerds$refds)))) +
  theme_bw(base_size = 35)+
  theme(legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),  # 移除默认边框
        axis.line = element_line(color = "black", linewidth = 0.8),  # 保留坐标轴线
        axis.text = element_text(color="black", size=27, family="sans"),
        axis.title = element_text(family="sans"),
        axis.title.x = element_text(margin = ggplot2::margin(t = -4)),  # X轴标题向上移动
        axis.title.y = element_text(margin = ggplot2::margin(r = -0.2)),  # Y轴标题向右移动
        axis.ticks.length=unit(0.35, "cm"),
        axis.ticks = element_line(linewidth = 0.75),
        plot.margin = unit(c(0.3, 0.1, -0.1, 0), "cm"))+
  # 添加次刻度线（需要ggh4x包）
  scale_x_continuous(limits = c(0, 183),
                     guide = ggh4x::guide_axis_minor(),breaks = c(0, 5, 10, 15, 20, 180), #根据数据调整坐标
                     minor_breaks = waiver()) +
  scale_y_continuous(guide = ggh4x::guide_axis_minor(),
                     minor_breaks = waiver()) +
  theme(ggh4x.axis.ticks.length.minor=rel(1/2))+
  scale_x_break(c(6, 17), scales = 0.66) + 
  scale_x_break(c(21, 179), scales = 0.65)+  # 添加断点（需要ggbreak包），需根据数据范围调整scale（基本上是右侧数值长度比左侧）
  theme(axis.line.x.top = element_blank(),
        axis.ticks.x.top = element_blank(),
        axis.text.x.top = element_blank())
dev.off()

# plot var vs mean value of ref #
refval<-c()
for(i in 1:30){
  enerdsp<-enerds[which(enerds$refID==i),9]
  refval<-c(refval,rep(mean(enerdsp),length(enerdsp)))
}
enerds<-cbind(enerds,refval)
library(ggbreak)
jpeg("var vs refval v5.jpg", width = 3900, height = 3600, res = 600)
ggplot(enerds, aes(x=refval, y=tv7rfvar5)) +
  stat_boxplot(aes(group=refval),geom="errorbar",width=2,size=1,position=position_identity(),color="grey")+
  geom_boxplot(aes(group=refval,fill=as.ordered(refval)),size=1,alpha=0.85,position=position_identity(),width=2,
               color="grey",outlier.shape = NA)+
  geom_point(aes(group=refval,color=as.ordered(refval)),position = position_jitter(width = 0.2),
             size=1.7) + 
  #ylim(0,3200)+
  labs(y="Estimated\nuncertainties",x=bquote(atop("Mean value of reference", "(" * mJ / cm^3 * ")")))+
  # 添加自定义颜色
  scale_fill_manual(values = rep(custom_colors, length.out = length(unique(enerds$refID)))) +
  scale_color_manual(values = rep(custom_colors, length.out = length(unique(enerds$refID)))) +
  theme_bw(base_size = 35)+
  theme(legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),  # 移除默认边框
        axis.line = element_line(color = "black", linewidth = 0.8),  # 保留坐标轴线
        axis.text = element_text(color="black", size=27, family="sans"),
        axis.title = element_text(family="sans"),
        axis.title.x = element_text(margin = ggplot2::margin(t = -2),hjust = 1), 
        axis.title.y = element_text(margin = ggplot2::margin(r = -0.2)),  # Y轴标题向右移动
        axis.ticks.length=unit(0.35, "cm"),
        axis.ticks = element_line(linewidth = 0.75),
        plot.margin = unit(c(0.3, 0.1, -0.1, 0), "cm"))+
  # 添加次刻度线（需要ggh4x包）
  scale_x_continuous(limits = c(0, 83),
                     guide = ggh4x::guide_axis_minor(),breaks = c(0, 20, 40, 60, 80), #根据数据调整坐标
                     minor_breaks = waiver()) +
  scale_y_continuous(guide = ggh4x::guide_axis_minor(),
                     minor_breaks = waiver()) +
  theme(ggh4x.axis.ticks.length.minor=rel(1/2))+
  scale_x_break(c(11, 29), scales = 4.96)+  # 添加断点（需要ggbreak包），需根据数据范围调整scale（基本上是右侧数值长度比左侧）
  theme(axis.line.x.top = element_blank(),
        axis.ticks.x.top = element_blank(),
        axis.text.x.top = element_blank())
dev.off()

##*********##


# ##** Predicting in virtual space **##
# btovs1<-read.csv("BTO-VS.csv")[,2:8]
# btovs1<-cbind(btovs1,read.csv("BTO-VS.csv")[,c("TA.B", "BD", "EN")])
# #将虚拟空间描述符值对应为训练集中百分数含量计算出值
# btovs1$BD<-btovs1$BD*100
# gc()
# #* prediction models *#
# preo<-krigmo(enerds[,9:12],t,p,64)
# prenu<-krigm(enerds[,9:12],t,p,0,64)
# preno<-krigm(enerds[,9:12],t,p,enerds[,"tv7rfvar5"],64)
prenosa<-krigm(enerds[,9:12],t,p,enerds[,"tv7rfvar5m"],64)
# 
# # Input: the number(/10000) of btovs   Output: number, mean and sigma2 of predictions
# prevs<-function(n){
#   gc()
#   preovs<-predict(preo, btovs1[((n-1)*10000+1):(n*10000), c(8:10)], type = "SK")
#   prenuvs<-predict(prenu, btovs1[((n-1)*10000+1):(n*10000), c(8:10)], type = "SK")
#   prenovs<-predict(preno, btovs1[((n-1)*10000+1):(n*10000), c(8:10)], type = "SK")
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
# preovs_la<-predict(preo, btovs1[1920001:1926974, c(8:10)], type = "SK")
# prenuvs_la<-predict(prenu, btovs1[1920001:1926974, c(8:10)], type = "SK")
# prenovs_la<-predict(preno, btovs1[1920001:1926974, c(8:10)], type = "SK")
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
# save.image("enerd7m.RData")
# 
# 
# #predict some selected samples
# btovss<-read.csv("selected VS.csv")[,2:8]
# btovss<-cbind(btovss,read.csv("selected VS.csv")[,c("TA.B", "BD", "EN")])
# #将虚拟空间描述符值对应为训练集中百分数含量计算出值
# btovss$BD<-btovss$BD*100
# 
# preovss<-predict(preo, btovss[, c(8:10)], type = "SK")
# prenuvss<-predict(prenu, btovss[, c(8:10)], type = "SK")
# prenovss<-predict(preno, btovss[, c(8:10)], type = "SK")
# prevssp<-data.frame(ind=c(1:12), om=preovss[["mean"]], os=preovss[["sd"]],
#                     num=prenuvss[["mean"]], nus=prenuvss[["sd"]],
#                     nom=prenovss[["mean"]], nos=prenovss[["sd"]])
# prevssp$div<-abs(prevssp$om-prevssp$num)+abs(prevssp$num-prevssp$nom)+
#   abs(prevssp$nom-prevssp$om)
# 
# 
# #predict some random selected samples
btovsrs<-read.csv("random selected VS +.csv")[,c(1:7,10)]
btovsrs<-cbind(btovsrs,read.csv("random selected VS +.csv")[,c("TA.B", "BD", "EN")])
#将虚拟空间描述符值对应为训练集中百分数含量计算出值
btovsrs$BD<-btovsrs$BD*100
# 
# library(Rtsne)
# intsnet<-as.matrix(rbind(enerds[,c(10:12)],btovsrs[,c(8:10)]))
# #normalize
# for(i in 1:3){
#   intsnet[,i]<-(intsnet[,i]-min(intsnet[,i])) / (max(intsnet[,i])-min(intsnet[,i]))
# }
# intsnet<-normalize_input(intsnet)
# set.seed(3)
# intsne<-Rtsne(intsnet)
# plot(x=intsne$Y[,1],y=intsne$Y[,2],
#      pch= c(rep(1,length(enerds[,1])),rep(17,length(btovsrs[,1]))),
#      col= c(rep("red",length(enerds[,1])),rep("black",length(btovsrs[,1]))), cex = 1.2,
#      #xlim = c(-20,28), ylim = c(-25,30),
#      xlab="tSNE_1",ylab="tSNE_2",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# 
preovsrs<-predict(preo, btovsrs[, c(9:11)], type = "SK")
prenuvsrs<-predict(prenu, btovsrs[, c(9:11)], type = "SK")
prenovsrs<-predict(preno, btovsrs[, c(9:11)], type = "SK")
prenosavsrs<-predict(prenosa, btovsrs[, c(9:11)], type = "SK")
prevsrsp<-data.frame(btovsrs, om=preovsrs[["mean"]], os=preovsrs[["sd"]],
                     num=prenuvsrs[["mean"]], nus=prenuvsrs[["sd"]],
                     nom=prenovsrs[["mean"]], nos=prenovsrs[["sd"]],
                     nosam=prenosavsrs[["mean"]], nosas=prenosavsrs[["sd"]])
write.table(prevsrsp,"exp compare.csv",append = T,sep = ",")
# 
# library(readxl)
# expcomp <- read_excel("experiments compare.xlsx", sheet = "enerd")
# expcomp<-as.matrix(expcomp)
# plot(rep(expcomp[,4],3),y=c(expcomp[,1],expcomp[,2],expcomp[,3]),
#      pch=c(rep(15,4),rep(16,4),rep(17,4)),
#      col= c(rep("grey",4),rep("red",4),rep("blue",4)), cex = 1.4,
#      xlim = c(30,58), ylim = c(30,58),
#      xlab="Measured ES density (mJ/cm3)", ylab="Predicted ES density (mJ/cm3)",
#      cex.lab=1.3,font.lab=2,mgp=c(2.4,1,0),font.axis=2)
# axis(1,lwd.ticks=2)
# axis(2,lwd.ticks=2)
# box(lwd=2)
# abline(0,1,lwd=2)



##** Fig 3b Predicting in sparse virtual space **##
btovs1<-read.csv("BTO-VS sparse.csv")[,2:8]
btovs1<-cbind(btovs1,read.csv("BTO-VS sparse.csv")[,c("TA.B", "BD", "EN")])
#将虚拟空间描述符值对应为训练集中百分数含量计算出值
btovs1$BD<-btovs1$BD*100
gc()

library(Rtsne)
vsstsned<-as.matrix(btovs1[,c(8:10)])
#normalize
for(i in 1:3){
  vsstsned[,i]<-(vsstsned[,i]-min(vsstsned[,i])) / (max(vsstsned[,i])-min(vsstsned[,i]))
}
vsstsned<-normalize_input(vsstsned)
set.seed(9)
vsstsne<-Rtsne(vsstsned)

vsspo<-predict(preo, btovs1[, c(8:10)], type = "SK")[["mean"]]
vsspno<-predict(preno, btovs1[, c(8:10)], type = "SK")[["mean"]]

#find and plot the extreme value on the top
exind<-order(vsspo, decreasing = TRUE)[1:5]

jpeg("Fig3b 1.jpg", width = 2783, height = 2800, res = 600)
ggplot(data.frame(x=c(vsstsne$Y[,1],vsstsne$Y[exind,1]),y=c(vsstsne$Y[,2],vsstsne$Y[exind,2]),
                  z=c(vsspo,vsspo[exind])), 
       aes(x = x, y = y, color = z)) +    
  geom_point(size = 3) +
  scale_y_continuous(name = "t-SNE 2") +
  scale_x_continuous(name = "t-SNE 1", limits = c(-39,69)) +
  scale_color_gradientn(colors = c("#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"),
                        limits = c(10, 75))+
  theme_bw(base_size = 35)+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),  # 移除默认边框
        axis.line = element_line(color = "black", linewidth = 0.8),  # 保留坐标轴线
        axis.text = element_blank(),
        axis.title = element_text(family="sans"),
        axis.title.y = element_text(margin = ggplot2::margin(r = 0)),  # Y轴标题向右移动
        axis.title.x = element_text(margin = ggplot2::margin(t = 4)),
        axis.ticks.length=unit(0, "cm"),
        legend.title = element_blank(), 
        legend.text = element_text(family="sans",size=27),
        legend.position = c(0.88,0.18),legend.background = element_blank(),
        plot.margin = unit(c(0.3, 0, 0, 0), "cm"))+
  annotate("text", x = 58.2, y = -5, label = expression(paste("mJ", bold("/"), "cm"^3)),
            size = 9, family = "sans")
dev.off()

jpeg("Fig3b 2.jpg", width = 2783, height = 2800, res = 600)
ggplot(data.frame(x=vsstsne$Y[,1],y=vsstsne$Y[,2],z=vsspno), 
       aes(x = x, y = y, color = z)) +    
  geom_point(size = 3) +
  scale_y_continuous(name = "t-SNE 2") +
  scale_x_continuous(name = "t-SNE 1", limits = c(-39,69)) +
  scale_color_gradientn(colors = c("#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"),
                        limits = c(10, 75))+
  theme_bw(base_size = 35)+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),  # 移除默认边框
        axis.line = element_line(color = "black", linewidth = 0.8),  # 保留坐标轴线
        axis.text = element_blank(),
        axis.title = element_text(family="sans"),
        axis.title.y = element_text(margin = ggplot2::margin(r = 0)),  # Y轴标题向右移动
        axis.title.x = element_text(margin = ggplot2::margin(t = 4)),
        axis.ticks.length=unit(0, "cm"),
        legend.title = element_blank(), 
        legend.text = element_text(family="sans",size=27),
        legend.position = c(0.88,0.18),legend.background = element_blank(),
        plot.margin = unit(c(0.3, 0, 0, 0), "cm"))+
  annotate("text", x = 58.2, y = -5, label = expression(paste("mJ", bold("/"), "cm"^3)),
           size = 9, family = "sans")
dev.off()