
#dataset = read.table("train_val_test.loss_q3_sov_history_summary",head=T,sep="\t")


options(echo=FALSE) # if you want see commands in output file
args <- commandArgs(trailingOnly = TRUE)
print(args)
print(length(args))
if(length(args) != 2)
{
  stop("The number of parameter is not correct!\n")
}

inputfile = args[1] #train_val_test.loss_q3_sov_history_summary
outputfile = args[2] #"Training_results_summary.jpeg" 



library(ggplot2)
dataset = read.table(inputfile,head=T,sep="\t")
dataset$Data =  factor(dataset$Data,levels=c("Train","Validation","Test"))

gg <- ggplot(dataset, aes(x=Epoch, y=Score, group=Data, color=Data)) +
  geom_line(size=1.5)+ geom_point(size=3)+ theme_bw()+ facet_wrap(~Metric, ncol = 1, scales = "free")+
  labs( x="Epoch", y ="SOV/Q3/Loss/Accuracy")+
  theme(plot.title = element_text(color="#666666", face="bold", size=35))  +
  theme(axis.title = element_text(color="black", face="bold", size=15),axis.text.y=element_text(color="#666666", face="bold",size=15),
        axis.text.x = element_text(angle = 0, hjust = 0.9,vjust=0.3, face="bold",size=10))+ 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank())+
  theme(panel.grid.major = element_line(colour = "black", linetype = "dotted"),panel.grid.minor.y = element_blank())+
  theme(strip.text.x = element_text(size = 12, face="bold"))+
  theme(legend.text = element_text( size=22), legend.key.size = unit(1,"cm"),legend.position="top", 
        legend.direction="horizontal",legend.title = element_blank(),
        legend.key = element_rect(fill="white"), legend.background = element_rect(fill=NA))+guides(col=guide_legend(ncol=3,bycol=T))+
  scale_x_continuous(breaks = seq(0,max(dataset$Epoch),4)) 

gg



jpeg(outputfile , width = 20, height = 15, units = 'in', res = 300)
gg
dev.off()