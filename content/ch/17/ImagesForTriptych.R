x = c(1,1,1,1,1,1,2,2,2,2,3,3,3,3,3,4,4)

xB = c(1,1,1,1,1,1,2,2,2,2,2,3,3,3,4,4,4)

y = c(rep(1, 200), rep(2,150), rep(3, 130), 
      rep(4, 50), rep(5, 10))


png(file="popHist.png", width=600, height=350)
line = par(lwd=2, mar=c(0.2,0.2,0.2,0.2))
hist(y, breaks = c(0.5,1.5,2.5,3.5,4.5,5.5),
axes = FALSE, xlab="", ylab="", main="", xlim = c(0,6))
abline(a=0, b=0, col="black", lwd=2)
dev.off()

png(file="sampHist.png", width=600, height=350)
line = par(lwd=2, mar=c(0.2,0.2,0.2,0.2))
hist(x, breaks = c(0.5,1.5,2.5,3.5,4.5),
axes = FALSE, xlab="", ylab="", main="", xlim = c(0,6))
abline(a=0, b=0, col="black", lwd=2)
dev.off()

png(file="bootsampHist.png", width=600, height=350)
line = par(lwd=2, mar=c(0.2,0.2,0.2,0.2))
hist(xB, breaks = c(0.5,1.5,2.5,3.5,4.5),
     axes = FALSE, xlab="", ylab="", main="", xlim = c(0,6))
abline(a=0, b=0, col="black", lwd=2)
dev.off()

z = rnorm(n=200)
png(file="samplingDist.png", width=600, height=350)
line = par(lwd=2, mar=c(0.2,0.2,0.2,0.2))
hist(z,
axes = FALSE, xlab="", ylab="", main="")
dev.off()

set.seed(42)
zB = rnorm(n=200)
png(file="bootsamplingDist.png", width=600, height=350)
line = par(lwd=2, mar=c(0.2,0.2,0.2,0.2))
hist(zB,
     axes = FALSE, xlab="", ylab="", main="")
dev.off()


x_rank = 1:200

png(file="popHistRank.png", width=600, height=350)
line = par(lwd=2, mar=c(2,0.2,0.2,0.2))
hist(x_rank, breaks = seq(0.5,200.5,5),
     axes = FALSE, xlab="", ylab="", main="", xlim = c(0,200))
axis(1)
#abline(a=0, b=0, col="black", lwd=2)
dev.off()

wiki = read.csv("~/textbook/content/datasets/Wikipedia.csv")
postprod = wiki$postproductivity
rankX = rank(postprod, ties.method = "average")
rankTreat = rankX[wiki$experiment==1]

png(file="sampHistRank.png", width=600, height=350)
line = par(lwd=2, mar=c(2,0.2,0.2,0.2))
hist(rankTreat, breaks = seq(0.5,200.5,5),
     axes = FALSE, xlab="", ylab="", main="", xlim = c(0,200))
axis(1)
#abline(a=0, b=0, col="black", lwd=2)
dev.off()

mean(rankTreat)
sd(rankTreat)

set.seed(42)
zB = replicate(100000, mean(sample(x_rank, 100)))
png(file="samplingDistRank.png", width=600, height=350)
line = par(lwd=2, mar=c(2,0.2,0.2,0.2))
hist(zB, breaks = seq(83.5,117.5,1),
     axes = FALSE, xlab="", ylab="", main="")
axis(1)
dev.off()

