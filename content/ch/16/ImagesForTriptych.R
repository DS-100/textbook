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
