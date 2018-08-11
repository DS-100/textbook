
## Probability Sampling

Unlike convenience sampling, probability sampling allows us to assign a precise
probability to the event that we draw a particular sample. We will begin by
reviewing simple random samples from Data 8, then introduce two alternative
methods of probability sampling: cluster sampling and stratified sampling.

Suppose we have a population of $6$ individuals. We've given each individual a
different letter from $A-F$.

### Simple Random Sample (SRS)

To take an simple random sample of size $2$ from this population, we can write
each letter $A-F$ on a single index card, place all the cards into a hat, mix
the cards well, and draw $2$ cards without looking. That is, a SRS is sampling
uniformly at random without replacement.

Here are all possible samples of size 2:

\\[
\begin{matrix}
AB & BC & CD & DE & EF\\\\
AC&BD&CE&DF&\\\\
AD&BE&CF&&\\\\
AE&BF&&&\\\\
AF&&&&\\\\
\end{matrix}
\\]

There are $15$ possible samples of size $2$ from our population of $6$. Another
way to count the number of possible samples is:

\\[
\binom{6}{2} = \frac{6!}{2!4!} = 15
\\]

Since in a SRS we sample uniformly at random, each of these $15$ samples are
equally likely to be chosen:

\\[ P(AB) = P(CD) = \ldots = P(DF) = \frac{1}{15} \\]

We can also use this chance mechanism to answer other questions about the
composition of the sample. For example:

\\[ P(A \text{ in sample}) = \frac{5}{15} = \frac{1}{3}\\]

Since $5$ out of $15$ of the possible samples listed above contain $A$.

By symmetry, we can say:

\\[ P(A \text{ in sample}) = P(F \text{ in sample}) = \frac{1}{3} \\]

Another way of computing $ P(A \text{ in sample}) $ is to recognize that for
$A$ to be in the sample, we either need to draw it as the first marble or as
the second marble.

\\[
\begin{aligned}
P(A \text{ in sample}) &= P(\text{A is first or A is second}) \\\\
&= P(\text{A is first}) + P(\text{A is second}) \\\\
&= \frac{1}{6} \cdot \frac{5}{5} + \frac{5}{6} \cdot \frac{1}{5} \\\\
&= \frac{1}{3} \\\\
\end{aligned}
\\]

### Cluster Sampling

In cluster sampling, we divide the population into clusters. Then, we use SRS
to select clusters at random instead of individuals.

As an example, suppose we take our population of $6$ individuals and we pair
each of them up: $ (A,B)\quad (C,D) \quad (E,F) $ to form 3 clusters of 2
individuals. Then, we use SRS to select one cluster to produce a sample of size 2.

As before, we can compute the probability that $A$ is in our sample:

\\[ P(A \text{ in sample}) = P(AB \text{ drawn}) = \frac{1}{3} \\]

Similarly the probability that any particular person appears in our sample is
$\frac{1}{3}$. Note that this is the same as our SRS. However, we see
differences when we look at the samples themselves. For example, in a SRS the
chance of getting $AB$ is the same as the chance of getting $AC$:
$\frac{1}{15}$. However, with this cluster sampling scheme:

\\[
\begin{aligned}
P(AB) &= \frac{1}{3} \\\\
P(AC) &= 0
\end{aligned}
\\]

Since $A$ and $C$ can never appear in the same sample if we only select one
cluster.

Cluster sampling is still probability sampling since we can assign a
probability to each potential sample. However, the resulting probabilities are
different than using a SRS depending on how the population is clustered.

Why use cluster sampling? Cluster sampling is most useful because it makes
sample collection easier. For example, it is much easier to poll towns of 100
people each than to poll thousands of people distributed across the entire US.
This is the reason why many polling agencies today use forms of cluster
sampling to conduct surveys.

The main downside of cluster sampling is that it tends to produce greater
variation in estimation. This typically means that we take larger samples when
using cluster sampling. Note that the reality is much more complicated than
this, but we will leave the details to a future course on sampling techniques.

### Stratified Sampling

In stratified sampling, we divide the population into strata, and then produce
one simple random sample per strata. In both cluster sampling and stratified
sampling we split the population into groups; in cluster sampling we use a
single SRS to select groups whereas in stratified sampling we use multiple
SRS's, one for each group.

We can divide our population of 6 individuals into the following strata:

\\[
\begin{matrix}
\text{Strata 1:} & \{A,B,C,D\} \\\\
\text{Strata 2:} & \{E,F\}
\end{matrix}
\\]

We use an SRS to select one individual from each strata to produce a sample of
size $2$. This gives us the following possible samples:

\\[
\begin{aligned}
(A,E)\quad (A,F) \quad (B,E) \quad (B,F) \quad (C,E) \quad (C,F)
\quad (D,E) \quad (D,F)
\end{aligned}
\\]

Again, we can compute the probability that $A$ is in our sample:

\\[
\begin{aligned}
P(A \text{ in sample}) &= P(A \text{ selected from Strata 1}) \\\\
&= \frac{1}{4}
\end{aligned}
\\]

However:

\\[
\begin{aligned}
P(AB) &= 0
\end{aligned}
\\]

since $A$ and $B$ cannot appear in the same sample.

Like cluster sampling, stratified sampling is also a probability sampling
method that produces different probabilities depending on the stratification of
the population. Note that like this example, the strata do not have to be the
same size. For example, we can stratify the US by occupation, then take samples
from each strata of size proportional to the distribution of occupations in the
US — if only 0.01% of people in the US are statisticians, we can ensure that
0.01% of our sample will be composed of statisticians. A simple random sample
might miss the poor statisticians altogether!

As you may have figured out, stratified sampling can perhaps be called the
proper way to conduct quota sampling. It allows the researcher to ensure that
subgroups of the population are well-represented in the sample without using
human judgement to select the individuals in the sample. This can often result
in less variation in estimation. However, stratified sampling is sometimes more
difficult to accomplish because we sometimes don't know how large each strata
is. In the previous example we have the advantage of the US census, but other
times we are not so fortunate.

## Why Probability Sampling?

As we have seen in Data 8, probability sampling enables us to quantify our
uncertainty about an estimation or prediction. It is only through this
precision that we can conduct inference and hypothesis testing. Be wary when
anyone gives you p-values or confidence levels without a proper explanation of
their sampling techniques.

Now that we understand probability sampling let us see how the humble SRS
compares against "big data".

