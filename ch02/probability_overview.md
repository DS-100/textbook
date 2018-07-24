
## Probability Overview

Many of the concepts that we touch upon in the course contain a certain degree of randomness. Probability is a tool that we can use to analyze these random phenomena.

Let's say we have a random experiment in which we toss two coins. The **sample space** $\Omega$ consists of all the possible outcomes of an experiment. In this experiment, our sample space consists of the outcomes: $\Omega = \{HH, HT, TH, TT\}$.

An **event** is any subset of the sample space to which a probability can be assigned. For example, "getting one tails and one heads" would be an event for the coin toss experiment mentioned above. The outcomes that make up the event are $\{HT, TH\}$, and the probability of this event occurring is $\frac{1}{2}$.

## Probability of Events

Let us consider a sample space in which **every outcome is equally likely to occur**. Then, the probability of an event occurring can be calculated as the ratio of the number of outcomes in the event to the total number of outcomes in the sample space. $ P(\text{an event happens}) = \frac{\text{# of outcomes that make the event happen}}{\text{# of total outcomes}} $.    
In the coin toss experiment, the events $\{HH, HT, TH, TT\}$ are all equally likely to occur, so to calculate the probability of getting one tails and one heads, we can count the total number of outcomes in the sample space (2) and divide by the total number of outcomes (4) : $P(\text{one heads and one tails}) = \frac{2}{4} = \frac{1}{2} $

There are a few important properties of probabilities of events:

1. The probability of an event is between 0 and 1: $0 \leq P(E) \leq 1$
2. The probabilities of all the outcomes in a sample space sum to 1: $\sum_{\omega \in \Omega} P(\omega) = 1$

## Addition Rule
Sometimes, we want to calculate the probabilities involving the occurrence of one event OR another. To do so, we add the probabilities of the events.

When two events $A$ and $B$ are disjoint (they don't share any common outcomes), the probability that either event $A$ or $B$ ($A \cup B$) occurs is the sum of the probability of each event. $P(A \cup B) = P(A) + P(B) \text{ if } A \cap B = \varnothing$. For instance, the events "getting two heads" and "getting two tails" with two coin tosses are disjoint. Therefore, the probability of the combined event "getting two heads or two tails" is equal to the sum of the individual events. $P(\text{getting two heads or two tails}) = P(\text{getting two heads}) + P(\text{getting two tails}) = \frac{1}{4} + \frac{1}{4} = \frac{1}{2}$

When two events $A$ and $B$ aren't disjoint (they share common outcomes), the probability that either event $A$ or $B$ ($A \cup B$) occurs is the sum of the probability of each event minus the probability of both events occurring ($A \cap B$). $P(A \cup B) = P(A) + P(B) - P(A \cap B) \text{ if } A \cap B \neq \varnothing$. The intuition behind this can better be visualized through a Venn diagram.

![probability_intro_venn](../assets/probability_intro_venn.png)

When we have two events $A$ and $B$ that have common outcomes and we want to find the probability that either event happens, we want the area inside the circles. However, if we add together the probabilities of $A$ occurring and $B$ occurring, we count the intersection of the circles, $ P(A \cap B) $, twice. That's why we need so subtract the intersection once.

For example, let's say we wish to find the probability of the events "getting two heads" or "getting at least 1 head" in two coin tosses. We can see that there are 3 outcomes in our sample space that have at least one head and 1 outcome that has two heads. If we were to add these probabilities together, we would have $ \frac{3}{4} + \frac{1}{4} = 1$ . Clearly, this result is wrong because we could get two tails, which isn't an outcome in either event. We did not account for the fact that the two events share an outcome; getting $ \{HH\} $ falls under both events, so we need to subtract the probability of that occurring. The correct probability of both events occurring would be $ \frac{3}{4} + \frac{1}{4} - \frac{1}{4} = \frac{3}{4} $.

## Multiplication Rule

Other times, we wish to calculate probabilities involving the occurrence of one event AND another. To do so, we multiply the probabilities of the events.

In some cases, the occurrence of the first event does not impact the probability of the second. We call these **independent events**. Suppose we want to calculate the probability of "getting heads on a fair coin toss" and "getting a six on a fair dice roll". We can take into account that getting heads on a coin toss doesn't have any effect on the result of the dice roll. $ \frac{1}{2} $ the time we will get heads, and we will only get a six $ \frac{1}{6} $ of the time. To find a fraction of a fraction, we can multiply them together. $P(\text{getting heads and a 6}) = \frac{1}{2} \* \frac{1}{6} = \frac{1}{12}$. In general for independent events $A$ and $B$, $P(A \cap B) = P(A)\*P(B)$.

In other cases, the occurrence of the first event does impact the probability of the second. We call these **dependent events**. For example, let's say we have a bag of 10 marbles, 5 blue marbles, and 5 red marbles. Consider the event of picking two marbles out of the bag without replacement. What is the probability that both marbles are blue? Half the marbles are blue at first, so the probability of the first marble being blue is $ \frac{5}{10} $. However, the probability of getting a blue marble has changed because the number of blue marbles and the total number of marbles both decreased by 1. The second time, there is only a $ \frac{4}{9} $ chance of picking a blue marble. Therefore, the probability that both marbles are blue is $ \frac{5}{10} * \frac{4}{9} = \frac{2}{9} $.

## General Multiplication Rule and Conditional Probability
In the previous example, we implicitly used the tools of **conditional probability** to calculate the probability of occurrence of two dependent events. Formally, the conditional probability of an event $A$ given an event $B$ is:

\\[ P(A \; \vert \; B) = \frac{P(A \cap B)}{P(B)} \\]

A closer look into this definition indicates that if we know that event $B$ occurred, we must restrict our sample space from $\Omega$ to the outcomes in $B$ since outcomes outside of $B$ cannot possibly occur. Thus, to calculate the conditional probability of $A$ given $B$, we only need to consider the probability of the outcomes in $A \cap B$ and scale appropriately by $P(B)$.

Going back to the picking marbles without replacement example, we define $A$ as the event that the first marble is blue and $B$ as the event that the second marble is blue. Then, $P(A) = \frac{5}{10}$, and $P(B \; \vert \; A) = \frac{4}{9}$. With the definition of conditional probability, we conclude that $P(A \cap B) = P(A) \* P(B \; \vert \; A) = \frac{5}{10} \* \frac{4}{9} = \frac{2}{9}$.

With conditional probability, we can develop a more formal intuition of independent events. We saw earlier that two events are independent if and only if
$P(A) \* P(B) = P(A \cap B)$. Equivalently, some algebraic manipulation with the definition of conditional probability yields that $P(A \; \vert \; B) = P(A)$ and $P(B \; \vert \; A) = P(A)$, which indicates that the one event's occurrence has no effect on the other event's occurrence.
