
# Modeling and Estimation

> Essentially, all models are wrong, but some are useful.
>
> — [George Box, Statistician (1919-2013)](https://www.wikiwand.com/en/George_E._P._Box)

We have covered question formulation, data cleaning, and exploratory data
analysis, the first three steps of the data science lifecycle. We have also
seen that EDA often reveals relationships between variables in our dataset. How
do we decide whether a relationship is real or spurious? How do we use these
relationships to make reliable predictions about the future? To answer these
questions we will need the mathematical tools for modeling and estimation.

A model is an **idealized** representation of a system. For example, if we drop
a ball off the Leaning Tower of Pisa, we expect the ball to drop to the ground
because we have a model of gravity. Our model of gravity also allows us to
predict how long it will take the ball to hit the ground using the laws of
projectile motion.

This model represents our system but is simplistic—for example, it leaves out
the effects of air resistance, the gravitational effects of other celestial
bodies, and the buoyancy of air. Because of these unconsidered factors, our
model will almost always make incorrect predictions in real life! Still, the
simple model of gravity is accurate enough in so many situations that it's
widely used and taught today.

In data science, we often use data to create models to estimate how the world
will behave in the future. How do we choose a model? How do we decide whether
we need a more complicated model? We will explore these questions in this
chapter.

