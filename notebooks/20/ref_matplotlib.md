
## matplotlib

### Types of Plots

| Function                                                                                   | Chapter            | Description                                                                                     |
| ------------------------------------------------------------------------------------------ | ------------------ | ----------------------------------------------------------------------------------------------- |
| [`plt.scatter(x, y)`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html)   | Data Visualization | Creates a scatter plot of the variable x against the variable y                                 |
| [`plt.plot(x, y)`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html)         | Data Visualization | Creates a line plot of the variable x against the variable y                                    |
| [`plt.hist(x, bins=None)`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) | Data Visualization | Creates a histogram of x. Bins argument can be an integer or sequence                           |
| [`plt.bar(x, height)`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.bar.html)      | Data Visualization | Creates a bar plot. `x` specifies x-coordinates of bars, `height` specifies heights of the bars |
| [`plt.axvline(x=0)`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axvline.html)    | Data Visualization | Creates a vertical line at the x value specified                                                |
| [`plt.axhline(y=0)`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axhline.html)    | Data Visualization | Creates a horizontal line at the y value specified                                              |

### Plot additions

| Function                                                                                         | Chapter            | Description                                                        |
| ------------------------------------------------------------------------------------------------ | ------------------ | ------------------------------------------------------------------ |
| [`%matplotlib inline`](http://ipython.readthedocs.io/en/stable/interactive/plotting.html)        | Data Visualization | Causes output of plotting commands to be displayed inline          |
| [`plt.figure(figsize=(3, 5))`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figure.html) | Data Visualization | Creates a figure with a width of 3 inches and a height of 5 inches |
| [`plt.xlim(xmin, xmax)`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.xlim.html)         | Data Visualization | Sets the x-limits of the current axes                              |
| [`plt.xlabel(label)`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.xlabel.html)          | Data Visualization | Sets an x-axis label of the current axes                           |
| [`plt.title(label)`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.title.html)            | Data Visualization | Sets a title of the current axes                                   |
| [`plt.legend(x, height)`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html)      | Data Visualization | Places a legend on the axes                                        |
| [`fig, ax = plt.subplots()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html) | Data Visualization | Creates a figure and set of subplots                               |
| [`plt.show()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.show.html)                   | Data Visualization | Displays a figure                                                  |

