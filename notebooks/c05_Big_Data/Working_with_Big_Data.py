# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Introduction to Dask
# There are many occasions when we have to work with datasets that are so big we can't just load all of it into memory. If we have to work with such a data, what is the solution? One of the libraries in python for this type of problems is __Dask__. Dask is a library for parallel computing. It helps us perform common pandas and numpy opperations on large datasets. In this tutorial we will learn about some of the features of Dask and how it can help us.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask
import dask.dataframe as dd
import os
import psutil

# For the beginning to see the difference in the performance of Dask vs Pandas, let's read a 70 MB csv file with both libraries. Of course 70 MB is not considered a large file and we can easily fit it into memory, but it is large enough to see the advantage of using Dask.

path = "../../data/processed/MNIST/train.csv"


# We are also defining a function to report memory usage, so we can see how each method affect the memory.

def memory_usage():
    """String with current memory usage in MB. Requires `psutil` package."""
    pid = os.getpid()
    mem_bytes = psutil.Process(pid).memory_info().rss
    print(f"[Process {pid} uses {mem_bytes/1024/1024:.1f}MB]")
    return mem_bytes / 1024 / 1024


memory_usage()

# You should be able to see the amount of used memory.<br>
# Now let's load the data with pandas and see how long it takes to load and how much memory it takes.

# %%time
df1 = pd.read_csv(path)

memory_usage()

# The memory usage has gone up by about 250 MB.<br>
# Now do the same with Dask.

# %%time
df2=dd.read_csv(path)

memory_usage()

# Dask read the file in a raction of a second and used only about 3 MB of memory. How is that possible?<br> It's because Dask dosn't load the data into memory. The data is still on the disk. It only reads the data when it needs to perform calculations.
#

# Now, let's calculate the mean for the first 100 columns.

# %%time
df1.iloc[:,:100].mean()

# As you can see pandas does the calculations in a fraction of a second. <br>
# Let's try Dask:

# %%time
avg = df2.iloc[:,100:200].mean()

# Dask is also did the operation very quickly. Let's have a look at the output.

avg

# It's not returning any numbers. What is happening?<br>
# The reason is Dask has not calculated the result yet. It only creates a dependency graph (also called Directed Acyclec Graph - DAG), which is basically how the calculations will take place. We need to execute the graph to see the result.

# %%time
avg.compute()

# Now we can see the results. Also, we can see that this step is the most time consuming step of all. It's because this is where Dask actually goes to disk and reads the data. If you add up the time for all the steps (reading the file and performing calculations) you will see that both take almost the same amount of time to do the operation, with pandas being slightly faster. This shows that Dask is not doing any magic. It's doing the same steps but it's doing it without using as much memory. However, Dask can perform operations in parallel using multiple cpu cores and even multiple machines. 

# We mentioned that Dask creates a dependency graph before doing the calculations. We can have a look at this graph and see how it is taking place:

# <div class='alert alert-danger'>To see the graph you need a library called <b>GraphViz</b>. If this library is not installed on your system you will not be able to see the graph.</div>

# Here is one we prepared earlier:
#
# ![](img/dask_graphviz.png)

avg.visualize()

# You can also create a progress bar for each computation:

avg = df2.iloc[:, 100:200].mean()
task = avg.mean()

from dask.diagnostics import ProgressBar

with ProgressBar():
    task.compute()

# Dask dataframe is very similar to pandas dataframe. Even though the functionalities are limited but you will find many methods from pandas dataframe in Dask as well.

task = df2.rolling(window=10).mean().max()

with ProgressBar():
    ma_max = task.compute()

ma_max

task = df2[["label", "pixel100", "pixel300", "pixel500"]].groupby("label").mean()

with ProgressBar():
    result = task.compute()

result

# ### Exercise
# Use Dask dataframe of MNIST and follow these steps:
# 1. Add a column to the dataframe which contains sum of all the pixels
# 2. Use Groupby and find the average of __sum__ column for each number (label)

# +
# Code Here
# -

# <details><summary>Solution</summary>
#
# ```Python
#     df2['sum']=df2.values[:,1:].sum(axis=1)
#     task = df2[['label','sum']].groupby('label').mean()
#     with ProgressBar():
#         result=task.compute() 
#     print(result)
# ```
#     
# </details>

# ## Dask Array
# Dask is not just used to replace pandas. There are also multiple numpy functions which can be replaced by Dask. Dask array is Dask equivalent of a numpy array. By doing so, we can perform the computations in parallel and get the results faster.

from dask import array

big_array = array.random.normal(size=(10000000, 100))

big_array

# This data takes 8 GB if we wanted to store it in RAM. But Dask only generates the numbers in chunks when it needs them. So at each steps it has to deal with a chunk which is 125 MB in this case.

task = (big_array * big_array).mean(axis=1)
with ProgressBar():
    res = task.compute()

# We can set the chunk size:

big_array = array.random.normal(size=(10000000, 100), chunks=(2 ** 19, 100))
big_array

task = (big_array * big_array).mean(axis=1)
with ProgressBar():
    res = task.compute()

# We can also apply most of common numpy functions to the array.

task = np.sin(big_array).mean(axis=0)
with ProgressBar():
    res = task.compute()

res

# ### Exercise
# Create two Dask random arrays of size 10,000,000-by-100. Find the difference between the two and pass it to `array.linalg.norm` using argument `axis=1`. Calculate the result and create a histogram of it.

from matplotlib.pyplot import hist

# +
# Code Here
# -

# <details><summary>Solution</summary>
#
# ```Python
#     x1 = array.random.random(size=(10000000,100))
#     x2 = array.random.random(size=(10000000,100))
#     y = x2-x1
#     d = array.linalg.norm(y,axis=1)
#     with ProgressBar():
#         result = d.compute()
#     hist(result,bins=100);
# ```
#     
# </details>

# ## Delayed
# Dask delayed is a method for parallelising code where you can't write your code directly as dataframe or array operation. `Dask.delayed` is an easy-to-use tool to quickly parallelise these tasks.

# Consider the following functions. The first one takes an input, waits for one second and returns the value. The second function takes two inputs, waits for one second and returns the sum. We are using these functions to represent tasks that are time consuming.

# +
from time import sleep


def task1(x):
    sleep(1)
    return x


def task2(x, y):
    sleep(1)
    return x + y


# -

# Now, if we pass two values separately into the first function and then pass the results into the second function, we will have the following code:

# %%time
x1 = task1(1)
x2 = task1(2)
y = task2(x1,x2)

# Since each of these functions are taking one second; therefore, the entire block takes three seconds. But the calculation for `x1` is totally independent of the calculation for `x2`. If we were able to do these operation simultaneously we could save time. This is where `Dask.delayed` comes into play. We need to convert the functions into `delayed` functions so Dask can handle parallelisation.

task1_delayed = dask.delayed(task1)
task2_delayed = dask.delayed(task2)

# And now insteam of the original function we use the delayed functions:

# %%time
x1 = task1_delayed(1)
x2 = task1_delayed(2)
y = task2_delayed(x1,x2)

# %%time
y.compute()


# And we saved one second! `x1` and `x2` where calculated in parallel, and then `y` was calculated using `x1` and `x2`.

# We can directly create delayed functions using `dask.delayed` decorator.

# +
@dask.delayed
def task1(x):
    sleep(1)
    return x


@dask.delayed
def task2(x, y):
    sleep(1)
    return x + y


# -

# %%time
x1 = task1(1)
x2 = task1(2)
y = task2(x1,x2)
y.compute()

# # Introduction to Numba

# ## What is Numba?
#
# Numba is a **just-in-time**, **type-specializing**, **function compiler** for accelerating **numerically-focused** Python.  That's a long list, so let's break down those terms:
#
#  * **function compiler**: Numba compiles Python functions, not entire applications, and not parts of functions.  Numba does not replace your Python interpreter, but is just another Python module that can turn a function into a (usually) faster function. 
#  * **type-specializing**: Numba speeds up your function by generating a specialized implementation for the specific data types you are using.  Python functions are designed to operate on generic data types, which makes them very flexible, but also very slow.  In practice, you only will call a function with a small number of argument types, so Numba will generate a fast implementation for each set of types.
#  * **just-in-time**: Numba translates functions when they are first called.  This ensures the compiler knows what argument types you will be using.  This also allows Numba to be used interactively in a Jupyter notebook just as easily as a traditional application
#  * **numerically-focused**: Currently, Numba is focused on numerical data types, like `int`, `float`, and `complex`.  There is very limited string processing support, and many string use cases are not going to work well on the GPU.  To get best results with Numba, you will likely be using NumPy arrays.
#

# ### First Steps
#
# Let's write our first Numba function and compile it for the **CPU**.  The Numba compiler is typically enabled by applying a *decorator* to a Python function.  Decorators are functions that transform Python functions.  Here we will use the CPU compilation decorator:

# +
from numba import jit
import math


@jit
def hypot(x, y):
    # Implementation from https://en.wikipedia.org/wiki/Hypot
    x = abs(x)
    y = abs(y)
    t = min(x, y)
    x = max(x, y)
    t = t / x
    return x * math.sqrt(1 + t * t)


# -

# The above code is equivalent to writing:
# ``` python
# def hypot(x, y):
#     x = abs(x);
#     y = abs(y);
#     t = min(x, y);
#     x = max(x, y);
#     t = t / x;
#     return x * math.sqrt(1+t*t)
#     
# hypot = jit(hypot)
# ```
# This means that the Numba compiler is just a function you can call whenever you want!
#
# Let's try out our hypotenuse calculation:

hypot(3.0, 4.0)

# The first time we call `hypot`, the compiler is triggered and compiles a machine code implementation for float inputs.  Numba also saves the original Python implementation of the function in the `.py_func` attribute, so we can call the original Python code to make sure we get the same answer:

hypot.py_func(3.0, 4.0)

# ### Benchmarking
#
# An important part of using Numba is measuring the performance of your new code.  Let's see if we actually sped anything up.  The easiest way to do this in the Jupyter notebook is to use the `%timeit` magic function.  Let's first measure the speed of the original Python:

# %timeit hypot.py_func(3.0, 4.0)

# The `%timeit` magic runs the statement many times to get an accurate estimate of the run time.

# %timeit hypot(3.0, 4.0)

# Numba did a pretty good job with this function.  It's 3x faster than the pure Python version.
#
# Of course, the `hypot` function is already present in the Python module:

# %timeit math.hypot(3.0, 4.0)

# Python's built-in is even faster than Numba!  This is because Numba does introduce some overhead to each function call that is larger than the function call overhead of Python itself.  Extremely fast functions (like the above one) will be hurt by this.
#
# (However, if you call one Numba function from another one, there is very little function overhead, sometimes even zero if the compiler inlines the function into the other one.)

# ### How does Numba work?
#
# The first time we called our Numba-wrapped `hypot` function, the following process was initiated:
#
# ![Numba Flowchart](img/numba_flowchart.png "The compilation process")
#
# We can see the result of type inference by using the `.inspect_types()` method, which prints an annotated version of the source code:

hypot.inspect_types()


# Note that Numba's type names tend to mirror the NumPy type names, so a Python `float` is a `float64` (also called "double precision" in other languages).  Taking a look at the data types can sometimes be important in GPU code because the performance of `float32` and `float64` computations will be very different on CUDA devices.  An accidental upcast can dramatically slow down a function.

# ### When Things Go Wrong
#
# Numba cannot compile all Python code.  Some functions don't have a Numba-translation, and some kinds of Python types can't be efficiently compiled at all (yet).  For example, Numba does not support `FrozenSet` (as of this tutorial):

# +
@jit
def cannot_compile(x):
    return "a" in x


cannot_compile(frozenset(("a", "b", "c")))


# -

# Wait, what happened??  By default, Numba will fall back to a mode, called "object mode," which does not do type-specialization.  Object mode exists to enable other Numba functionality, but in many cases, you want Numba to tell you if type inference fails.  You can force "nopython mode" (the other compilation mode) by passing arguments to the decorator:

# +
@jit(nopython=True)
def cannot_compile(x):
    return "a" in x


cannot_compile(frozenset(("a", "b", "c")))


# -

# Now we get an exception when Numba tries to compile the function, with an error that says:
# ```
# - argument 0: cannot determine Numba type of <class 'frozenset'>
# ```
# which is the underlying problem. Numba doesn't know about frozenset. There are classes that we use regularly in our code but they might not be defined in Numba. An example of a common class that you cannot use in Numba is pandas data frames. <br>Now the question is: what does Numba support? Some of the types/classes that are supported by Numba are listed below:
# * Numbers (integers, floats, etc)
# * Numpy arrays
# * Strings
# * Lists and tuples (note that a list/tuple of numbers or strings is supported but a list of lists is not)

# So, if we want the last example to be compiled successfully by Numba jit, we need to use a tuple or a list.

# +
@jit(nopython=True)
def can_compile(x):
    return "a" in x


can_compile(("a", "b", "c"))
# -

# ### Exercise
# Gregory–Leibniz infinite series converges to $\pi$:
# $$\pi = \frac{4}{1} - \frac{4}{3} + \frac{4}{5} - \frac{4}{7} + \frac{4}{9} - \frac{4}{11} + \frac{4}{13} - \cdots$$
#
# Write a Numba function which calculates the sum of first $n$ terms in this series. Then test its speed agains normal Python function for $ n = 1000000$.

# +
# Code Here
# -

# <details><summary>Solution</summary>
#
# ```Python
#     @jit
#     def gl_pi(n):
#         pi = 0
#         for i in range(n):
#             if i%2 ==0:
#                 pi += 4/(2*i+1)
#             else:
#                 pi -= 4/(2*i+1)
#         return pi 
# ```
#
# <b>Numba function speed test:</b>
# ```Python
#     %timeit gl_pi(1000000) 
# ```
#     
# <b>Normal Python function speed test:</b>
# ```Python
#     %timeit gl_pi.py_func(1000000) 
# ```
#     
# </details>

# # References
# The following sources where used for creation of this notebook:
# - https://github.com/NCAR/ncar-python-tutorial
# - https://github.com/stevesimmons/pydata-ams2017-pandas-and-dask-from-the-inside
# - https://github.com/numba/euroscipy2019-numba

# # Further Reading
# - [Dask documentation](https://docs.dask.org/en/latest/)
# - [Why Dask?](https://docs.dask.org/en/latest/why.html)
# - [Distributed Machine Learning with Python and Dask](https://towardsdatascience.com/distributed-machine-learning-with-python-and-dask-2d6bae91a726)
# - [Speeding up your Algorithms — Dask](https://towardsdatascience.com/speeding-up-your-algorithms-part-4-dask-7c6ed79994ef)
# - [Speeding Up your Algorithms — Numba](https://towardsdatascience.com/speed-up-your-algorithms-part-2-numba-293e554c5cc1)
