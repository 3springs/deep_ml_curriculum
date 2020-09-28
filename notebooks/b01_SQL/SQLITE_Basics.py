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
#     display_name: py37_pytorch
#     language: python
#     name: conda-env-py37_pytorch-py
# ---



# # SQLite Basics
# This notebook includes basic instructions about how to access a database, and read/write data from/to it.<br>
# Here, we are using **SQLite** which is light-weight database management system. To communicate with a database in SQLite we use Structured Query Language a.k.a **SQL**. We use SQL to send a querry to the database manager. Then the database manager processes the query and sends the result back. The querry is the instructions that tells database manager what to do.
# In python we can use **sqlite3** package for this putpose.

import sqlite3

# ## Connecting to a database
# Let's start by connecting to a database. We can do that using `sqlite.connect` which returns a connection object.

# create a database connection to the SQLite database specified by the db_file:

db_file = "Sales.db"
conn = sqlite3.connect(db_file)

# If you didn't get any error, it means you have successfully connected to the database. <br>
# **Note:** If the file doesn't exist sqlite will create an empty database with the given name.<br><br>
# Now we can start executing a query. Let's start by the simplest query:<br>
# <code>
#     SELECT * FROM {name of table}
# </code>
# This query returns all the rows and columns of a specific table in the database. To use this query first we need to know the names of the tables in the database and in general the structure of the database. 
# <img src="DB.png" alt="Image not found" width="50%">
# Let's get a list of all the customers

query = """
SELECT * FROM Customers;

"""

# **Note:** It is customary to use SQL key word all in caps but that is not necessary. "SELECT" and "select" do the same job.<br>
# <br>Now that we have the query, let's send it to the database manager. To do that, we need to create a cursor object and ask the cursor to execute the query and return all the information.

cur = conn.cursor()
cur.execute(query)
output = cur.fetchall()
print(output)

# The output is a list of rows in the table. Each row is returned in form of a tuple. Let's have a look at the first row:

output[0]


# Let's turn it into a function to avoid writing the same lines of code over and over. This function executes a query and prints all the rows:

def execute_query(conn, query):
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    for i, row in enumerate(rows):
        print(f"{i+1}. {row}")


# ## SQL clauses
# ### SELECT
# SELECT is the most common clause in SQL. Using SELECT you can specify which columns should be returned. 
#

# Returns a list of first and last names of all customers

query = """
SELECT FirstName, Surname FROM Customers;
"""
execute_query(conn, query)

# Returns a list of first name, last name, and email for all customers

query = """
SELECT FirstName, Surname, Email FROM Customers;
"""
execute_query(conn, query)

# We need to specify which columns we want to be returned. If we need all the columns we use `*`.

# One issue you might have is not knowing what the names of the columns are. So how do we get the name of the columns? There are a few ways to do that. Probably, the simplest one is using `.description` property of the cursor. When you execute a query you can find the names of the columns in `.description` property of the cursor. To have a full list of columns, use `SELECT *`.

query = """
SELECT * from Customers
"""

cur = conn.cursor()
cur.execute(query)
cur.description

# **Note:** you don't need to use `.fetchall()` method. Just executing the query will give the names of the columns.

# Another way is using the query below which will return names of all the tables, names of columns in each table, and the type of data in each column.

query = "SELECT * FROM sqlite_master WHERE type = 'table'"
cur = conn.cursor()
cur.execute(query)
output = cur.fetchall()
for row in output:
    for c in row:
        print(c)
    print("-" * 50)

# ### ORDER BY
# By default the output of a query has no specific order. `ORDER BY` clause is used to sort data in the output. You can sort the data based on one or more columns. We can also specify whether we want the sorted output to be ascending or descending. This can be achieved by adding the following code to the query:<br>
# `ORDER BY {name of column} {ASC or DESC}`<br>
# ASC: Ascending<br>
# DESC: Descending
#
#

# The query below returns a list of first name, last name, Country and email for all customers.

query = """
SELECT FirstName, Surname, Country, Email FROM Customers
ORDER BY Country ASC,Surname ASC, FirstName ASC;
"""
execute_query(conn, query)

# **Note:** In the query above, the output is ordered by Country and then last and first name.

# ### DISTINCT
# This clause is used to remove duplicates in the result. It appears immediately after `SELECT` followed by a column name or a list of columns. If only one column is specified only the values on that column is used to identify duplicates. If multiple columns are specified, the combination of them will be used to identify duplicates.
# <code>
# SELECT DISTINCT {column}
# FROM {table};
# </code>

query = """
SELECT DISTINCT City
FROM Customers
ORDER BY City;
"""
execute_query(conn, query)

# ### WHERE
# We can add condition to the query sing `WHERE` clause.
# <code>
# SELECT column_list
# FROM table
# WHERE search_condition;
# </code>
# <br>
# SQLite uses the following steps:
# 1. Check the table in the FROM clause.
# 2. Evaluate the conditions in the WHERE clause to get the rows that met these conditions.
# 3. Make the final result set based on the rows in the previous step with columns in the SELECT clause.
#
# The search condition in the WHERE has the following form:<br>
# left_expression  COMPARISON_OPERATOR  right_expression<br><br>
# e.g. 
# <code> 
# WHERE column_1>5
#     
# WHERE column_2 IN (1,2,3)
#
# </code>
# **List of comparison operators:**
# <table>
# <thead>
#   <tr>
#     <th>Operator</th>
#     <th>Meaning</th>
#   </tr>
# </thead>
# <tbody>
#   <tr>
#     <td>=</td>
#     <td>Equal to</td>
#   </tr>
#   <tr>
#     <td>&lt;&gt; or !=</td>
#     <td>Not equal to</td>
#   </tr>
#   <tr>
#     <td>&lt;</td>
#     <td>Less than</td>
#   </tr>
#   <tr>
#     <td>&gt;</td>
#     <td>Greater than</td>
#
#   </tr>
#   <tr>
#     <td>&lt;=</td>
#     <td>Less than or equal to</td>
#
#   </tr>
#   <tr>
#     <td>&gt;=</td>
#     <td>Greater than or equal to</td>
#
#   </tr>
# </tbody>
# </table>
#
# **List of logical operators:**
# <table>
# <thead>
#   <tr>
#     <th>Operator</th>
#     <th>Meaning</th>
#   </tr>
# </thead>
# <tbody>
#   <tr>
#     <td>ALL</td>
#     <td>returns 1 if all expressions are 1.</td>
#   </tr>
#   <tr>
#     <td>AND</td>
#     <td>returns 1 if both expressions are 1, and 0 if one of the expressions is 0.</td>
#   </tr>
#   <tr>
#     <td>ANY</td>
#     <td>returns 1 if any one of a set of comparisons is 1.</td>
#   </tr>
#   <tr>
#     <td>BETWEEN</td>
#     <td>returns 1 if a value is within a range.</td>
#   </tr>
#   <tr>
#     <td>EXISTS</td>
#     <td>returns 1 if a subquery contains any rows.</td>
#   </tr>
#   <tr>
#     <td>IN</td>
#     <td>returns 1 if a value is in a list of values.</td>
#   </tr>
#   <tr>
#     <td>LIKE</td>
#     <td>returns 1 if a value matches a pattern</td>
#   </tr>
#   <tr>
#     <td>NOT</td>
#     <td>reverses the value of other operators such as NOT EXISTS, NOT IN, NOT BETWEEN, etc.</td>
#   </tr>
#   <tr>
#     <td>OR</td>
#     <td>returns true if either expression is 1</td>
#   </tr>
# </tbody>
# </table>



# Let's create a list of all the Products having StockCode, Category, Weight, and UnitPrice in the result and sorted from high to low by the price.

query = """
SELECT StockCode,Category,Weight, UnitPrice FROM Products
ORDER BY UnitPrice DESC;
"""
execute_query(conn, query)

# Now, let's do the same thing but this time only the products with price above $15.

query = """
SELECT StockCode,Category,Weight, UnitPrice FROM Products
WHERE UnitPrice>15
ORDER BY UnitPrice DESC;
"""
execute_query(conn, query)

# The database puts `NULL` in the table where there is no data. When this data is transfered to Python `NULL` is converted to `None`.<br>
# Let's get a list of customers without a Fax number.

query = """
SELECT Surname,Country,Fax FROM Customers
WHERE Fax IS NULL;
"""
execute_query(conn, query)

# **Note:** `WHERE` must come before `ORDER BY`.

# Only Customers from Australia:

query = """
SELECT CustomerId, FirstName, Surname FROM Customers
WHERE Country="Australia"
ORDER BY Surname DESC;
"""
execute_query(conn, query)

# ### LIMIT
# This clause is used to constrain the number of rows in the result.

# This query shows the 10 most expensive products.

query = """
SELECT StockCode,Category,Weight, UnitPrice FROM Products
ORDER BY UnitPrice DESC
LIMIT 10;
"""
execute_query(conn, query)

# You can use `OFFSET` to skip a few rows.

# This query shows the next top 10 products with highest price. Since it is skipping the first 10 the result would be numbers 11-20.

query = """
SELECT StockCode,Category,Weight, UnitPrice FROM Products
ORDER BY UnitPrice DESC
LIMIT 10 OFFSET 10;
"""
execute_query(conn, query)

# **Note:** `LIMIT` should always be used with `ORDER BY` so the rows are always in a specific order.

# ## Joins
# As you have probably noticed the data in this database is in various tables. Each table contains a specific part of data. For instance, one table has the information about invoices and another table has information about customers. These tables are linked together so we can find which invoice belongs to which customer. To do this we use various types of `JOIN`. Each join clause determines how SQLite uses data from one table to match with rows in another table.<br>
# Tables are connected using unique identifiers. For instance, __Sales__ table has a column called _CustomerId_. The same column name can be found in __Customers__ table. By looking up the customer id from invoices in the customers table we can find the information about the customer of each invoice.`JOIN` used this identifiers to connect the tables and look up information.
#
#
#

# ### INNER JOIN

# +
query = """
SELECT PurchaseDate,FirstName, Surname, ProductID, Quantity
FROM Sales
INNER JOIN Customers 
    ON Sales.CustomerId = Customers.CustomerId
ORDER BY PurchaseDate
LIMIT 20;
"""

execute_query(conn, query)
# -

# This query shows the purchase history of customers.

query = """
SELECT Surname, ProductID, Quantity, PurchaseDate
FROM Customers
INNER JOIN Sales ON
    Customers.CustomerID = Sales.CustomerID
ORDER BY Surname;
"""
execute_query(conn, query)

# ### LEFT  JOIN
# Also called `LEFT OUTER JOIN`

# This query shows the purchase history of customers:

# +
query = """
SELECT Surname, ProductID, Quantity, PurchaseDate
FROM Customers
LEFT JOIN Sales ON
    Customers.CustomerID = Sales.CustomerID
ORDER BY Surname;
"""

execute_query(conn, query)
# -

# **Note:** The top two cells are both showing the purchase history of customers. But one is using `INNER JOIN` and the other is using `LEFT JOIN`. So what's the difference? If you pay close attention you will see that the two tables don't have the same number of rows. The main difference between the two is that `LEFT JOIN` conserves all the entries on the left (customers) and if it can't find any purchases for that customer it will return `NULL`. However, `INNER JOIN` tries to find a match. If a customer doesn't have any purchases in the list it will not be shown in th final result.<br>
# `INNER JOIN` and `LEFT JOIN` are the most common types of join. However, there are other types such as `CROSS JOIN`, `FULL OUTER JOIN`, `RIGHT JOIN`, etc. You can find more information about these types of join on https://www.sqlitetutorial.net/.
#

# One trick that makes writing a query faster is using aliases to tables and column. The query below is the same as the one above. The only difference is use of aliases.

# +
query = """
SELECT Surname, ProductID, Quantity, PurchaseDate
FROM Customers as c
LEFT JOIN Sales as s ON
    c.CustomerID = s.CustomerID
ORDER BY Surname;
"""

execute_query(conn, query)
# -

# After `SELECT` we can also use math operation on the columns.

query = """
SELECT InvoiceNumber, StockCode, Quantity, UnitPrice, Quantity*UnitPrice as Total,  CustomerID
FROM Sales as s
INNER JOIN Products as p
    ON s.ProductId = p.ProductId
ORDER BY InvoiceNumber DESC
"""
execute_query(conn, query)

# ### GROUP BY
# This clause allows us to summerise the result of a query. It returns only one row for every group of rows. The rows are summerised using an aggregate function such as MIN, MAX, COUNT, AVG, or SUM.
#

query = """
SELECT InvoiceNumber, PurchaseDate, COUNT(StockCode) as ProductsCount,SUM(Quantity) as ItemsCount
FROM Sales as s
INNER JOIN Products as p
    ON s.ProductID = p.ProductID
GROUP BY InvoiceNumber
LIMIT 20
"""
execute_query(conn, query)

# In the example above we first join `Sales` with `Products` to get a list of purchases with product details. Then we group them by invoice number, and aggregate the total invoice once using count (to find how many types of product was purchased) and once using sum (to find how many items were purchased). <br>

# ### HAVING
# This clause adds conditions to the result of group by. It can be used as follows:
# <code> 
# SELECT column_1, column_2, aggregate_function (column_3)
# FROM table
# GROUP BY
# 	column_1, column_2
# HAVING search_condition;
# </code>
#
# <br> __<font color="red">Note that the HAVING clause is applied after GROUP BY clause, whereas the WHERE clause is applied before the GROUP BY clause.</font>__

# Let's repeat the last example, but this time return only the invoices with more than one type of product in them.

# +
query = """
SELECT InvoiceNumber, PurchaseDate, COUNT(StockCode) as ProductsCount,SUM(Quantity) as ItemsCount
FROM Sales as s
INNER JOIN Products as p
    ON s.ProductID = p.ProductID
GROUP BY InvoiceNumber
HAVING ProductsCount>1
LIMIT 20
"""

execute_query(conn, query)
# -

# ### Subqueries

# Subqueries are queries within queries. We can use a query to grab data and arrange it in a table and the write another query to work on the table we just created. This table is not saved in the database but is available until the end of the query.

# Let's use what we have learned so far and create a list of top 10 invoices with the higherst values.

query = """
SELECT InvoiceNumber, Quantity*UnitPrice as Total,  CustomerID
FROM Sales as s
INNER JOIN Products as p
    ON s.ProductId = p.ProductId
GROUP BY InvoiceNumber
ORDER BY Total DESC
LIMIT 10

"""
execute_query(conn, query)

# Now If we wanted to have Customers details in the result as well, we could use this as a subquery.

query = """
SELECT InvoiceNumber, FirstName, Surname, Mobile, Total
FROM
    (SELECT InvoiceNumber, Quantity*UnitPrice as Total,  CustomerID
    FROM Sales as s
    INNER JOIN Products as p
        ON s.ProductId = p.ProductId
    GROUP BY InvoiceNumber
    ORDER BY Total DESC
    LIMIT 10) as invoices
INNER JOIN Customers as c
    ON invoices.CustomerID = c.CustomerID
"""
execute_query(conn, query)

# ## CRUD
# CRUD is an acronym for four main database operations: Create, Read, Update, and Delete.<br>
# We have already used `SELECT` to read data from database. Now we can learn about other operations. Let's start by adding data to a database. But before that we need to create a new database and a new table inside it.
#

db_file = "sampledb.db"
conn = sqlite3.connect(db_file)

# __Note:__ When no file is found by the name we have entered, it will create a new database by that name.

# We can create a table using `CREATE TABLE` followed by the name of the table. Then, we define name of each column, the type of data it contains, and its default value. <br>
# We can use `NOT NULL` to specify that the data for a certain column cannot be left empty (NULL).
#

query = """
CREATE TABLE employees (
    FirstName nvarchar(25) NOT NULL,
    Surname nvarchar(25) NOT NULL,
    PhoneNumber nvarchar(25) NOT NULL,
    Email nvarchar(40) DEFAULT ""
    )

"""
execute_query(conn, query)

# __Note:__ FirstName, Surname, and PhoneNumber must be entered. But if the email is not entered it will be set to default value which is an empty string.

# ### Data types in SQL
# We need to specify the type of data each column contains. The types of data supported by SQL are as follows:
# - __Exact numeric:__ BOOLEAN, TINYINT, SMALLINT, INT, BIGINT
# - __Approximate Numeric:__ FLOAT, DOUBLE
# - __String:__ CHARACTER, VARCHAR, NCHAR, NVARCHAR 
# - __Date/Time:__ DATE, DATETIME, TIME, YEAR
#
# For details about each type visit [sqlite.org](https://www.sqlite.org/datatype3.html)

# Now let's see how the table looks like.

query = """
SELECT * FROM employees
"""
execute_query(conn, query)

# Nothing is printed. Because the table is empty. To ensure the table has been created, let's use `.description` to get a list of columns in the table.

cur = conn.cursor()
cur.execute(query)
cur.description

# If you see the name of columns, then the table has been successfully created.

# To add new rows to the table, we use `INSERT INTO` followed by the name of the table and the values we want to add. The values need to be in the same order as the columns.

query = """
INSERT INTO employees
VALUES ("Darth","Vader","456123789","d.vader@deathstar.com")

"""
execute_query(conn, query)

execute_query(conn, "SELECT * FROM employees")

# Now, let's add another row but this time we won't enter a value for email. Since we defined a default value for email, we expect to see it appear for the new row.

query = """
INSERT INTO employees
VALUES ("Luke","Skywalker","456123789")

"""
execute_query(conn, query)

# There is an error: _`table employees has 4 columns but 3 values were supplied`_ <br>
# The reason is the database manager doesn't know which value you have not entered.
# The correct way to add data is not only specify the values, but also the name of the columns. This way we are clear in our instruction that which value belongs to which column.

query = """
INSERT INTO employees(FirstName,Surname,PhoneNumber)
VALUES ("Luke","Skywalker","456123789")

"""
execute_query(conn, query)

execute_query(conn, "SELECT * FROM employees")

# We can add multiple rows of data simultaneously. Each row must be separated by __`,`__

query = """
INSERT INTO employees(FirstName,Surname,PhoneNumber,Email)
VALUES ("Leia","Organa","404200501","princess@aldraan.gov"),
     ("Han","Solo","437294558","m.falcon@smagglers.com")
"""
execute_query(conn, query)

execute_query(conn, "SELECT * FROM employees")

# ### Update
# Next operation to learn is `UPDATE`. It is used for change the values in the table.<br>
# A simple form of updating is presented below. Right after `UPDATE` we specify the name of the table. Then use `SET` to specify the new value for the column. Using `WHERE` clause we can conditions and control which rows should be updated.

# The query below sets the value of Email column to NULL for the rows where email address is empty.

query = """
UPDATE employees 
SET Email = NULL
WHERE Email=""
"""
execute_query(conn, query)

execute_query(conn, "SELECT * FROM employees")

# We can see that the empty email address is now updated to `None`.<br>
# __Note:__ `None` is Python equivalent of `NULL`.
#

# ### Delete
# Finally, to delete data from table we use `DELETE`. We need to specify the name of the table and conditions of the row(s).

query = """
DELETE FROM employees
WHERE Surname="Vader"
"""
execute_query(conn, query)
execute_query(conn, "SELECT * FROM employees")

# Adding the condition is important. If no condition is specified all the rows will be removed.

query = """
DELETE FROM employees
"""
execute_query(conn, query)
execute_query(conn, "SELECT * FROM employees")

# As you can see the query doesn't return anything, which suggests the table is empty.

# When your work is done on a database make sure you close the connection.

conn.close()



# ## Exrecises
# To make sure you learned writing a query try creating the following tables:
# - How many customers are there in each country?
# - How many products are in each category?
# - A list of customers ordered by their total purchase (it can be across multiple invoices)
# - A list of products orders by the number of time they have been ordered
# - A list of total sales per country



# ## Further reading
# - [SQLITE documentation](https://www.sqlite.org)
# - [SQLITE tutorial](https://www.sqlitetutorial.net/sqlite-python)
# - [W3schools](https://www.w3schools.com/sql)
