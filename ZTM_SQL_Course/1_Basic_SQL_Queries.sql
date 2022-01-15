--A word on formatting, spaces should be used to line up the code so that the root keywords all end on the same character boundary. This forms a river down the middle making it easy for the readers eye to scan over the code and separate the keywords from the implementation detail.

---------------------------- SELECT ---------------------------- 

--selecting all data from our departments table
--SELECT * FROM departments;

--how many times has employee 10001 had a raise?
--SELECT * 
--  FROM salaries 
-- WHERE emp_no=10001;

--what title does 10006 have?
--SELECT title 
--  FROM titles
-- WHERE emp_no=10006;

--ranaming columns, note this only renamed the column of the returned data, not the actual database
--SELECT emp_no AS "Employee Number", birth_date AS "Birthday" FROM employees;


-------------------------- FUNCTIONS -------------------------- 
--mixture of aggregate and scalar functions in the following

--CONCAT, stitching together first and last name, note double quote refer to a column name, single quote refer simply to text, hence we're using ' ' to denote a space in the soncat function. 
--SELECT emp_no, 
--       CONCAT(first_name, ' ', last_name) AS "Full Name" 
-- FROM employees;

--COUNT, how many people work here?
--SELECT COUNT(emp_no) FROM employees;

--AVG, whats the average salary of an employee.
--SELECT AVG(salary) FROM salaries

--MAX, whats the maximum salary paid?
--SELECT MAX(salary) FROM salaries

--MIN, whats you birthday of the oldest employee?
--SELECT MIN(birth_date) FROM employees

--SUM, whats the total amount the company pays per year?
--SELECT SUM(salary) FROM salaries


