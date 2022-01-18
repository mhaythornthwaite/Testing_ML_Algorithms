
-- Exercise - how many female customers do we have from the state of Orgegon and New York?
--SELECT COUNT(firstname) FROM customers
-- WHERE gender = 'F' AND state = 'OR'
--    OR gender = 'F' AND state = 'NY';

-- Here is a shorter way of writing the above
--SELECT COUNT(firstname) FROM customers
-- WHERE gender = 'F' AND (state = 'OR' OR  state = 'NY');

-- What if we want everything expect from something.
--   SELECT * FROM customers
--WHERE NOT gender = 'M'

-- How many customers arent 55 or 54? 
--   SELECT COUNT(firstname) FROM customers
--WHERE NOT age = 55 AND NOT age = 54;

-- Typically when using numbers, we wish to use the != to denote something is not. Where we're dealing with anything else we use the WHERE NOT
--SELECT COUNT(firstname) FROM customers
-- WHERE age != 55 AND age != 54;

-- How many customers are 44 or over?
--SELECT COUNT(age) FROM customers
-- WHERE age >= 44;

-- What is the average income between the ages of 20 and 50? (Excluding 20 and 50)
--SELECT AVG(income) FROM customers
-- WHERE age > 20 AND age < 50



