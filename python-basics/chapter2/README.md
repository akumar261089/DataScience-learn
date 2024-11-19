### **Detailed Tutorial: Expressions and Variables in Python**

---

### **1. Variables in Python**
A **variable** is a named reference to a memory location used to store data. You can assign values to variables using the assignment operator `=`.

#### **Features of Variables in Python**
1. **Dynamic Typing**: You don't need to declare the type of a variable explicitly. The type is inferred at runtime.
2. **Case-Sensitive**: Variable names are case-sensitive (`Age` and `age` are different variables).
3. **Naming Rules**:
   - Must start with a letter or an underscore `_`.
   - Can contain letters, numbers, and underscores (`_`).
   - Cannot use Python's reserved keywords like `if`, `for`, etc.

---

#### **Creating Variables**
```python
# Assigning values
name = "Alice"  # String
age = 25        # Integer
height = 5.7    # Float
is_student = True  # Boolean

print(name, age, height, is_student)
# Output: Alice 25 5.7 True
```

---

#### **Reassigning Variables**
Variables can be reassigned to different values or even different types (dynamic typing).

```python
x = 10
print(x)  # Output: 10

x = "Hello"
print(x)  # Output: Hello
```

---

#### **Multiple Assignments**
You can assign multiple variables in a single line.

```python
a, b, c = 1, 2, 3
print(a, b, c)  # Output: 1 2 3

# Assign the same value to multiple variables
x = y = z = 100
print(x, y, z)  # Output: 100 100 100
```

---

#### **Deleting Variables**
You can delete a variable using the `del` keyword.

```python
x = 10
del x
# print(x)  # Error: NameError: name 'x' is not defined
```

---

### **2. Expressions in Python**
An **expression** is a combination of variables, values, and operators that Python evaluates to produce a result.

#### **Types of Expressions**
1. **Arithmetic Expressions**
2. **Comparison Expressions**
3. **Logical Expressions**
4. **Membership Expressions**
5. **Identity Expressions**

---

#### **Arithmetic Expressions**
Used to perform mathematical operations. Common operators:

| Operator | Description         | Example    | Result |
|----------|---------------------|------------|--------|
| `+`      | Addition            | `3 + 2`    | `5`    |
| `-`      | Subtraction         | `5 - 3`    | `2`    |
| `*`      | Multiplication      | `4 * 2`    | `8`    |
| `/`      | Division            | `10 / 2`   | `5.0`  |
| `//`     | Floor Division      | `7 // 2`   | `3`    |
| `%`      | Modulus (Remainder) | `10 % 3`   | `1`    |
| `**`     | Exponentiation      | `2 ** 3`   | `8`    |

**Examples:**
```python
x = 10
y = 3

print(x + y)  # Output: 13
print(x // y)  # Output: 3
print(x ** y)  # Output: 1000
```

---

#### **Comparison Expressions**
Used to compare values and return a boolean (`True` or `False`).

| Operator | Description         | Example   | Result  |
|----------|---------------------|-----------|---------|
| `==`     | Equal to            | `5 == 5`  | `True`  |
| `!=`     | Not equal to        | `5 != 3`  | `True`  |
| `>`      | Greater than        | `5 > 3`   | `True`  |
| `<`      | Less than           | `3 < 5`   | `True`  |
| `>=`     | Greater or equal    | `5 >= 5`  | `True`  |
| `<=`     | Less or equal       | `3 <= 5`  | `True`  |

**Examples:**
```python
a = 7
b = 10

print(a > b)  # Output: False
print(a != b)  # Output: True
```

---

#### **Logical Expressions**
Used to combine multiple conditions. Operators:

| Operator | Description               | Example            | Result |
|----------|---------------------------|--------------------|--------|
| `and`    | True if both are True     | `True and False`   | `False`|
| `or`     | True if at least one True | `True or False`    | `True` |
| `not`    | Negates a condition       | `not True`         | `False`|

**Examples:**
```python
a = 5
b = 10

print(a < b and b > 0)  # Output: True
print(a > b or b > 0)   # Output: True
print(not (a > b))      # Output: True
```

---

#### **Membership Expressions**
Used to test if a value is in a collection (`list`, `tuple`, `string`, etc.).

| Operator | Description         | Example         | Result |
|----------|---------------------|-----------------|--------|
| `in`     | True if present     | `'a' in 'apple'`| `True` |
| `not in` | True if not present | `'z' not in 'apple'` | `True` |

**Examples:**
```python
x = "hello"
print('h' in x)      # Output: True
print('z' not in x)  # Output: True
```

---

#### **Identity Expressions**
Used to compare memory locations of two objects.

| Operator | Description                | Example       | Result |
|----------|----------------------------|---------------|--------|
| `is`     | True if same object        | `x is y`      | `True` |
| `is not` | True if not the same object| `x is not y`  | `True` |

**Examples:**
```python
x = [1, 2, 3]
y = x
z = [1, 2, 3]

print(x is y)       # Output: True (Same object)
print(x is z)       # Output: False (Different objects)
print(x == z)       # Output: True (Values are equal)
```

---

### **3. Combining Variables and Expressions**

You can use variables within expressions:

```python
a = 10
b = 3

result = (a + b) * 2  # Arithmetic expression
print(result)         # Output: 26

is_greater = a > b    # Comparison expression
print(is_greater)     # Output: True
```

---

### **4. Built-in Functions for Variables and Expressions**
Python provides several functions to manipulate or inspect variables and expressions:

| Function   | Description                | Example                   | Result      |
|------------|----------------------------|---------------------------|-------------|
| `type()`   | Returns type of variable   | `type(5)`                 | `<class 'int'>` |
| `id()`     | Returns memory location    | `id(5)`                   | e.g., `140123456` |
| `int()`    | Converts to integer        | `int(5.5)`                | `5`         |
| `float()`  | Converts to float          | `float(5)`                | `5.0`       |
| `str()`    | Converts to string         | `str(10)`                 | `'10'`      |

---

### **5. Common Mistakes to Avoid**
1. **Uninitialized Variables**: Using a variable without assigning a value will raise an error.
   ```python
   print(x)  # Error: NameError: name 'x' is not defined
   ```

2. **Type Errors**: Mixing incompatible types in expressions.
   ```python
   print(10 + "Python")  # Error: TypeError
   ```

3. **Case Sensitivity**:
   ```python
   age = 25
   print(Age)  # Error: NameError
   ```

---

### **Practice Problems**
1. Assign values to variables `x`, `y`, and `z`. Compute their sum, difference, product, and quotient.
2. Check if a given number is odd or even using a comparison expression.
3. Write a program that checks if a character is present in a string using membership expressions.

---

### **Detailed Tutorial: String Operations in Python**

Strings are one of the most widely used data types in Python. A **string** is a sequence of characters enclosed in single quotes `'`, double quotes `"`, or triple quotes `'''` / `"""`.

---

### **1. Creating Strings**

#### **Single-Line Strings**
```python
string1 = 'Hello'
string2 = "Python"
print(string1, string2)
```

#### **Multi-Line Strings**
Use triple quotes for multi-line strings.
```python
multiline_string = '''This is a
multi-line string.'''
print(multiline_string)
```

---

### **2. Accessing Strings**

#### **Indexing**
You can access individual characters in a string using **indices** (starting from 0).
```python
string = "Python"
print(string[0])  # Output: P
print(string[-1])  # Output: n (last character)
```

#### **Slicing**
Extract a portion of a string using slicing syntax: `string[start:end:step]`
```python
string = "Python"
print(string[0:3])   # Output: Pyt (characters from index 0 to 2)
print(string[:3])    # Output: Pyt (start defaults to 0)
print(string[2:])    # Output: thon (till the end)
print(string[::-1])  # Output: nohtyP (reversed string)
```

---

### **3. String Operations**

#### **Concatenation**
Combine two or more strings using the `+` operator.
```python
str1 = "Hello"
str2 = "World"
result = str1 + " " + str2
print(result)  # Output: Hello World
```

#### **Repetition**
Repeat a string multiple times using the `*` operator.
```python
string = "Ha"
print(string * 3)  # Output: HaHaHa
```

#### **Membership**
Check if a substring exists in a string using `in` or `not in`.
```python
string = "Python is fun"
print("Python" in string)  # Output: True
print("Java" not in string)  # Output: True
```

---

### **4. String Methods**

Python provides numerous built-in methods for manipulating strings.

#### **Case Conversion**
| Method              | Description                          | Example                         |
|---------------------|--------------------------------------|---------------------------------|
| `.upper()`          | Converts to uppercase               | `"hello".upper()` -> `HELLO`   |
| `.lower()`          | Converts to lowercase               | `"HELLO".lower()` -> `hello`   |
| `.capitalize()`     | Capitalizes the first character     | `"hello".capitalize()` -> `Hello` |
| `.title()`          | Capitalizes each word               | `"hello world".title()` -> `Hello World` |
| `.swapcase()`       | Swaps uppercase and lowercase       | `"HeLLo".swapcase()` -> `hEllO` |

#### **Trimming Whitespace**
| Method              | Description                          | Example                         |
|---------------------|--------------------------------------|---------------------------------|
| `.strip()`          | Removes leading/trailing spaces     | `" hello ".strip()` -> `hello` |
| `.lstrip()`         | Removes leading spaces              | `" hello ".lstrip()` -> `hello ` |
| `.rstrip()`         | Removes trailing spaces             | `" hello ".rstrip()` -> ` hello` |

#### **Searching and Replacing**
| Method              | Description                          | Example                         |
|---------------------|--------------------------------------|---------------------------------|
| `.find()`           | Returns the first index of substring | `"hello".find("e")` -> `1`     |
| `.rfind()`          | Returns the last index of substring  | `"hello".rfind("l")` -> `3`    |
| `.count()`          | Counts occurrences of substring     | `"hello".count("l")` -> `2`    |
| `.replace()`        | Replaces a substring with another   | `"hello".replace("l", "x")` -> `hexxo` |

#### **Splitting and Joining**
| Method              | Description                          | Example                         |
|---------------------|--------------------------------------|---------------------------------|
| `.split()`          | Splits string into a list           | `"a,b,c".split(",")` -> `['a', 'b', 'c']` |
| `.rsplit()`         | Splits from the right               | `"a,b,c".rsplit(",", 1)` -> `['a,b', 'c']` |
| `.join()`           | Joins elements of a list into string| `" ".join(['Hello', 'World'])` -> `Hello World` |

#### **Checking String Content**
| Method              | Description                          | Example                         |
|---------------------|--------------------------------------|---------------------------------|
| `.startswith()`     | Checks if string starts with prefix | `"hello".startswith("he")` -> `True` |
| `.endswith()`       | Checks if string ends with suffix   | `"hello".endswith("lo")` -> `True` |
| `.isalpha()`        | Checks if all characters are letters| `"hello".isalpha()` -> `True`  |
| `.isdigit()`        | Checks if all characters are digits | `"123".isdigit()` -> `True`    |
| `.isalnum()`        | Checks if alphanumeric             | `"hello123".isalnum()` -> `True` |
| `.isspace()`        | Checks if all characters are spaces | `"   ".isspace()` -> `True`    |

---

### **5. String Formatting**

#### **Concatenation**
Using `+` to insert variables into strings:
```python
name = "Alice"
age = 25
print("Name: " + name + ", Age: " + str(age))
```

#### **f-Strings** (Recommended for Python 3.6+)
```python
name = "Alice"
age = 25
print(f"Name: {name}, Age: {age}")
# Output: Name: Alice, Age: 25
```

#### **Using `.format()`**
```python
name = "Alice"
age = 25
print("Name: {}, Age: {}".format(name, age))
# Output: Name: Alice, Age: 25
```

#### **Using `% Formatting**
```python
name = "Alice"
age = 25
print("Name: %s, Age: %d" % (name, age))
# Output: Name: Alice, Age: 25
```

---

### **6. Escape Sequences**

Escape sequences allow you to include special characters in strings.
| Escape Sequence | Meaning                  | Example                |
|-----------------|--------------------------|------------------------|
| `\'`           | Single quote             | `'It\'s Python'`      |
| `\"`           | Double quote             | `"He said \"Hi\""`    |
| `\\`           | Backslash               | `print("C:\\path")`   |
| `\n`           | New line                | `"Hello\nWorld"`       |
| `\t`           | Tab                     | `"Hello\tWorld"`       |

---

### **7. Raw Strings**

Use `r` to treat backslashes as literal characters:
```python
path = r"C:\Users\Alice"
print(path)  # Output: C:\Users\Alice
```

---

### **8. Advanced String Operations**

#### **String Reversal**
```python
string = "Python"
print(string[::-1])  # Output: nohtyP
```

#### **Checking for Palindromes**
```python
string = "madam"
is_palindrome = string == string[::-1]
print(is_palindrome)  # Output: True
```

#### **Character Frequency**
```python
string = "hello"
frequency = {char: string.count(char) for char in set(string)}
print(frequency)  # Output: {'h': 1, 'e': 1, 'l': 2, 'o': 1}
```

---

### **Practice Problems**
1. Write a program to count vowels in a string.
2. Reverse a string without using slicing.
3. Check if a given string is a palindrome.
4. Extract words from a sentence and sort them alphabetically.
5. Replace all occurrences of a substring in a string with another substring.

---
