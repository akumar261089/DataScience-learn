Sure! Below is a detailed Python tutorial on **Conditions**, **Loops**, **Functions**, and **Exceptions**. Each section includes explanations, examples, and sample exercises to help reinforce learning.

---

# **Python Tutorial: Conditions, Loops, Functions, and Exceptions**

## 1. **Conditions in Python**
Conditions allow you to execute different blocks of code based on specific conditions. In Python, the `if`, `elif`, and `else` statements are used for conditional execution.

### 1.1 **Basic Syntax**

```python
if condition:
    # code to execute if condition is True
elif another_condition:
    # code to execute if another_condition is True
else:
    # code to execute if none of the conditions are True
```

### 1.2 **Comparison Operators**

- `==`: Equal to
- `!=`: Not equal to
- `>`: Greater than
- `<`: Less than
- `>=`: Greater than or equal to
- `<=`: Less than or equal to

### 1.3 **Logical Operators**

- `and`: Returns True if both conditions are true
- `or`: Returns True if at least one condition is true
- `not`: Reverses the condition (True becomes False, False becomes True)

### 1.4 **Example: Basic Conditions**

```python
age = 20

if age >= 18:
    print("You are an adult.")
else:
    print("You are a minor.")
```

### 1.5 **Exercise:**
Write a program that asks for the user's age and prints whether they are eligible to vote (18 or older).

---

## 2. **Loops in Python**
Loops allow you to repeat a block of code multiple times. In Python, the primary loop structures are `for` and `while`.

### 2.1 **For Loop**

A `for` loop is commonly used to iterate over a sequence (like a list, tuple, or string).

```python
for item in sequence:
    # Code to execute for each item
```

### 2.2 **While Loop**

A `while` loop repeatedly executes a block of code as long as the specified condition remains True.

```python
while condition:
    # Code to execute as long as condition is True
```

### 2.3 **Breaking and Continuing Loops**

- `break`: Exits the loop completely.
- `continue`: Skips the current iteration and proceeds with the next one.

### 2.4 **Example: Using `for` Loop**

```python
for i in range(1, 6):
    print(i)
```

### 2.5 **Example: Using `while` Loop**

```python
counter = 1
while counter <= 5:
    print(counter)
    counter += 1
```

### 2.6 **Exercise:**
Write a program that prints all the even numbers between 1 and 20 using a loop.

---

## 3. **Functions in Python**

Functions allow you to encapsulate reusable pieces of code. They help in making your code modular and easier to manage.

### 3.1 **Defining a Function**

```python
def function_name(parameters):
    # Code to execute
    return result
```

- `def`: Keyword to define a function
- `return`: Used to return a value from the function

### 3.2 **Example: Simple Function**

```python
def greet(name):
    return f"Hello, {name}!"

message = greet("Alice")
print(message)  # Output: Hello, Alice!
```

### 3.3 **Functions with Multiple Parameters**

```python
def add_numbers(a, b):
    return a + b

result = add_numbers(3, 4)
print(result)  # Output: 7
```

### 3.4 **Default Parameters**

You can set default values for parameters when defining the function.

```python
def greet(name="Guest"):
    return f"Hello, {name}!"
```

### 3.5 **Variable-Length Arguments**

Using `*args` and `**kwargs`, you can pass a variable number of arguments to a function.

- `*args`: Allows passing a variable number of positional arguments
- `**kwargs`: Allows passing a variable number of keyword arguments

```python
def print_names(*names):
    for name in names:
        print(name)

print_names("Alice", "Bob", "Charlie")
```

### 3.6 **Exercise:**
Write a function that takes a number and returns whether it is prime or not.

---

## 4. **Exceptions in Python**

Exceptions are errors that occur during the execution of a program. Python provides a robust way to handle these errors using `try`, `except`, `else`, and `finally` blocks.

### 4.1 **Basic Try-Except Block**

```python
try:
    # Code that might raise an exception
except ExceptionType:
    # Code to handle the exception
```

### 4.2 **Handling Specific Exceptions**

You can handle different types of exceptions separately.

```python
try:
    # Code that might raise an exception
except ValueError:
    print("Invalid value")
except ZeroDivisionError:
    print("Cannot divide by zero")
```

### 4.3 **Else and Finally**

- `else`: If no exceptions occur, this block is executed.
- `finally`: This block always executes, regardless of whether an exception occurred or not.

```python
try:
    result = 10 / 2
except ZeroDivisionError:
    print("Cannot divide by zero")
else:
    print("Division successful")
finally:
    print("This will run no matter what.")
```

### 4.4 **Example: Division with Exception Handling**

```python
def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        return "Error: Cannot divide by zero"
    except TypeError:
        return "Error: Both arguments must be numbers"
    else:
        return result

print(divide(10, 2))  # Output: 5.0
print(divide(10, 0))  # Output: Error: Cannot divide by zero
print(divide(10, "a"))  # Output: Error: Both arguments must be numbers
```

### 4.5 **Raising Exceptions**

You can also raise exceptions manually using the `raise` keyword.

```python
def check_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age

# check_age(-1)  # Will raise ValueError: Age cannot be negative
```

### 4.6 **Exercise:**
Write a program that takes two numbers as input and divides them, but handle exceptions for invalid input and division by zero.

---

## Conclusion

- **Conditions**: Used to execute code based on specific conditions using `if`, `elif`, and `else`.
- **Loops**: Used to repeat code. `for` loops are used for iterating over sequences, and `while` loops are used when you want to repeat as long as a condition is true.
- **Functions**: Help in organizing your code into reusable blocks. Functions can take parameters and return values. Python also supports default arguments and variable-length arguments.
- **Exceptions**: Allow you to handle runtime errors gracefully using `try`, `except`, `else`, and `finally`. You can also raise your own exceptions when necessary.

---

Certainly! Here's an added section on **Function Decorators** in Python, which are a powerful feature to modify or extend the behavior of functions or methods without changing their actual code.

---

## 5. **Function Decorators in Python**

### 5.1 **What is a Decorator?**

A **decorator** is a function that takes another function as an argument and extends or modifies its behavior without modifying its code directly. It is often used for logging, access control, memoization, and other common tasks.

In Python, decorators are implemented using functions that return another function. You can use the `@decorator_name` syntax to apply a decorator to a function.

### 5.2 **Basic Syntax of a Decorator**

To define a decorator, you need to create a function that takes another function as its argument, performs some actions, and returns a new function (which may call the original function).

#### Basic Structure:
```python
def decorator(func):
    def wrapper():
        # Code to run before the original function
        print("Before function call")
        
        # Call the original function
        func()
        
        # Code to run after the original function
        print("After function call")
    
    return wrapper
```

You then apply the decorator to a function using the `@` symbol.

#### Example of Using a Decorator:
```python
def greet():
    print("Hello, world!")

# Applying the decorator manually
greet = decorator(greet)
greet()

# Or more commonly, using the @ syntax
@decorator
def greet():
    print("Hello, world!")
```

### 5.3 **Example: Simple Decorator**

Here’s a simple decorator that prints "Before" and "After" messages around the execution of a function:

```python
def decorator(func):
    def wrapper():
        print("Before function call")
        func()  # Call the original function
        print("After function call")
    return wrapper

@decorator
def say_hello():
    print("Hello!")

say_hello()
```

**Output:**
```
Before function call
Hello!
After function call
```

In this example, the `say_hello` function is decorated with `@decorator`, which means it will now be wrapped by the `wrapper` function inside the `decorator`.

### 5.4 **Decorators with Arguments**

If your decorated function takes arguments, the decorator needs to accept those arguments as well. You can modify the `wrapper` function to take `*args` and `**kwargs` to forward any arguments passed to the original function.

#### Example: Decorator with Arguments

```python
def decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)  # Call the original function with arguments
        print("After function call")
        return result  # Return the result of the original function
    return wrapper

@decorator
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
```

**Output:**
```
Before function call
Hello, Alice!
After function call
```

In this example, the `greet` function takes an argument `name`, and the decorator properly passes it along to the original function.

### 5.5 **Returning Values from a Decorated Function**

Decorators can also modify or return values from the function they wrap. You can use the `return` keyword to return the result of the decorated function.

#### Example: Decorator Returning a Value

```python
def decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)  # Call the original function
        print("After function call")
        return result * 2  # Modify the return value
    return wrapper

@decorator
def add(a, b):
    return a + b

result = add(3, 5)
print(result)  # Output: 16
```

**Explanation:** In this case, the decorator not only wraps the `add` function but also modifies the returned result by multiplying it by 2.

### 5.6 **Chaining Multiple Decorators**

You can apply multiple decorators to a single function by stacking them. The decorators are applied from the innermost one to the outermost one.

```python
def decorator1(func):
    def wrapper():
        print("Decorator 1 before")
        func()
        print("Decorator 1 after")
    return wrapper

def decorator2(func):
    def wrapper():
        print("Decorator 2 before")
        func()
        print("Decorator 2 after")
    return wrapper

@decorator1
@decorator2
def say_hello():
    print("Hello!")

say_hello()
```

**Output:**
```
Decorator 1 before
Decorator 2 before
Hello!
Decorator 2 after
Decorator 1 after
```

In this case, the `say_hello` function is decorated first by `decorator2` and then by `decorator1`. The order of application is from the bottom upwards.

### 5.7 **Built-in Python Decorators**

Python provides some useful built-in decorators that can be used directly:

- **`@staticmethod`**: Used to define static methods that don't require a class instance.
- **`@classmethod`**: Used to define class methods that receive the class (`cls`) as the first argument.
- **`@property`**: Allows you to define a method as a property, meaning it behaves like an attribute.

#### Example: `@staticmethod` and `@classmethod`

```python
class MyClass:
    @staticmethod
    def static_method():
        print("This is a static method.")

    @classmethod
    def class_method(cls):
        print(f"This is a class method. The class is {cls.__name__}.")

# Using the static method and class method
MyClass.static_method()
MyClass.class_method()
```

**Output:**
```
This is a static method.
This is a class method. The class is MyClass.
```

### 5.8 **Exercise:**

Write a decorator that logs the execution time of a function. The decorator should print how long the function took to execute in seconds.

**Hint**: Use the `time` module and `time.time()` to measure the start and end time.

---

## Conclusion on Decorators

Decorators are a powerful and elegant feature in Python that allows you to enhance or modify the behavior of functions or methods. They are widely used in frameworks, such as Flask and Django, for routing, logging, access control, and more.

Key points to remember:
- A decorator is a function that takes another function and extends or modifies its behavior.
- You can pass arguments and return values from decorated functions.
- Python has several built-in decorators like `@staticmethod`, `@classmethod`, and `@property`.
- Decorators can be chained together for more complex functionality.

With this knowledge of function decorators, you can write cleaner and more modular code in Python. Feel free to explore and experiment with decorators in your projects!

---
