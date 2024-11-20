

### Detailed Tutorial on Data Types in Python

Python provides various **built-in data types** that allow developers to handle different types of data efficiently. Here's an in-depth look at Python's data types:

---

### **1. Numeric Types**
Python supports three types of numbers:

#### **a. Integers (`int`)**
- Whole numbers (positive, negative, or zero) without a fractional part.
- Examples: `10`, `-5`, `0`

```python
a = 5       # Positive integer
b = -10     # Negative integer
c = 0       # Zero
print(type(a))  # Output: <class 'int'>
```

#### **b. Floating-Point Numbers (`float`)**
- Numbers with a decimal point or scientific notation.
- Examples: `3.14`, `-0.001`, `1.5e3` (which means \(1.5 \times 10^3 = 1500\)).

```python
x = 3.14    # Decimal number
y = -0.01   # Negative float
z = 1.2e3   # Scientific notation (1200.0)
print(type(x))  # Output: <class 'float'>
```

#### **c. Complex Numbers (`complex`)**
- Numbers with a real and imaginary part, denoted by `j`.
- Examples: `3+4j`, `-2-5j`.

```python
z = 2 + 3j
print(z.real)  # Output: 2.0
print(z.imag)  # Output: 3.0
print(type(z))  # Output: <class 'complex'>
```

---

### **2. Sequence Types**
Sequences are ordered collections of items. Common sequence types in Python are:

#### **a. Strings (`str`)**
- Immutable sequences of characters.
- Enclosed in single `'` or double `"` quotes.

```python
s = "Hello, World!"
print(s[0])    # Access the first character: 'H'
print(s[-1])   # Access the last character: '!'
print(s[0:5])  # Slicing: 'Hello'
print(type(s))  # Output: <class 'str'>
```

#### **b. Lists (`list`)**
- Ordered, mutable collections of items (can contain mixed types).

```python
lst = [1, "Python", 3.14, True]
lst[1] = "Java"  # Modify the second element
print(lst)  # Output: [1, 'Java', 3.14, True]
print(type(lst))  # Output: <class 'list'>
```

#### **c. Tuples (`tuple`)**
- Ordered, immutable collections of items.

```python
tup = (1, "Python", 3.14)
# tup[1] = "Java"  # Error: Tuples are immutable
print(tup)  # Output: (1, 'Python', 3.14)
print(type(tup))  # Output: <class 'tuple'>
```

#### **d. Ranges (`range`)**
- Represent sequences of numbers, commonly used in loops.

```python
r = range(1, 10, 2)  # Start: 1, Stop: 10, Step: 2
print(list(r))  # Output: [1, 3, 5, 7, 9]
print(type(r))  # Output: <class 'range'>
```

---

### **3. Mapping Type**
#### **Dictionaries (`dict`)**
- Unordered collections of key-value pairs.
- Keys must be unique and immutable (strings, numbers, or tuples).

```python
d = {"name": "Alice", "age": 25}
d["age"] = 26  # Update value
print(d)  # Output: {'name': 'Alice', 'age': 26}
print(type(d))  # Output: <class 'dict'>
```

---

### **4. Set Types**
Unordered collections of unique elements.

#### **a. Sets (`set`)**
- Mutable and unordered.

```python
s = {1, 2, 3, 3, 4}
s.add(5)
print(s)  # Output: {1, 2, 3, 4, 5} (duplicates removed)
print(type(s))  # Output: <class 'set'>
```

#### **b. Frozensets (`frozenset`)**
- Immutable version of sets.

```python
fs = frozenset([1, 2, 3, 3, 4])
# fs.add(5)  # Error: Frozensets are immutable
print(fs)  # Output: frozenset({1, 2, 3, 4})
print(type(fs))  # Output: <class 'frozenset'>
```

---

### **5. Boolean Type (`bool`)**
- Represents truth values: `True` and `False`.

```python
is_valid = True
print(type(is_valid))  # Output: <class 'bool'>
print(1 == 1)  # Output: True
print(1 > 2)   # Output: False
```

---

### **6. Binary Types**
Used to work with binary data.

#### **a. Bytes (`bytes`)**
- Immutable sequences of bytes.

```python
b = b"Python"
print(b)  # Output: b'Python'
print(type(b))  # Output: <class 'bytes'>
```

#### **b. Byte Arrays (`bytearray`)**
- Mutable version of `bytes`.

```python
ba = bytearray(b"Python")
ba[0] = 80  # Modify the first byte
print(ba)  # Output: bytearray(b'Python')
print(type(ba))  # Output: <class 'bytearray'>
```

#### **c. Memoryview (`memoryview`)**
- Provides memory access without copying the data.

```python
mv = memoryview(bytearray("Python", "utf-8"))
print(mv[0])  # Output: 80
print(type(mv))  # Output: <class 'memoryview'>
```

---

### **Type Conversion**
You can convert one data type to another using **typecasting functions**:

```python
# Integer to float
a = float(5)
print(a)  # Output: 5.0

# String to integer
b = int("10")
print(b)  # Output: 10

# List to tuple
c = tuple([1, 2, 3])
print(c)  # Output: (1, 2, 3)
```

---

### **Common Operations**
Some operations applicable across data types:

| Operation   | Data Types           | Example                     | Output        |
|-------------|----------------------|-----------------------------|---------------|
| Concatenation | Strings, Lists, Tuples | `"Hi" + " there!"`         | `"Hi there!"` |
| Repetition  | Strings, Lists, Tuples | `["a"] * 3`               | `["a", "a", "a"]` |
| Membership  | Strings, Lists, Dicts | `'a' in 'apple'`           | `True`        |
| Length      | Strings, Lists, Dicts | `len([1, 2, 3])`           | `3`           |

---

### **Python Tutorial: Lists, Tuples, Dictionaries, and Sets**

---

### **1. Lists in Python**
A **list** is an ordered, mutable collection that allows duplicate elements.

#### **Creating a List**
```python
# List of integers
numbers = [1, 2, 3, 4]

# List with mixed data types
mixed = [1, "Hello", 3.14, True]

# Nested lists
nested = [1, [2, 3], [4, [5, 6]]]

print(numbers, mixed, nested)
```

#### **Accessing List Elements**
```python
# Indexing
numbers = [10, 20, 30, 40]
print(numbers[0])   # Output: 10
print(numbers[-1])  # Output: 40

# Slicing
print(numbers[1:3])  # Output: [20, 30]
print(numbers[::-1]) # Output: [40, 30, 20, 10]
```

#### **Modifying Lists**
```python
# Add elements
numbers.append(50)          # Adds at the end
numbers.insert(2, 25)       # Inserts 25 at index 2

# Remove elements
numbers.remove(30)          # Removes the first occurrence of 30
last = numbers.pop()        # Removes and returns the last element
del numbers[1]              # Deletes element at index 1

# Modify element
numbers[0] = 15

print(numbers)
```

#### **List Operations**
| Operation       | Syntax                         | Example                         |
|-----------------|--------------------------------|---------------------------------|
| Concatenation   | `list1 + list2`               | `[1, 2] + [3, 4]` -> `[1, 2, 3, 4]` |
| Repetition      | `list * n`                    | `[1, 2] * 2` -> `[1, 2, 1, 2]` |
| Membership      | `element in list`             | `2 in [1, 2, 3]` -> `True`     |
| Length          | `len(list)`                  | `len([1, 2, 3])` -> `3`        |
| Iteration       | `for item in list`            | `for x in [1, 2]: print(x)`    |

#### **Useful List Methods**
| Method          | Description                     | Example                         |
|-----------------|---------------------------------|---------------------------------|
| `.append(x)`    | Adds an element to the end      | `[1, 2].append(3)` -> `[1, 2, 3]` |
| `.extend(x)`    | Extends list with another list  | `[1, 2].extend([3, 4])` -> `[1, 2, 3, 4]` |
| `.sort()`       | Sorts list in ascending order   | `[3, 1].sort()` -> `[1, 3]`     |
| `.reverse()`    | Reverses the list order         | `[1, 2].reverse()` -> `[2, 1]` |
| `.index(x)`     | Returns the index of element    | `[1, 2, 3].index(2)` -> `1`    |
| `.count(x)`     | Counts occurrences of an element| `[1, 2, 2].count(2)` -> `2`    |

---

### **2. Tuples in Python**
A **tuple** is an ordered, immutable collection that allows duplicate elements.

#### **Creating Tuples**
```python
# Empty tuple
empty_tuple = ()

# Tuple with elements
tuple1 = (1, 2, 3)
tuple2 = 1, 2, 3  # Parentheses are optional

# Single-element tuple (comma is required)
single_element = (5,)

print(tuple1, tuple2, single_element)
```

#### **Accessing Tuple Elements**
Tuples support indexing and slicing like lists.
```python
my_tuple = (10, 20, 30, 40)
print(my_tuple[1])  # Output: 20
print(my_tuple[:2]) # Output: (10, 20)
```

#### **Tuple Operations**
| Operation       | Syntax                         | Example                         |
|-----------------|--------------------------------|---------------------------------|
| Concatenation   | `tuple1 + tuple2`             | `(1, 2) + (3, 4)` -> `(1, 2, 3, 4)` |
| Repetition      | `tuple * n`                   | `(1, 2) * 2` -> `(1, 2, 1, 2)` |
| Membership      | `element in tuple`            | `2 in (1, 2, 3)` -> `True`     |
| Length          | `len(tuple)`                 | `len((1, 2, 3))` -> `3`        |

#### **Tuple Unpacking**
```python
a, b, c = (10, 20, 30)
print(a, b, c)  # Output: 10 20 30
```

---

### **3. Dictionaries in Python**
A **dictionary** is an unordered, mutable collection of key-value pairs.

#### **Creating a Dictionary**
```python
# Empty dictionary
empty_dict = {}

# Dictionary with key-value pairs
my_dict = {
    "name": "Alice",
    "age": 25,
    "is_student": True
}

print(my_dict)
```

#### **Accessing and Modifying Elements**
```python
# Accessing values
print(my_dict["name"])      # Output: Alice
print(my_dict.get("age"))   # Output: 25

# Modifying values
my_dict["age"] = 26

# Adding a new key-value pair
my_dict["city"] = "New York"

# Removing a key-value pair
del my_dict["is_student"]

print(my_dict)
```

#### **Dictionary Operations**
| Operation       | Syntax                         | Example                         |
|-----------------|--------------------------------|---------------------------------|
| Membership      | `key in dict`                 | `"name" in my_dict` -> `True`  |
| Keys            | `dict.keys()`                | `{"a": 1}.keys()` -> `dict_keys(['a'])` |
| Values          | `dict.values()`              | `{"a": 1}.values()` -> `dict_values([1])` |
| Items           | `dict.items()`               | `{"a": 1}.items()` -> `dict_items([('a', 1)])` |

#### **Useful Dictionary Methods**
| Method          | Description                     | Example                         |
|-----------------|---------------------------------|---------------------------------|
| `.get(key, default)`| Returns value or default if key missing | `{"a": 1}.get("b", 0)` -> `0` |
| `.pop(key)`     | Removes and returns value       | `{"a": 1}.pop("a")` -> `1`     |
| `.update(dict)` | Updates dictionary with another | `{"a": 1}.update({"b": 2})` -> `{"a": 1, "b": 2}` |

---

### **4. Sets in Python**
A **set** is an unordered, mutable collection of unique elements.

#### **Creating a Set**
```python
# Empty set
empty_set = set()

# Set with elements
my_set = {1, 2, 3, 3}  # Duplicate elements are removed
print(my_set)  # Output: {1, 2, 3}
```

#### **Accessing Elements**
Sets are unordered, so you cannot access elements by index. However, you can check membership:
```python
print(1 in my_set)  # Output: True
```

#### **Set Operations**
| Operation       | Syntax                         | Example                         |
|-----------------|--------------------------------|---------------------------------|
| Union           | `set1 | set2`                | `{1, 2} | {2, 3}` -> `{1, 2, 3}` |
| Intersection    | `set1 & set2`                | `{1, 2} & {2, 3}` -> `{2}`      |
| Difference      | `set1 - set2`                | `{1, 2} - {2, 3}` -> `{1}`      |
| Symmetric Diff. | `set1 ^ set2`                | `{1, 2} ^ {2, 3}` -> `{1, 3}`   |

#### **Useful Set Methods**
| Method          | Description                     | Example                         |
|-----------------|---------------------------------|---------------------------------|
| `.add(x)`       | Adds an element to the set      | `{1, 2}.add(3)` -> `{1, 2, 3}` |
| `.remove(x)`    | Removes an element (raises error)| `{1, 2}.remove(2)` -> `{1}`    |
| `.discard(x)`   | Removes an element (no error)  | `{1, 2}.discard(3)` -> `{1, 2}` |
| `.clear()`      | Removes all elements            | `{1, 2}.clear()` -> `set()`    |

---

### **Comparison of Lists, Tuples, Dicts, and Sets**

| Feature           | List        | Tuple       | Dictionary        | Set           |
|-------------------|-------------|-------------|-------------------|---------------|
| Ordered           | Yes         | Yes         | No                | No            |
| Mutable           | Yes         | No          | Yes               | Yes           |
| Duplicate Allowed | Yes         | Yes         | Keys: No, Values: Yes | No        |

Feel free to experiment with these examples in your Python environment!
