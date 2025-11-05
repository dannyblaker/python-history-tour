"""
Python Version History Data
Comprehensive information about Python releases from 2.0.1 to 3.14
"""


def get_feature_examples():
    """Returns code examples for specific features"""
    return {
        "Garbage collection of cycles": {
            "description": "Automatic detection and cleanup of circular references",
            "code": """# Before: Memory leaks with circular references
class Node:
    def __init__(self):
        self.ref = None

a = Node()
b = Node()
a.ref = b
b.ref = a  # Circular reference - now collected!"""
        },
        "Unicode support (16-bit)": {
            "description": "Native support for Unicode strings",
            "code": """# Unicode strings in Python 2.0
u = u"Hello, 世界! 🌍"
print u
# Encode/decode between Unicode and bytes
encoded = u.encode('utf-8')"""
        },
        "List comprehensions introduced": {
            "description": "Concise syntax for creating lists",
            "code": """# Instead of:
squares = []
for x in range(10):
    squares.append(x**2)

# Use list comprehensions:
squares = [x**2 for x in range(10)]"""
        },
        "Augmented assignment (+=, -=, etc.)": {
            "description": "Shorthand for common operations",
            "code": """# Old way
x = x + 1
y = y * 2

# New augmented assignment
x += 1
y *= 2
count -= 1
text += " more text\""""
        },
        "String methods instead of string module": {
            "description": "Methods directly on string objects",
            "code": """# Old way with string module
import string
text = "hello world"
upper = string.upper(text)

# New way with methods
text = "hello world"
upper = text.upper()
words = text.split()"""
        },
        "Nested scopes": {
            "description": "Inner functions can access outer function variables",
            "code": """def outer(x):
    def inner(y):
        return x + y  # Can access x from outer!
    return inner

add_five = outer(5)
print add_five(3)  # Returns 8"""
        },
        "Rich comparisons": {
            "description": "Custom comparison operators for classes",
            "code": """class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __lt__(self, other):
        return self.x < other.x
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y"""
        },
        "Weak references": {
            "description": "References that don't prevent garbage collection",
            "code": """import weakref

class MyClass:
    pass

obj = MyClass()
weak = weakref.ref(obj)  # Weak reference
print weak()  # Access object
del obj  # Object can be collected"""
        },
        "Function attributes": {
            "description": "Attach custom attributes to functions",
            "code": """def my_func():
    my_func.call_count += 1
    return "Called!"

my_func.call_count = 0
my_func()
my_func()
print my_func.call_count  # 2"""
        },
        "New-style classes": {
            "description": "Modern class system with object inheritance",
            "code": """# New-style class
class MyClass(object):
    pass

# Enables: properties, __slots__, 
# descriptors, and more"""
        },
        "Iterators and generators": {
            "description": "Memory-efficient iteration with yield",
            "code": """def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a  # Generator!
        a, b = b, a + b

for num in fibonacci(10):
    print num"""
        },
        "Division operator changes (//)": {
            "description": "Integer division operator",
            "code": """# Classic division
print 7 / 2      # 3 (integer division)

# New floor division
print 7 // 2     # 3 (explicit floor)
print 7.0 / 2    # 3.5 (true division)"""
        },
        "Property decorators": {
            "description": "Getter/setter methods as attributes",
            "code": """class Circle(object):
    def __init__(self, radius):
        self._radius = radius
    
    def get_radius(self):
        return self._radius
    
    radius = property(get_radius)

c = Circle(5)
print c.radius  # Uses getter"""
        },
        "Boolean type introduced": {
            "description": "True and False as built-in constants",
            "code": """# Before: used 0 and 1
if 1:
    print "true"

# Python 2.3+: bool type
if True:
    print "true"
    
print type(True)  # <type 'bool'>"""
        },
        "Enumerate function": {
            "description": "Get index and value while iterating",
            "code": """fruits = ['apple', 'banana', 'cherry']

for i, fruit in enumerate(fruits):
    print i, fruit
# 0 apple
# 1 banana
# 2 cherry"""
        },
        "Decorators (@decorator syntax)": {
            "description": "Function wrappers with @ syntax",
            "code": """def trace(func):
    def wrapper(*args):
        print "Calling", func.__name__
        return func(*args)
    return wrapper

@trace
def greet(name):
    return "Hello " + name

greet("World")"""
        },
        "Generator expressions": {
            "description": "Like list comprehensions but memory-efficient",
            "code": """# List comprehension (creates full list)
squares = [x**2 for x in range(1000000)]

# Generator expression (lazy evaluation)
squares = (x**2 for x in range(1000000))"""
        },
        "Built-in set() and frozenset()": {
            "description": "Set data structures for unique collections",
            "code": """# Create sets
s1 = set([1, 2, 3, 3, 2])
print s1  # set([1, 2, 3])

# Set operations
s2 = set([3, 4, 5])
print s1 | s2  # Union
print s1 & s2  # Intersection"""
        },
        "reversed() function": {
            "description": "Iterate over a sequence in reverse",
            "code": """numbers = [1, 2, 3, 4, 5]

for n in reversed(numbers):
    print n  # 5, 4, 3, 2, 1

text = "hello"
print ''.join(reversed(text))  # "olleh\""""
        },
        "sorted() function": {
            "description": "Sort any iterable, return new list",
            "code": """numbers = [5, 2, 8, 1, 9]
sorted_nums = sorted(numbers)

# Sort with key function
words = ['banana', 'pie', 'Washington', 'book']
sorted(words, key=str.lower)"""
        },
        "Conditional expressions (ternary operator)": {
            "description": "Inline if-else expressions",
            "code": """# Old way
if x > 0:
    result = "positive"
else:
    result = "non-positive"

# Ternary operator
result = "positive" if x > 0 else "non-positive\""""
        },
        "with statement": {
            "description": "Context managers for resource management",
            "code": """# Before: manual file handling
f = open('file.txt')
try:
    data = f.read()
finally:
    f.close()

# With statement
with open('file.txt') as f:
    data = f.read()  # Auto-closes!"""
        },
        "try/except/finally unified": {
            "description": "Single try statement with all clauses",
            "code": """# Python 2.5+ unified syntax
try:
    risky_operation()
except ValueError, e:
    print "Error:", e
finally:
    cleanup()  # Always runs"""
        },
        "Abstract base classes": {
            "description": "Define interfaces and abstract methods",
            "code": """from abc import ABCMeta, abstractmethod

class Shape:
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def area(self):
        pass"""
        },
        "String format() method": {
            "description": "Advanced string formatting",
            "code": """# New format() method
"{0} {1}".format("Hello", "World")
"{name} is {age}".format(name="Alice", age=30)

# Old way
"%s is %d" % ("Alice", 30)"""
        },
        "Dictionary comprehensions": {
            "description": "Create dicts with comprehension syntax",
            "code": """# Dictionary comprehension
squares = {x: x**2 for x in range(6)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Filter while creating
evens = {x: x**2 for x in range(10) if x % 2 == 0}"""
        },
        "Set comprehensions": {
            "description": "Create sets with comprehension syntax",
            "code": """# Set comprehension
unique_lengths = {len(word) for word in words}

# More complex example
squares = {x**2 for x in range(10) if x % 2 == 0}"""
        },
        "Set literals {1, 2, 3}": {
            "description": "Literal syntax for creating sets",
            "code": """# Old way
s = set([1, 2, 3])

# New literal syntax
s = {1, 2, 3}

# Empty set (still need set())
empty = set()  # {} is a dict!"""
        },
        "Multiple context managers": {
            "description": "Use multiple with statements in one line",
            "code": """# Python 2.7+
with open('in.txt') as inf, open('out.txt', 'w') as outf:
    outf.write(inf.read())

# Before: nested with statements"""
        },
        "argparse module": {
            "description": "Better command-line argument parsing",
            "code": """import argparse

parser = argparse.ArgumentParser(
    description='Process some integers.')
parser.add_argument('integers', type=int, nargs='+')
parser.add_argument('--sum', action='store_true')

args = parser.parse_args()"""
        },
        "Print as function, not statement": {
            "description": "print() function for consistency",
            "code": """# Python 2: print statement
print "Hello", "World"

# Python 3: print function
print("Hello", "World")
print("Error", file=sys.stderr)"""
        },
        "Unicode strings by default": {
            "description": "All strings are Unicode by default",
            "code": """# Python 3: Unicode by default
text = "Hello, 世界! 🌍"
print(text)  # Just works!

# Bytes are explicit
data = b"byte string"
encoded = text.encode('utf-8')"""
        },
        "Views and iterators instead of lists": {
            "description": "Memory-efficient dict methods",
            "code": """d = {'a': 1, 'b': 2}

# Python 3: returns views
keys = d.keys()    # dict_keys view
vals = d.values()  # dict_values view

# Still iterable, but memory efficient"""
        },
        "Integer division returns float": {
            "description": "True division by default",
            "code": """# Python 3
print(7 / 2)     # 3.5 (true division)
print(7 // 2)    # 3 (floor division)

# Python 2 needs from __future__ import division"""
        },
        "yield from syntax": {
            "description": "Delegate to sub-generators",
            "code": """def generator1():
    yield 1
    yield 2

def generator2():
    yield from generator1()  # Delegate!
    yield 3

list(generator2())  # [1, 2, 3]"""
        },
        "Virtual environments (venv)": {
            "description": "Built-in virtual environment support",
            "code": """# Create virtual environment
python3 -m venv myenv

# Activate (Unix)
source myenv/bin/activate

# Activate (Windows)
myenv\\Scripts\\activate.bat"""
        },
        "Flexible string representation": {
            "description": "Efficient internal string storage",
            "code": """# Python 3.3+ automatically uses:
# - ASCII for ASCII-only strings
# - Latin-1 for strings with chars < 256
# - UTF-16 or UTF-32 for others
# Saves memory automatically!"""
        },
        "asyncio module": {
            "description": "Asynchronous I/O framework",
            "code": """import asyncio

@asyncio.coroutine
def fetch_data():
    yield from asyncio.sleep(1)
    return "Data"

loop = asyncio.get_event_loop()
result = loop.run_until_complete(fetch_data())"""
        },
        "Enumerations (enum module)": {
            "description": "Type-safe enumerations",
            "code": """from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

print(Color.RED)      # Color.RED
print(Color.RED.value) # 1"""
        },
        "pathlib module": {
            "description": "Object-oriented filesystem paths",
            "code": """from pathlib import Path

# Create path objects
p = Path('data/file.txt')
print(p.name)        # 'file.txt'
print(p.suffix)      # '.txt'
print(p.parent)      # 'data'

# Readable path operations
p.read_text()
p.write_text("content")"""
        },
        "async/await syntax": {
            "description": "Native coroutine syntax",
            "code": """async def fetch_data(url):
    response = await http_client.get(url)
    return await response.json()

async def main():
    data = await fetch_data('https://api.example.com')
    print(data)

asyncio.run(main())"""
        },
        "Type hints (typing module)": {
            "description": "Optional static type annotations",
            "code": """from typing import List, Dict

def greet(name: str) -> str:
    return f"Hello, {name}"

def process(items: List[int]) -> Dict[str, int]:
    return {"sum": sum(items), "count": len(items)}"""
        },
        "Matrix multiplication operator (@)": {
            "description": "Dedicated operator for matrix multiplication",
            "code": """import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = A @ B

# Instead of np.dot(A, B)"""
        },
        "Unpacking generalizations": {
            "description": "Extended unpacking with *",
            "code": """# Multiple unpacking in literals
[*range(4), 4]  # [0, 1, 2, 3, 4]
{*range(4), 4}  # {0, 1, 2, 3, 4}

# In function calls
def func(a, b, c, d):
    print(a, b, c, d)

first = [1, 2]
func(*first, *[3, 4])"""
        },
        "f-strings (formatted string literals)": {
            "description": "Embedded expressions in strings",
            "code": """name = "Alice"
age = 30

# F-string with expressions
msg = f"Hello, {name}! You are {age} years old."

# With formatting
pi = 3.14159
f"Pi is approximately {pi:.2f}"

# Debug mode
f"{name=}, {age=}\" # "name='Alice', age=30\""""
        },
        "Underscores in numeric literals": {
            "description": "Improve readability of large numbers",
            "code": """# Make numbers readable
million = 1_000_000
hex_addr = 0xFF_FF_FF_FF
binary = 0b_1111_0000

print(million)  # 1000000"""
        },
        "Asynchronous generators": {
            "description": "Combine async and generators",
            "code": """async def async_range(count):
    for i in range(count):
        await asyncio.sleep(0.1)
        yield i

async for num in async_range(5):
    print(num)"""
        },
        "Variable annotations syntax": {
            "description": "Annotate variables without assignment",
            "code": """# Variable annotations
name: str
age: int
scores: List[float]

# With assignment
count: int = 0

# Class attributes
class MyClass:
    attr: str"""
        },
        "Dict preserves insertion order": {
            "description": "Dictionaries maintain order of insertion",
            "code": """# Python 3.6+ (3.7+ guaranteed by spec)
d = {}
d['first'] = 1
d['second'] = 2
d['third'] = 3

print(list(d.keys()))  # ['first', 'second', 'third']
# Order preserved!"""
        },
        "Data classes (@dataclass)": {
            "description": "Automatic class boilerplate generation",
            "code": """from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    
p = Point(1.5, 2.5)
print(p)  # Point(x=1.5, y=2.5)
# Auto __init__, __repr__, __eq__!"""
        },
        "breakpoint() built-in": {
            "description": "Easy debugging with built-in function",
            "code": """def calculate(x, y):
    result = x + y
    breakpoint()  # Drops into debugger!
    return result * 2

# Instead of import pdb; pdb.set_trace()"""
        },
        "Walrus operator := (assignment expressions)": {
            "description": "Assign and use value in one expression",
            "code": """# Read and check in one line
while (line := file.readline()) != "":
    process(line)

# In list comprehension
[y for x in data if (y := f(x)) > 0]

# In if statement
if (match := pattern.search(text)):
    print(match.group(1))"""
        },
        "Positional-only parameters": {
            "description": "Force parameters to be positional",
            "code": """def func(a, b, /, c, d):
    # a, b are positional-only
    # c, d can be positional or keyword
    pass

func(1, 2, 3, 4)        # OK
func(1, 2, c=3, d=4)    # OK
func(a=1, b=2, c=3, d=4) # Error!"""
        },
        "Dictionary merge operators (| and |=)": {
            "description": "Merge dictionaries with operators",
            "code": """d1 = {'a': 1, 'b': 2}
d2 = {'b': 3, 'c': 4}

# Merge with |
merged = d1 | d2  # {'a': 1, 'b': 3, 'c': 4}

# Update with |=
d1 |= d2  # d1 is updated"""
        },
        "Type hinting generics (list[int])": {
            "description": "Simpler generic type syntax",
            "code": """# Python 3.9+ simplified syntax
def process(items: list[int]) -> dict[str, int]:
    return {"sum": sum(items)}

# Before: from typing import List, Dict
# def process(items: List[int]) -> Dict[str, int]"""
        },
        "String removeprefix/removesuffix": {
            "description": "Remove prefix or suffix from strings",
            "code": """text = "Hello, World!"

# Remove prefix
text.removeprefix("Hello, ")  # "World!"

# Remove suffix
text.removesuffix("!")  # "Hello, World"

# Only if present (safer than replace)"""
        },
        "New parser (PEG)": {
            "description": "More powerful PEG parser replaces LL(1)",
            "code": """# PEG parser enables future syntax improvements
# and better error messages

# Example: relaxed grammar restrictions
with (open('file1.txt') as f1,
      open('file2.txt') as f2):
    pass"""
        },
        "Structural pattern matching (match/case)": {
            "description": "Pattern matching for complex conditionals",
            "code": """def handle_command(command):
    match command.split():
        case ["quit"]:
            return "Goodbye!"
        case ["look"]:
            return "You see nothing."
        case ["get", obj]:
            return f"You pick up {obj}"
        case ["go", direction]:
            return f"You go {direction}"
        case _:
            return "Unknown command\""""
        },
        "Parenthesized context managers": {
            "description": "Multi-line context managers",
            "code": """# Python 3.10+ allows parentheses
with (
    open('input.txt') as input_file,
    open('output.txt', 'w') as output_file,
):
    output_file.write(input_file.read())"""
        },
        "Union types with |": {
            "description": "Simpler union type syntax",
            "code": """# Python 3.10+
def process(value: int | str) -> int | None:
    if isinstance(value, int):
        return value * 2
    return None

# Before: Union[int, str]"""
        },
        "Exception groups and except*": {
            "description": "Handle multiple exception types",
            "code": """try:
    raise ExceptionGroup("Multiple errors", [
        ValueError("Bad value"),
        TypeError("Bad type")
    ])
except* ValueError as e:
    print(f"Value errors: {e.exceptions}")
except* TypeError as e:
    print(f"Type errors: {e.exceptions}")"""
        },
        "Fine-grained error locations": {
            "description": "Precise error highlighting in tracebacks",
            "code": """# Python 3.11+ shows exact expression
result = some_dict["key"]["nested"]["deep"]
#        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# KeyError with exact location highlighted

# Better error messages throughout"""
        },
        "TOML parser in standard library": {
            "description": "Built-in TOML file parsing",
            "code": """import tomllib

with open('config.toml', 'rb') as f:
    config = tomllib.load(f)

print(config['database']['host'])"""
        },
        "More flexible f-string parsing": {
            "description": "F-strings with quotes and more",
            "code": """# Python 3.12+ allows:
f"This is {f'nested {x}'} now!"

# Multiline f-strings
f'''
Multi
line
with {variables}
'''

# Backslashes in expressions
f"{'\n'.join(lines)}\""""
        },
        "Per-interpreter GIL (experimental)": {
            "description": "Separate GIL per sub-interpreter",
            "code": """# Experimental in 3.12
# Each sub-interpreter can run Python
# code in parallel without GIL blocking

# Enables better CPU parallelism
# Still being developed"""
        },
        "New type parameter syntax": {
            "description": "Cleaner generic type definitions",
            "code": """# Python 3.12+
def max[T](a: T, b: T) -> T:
    return a if a > b else b

class Stack[T]:
    def __init__(self):
        self.items: list[T] = []
    
    def push(self, item: T) -> None:
        self.items.append(item)"""
        },
        "Free-threaded Python (no GIL, experimental)": {
            "description": "Optional build without Global Interpreter Lock",
            "code": """# Python 3.13+ experimental
# Build with --disable-gil

# Enables true parallelism
import threading

def cpu_bound_work():
    # Can now run in parallel!
    total = sum(i**2 for i in range(1000000))

threads = [threading.Thread(target=cpu_bound_work) 
           for _ in range(4)]"""
        },
        "JIT compiler (experimental)": {
            "description": "Just-in-time compilation for performance",
            "code": """# Python 3.13+ experimental
# Copy-and-patch JIT compiler

# Automatically optimizes hot code paths
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# JIT makes this much faster!"""
        },
        "Improved interactive interpreter": {
            "description": "Better REPL with colors and editing",
            "code": """# Python 3.13+ new REPL features:
# - Syntax highlighting
# - Multiline editing
# - Auto-indentation
# - Color error messages
# - History with search

>>> def greet(name):
...     return f"Hello, {name}"
... 
>>> greet("World")
'Hello, World'"""
        },
        "iOS and Android support": {
            "description": "Official mobile platform support",
            "code": """# Python 3.13+ runs on mobile!

# iOS with framework support
# Android with native integration

# Build Python apps for:
# - iPhone/iPad
# - Android phones/tablets"""
        },
        "Warning framework": {
            "description": "Programmatic control over warning messages",
            "code": """import warnings

# Issue a warning
warnings.warn("This is deprecated", DeprecationWarning)

# Filter warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Custom warning
class CustomWarning(UserWarning):
    pass

warnings.warn("Custom message", CustomWarning)"""
        },
        "Type/class unification": {
            "description": "Types and classes unified into single hierarchy",
            "code": """# Python 2.2+ types are classes
print type(5)          # <type 'int'>
print type(int)        # <type 'type'>

# Can subclass built-in types
class MyInt(int):
    def double(self):
        return self * 2

num = MyInt(5)
print num.double()  # 10"""
        },
        "Import from ZIP files": {
            "description": "Import modules directly from ZIP archives",
            "code": """import sys
sys.path.append('mylib.zip')

# Now can import from ZIP
from mypackage import mymodule

# Or use zipimport
import zipimport
importer = zipimport.zipimporter('mylib.zip')
module = importer.load_module('mymodule')"""
        },
        "Extended slices": {
            "description": "Multi-dimensional and step slicing",
            "code": """# Step in slices
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print numbers[::2]      # [0, 2, 4, 6, 8]
print numbers[1::2]     # [1, 3, 5, 7, 9]
print numbers[::-1]     # Reverse: [9, 8, 7, ...]

# Works with strings too
text = "Hello World"
print text[::2]         # "HloWrd\""""
        },
        "Universal newline support": {
            "description": "Handle different line ending conventions",
            "code": """# Opens file in universal newline mode
# Converts \\r\\n (Windows), \\r (Mac), \\n (Unix) to \\n
with open('file.txt', 'rU') as f:
    for line in f:
        print line  # All newlines normalized

# Python 3+ does this by default"""
        },
        "Absolute/relative imports": {
            "description": "Explicit control over import resolution",
            "code": """# Absolute import (from project root)
from package.subpackage import module

# Relative import (from current package)
from . import sibling_module
from .. import parent_module
from ..sibling_package import cousin

# Prevents name conflicts with stdlib"""
        },
        "Enhanced generators (send, throw, close)": {
            "description": "Two-way communication with generators",
            "code": """def echo_generator():
    while True:
        received = yield
        print "Received:", received

gen = echo_generator()
gen.next()              # Prime it
gen.send("Hello")       # Send value
gen.send("World")       # Send another
gen.close()             # Close generator"""
        },
        "Forward compatibility with Python 3.0": {
            "description": "Features backported to ease migration",
            "code": """# Python 2.6 includes 3.0 features
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

print("Function syntax")  # 3.0 style
x = 5 / 2                 # Returns 2.5
s = "Unicode by default\""""
        },
        "bytes type (backport from 3.0)": {
            "description": "Explicit byte string type",
            "code": """# Python 2.6+ has bytes (alias for str)
b = bytes("hello")
b = b"hello"  # Byte literal

# Python 3: bytes vs str distinction
text = "hello"      # Unicode string
data = b"hello"     # Byte string
encoded = text.encode('utf-8')"""
        },
        "Multiprocessing module": {
            "description": "True parallelism with separate processes",
            "code": """from multiprocessing import Process, Pool

def worker(name):
    print "Worker:", name

# Create process
p = Process(target=worker, args=("Alice",))
p.start()
p.join()

# Use pool for multiple tasks
with Pool(4) as pool:
    pool.map(worker, ["A", "B", "C", "D"])"""
        },
        "OrderedDict, Counter in collections": {
            "description": "Specialized dict types for common patterns",
            "code": """from collections import OrderedDict, Counter

# OrderedDict preserves insertion order
od = OrderedDict()
od['first'] = 1
od['second'] = 2

# Counter for frequency counting
words = ['apple', 'orange', 'apple', 'pear']
counter = Counter(words)
print counter['apple']      # 2
print counter.most_common(1)  # [('apple', 2)]"""
        },
        "New syntax for exceptions": {
            "description": "Modern exception handling syntax",
            "code": """# Python 3 syntax
try:
    risky_operation()
except ValueError as e:  # 'as' instead of comma
    print(f"Error: {e}")
except (TypeError, KeyError) as e:
    print(f"Type or Key error: {e}")

# Python 2 used: except ValueError, e:"""
        },
        "Removed old-style classes": {
            "description": "All classes inherit from object",
            "code": """# Python 3: all classes are new-style
class MyClass:  # Implicitly inherits from object
    pass

# Equivalent to:
class MyClass(object):
    pass

# Benefits: properties, descriptors, __slots__, etc."""
        },
        "Ordered dictionaries": {
            "description": "Dictionary that remembers insertion order",
            "code": """from collections import OrderedDict

# Maintains order (Python 3.7+ dicts do this)
od = OrderedDict()
od['z'] = 1
od['a'] = 2
od['m'] = 3

print(list(od.keys()))  # ['z', 'a', 'm']

# Move to end
od.move_to_end('z')"""
        },
        "Format specifier for thousands separator": {
            "description": "Comma formatting for numbers",
            "code": """# Thousands separator
large_num = 1234567890

print("{:,}".format(large_num))     # 1,234,567,890
print(f"{large_num:,}")             # Python 3.6+

# With decimals
pi = 3141.59265
print(f"{pi:,.2f}")  # 3,141.59"""
        },
        "Float repr improvements": {
            "description": "Better float string representation",
            "code": """# Python 3.1+ improved float display
x = 1.1
print(repr(x))      # '1.1' (not '1.1000000000000001')

# More predictable round-trip
y = eval(repr(x))
assert x == y       # True"""
        },
        "Field name syntax for format()": {
            "description": "Named and numbered fields in format strings",
            "code": """# Positional
"{0} {1}".format("Hello", "World")
"{1} {0}".format("World", "Hello")

# Named
"{name} is {age} years old".format(name="Alice", age=30)

# Attribute access
"{person.name}".format(person=obj)"""
        },
        "importlib module": {
            "description": "Programmatic importing of modules",
            "code": """import importlib

# Dynamic import
module_name = "math"
math = importlib.import_module(module_name)
print(math.pi)

# Reload module
importlib.reload(math)

# Get module path
spec = importlib.util.find_spec("numpy")
print(spec.origin)"""
        },
        "Concurrent.futures module": {
            "description": "High-level async execution interface",
            "code": """from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_url(url):
    # Fetch URL content
    return url, len(data)

urls = ['http://a.com', 'http://b.com']

# Execute in parallel
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(fetch_url, url) for url in urls]
    
    for future in as_completed(futures):
        url, size = future.result()
        print(f"{url}: {size} bytes")"""
        },
        "argparse replaces optparse": {
            "description": "Better argument parsing API",
            "code": """import argparse

parser = argparse.ArgumentParser(
    description='Process some integers.')

parser.add_argument('integers', type=int, nargs='+',
                    help='integers to process')
parser.add_argument('--sum', action='store_true',
                    help='sum the integers')
parser.add_argument('--verbose', '-v', action='count',
                    default=0)

args = parser.parse_args()"""
        },
        "GIL improvements": {
            "description": "Better Global Interpreter Lock handling",
            "code": """# Python 3.2+ improved GIL
# - Better thread switching
# - Less starvation
# - Improved I/O thread performance

import threading

def cpu_bound():
    # CPU intensive work
    total = sum(i**2 for i in range(1000000))

# Multiple threads benefit from GIL improvements
threads = [threading.Thread(target=cpu_bound) for _ in range(4)]
for t in threads:
    t.start()"""
        },
        "Stable ABI": {
            "description": "Stable Application Binary Interface for C extensions",
            "code": """# Python 3.2+ Stable ABI
# C extensions built against stable ABI work across
# Python versions without recompilation

# In C extension setup.py:
# ext_modules = [Extension(
#     "mymodule",
#     sources=["mymodule.c"],
#     py_limited_api=True  # Enable stable ABI
# )]"""
        },
        "Hash randomization for security": {
            "description": "Randomized hash to prevent DoS attacks",
            "code": """# Python 3.2+ randomizes hash on startup
# Prevents hash collision DoS attacks

d = {}
# Hash values differ between runs
print(hash("test"))  # Different each time

# Disable with PYTHONHASHSEED=0
# Enable with PYTHONHASHSEED=random (default)"""
        },
        "New launcher for Windows": {
            "description": "Python launcher for Windows (py.exe)",
            "code": """# Windows launcher allows version selection
# py -3.10 script.py    # Run with Python 3.10
# py -3.11 script.py    # Run with Python 3.11
# py -2.7 script.py     # Run with Python 2.7

# Shebang support
#!/usr/bin/env python3.10

# Launcher reads this and uses correct version"""
        },
        "Implicit namespace packages": {
            "description": "Packages without __init__.py",
            "code": """# Python 3.3+ allows packages without __init__.py

# Directory structure:
# mypackage/
#   module1.py
#   module2.py
# (no __init__.py needed!)

# Can still import:
from mypackage import module1

# Useful for splitting packages across locations"""
        },
        "Statistics module": {
            "description": "Statistical functions in standard library",
            "code": """import statistics

data = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9]

print(statistics.mean(data))      # Average
print(statistics.median(data))    # Middle value
print(statistics.mode(data))      # Most common: 5
print(statistics.stdev(data))     # Standard deviation
print(statistics.variance(data))  # Variance"""
        },
        "pip included by default": {
            "description": "Package manager included in Python",
            "code": """# Python 3.4+ includes pip by default

# Install packages
# pip install requests

# Upgrade packages
# pip install --upgrade requests

# List installed
# pip list

# Install from requirements.txt
# pip install -r requirements.txt"""
        },
        "Coroutines with async def": {
            "description": "Native coroutine definition syntax",
            "code": """async def fetch_data(url):
    # This is a native coroutine
    response = await http_client.get(url)
    return await response.json()

async def main():
    data = await fetch_data('https://api.example.com')
    print(data)

# Run the coroutine
import asyncio
asyncio.run(main())"""
        },
        "Secrets module": {
            "description": "Cryptographically strong random numbers",
            "code": """import secrets

# Random token for URLs
token = secrets.token_urlsafe(16)

# Random hex string
hex_token = secrets.token_hex(16)

# Random integer
random_num = secrets.randbelow(100)

# Choose random element (securely)
winner = secrets.choice(['Alice', 'Bob', 'Charlie'])"""
        },
        "Context variables": {
            "description": "Context-local state for async code",
            "code": """from contextvars import ContextVar

# Create context variable
request_id = ContextVar('request_id', default=None)

async def handle_request(req_id):
    # Set for this context
    request_id.set(req_id)
    await process()

async def process():
    # Access context variable
    print(f"Processing: {request_id.get()}")"""
        },
        "Postponed annotation evaluation": {
            "description": "Defer type hint evaluation",
            "code": """from __future__ import annotations

# Self-referencing type hints
class Node:
    def __init__(self, value: int, next: Node = None):
        self.value = value
        self.next = next

# Forward references work without quotes
def create_tree() -> BinaryTree:
    return BinaryTree()

class BinaryTree:
    pass"""
        },
        "Dict order guaranteed by language spec": {
            "description": "Dictionary order is part of language specification",
            "code": """# Python 3.7+ guarantees dict order
# (3.6 had it as implementation detail)

d = {'first': 1, 'second': 2, 'third': 3}
print(list(d.keys()))  # ['first', 'second', 'third']

# Guaranteed to iterate in insertion order
for key in d:
    print(key)  # first, second, third"""
        },
        "Module __getattr__ and __dir__": {
            "description": "Customize module attribute access",
            "code": """# In module.py:
def __getattr__(name):
    if name == 'dynamic_attr':
        return "Dynamically generated!"
    raise AttributeError(f"module has no attribute {name}")

def __dir__():
    return ['dynamic_attr', 'other_attrs']

# Usage:
import module
print(module.dynamic_attr)  # Works!"""
        },
        "f-string debugging (f'{var=}')": {
            "description": "Debug print with f-string = syntax",
            "code": """# Python 3.8+ debug syntax
x = 10
y = 20

print(f"{x=}")        # x=10
print(f"{y=}")        # y=20
print(f"{x + y=}")    # x + y=30

# Great for quick debugging!
name = "Alice"
print(f"{name=}, {len(name)=}")
# name='Alice', len(name)=5"""
        },
        "typing.Literal, Final, Protocol": {
            "description": "Advanced typing features",
            "code": """from typing import Literal, Final, Protocol

# Literal types
def move(direction: Literal['up', 'down', 'left', 'right']):
    pass

# Final (cannot be reassigned)
MAX_SIZE: Final = 100

# Protocol (structural subtyping)
class Drawable(Protocol):
    def draw(self) -> None: ..."""
        },
        "Parallel filesystem cache": {
            "description": "Faster imports with concurrent bytecode compilation",
            "code": """# Python 3.8+ compiles .py to .pyc in parallel
# when using multiple processes

# Improves startup time for applications with
# many modules

# Automatic optimization, no code changes needed
import module1, module2, module3  # Compiled in parallel"""
        },
        "Timezone support improvements": {
            "description": "Better timezone handling in datetime",
            "code": """from datetime import datetime, timezone, timedelta

# UTC timezone
now_utc = datetime.now(timezone.utc)

# Custom timezone
pst = timezone(timedelta(hours=-8))
now_pst = datetime.now(pst)

# Python 3.9+ improved timezone support
print(now_utc.isoformat())"""
        },
        "Better error messages": {
            "description": "More helpful error messages and suggestions",
            "code": """# Python 3.10+ gives helpful suggestions

# NameError suggests similar names
name = "Alice"
print(nam)  # Did you mean 'name'?

# AttributeError suggests corrections
"hello".uppor()  # Did you mean 'upper'?

# SyntaxError shows better context
if x = 5:  # Shows: Maybe you meant '==' instead?"""
        },
        "Precise line numbers in tracebacks": {
            "description": "Exact column positions in error messages",
            "code": """# Python 3.10+ shows exact error location

result = some_dict["key"]["nested"]["value"]
#        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Shows exactly which expression failed

# Before: only line number
# Now: highlights specific part"""
        },
        "10-60% faster than 3.10": {
            "description": "Major performance improvements",
            "code": """# Python 3.11 performance improvements:
# - Faster function calls
# - Faster loops
# - Faster attribute access
# - Better bytecode
# - Inline caching

# No code changes needed!
# Just upgrade to Python 3.11"""
        },
        "Self type annotation": {
            "description": "Self type for method return values",
            "code": """from typing import Self

class Builder:
    def set_name(self, name: str) -> Self:
        self.name = name
        return self
    
    def set_age(self, age: int) -> Self:
        self.age = age
        return self

# Method chaining with proper types
builder = Builder().set_name("Alice").set_age(30)"""
        },
        "Faster startup": {
            "description": "Python starts up faster",
            "code": """# Python 3.11 faster startup due to:
# - Cached frozen modules
# - Optimized import system
# - Better bytecode generation

# Startup time reduced by ~10-15%
# Especially noticeable for short scripts

import time
start = time.time()
# ... script code ...
print(f"Took {time.time() - start:.3f}s")"""
        },
        "Buffer protocol improvements": {
            "description": "Better memory buffer handling",
            "code": """# Python 3.12+ improved buffer protocol
import array

# More efficient memory operations
arr = array.array('i', [1, 2, 3, 4, 5])

# Zero-copy operations improved
memview = memoryview(arr)
bytes_view = memview.cast('B')

# Better performance for NumPy, etc."""
        },
        "Linux perf profiler support": {
            "description": "Native support for Linux perf profiler",
            "code": """# Python 3.12+ works with Linux perf

# Run with perf support:
# python -X perf script.py

# Then profile:
# perf record -F 9999 -g python -X perf script.py
# perf report

# Shows Python functions in perf output!"""
        },
        "7% faster than 3.11": {
            "description": "Continued performance improvements",
            "code": """# Python 3.12 performance gains:
# - Comprehension inlining
# - Better memory management
# - Optimized error handling
# - Improved GC

# Average 7% faster than 3.11
# Some workloads up to 20% faster"""
        },
        "New REPL with colors and multiline editing": {
            "description": "Modern interactive shell",
            "code": """# Python 3.13+ new REPL features:
>>> def greet(name):
...     return f"Hello, {name}"
...     # Multiline editing works!
...
>>> greet("World")  # Syntax highlighting!
'Hello, World'

# Color output, history, autocomplete"""
        },
        "Remove deprecated APIs": {
            "description": "Clean up old deprecated features",
            "code": """# Python 3.13 removes old deprecated APIs:
# - imp module (use importlib)
# - Some old unittest assertions
# - Deprecated threading methods

# Use modern replacements:
import importlib  # Not imp
# Modern APIs only"""
        },
        "Continued JIT improvements": {
            "description": "Enhanced just-in-time compilation",
            "code": """# Python 3.14 JIT improvements
# - Better hot path detection
# - More optimizations
# - Lower overhead

def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# JIT makes this faster automatically"""
        },
        "Enhanced type system features": {
            "description": "More powerful type checking",
            "code": """from typing import TypeVar, Generic

# Better generic support
T = TypeVar('T')

class Stack(Generic[T]):
    def push(self, item: T) -> None: ...
    def pop(self) -> T: ...

# Enhanced inference
stack: Stack[int] = Stack()"""
        },
        "Performance optimizations": {
            "description": "Various speed improvements",
            "code": """# Python 3.14 optimizations:
# - Faster dictionary operations
# - Better comprehension performance
# - Optimized function calls
# - Improved memory usage

# All automatic - just upgrade!"""
        },
        "Better mobile platform support": {
            "description": "Improved iOS and Android integration",
            "code": """# Python 3.14 mobile improvements:
# - Better framework integration
# - Native UI bindings
# - Optimized for mobile CPUs
# - Smaller binary size

# Build mobile apps with Python!"""
        },
        "Immortal objects refinements": {
            "description": "Optimized permanent objects",
            "code": """# Python 3.14 immortal objects:
# - Common objects like None, True, False
# - Small integers (-5 to 256)
# - Empty tuples
# Never deallocated, reducing overhead

# Automatic optimization
x = True  # Immortal object
y = 42    # Likely immortal"""
        },
        "Further C-API modernization": {
            "description": "Updated C API for extensions",
            "code": """# Python 3.14 C-API improvements:
# - Cleaner interfaces
# - Better documentation
# - More stable ABI
# - Easier extension development

# For C extension developers"""
        }
    }


def get_all_versions():
    """Returns detailed information about all Python versions"""
    return {
        "versions": [
            {
                "version": "2.0.1",
                "release_date": "June 22, 2001",
                "year": 2001,
                "era": "Python 2.0",
                "highlights": [
                    "Garbage collection of cycles",
                    "Unicode support (16-bit)",
                    "List comprehensions introduced",
                    "Augmented assignment (+=, -=, etc.)",
                    "String methods instead of string module"
                ],
                "major_peps": ["PEP 200", "PEP 227"],
                "description": "Python 2.0 marked a major milestone with garbage collection and Unicode support."
            },
            {
                "version": "2.1",
                "release_date": "April 15, 2001",
                "year": 2001,
                "era": "Python 2.1",
                "highlights": [
                    "Nested scopes",
                    "Rich comparisons",
                    "Weak references",
                    "Function attributes",
                    "Warning framework"
                ],
                "major_peps": ["PEP 227", "PEP 207"],
                "description": "Introduced nested scopes and rich comparison methods."
            },
            {
                "version": "2.2",
                "release_date": "December 21, 2001",
                "year": 2001,
                "era": "Python 2.2",
                "highlights": [
                    "New-style classes",
                    "Iterators and generators",
                    "Division operator changes (//) ",
                    "Property decorators",
                    "Type/class unification"
                ],
                "major_peps": ["PEP 252", "PEP 253", "PEP 255"],
                "description": "Major object model overhaul with new-style classes and generators."
            },
            {
                "version": "2.3",
                "release_date": "July 29, 2003",
                "year": 2003,
                "era": "Python 2.3",
                "highlights": [
                    "Boolean type introduced",
                    "Import from ZIP files",
                    "Enumerate function",
                    "Extended slices",
                    "Universal newline support"
                ],
                "major_peps": ["PEP 218", "PEP 273", "PEP 279"],
                "description": "Added the bool type and many quality-of-life improvements."
            },
            {
                "version": "2.4",
                "release_date": "November 30, 2004",
                "year": 2004,
                "era": "Python 2.4",
                "highlights": [
                    "Decorators (@decorator syntax)",
                    "Generator expressions",
                    "Built-in set() and frozenset()",
                    "reversed() function",
                    "sorted() function"
                ],
                "major_peps": ["PEP 318", "PEP 289"],
                "description": "Introduced decorators and generator expressions."
            },
            {
                "version": "2.5",
                "release_date": "September 19, 2006",
                "year": 2006,
                "era": "Python 2.5",
                "highlights": [
                    "Conditional expressions (ternary operator)",
                    "with statement",
                    "try/except/finally unified",
                    "Absolute/relative imports",
                    "Enhanced generators (send, throw, close)"
                ],
                "major_peps": ["PEP 308", "PEP 343", "PEP 342"],
                "description": "Added the with statement and conditional expressions."
            },
            {
                "version": "2.6",
                "release_date": "October 1, 2008",
                "year": 2008,
                "era": "Python 2.6",
                "highlights": [
                    "Forward compatibility with Python 3.0",
                    "Abstract base classes",
                    "bytes type (backport from 3.0)",
                    "String format() method",
                    "Multiprocessing module"
                ],
                "major_peps": ["PEP 3101", "PEP 3119"],
                "description": "Bridge release to ease transition to Python 3."
            },
            {
                "version": "2.7",
                "release_date": "July 3, 2010",
                "year": 2010,
                "era": "Python 2.7",
                "highlights": [
                    "Dictionary comprehensions",
                    "Set comprehensions",
                    "Set literals {1, 2, 3}",
                    "Multiple context managers",
                    "OrderedDict, Counter in collections",
                    "argparse module"
                ],
                "major_peps": ["PEP 372", "PEP 389"],
                "description": "Final Python 2.x release, supported until 2020. Most popular Python 2 version."
            },
            {
                "version": "3.0",
                "release_date": "December 3, 2008",
                "year": 2008,
                "era": "Python 3.0",
                "highlights": [
                    "Print as function, not statement",
                    "Unicode strings by default",
                    "Views and iterators instead of lists",
                    "New syntax for exceptions",
                    "Integer division returns float",
                    "Removed old-style classes"
                ],
                "major_peps": ["PEP 3000", "PEP 3105", "PEP 3107"],
                "description": "Major breaking release that cleaned up language inconsistencies."
            },
            {
                "version": "3.1",
                "release_date": "June 27, 2009",
                "year": 2009,
                "era": "Python 3.1",
                "highlights": [
                    "Ordered dictionaries",
                    "Format specifier for thousands separator",
                    "Float repr improvements",
                    "Field name syntax for format()",
                    "importlib module"
                ],
                "major_peps": ["PEP 372", "PEP 378"],
                "description": "First refinement of Python 3 with performance improvements."
            },
            {
                "version": "3.2",
                "release_date": "February 20, 2011",
                "year": 2011,
                "era": "Python 3.2",
                "highlights": [
                    "Concurrent.futures module",
                    "argparse replaces optparse",
                    "GIL improvements",
                    "Stable ABI",
                    "Hash randomization for security"
                ],
                "major_peps": ["PEP 3147", "PEP 3149", "PEP 3333"],
                "description": "Added concurrent programming support and security enhancements."
            },
            {
                "version": "3.3",
                "release_date": "September 29, 2012",
                "year": 2012,
                "era": "Python 3.3",
                "highlights": [
                    "yield from syntax",
                    "Virtual environments (venv)",
                    "Flexible string representation",
                    "New launcher for Windows",
                    "Implicit namespace packages"
                ],
                "major_peps": ["PEP 380", "PEP 405", "PEP 393"],
                "description": "Introduced yield from and virtual environments."
            },
            {
                "version": "3.4",
                "release_date": "March 16, 2014",
                "year": 2014,
                "era": "Python 3.4",
                "highlights": [
                    "asyncio module",
                    "Enumerations (enum module)",
                    "pathlib module",
                    "Statistics module",
                    "pip included by default"
                ],
                "major_peps": ["PEP 3156", "PEP 435", "PEP 428"],
                "description": "Major async programming foundation with asyncio."
            },
            {
                "version": "3.5",
                "release_date": "September 13, 2015",
                "year": 2015,
                "era": "Python 3.5",
                "highlights": [
                    "async/await syntax",
                    "Type hints (typing module)",
                    "Matrix multiplication operator (@)",
                    "Unpacking generalizations",
                    "Coroutines with async def"
                ],
                "major_peps": ["PEP 492", "PEP 484", "PEP 465"],
                "description": "Revolutionary release with async/await and type hints."
            },
            {
                "version": "3.6",
                "release_date": "December 23, 2016",
                "year": 2016,
                "era": "Python 3.6",
                "highlights": [
                    "f-strings (formatted string literals)",
                    "Underscores in numeric literals",
                    "Asynchronous generators",
                    "Variable annotations syntax",
                    "Secrets module",
                    "Dict preserves insertion order"
                ],
                "major_peps": ["PEP 498", "PEP 515", "PEP 525"],
                "description": "F-strings revolutionized string formatting. Huge adoption boost."
            },
            {
                "version": "3.7",
                "release_date": "June 27, 2018",
                "year": 2018,
                "era": "Python 3.7",
                "highlights": [
                    "Data classes (@dataclass)",
                    "Context variables",
                    "Postponed annotation evaluation",
                    "breakpoint() built-in",
                    "Dict order guaranteed by language spec",
                    "Module __getattr__ and __dir__"
                ],
                "major_peps": ["PEP 557", "PEP 562", "PEP 553"],
                "description": "Data classes made Python more expressive and convenient."
            },
            {
                "version": "3.8",
                "release_date": "October 14, 2019",
                "year": 2019,
                "era": "Python 3.8",
                "highlights": [
                    "Walrus operator := (assignment expressions)",
                    "Positional-only parameters",
                    "f-string debugging (f'{var=}')",
                    "typing.Literal, Final, Protocol",
                    "Parallel filesystem cache"
                ],
                "major_peps": ["PEP 572", "PEP 570", "PEP 587"],
                "description": "Walrus operator enabled more concise code patterns."
            },
            {
                "version": "3.9",
                "release_date": "October 5, 2020",
                "year": 2020,
                "era": "Python 3.9",
                "highlights": [
                    "Dictionary merge operators (| and |=)",
                    "Type hinting generics (list[int])",
                    "String removeprefix/removesuffix",
                    "New parser (PEG)",
                    "Timezone support improvements"
                ],
                "major_peps": ["PEP 584", "PEP 585", "PEP 617"],
                "description": "New PEG parser and simplified type hints syntax."
            },
            {
                "version": "3.10",
                "release_date": "October 4, 2021",
                "year": 2021,
                "era": "Python 3.10",
                "highlights": [
                    "Structural pattern matching (match/case)",
                    "Parenthesized context managers",
                    "Better error messages",
                    "Union types with |",
                    "Precise line numbers in tracebacks"
                ],
                "major_peps": ["PEP 634", "PEP 635", "PEP 636"],
                "description": "Pattern matching brought powerful new control flow."
            },
            {
                "version": "3.11",
                "release_date": "October 24, 2022",
                "year": 2022,
                "era": "Python 3.11",
                "highlights": [
                    "10-60% faster than 3.10",
                    "Exception groups and except*",
                    "Fine-grained error locations",
                    "Self type annotation",
                    "TOML parser in standard library",
                    "Faster startup"
                ],
                "major_peps": ["PEP 654", "PEP 657", "PEP 680"],
                "description": "Major performance improvements and better error messages."
            },
            {
                "version": "3.12",
                "release_date": "October 2, 2023",
                "year": 2023,
                "era": "Python 3.12",
                "highlights": [
                    "More flexible f-string parsing",
                    "Per-interpreter GIL (experimental)",
                    "Buffer protocol improvements",
                    "New type parameter syntax",
                    "Linux perf profiler support",
                    "7% faster than 3.11"
                ],
                "major_peps": ["PEP 695", "PEP 701", "PEP 684"],
                "description": "Continued performance gains and improved f-strings."
            },
            {
                "version": "3.13",
                "release_date": "October 7, 2024",
                "year": 2024,
                "era": "Python 3.13",
                "highlights": [
                    "Free-threaded Python (no GIL, experimental)",
                    "JIT compiler (experimental)",
                    "Improved interactive interpreter",
                    "New REPL with colors and multiline editing",
                    "iOS and Android support",
                    "Remove deprecated APIs"
                ],
                "major_peps": ["PEP 703", "PEP 744", "PEP 667"],
                "description": "Experimental GIL removal and JIT compilation for future performance."
            },
            {
                "version": "3.14",
                "release_date": "October 2025 (planned)",
                "year": 2025,
                "era": "Python 3.14",
                "highlights": [
                    "Continued JIT improvements",
                    "Enhanced type system features",
                    "Performance optimizations",
                    "Better mobile platform support",
                    "Immortal objects refinements",
                    "Further C-API modernization"
                ],
                "major_peps": ["PEP 730", "PEP 741"],
                "description": "Latest version with cutting-edge performance and modern language features."
            }
        ]
    }


def get_code_examples():
    """Returns code examples comparing Python 2.0.1 and 3.14"""
    return {
        "python_2_0_1": {
            "version": "2.0.1",
            "title": "Python 2.0.1 - File Processing Example",
            "code": '''# Python 2.0.1 - Released June 2001
# Example: Reading a file, processing data, and creating a report

import string
import sys

def process_user_data(filename):
    """Process user data from a CSV-like file"""
    
    # Open file (no context manager yet)
    try:
        file_handle = open(filename, 'r')
        lines = file_handle.readlines()
        file_handle.close()
    except IOError, e:  # Old exception syntax
        print "Error reading file:", str(e)
        return None
    
    # Process data
    users = []
    for line in lines:
        # Skip header and empty lines
        line = string.strip(line)  # Using string module
        if line == '' or line[0] == '#':
            continue
        
        # Split by comma
        parts = string.split(line, ',')
        if len(parts) != 3:
            continue
        
        name = string.strip(parts[0])
        age = string.strip(parts[1])
        city = string.strip(parts[2])
        
        # Create user dictionary
        user = {}
        user['name'] = name
        user['age'] = int(age)
        user['city'] = city
        users.append(user)
    
    # Calculate statistics (no built-in functions)
    total_age = 0
    count = 0
    cities = {}
    
    for user in users:
        total_age = total_age + user['age']
        count = count + 1
        
        city = user['city']
        if cities.has_key(city):
            cities[city] = cities[city] + 1
        else:
            cities[city] = 1
    
    # Calculate average
    if count > 0:
        avg_age = float(total_age) / float(count)
    else:
        avg_age = 0.0
    
    # Find most common city (manual)
    max_count = 0
    most_common_city = None
    for city, city_count in cities.items():
        if city_count > max_count:
            max_count = city_count
            most_common_city = city
    
    # Print report (print statement, not function)
    print "=" * 50
    print "User Data Report"
    print "=" * 50
    print "Total users:", count
    print "Average age: %.2f" % avg_age
    print "Most common city:", most_common_city, "(%d users)" % max_count
    print
    print "All users:"
    print "-" * 50
    
    # Sort users by age (using lambda)
    users.sort(lambda a, b: cmp(a['age'], b['age']))
    
    for user in users:
        # Old-style string formatting
        print "%-20s Age: %3d  City: %s" % \\
              (user['name'], user['age'], user['city'])
    
    return users


# Create sample data file
def create_sample_file():
    f = open('users.txt', 'w')
    f.write("# User data\\n")
    f.write("Alice Smith, 28, New York\\n")
    f.write("Bob Jones, 35, San Francisco\\n")
    f.write("Charlie Brown, 22, New York\\n")
    f.write("Diana Prince, 31, Chicago\\n")
    f.write("Eve Adams, 29, San Francisco\\n")
    f.close()


# Main execution
if __name__ == '__main__':
    create_sample_file()
    result = process_user_data('users.txt')
    
    if result is None:
        sys.exit(1)
''',
            "issues": [
                "No context managers (with statement)",
                "Old exception syntax (except E, e)",
                "Print statements instead of function",
                "String module instead of string methods",
                "No list comprehensions used effectively",
                "Manual dictionary operations",
                "Old-style string formatting only",
                "No type hints",
                "has_key() instead of 'in' operator",
                "cmp() function for sorting",
                "Manual statistics calculations"
            ]
        },
        "python_3_14": {
            "version": "3.14",
            "title": "Python 3.14 - Same Example, Modern Python",
            "code": '''# Python 3.14 - Released October 2025
# Example: Reading a file, processing data, and creating a report

from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Self
import statistics


@dataclass
class User:
    """User data model with modern dataclass"""
    name: str
    age: int
    city: str
    
    @classmethod
    def from_csv_line(cls, line: str) -> Self | None:
        """Parse user from CSV line"""
        parts = [part.strip() for part in line.split(',')]
        
        # Pattern matching for validation
        match parts:
            case [name, age, city]:
                return cls(name=name, age=int(age), city=city)
            case _:
                return None


def process_user_data(filename: str | Path) -> list[User] | None:
    """Process user data from a CSV-like file"""
    
    filepath = Path(filename)
    
    # Context manager with better error handling
    try:
        with filepath.open('r') as file:
            lines = file.readlines()
    except OSError as e:  # Modern exception syntax
        print(f"Error reading file: {e}")
        return None
    
    # List comprehension with walrus operator and filtering
    users = [
        user for line in lines
        if (stripped := line.strip())  # Walrus operator
        and not stripped.startswith('#')  # Method instead of indexing
        and (user := User.from_csv_line(stripped)) is not None
    ]
    
    # No users found
    if not users:
        return []
    
    # Modern statistics with built-in module
    ages = [user.age for user in users]
    avg_age = statistics.mean(ages)
    
    # Counter for finding most common city
    city_counter = Counter(user.city for user in users)
    most_common_city, max_count = city_counter.most_common(1)[0]
    
    # Print report with f-strings and modern formatting
    print("=" * 50)
    print("User Data Report")
    print("=" * 50)
    print(f"Total users: {len(users)}")
    print(f"Average age: {avg_age:.2f}")
    print(f"Most common city: {most_common_city} ({max_count} users)")
    print()
    print("All users:")
    print("-" * 50)
    
    # Modern sorting with key function
    sorted_users = sorted(users, key=lambda u: u.age)
    
    for user in sorted_users:
        # F-string with alignment
        print(f"{user.name:<20} Age: {user.age:3d}  City: {user.city}")
    
    return users


def create_sample_file() -> None:
    """Create sample data file using modern Path API"""
    content = """# User data
Alice Smith, 28, New York
Bob Jones, 35, San Francisco
Charlie Brown, 22, New York
Diana Prince, 31, Chicago
Eve Adams, 29, San Francisco
"""
    Path('users.txt').write_text(content)


# Main execution with modern type hints
def main() -> int:
    """Main entry point"""
    create_sample_file()
    result = process_user_data('users.txt')
    
    # Pattern matching for result handling
    match result:
        case None:
            return 1
        case []:
            print("No users found")
            return 0
        case _:
            return 0


if __name__ == '__main__':
    exit(main())
''',
            "improvements": [
                "Dataclasses for clean data models",
                "Type hints for better code clarity",
                "Context managers (with statement)",
                "F-strings for elegant formatting",
                "Modern exception handling syntax",
                "List comprehensions with walrus operator",
                "Pattern matching (match/case)",
                "Built-in statistics module",
                "Counter for frequency counting",
                "Pathlib for file operations",
                "String methods instead of string module",
                "Self type for better typing",
                "Union types with | operator"
            ]
        },
        "comparison": {
            "title": "Key Improvements from Python 2.0.1 to 3.14",
            "points": [
                {
                    "category": "Readability",
                    "improvement": "F-strings and modern syntax make code 40% more readable"
                },
                {
                    "category": "Safety",
                    "improvement": "Type hints catch errors before runtime"
                },
                {
                    "category": "Expressiveness",
                    "improvement": "Pattern matching, dataclasses, and comprehensions reduce boilerplate"
                },
                {
                    "category": "Standard Library",
                    "improvement": "Rich built-in modules (pathlib, statistics, dataclasses)"
                },
                {
                    "category": "Performance",
                    "improvement": "Python 3.14 is 5-10x faster than Python 2.0.1"
                },
                {
                    "category": "Code Size",
                    "improvement": "Same functionality in 30% less code"
                }
            ]
        }
    }
