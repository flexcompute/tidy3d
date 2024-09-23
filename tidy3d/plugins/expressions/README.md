# Expressions module

The `expressions` module provides a way to construct and serialize complex mathematical expressions involving simulation metrics in Tidy3D.
It allows users to define objective functions for optimization tasks, for example in conjunction with the `invdes` plugin.
This module is essential for creating expressions that can be easily saved, loaded, and passed between different components of a simulation workflow.

## Introduction

The `expressions` module is designed to facilitate the construction of serializable mathematical expressions involving simulation metrics like mode power, mode coefficients, field intensity, and more.
These expressions can be combined using standard arithmetic operators.
This functionality is important when defining objective functions for optimization routines, especially in inverse design tasks where the objective functions may need to be transmitted or stored.

### Creating metrics

To start using the `expressions` module, you can create metrics using predefined metric classes such as `ModeAmp` and `ModePower`.

Metrics are a special instance of a `Variable`, which itself is a subclass of `Expression`, that take a `SimulationData` as input and are intended to return a single scalar.

```python
from tidy3d.plugins.expressions import ModeAmp, ModePower

# Create a ModeAmp metric
mode_coeff = ModeAmp(monitor_name="monitor1", freqs=[1.0])

# Create a ModePower metric
mode_power = ModePower(monitor_name="monitor2", freqs=[1.0])
```

### Combining metrics with operators

Metrics can be combined using standard arithmetic operators like `+`, `-`, `*`, `/`, `**`, and functions like `abs()`.
Together, they will form an `Expression`.

```python
# Define an objective function using metrics
f = abs(mode_coeff) - mode_power / 2
```

### Functions

The `expressions` module also provides a set of mathematical functions that can be used to create more complex expressions.

```python
from tidy3d.plugins.expressions import Sin, Cos

f = Sin(mode_coeff) + Cos(mode_power)
```

### Evaluating expressions

Once you have a metric, you can evaluate it using simulation data.

```python
# Assume "data" is a SimulationData object obtained from a simulation
result = f.evaluate(data)

# ...or just
result = f(data)
```

### Serializing and deserializing expressions

Expressions can be serialized to a file using the `to_file` method and deserialized using the `from_file` class method.

```python
# Serialize the metric to a file
f.to_file("metric_expression.hdf5")

# Deserialize the metric from a file
from tidy3d.plugins.expressions import Expression
loaded_expr = Expression.from_file("metric_expression.hdf5")
```

## Supported features

### Operators and functions

The `expressions` module supports various operators and functions to build complex expressions:

| Type       | Name       | Description                                      | Class Name       |
|------------|------------|--------------------------------------------------|------------------|
| Operator   | `+`        | Addition                                         | `Add`            |
| Operator   | `-`        | Subtraction                                      | `Subtract`       |
| Operator   | `*`        | Multiplication                                   | `Multiply`       |
| Operator   | `/`        | Division                                         | `Divide`         |
| Operator   | `**`       | Power                                            | `Power`          |
| Operator   | `//`       | Floor Division                                   | `FloorDivide`    |
| Operator   | `%`        | Modulus                                          | `Modulus`        |
| Operator   | `@`        | Matrix Multiplication                            | `MatMul`         |
| Operator   | `abs`      | Absolute Value                                   | `Abs`            |
| Function   | `Sin`      | Sine function                                    | `Sin`            |
| Function   | `Cos`      | Cosine function                                  | `Cos`            |
| Function   | `Tan`      | Tangent function                                 | `Tan`            |
| Function   | `Exp`      | Exponential function                             | `Exp`            |
| Function   | `Log`      | Natural logarithm function                       | `Log`            |
| Function   | `Log10`    | Base-10 logarithm function                       | `Log10`          |
| Function   | `Sqrt`     | Square root function                             | `Sqrt`           |

### Variables and metrics

The module provides predefined metrics and variables for constructing expressions.
Metrics represent specific simulation data, while variables act as placeholders.

| Type       | Name       | Description                                      |
|------------|------------|--------------------------------------------------|
| Metric     | `ModeAmp`  | Metric for calculating the mode coefficient      |
| Metric     | `ModePower`| Metric for calculating the mode power            |
| Variable   | `Constant` | Fixed value in expressions                       |
| Variable   | `Variable` | Placeholder for values in expressions            |

## Examples

### `ModeAmp` and `ModePower`

In this example, we create metrics using only `ModeAmp` and `ModePower` and combine them to define an objective function.

```python
from tidy3d.plugins.expressions import ModeAmp, ModePower

# Create metrics
mode_coeff = ModeAmp(monitor_name="monitor1", freqs=[1.0])
mode_power = ModePower(monitor_name="monitor2", freqs=[1.0])

# Define an objective function using metrics
f = abs(mode_coeff) - mode_power / 2

# Display the expression
print(f)
```

**Expected output:**

```text
(abs(ModeAmp("monitor1")) - (ModePower("monitor2") / 2))
```

**Evaluating the expression:**

```python
# Assume "data" is a SimulationData object obtained from a simulation
result = f.evaluate(data)

# Display the result
print(result)
```

### Using variables

The `expressions` module provides `Variable` and `Constant` classes to represent placeholders and fixed values in your expressions.
Variables can be used to parameterize expressions, allowing you to supply values at evaluation time.
Note that a `Metric` such as `ModeAmp` is a subclass of `Variable`, so it can be used in the same way.

#### Unnamed variables (positional arguments)

If you create a `Variable` without a name, it expects its value to be provided as a single positional argument during evaluation.

```python
from tidy3d.plugins.expressions import Variable

# Create an unnamed variable
x = Variable()

# Use the variable in an expression
expr = x**2 + 2 * x + 1

# Evaluate the expression with a positional argument
result = expr(3)

print(result)  # Outputs: 126
```

#### Named variables (keyword arguments)

If you create a `Variable` with a name, it expects its value to be provided as a keyword argument during evaluation.

```python
from tidy3d.plugins.expressions import Variable

# Create named variables
x = Variable(name="x")
y = Variable(name="y")
expr = x**2 + y**2

# Evaluate the expression with keyword arguments
result = expr(x=3, y=4)

print(result)  # Outputs: 25
```

#### Mixing Positional and Keyword Arguments

You can mix positional and keyword arguments when evaluating expressions that contain both unnamed and named variables.

```python
# Create a mix of named and unnamed variables
x = Variable(name="x")
y = Variable()
expr = x**2 + y**2

result = expr(3, y=4)  # x = 3, y = 4

print(result)  # Outputs: 25
```

#### A `Metric` is a `Variable`

A `Metric` such as `ModeAmp` is a subclass of `Variable`, so all the rules for `Variable` apply to `Metric` as well.

from tidy3d.plugins.expressions import ModeAmp, ModePower

```python
# Define two named metrics
mode_coeff1 = ModeAmp(name="mode_coeff1", monitor_name="monitor1", freqs=[1.0])
mode_power1 = ModePower(name="mode_power1", monitor_name="monitor2", freqs=[1.0])

# Create an expression using the metrics
expr = mode_coeff1 + mode_power1

# Assume "data1" and "data2" are SimulationData objects obtained from different simulations
result = expr(mode_coeff1=data1, mode_power1=data2)
```

#### Important notes on variable evaluation

- **Unnamed Variables**: If you have an unnamed variable (`Variable()`), you must provide exactly one positional argument during evaluation.
Providing multiple positional arguments will raise a `ValueError`.

- **Named Variables**: For each named variable, you must provide a corresponding keyword argument.
If a named variable is missing during evaluation, a `ValueError` will be raised.

- **Multiple Unnamed Variables**: If your expression contains multiple unnamed variables, you cannot provide multiple positional arguments.
You should assign names to the variables and provide their values as keyword arguments instead.

## Developer notes

### Extending expressions

To implement new metrics, follow these steps:

1. **Subclass the `Metric` base class:**

   Create a new class that inherits from `Metric` and implement the required methods.

   ```python
   from tidy3d.plugins.expressions.metrics import Metric

   class CustomMetric(Metric):
       monitor_name: str

       def evaluate(self, data: SimulationData) -> NumberType:
           # Implement custom evaluation logic
           pass
   ```

2. **Define attributes and methods:**

   - Add any necessary attributes with type annotations.
   - Implement the `evaluate` method, which defines how the metric is calculated from simulation data.

3. **Update forward references:**

   If your metrics reference other metrics or expressions, ensure you handle forward references appropriately.

### Extending operators

To extend the `expressions` module with additional operators:

1. **Create a new operator class:**

   Subclass `UnaryOperator` or `BinaryOperator` depending on the operator's arity.

   ```python
   from tidy3d.plugins.expressions.operators import BinaryOperator

   class CustomOperator(BinaryOperator):
       _symbol = "??"  # Replace with the operator symbol
       _format = "({left} {symbol} {right})"

       def evaluate(self, x, y):
           # Implement the operator logic
           pass
   ```

2. **Implement required methods:**

   - Define the `_symbol` class variable that represents the operator symbol.
   - Implement the `evaluate` method with the operator's logic.

3. **Update operator overloads:**

   If you want to enable the use of the operator with standard syntax (e.g., using `@` for matrix multiplication), override the corresponding magic methods in the `Expression` base class.

   ```python
   class Expression:
       # Existing methods...

       def __matmul__(self, other):
           return MatMul(left=self, right=other)
   ```

4. **Register the operator (if necessary):**

   Ensure the new operator is recognized by the serialization mechanism. If your operator introduces a new type, update any type maps or registries used in the `expressions` module.
