# Transformation nodes

### AbsoluteValue
#### Memory blocks
- Temp - TODO
- Output - absolute value of input
#### Tasks
- Absolute Value TODO
    + ScalarNormalization
    + VectorNormalization

### Reduction
Description TODO
#### Settings
- Mode - which type of reduction you want to perform (TODO more details)
#### Memory blocks
- Output - result of reduction
#### Tasks
- Reduction - performs reduction

### GoniometricFunction
Performs goniometric transformation of input values
#### Memory blocks
- Output
#### Tasks
- Goniometric
    + Type - goniometric function (TODO details)

### PolynomialFunction
Performs polynomial transformation of input values
#### Memory blocks
- Output
#### Tasks
- Polynomial Function
    + A0,A1,A2,A3 - Polynomial function quotients. Function will be A0 + A1\*x + A2\*x^2 +A3\*x^3

### LowHighFilter TODO
Description
#### Memory blocks
- Output
#### Tasks TODO
- Range Restriction
    + Minimum, Maximum
    + RangeRestrOperation
- Find max value
- Round
- Floor

### Threshold
Description
#### Settings
- Levels - TODO
#### Memory blocks
- Output
#### Tasks
- Threshold
    + Minimum, Maximum - TODO

### Filter2D
Description
#### Memory blocks
- Temp
- Output
#### Tasks TODO
- Variance 3x3
- Edge detection
- Gaussian Blur (3x3)
- Sobel Edge Detection (3x3)

### Analyze2D TODO
Description
#### Memory blocks
- Temp
- Derivatives
- LastInput
- GlobalRow
- Output
#### Tasks
- Optical Flow (Lucas-Kanade)
    + SubtractGlobalFlow

### Resize2D
Scales (up/down) input "image" by given factor
#### Settings
- Factor - scaling factor (>1 scale up, <1 scale down)
- FactorHeight - TODO
#### Memory blocks
- Output
#### Tasks
- ImageScale - performs scaling

### Accumulator
Can delay or accumulate values
#### Settings
- DelayMemorySize - TODO
#### Memory blocks
- DelayedInputs - used for saving delayed values
- Output
#### Tasks - TODO
- Delay
    + InitialValue
    + UseFirstInput
- Approach Value
    + ApproachMethod
    + Delta
    + Factor
    + Target
- Quantized Copy
    + TimePeriod

### Hash TODO
Description
#### Settings
- HashMethod
- SPARSE_SIZE
#### Memory blocks
- Output
#### Tasks
- MD5

---
# Data flow control nodes TODO
### Join
Joins multiple memory blocks into one
#### Settings
- Operation
    + Addition - sums all the inputs
    + Subtraction -
    + Multiplication - multiply all the inputs
    + AND
    + OR
    + OR_threshold
    + XOR
    + XNOR
    + IMP
    + Permutation
    + Inv_Permutation
    + DotProduct
    + CosineDistance
    + MatMultiplication
    + StackInputs - concatenate all input MBs to one MB
- InputBranches - Number of input MBs
- OutputColHint - ColumnHint of Output MB
#### Memory blocks
- Output
- InputBlocksPointers
- Temp
#### Tasks
- Init memory mapping
- Perform join operation

### Fork
Description
#### Settings
- Branches - define, how to divide input MB
#### Memory blocks
- Output
#### Tasks
- Perform fork operation

### GateInput
Description
#### Settings
#### Memory blocks
- Output
#### Tasks
- listOfTasks

### DataGate
Description
#### Settings
#### Memory blocks
- Output
#### Tasks
- listOfTasks

---
### NodeName
Description
#### Settings
#### Memory blocks
- Output
#### Tasks
- listOfTasks