## Matrix Node

These sample projects show several [examples](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/tree/master/Matrix) to peform [matrix oeprations](../guides/matrix.md).


### Addition

Brain: [Matrix/Matrix_Addition](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Matrix/Matrix_Addition.brain)

Brain file with several examples of the addtion operation.
It contains :

 * Summation of matrix and number. The number is defiend as a memory block as well as as well as `Params\DataInput0` paramter.
 * Row or column wise addition of vector and matrix.
 * Summation of two matrices. The JoinNode contais operation for addition of two matrices too. Thus, the example shows comparison of these two methods. Output of the MatrixNode and the JoinNode are compared in the new JoinNode using the DistanceSquere operations. The difference (visulizaed using the observer) is zero.

Note that dimensions of inputs must always correspond.

![](../img/matrix_ex_add.PNG)

---

### Multiplication

Brain: [Matrix/Matrix_Addition](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Matrix/Matrix_Addition.brain)


Brain file with several multiplication examples. Again, note that dimension of inputs have to correspond to the desired operation with matrices. The sample project contains examples with:

 * Vector times number
 * Vector time matrix
 * Matrix times matrix. Again, the JoinNode contais operation for matrix multiplication too. The example shows comparison of these two methods. Output of the MatrixNode and the JoinNode are compared in the new JoinNode using the DistanceSquere operations. The difference (visulizaed using the observer) is zero.


![](../img/matrix_ex_multipl.PNG)

---

### Log, Exp, Round

Brain: [Matrix/Matrix_Addition](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Matrix/Matrix_LogExpRound.brain)

Examples with Round, Exp, Log operations on the matrix. Note that only one input is used in this case. The MatrixNode now applied desired function on that input MemoryBlock.

![](../img/matrix_ex_AbsExp.PNG)

---

### Get Row / Column

Brain: [Matrix/Matrix_Addition](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Matrix/Matrix_getRowCol.brain)

This brain file sample shows how to use MatrixNode for getting the desired row or column of the matrix, defined by the id (which row/column I want to get). The observers in the figure bellow shows an example where we want to get row id ,,1'' (so second row because counting starts with 0) of the matrix shown in the middle observer. The result (node's output) is shown in the last observer.

![](../img/matrix_ex_getRowCol.PNG)
