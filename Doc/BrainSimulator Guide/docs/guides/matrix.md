# Matrix node

Purpose of this node is to simplify vanilla matrix operations such as addition or multiplication.

## List of features

 * The node allows multiplication or addition with different input sizes, so in addition to to matrix multiplication
$\mathbf{A} \cdot \mathbf{B}$ (where $\mathbf{A}$,$\mathbf{B}$ are matrices), it directly supports $\mathbf{v}^{\mathsf{T}} \cdot \mathbf{A}$ ($\mathbf{v}$ is vector), or $\mathbf{A} \cdot \mathbf{v}$, $\mathbf{v}^{\mathsf{T}} \cdot \mathbf{w}$, $\mathbf{v} \cdot \mathbf{w}^{\mathsf{T}}$, or $ c \cdot \mathbf{A}$.
 * For several opts (getRow, $c \cdot \mathbf{A}$), two input types are supported: 1) a memory block from another node; 2) user writes number in `ExecuteParams/DataInput` task property.
 * The node is a layer above the `MatrixAutoOps` class, so you can use it in your code.



## Operations

The node is always expected that one input goes to **A** and, if second input requires, it goes to **B**.

 | Operation | Input  | Comment |
 | - | -  | - |
 | **Multiplication**<br> $\mathbf A \cdot \mathbf B$      | Two MB | Each input can matrix, vector, or number. |
 | **Multiplication**<br> $\mathbf A \cdot$`DataInput0`    | One MB + 'DataInput0'  | MB Input can be matrix, vector, or number.  |
 | **Addition**<br> $\mathbf A + \mathbf B$                | Two MB  | Each input can matrix, vector, or number. If matrix/vector used as input, the node performs row/column-wise addition. |
 | **Addition**<br> $\mathbf A +$`DataInput0`              | One MB + 'DataInput0'  | Input into `A` can be matrix, vector, or number.  |
 | **Substraction**<br> $\mathbf A - \mathbf B$            | Two MB  | Each input can matrix, vector, or number. If matrix/vector used as input, the node performs row/column-wise addition. |
 | **Substraction** <br> $\mathbf A -$`DataInput0`         | One MB + 'DataInput0'  | Input into `A` can be matrix, vector, or number. Performs A$-$DataInput0.  |
 | **MultiplElemntWise**<br> $\mathbf A \circ \mathbf B$   | Two MB  | Element-wise product, each input can matrix, vector, or number. |
 | **DotProd**<br> $\mathbf A^{\mathsf T} \cdot \mathbf B$ | Two MB | Each input is a vector. If same inputs (with same, correct, dimension) are provied for **Multiplication** operation, the multiplication also performs dot product. |
 | **MinIndex**<br> $\textrm{arg min}_i ~\mathbf A_i$      | One MB |  Returns the index of the min value in the vector. |
 | **MaxIndex**<br> $\textrm{arg max}_i ~\mathbf A_i$      | One MB |  Returns the index of the max value in the vector. |
 | **GetCol**<br> $ \mathbf A_{i,:}$                       | Two MB |  First MB input is matrix, second input is number that defines the column to get (first one has index 0). |
 | **GetCol**<br> $ \mathbf A_{i,:}$                       | One MB + 'DataInput0' |  MB input is matrix,  'DataInput0' defines the column to get (first one has index 0). |
 | **GetRow**<br> $ \mathbf A_{:,i}$                       | Two MB |  First MB input is matrix, second input is number that defines the row to get (first one has index 0). |
 | **GetRow**<br> $ \mathbf A_{:,i}$                       | One MB + 'DataInput0' |  MB input is matrix,  'DataInput0' defines the row to get (first one has index 0). |
 | **Minus**<br>  $ -\mathbf A$                            | One MB |  MB input is matrix. |
 | **Norm2**<br>  $ \Vert \mathbf A \Vert_2 $              | One MB |  Returns Norm2 of the input MB. |
 | **Normalize**<br>  $ \frac{1}{\Vert \mathbf A \Vert_2} \mathbf A$   | One MB |  Normalizes the input MB. |
 | **Exp, Log, Abs, Round, Flooe, Ceil**                   | One MB |  Performs the desired operation on each element in the input MB. |



## Using as a node in Brain Simulator


Here are few representative examples how to use Matrix node, more can be found in the Sample Projects, where you can play with parameters and copy-past nodes directly into your projects.

* Two memory blocks that represents matrices are multiplied, below observers show what is inside the memory blocks.

![](img_examples/matrix_multi01.PNG)

* Two memory blocks that represents a matrix and a vector are multiplied, below observers again show what is inside the memory blocks.

![](img_examples/matrix_multi02.PNG)

* Two memory blocks that respresents a matrix and a vector are summed. The algorithm cannot perform element-wise addition because of the memory block sizes, but the number of columns of the matrix and the vector correspond. Thus the algorithm performs element-wise addition for each row.

![](img_examples/matrix_add01.PNG)

* The matrix node allows user to insert the constant as DataInput0 (see orange circle) and the values in the memory block A will be increase by it, as shown in figure below. Note that if you will choose 

![](img_examples/matrix_add02.PNG)



## Using it in your C# code

You need create the Matrix object
``` csharp
MyMatrixAutoOps mat_operation;
```

In the `Init` part, it is necassary to create an instance of the object and set-up the desired operations:
``` csharp
mat_operation = new MyMatrixAutoOps(Owner, Matrix.MatOperation.Multiplication | Matrix.MatOperation.Addition, A);
```

Inside details such as cublas, CPU, or kernels are hidden. So, you can directly multiply memory blocks `A` and `B` and save the result into memory the block `C`,
``` csharp
mat_operation.Run(Matrix.MatOperation.Multiplication, A, B, C);
```
or multiply `A` with number 10.4,

``` csharp
mat_operation.Run(Matrix.MatOperation.Multiplication, A, 10.4f, C);
```

Becuase the `Addition` operation was initlized too, the user can proceed,
``` csharp
mat_operation.Run(Matrix.MatOperation.Addition, A, B, C);
```

Complete code in the Brain Simulator node will look like:
``` csharp
public class MyExecuteTask : MyTask<MyNode>
{
  private MyMatrixAutoOps mat_operation;

  public override void Init(int nGPU)
  {
    mat_operation = new MyMatrixAutoOps(Owner, Matrix.MatOperation.Multiplication | Matrix.MatOperation.Addition, Owner.A);
  }
  public override void Execute()
  {
    mat_operation.Run(Matrix.MatOperation.Multiplication, Owner.A, Owner.B, Owner.C);
    mat_operation.Run(Matrix.MatOperation.Multiplication, Owner.C, 10.3f, Owner.B);
    mat_operation.Run(Matrix.MatOperation.Addition, Owner.A, Owner.B, Owner.C);  // C = A+A*B*10.3
  }
}
```

