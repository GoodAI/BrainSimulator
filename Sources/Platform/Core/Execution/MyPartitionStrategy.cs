using GoodAI.Core.Task;

namespace GoodAI.Core.Execution
{
    public interface IMyPartitionStrategy
    {
        MyExecutionPlan[] Divide(MyExecutionPlan executionPlan);    ///<Performs partitioning
    }

    /// Puts all tasks on one GPU
    public class MyAllInOneGPUPartitioning : IMyPartitionStrategy
    {
        private int numOfPieces;
        private int selected;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="numGPUs">Number of available GPUs</param>
        /// <param name="selected">ID of GPU where tasks will run</param>
        public MyAllInOneGPUPartitioning(int numGPUs, int selected)
        {
            numOfPieces = numGPUs;
            this.selected = selected;
        }

        /// <summary>
        /// Performs partitioning
        /// </summary>
        /// <param name="executionPlan">Generic execution plan</param>
        /// <returns>Execution plans for all GPUs</returns>
        public MyExecutionPlan[] Divide(MyExecutionPlan executionPlan)
        {           
            MyExecutionBlock.IteratorAction setGpuAction = delegate(IMyExecutable e)
            {
                if (e is MyTask)
                {
                    (e as MyTask).GenericOwner.GPU = selected;
                }
            };

            executionPlan.InitStepPlan.Iterate(true, setGpuAction);
            executionPlan.StandardStepPlan.Iterate(true, setGpuAction);

            MyExecutionPlan[] partitioning = new MyExecutionPlan[numOfPieces];
           
            for (int i = 0; i < numOfPieces; i++) 
            {
                if (i == selected)
                {                    
                    partitioning[i] = executionPlan;
                }
                else
                {
                    partitioning[i] = new MyExecutionPlan()
                    {
                        InitStepPlan = new MyExecutionBlock() { Name = executionPlan.InitStepPlan.Name },
                        StandardStepPlan = new MyExecutionBlock() { Name = executionPlan.StandardStepPlan.Name }
                    };
                }                
            }

            for (int i = 0; i < numOfPieces; i++)
            {
                partitioning[i].InitStepPlan.Name += " (GPU " + i + ")";
                partitioning[i].StandardStepPlan.Name += " (GPU " + i + ")";
            }

            return partitioning;
        }
    }

    /* Obsolete
    public class MyEvenPartitioning : IMyPartitionStrategy
    {
        private int numOfPieces;
        public MyEvenPartitioning(int numGPUs)
        {
            numOfPieces = numGPUs;
        }

        public MyExecutionBlock[] Divide(MyProject project)
        {            
            List<MyWorkingNode>[] result = new List<MyWorkingNode>[numOfPieces];

            for (int i = 0; i < numOfPieces; i++)
            {
                result[i] = new List<MyWorkingNode>();
            }

            project.World.GPU = 0;
            result[0].Add(project.World);

            int lastGPU = numOfPieces - 1;
            //collect all nodes
            MyNodeGroup.IteratorAction action = delegate(MyNode node)
            {
                if (node is MyWorkingNode)
                {                    
                    lastGPU = (lastGPU + 1) % numOfPieces;
                    node.GPU = lastGPU;

                    result[node.GPU].Add(node as MyWorkingNode);
                }
            };
            project.Network.Iterate(true, action);

            return result;
        }
    }

    public class MySequentialOnFirstPartitioning : IMyPartitionStrategy
    {
        private int numOfPieces;
        public MySequentialOnFirstPartitioning(int numGPUs)
        {
            numOfPieces = numGPUs;
        }

        public List<MyWorkingNode>[] Divide(MyProject project)
        {
            List<MyWorkingNode>[] result = new List<MyWorkingNode>[numOfPieces];

            for (int i = 0; i < numOfPieces; i++)
            {
                result[i] = new List<MyWorkingNode>();
            }

            project.World.GPU = 0;
            result[0].Add(project.World);

            int lastGPU = numOfPieces - 1;
            //collect all nodes
            MyNodeGroup.IteratorAction action = delegate(MyNode node)
            {
                if (node is MyWorkingNode)
                {
                    if (node.Sequential)
                    {
                        node.GPU = 0;
                        result[0].Add(node as MyWorkingNode);
                    }
                    else
                    {
                        lastGPU = (lastGPU + 1) % numOfPieces;
                        node.GPU = lastGPU;

                        result[node.GPU].Add(node as MyWorkingNode);
                    }
                }
            };
            project.Network.Iterate(true, action);

            return result;
        }
    }
     */
}
