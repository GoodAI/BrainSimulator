

namespace BrainSimulator
{
    class Program
    {
        static void Main(string[] args)
        {
            /*
            //MyLog.GrabConsole();
            MyLog.Level = MyLogLevel.DEBUG;

            int neuronsCount = 8;
            int inputsCount = neuronsCount * 2;

            //NETWORK------------------------------------------------------------------------------
            MyNetwork network = new MyNetwork();
                        
            //NODES--------------------------------------------------------------------------------
            
            //INPUT and OUTPUT nodes are not needed as they are initialized in NETWORK constructor

            //create external input to the simlation (is not part of the network!)
            MyTestingWorld externalInputNode = new MyTestingWorld()
            {
                OutputSize = neuronsCount * 2
            };                                    

            //create neurons group node
            MyNeuronsNode neuronsNode = new MyNeuronsNode()
            {
                NeuronsCount = neuronsCount,
                OutputSize = neuronsCount
            };
            network.AddChild(neuronsNode);

            //create transformation node
            MyTestTransform transformationNode = new MyTestTransform()
            {
                OutputSize = neuronsCount
            };            
            network.AddChild(transformationNode);
            
            //CONNECTIONS-------------------------------------------------------------------------

            //connect external input to the global network input
            MyConnection externalInputToNetwork = new MyConnection(externalInputNode, network);
            externalInputToNetwork.Connect();
            
            //connect external input to transformation
            MyConnection inputToTransform = new MyConnection(network.GroupInputNode, transformationNode);
            inputToTransform.Connect();

            //connect transformation output to neurons group node
            MyConnection transformToNeurons = new MyConnection(transformationNode, neuronsNode);
            transformToNeurons.Connect();            

            //connect neurons group node to output node
            MyConnection neuronsGroupToOutput = new MyConnection(neuronsNode, network.GroupOutputNode);
            neuronsGroupToOutput.Connect();
            
            //EXECUTION ----------------------------------------                        

            MyMemoryManager.Instance.AllocateBlocks(externalInputNode, false, true, true);
            MyMemoryManager.Instance.AllocateBlocks(network, true, true, true);
            
            MyKernelFactory.Instance.ClearTaskInfo();
            MyKernelFactory.Instance.Schedule(externalInputNode);
            MyKernelFactory.Instance.Schedule(network);            

            MyKernelFactory.Instance.Execute();
            MyKernelFactory.Instance.Execute();

            MyMemoryManager.Instance.FreeBlocks(externalInputNode, false, true, true);
            MyMemoryManager.Instance.FreeBlocks(network, true, true, true);
            
            Console.ReadKey();
            */
        }
    }
}
