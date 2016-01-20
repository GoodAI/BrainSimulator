using GoodAI.Core.Memory;
using GoodAI.Core.Signals;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Core.Nodes
{
    /// <author>GoodAI</author>
    /// <meta>df</meta>
    /// <status>Working</status>
    /// <summary>Alters signal on the data connection according to signal input.</summary>
    /// <description>Select a signal name and threshold to raise/drop selected signal on the data output.</description>
    public class MySignalNode : MyWorkingNode
    {
        public MyProxySignal ProxySignal { get; set; }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> Signal
        {
            get { return GetInput(1); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyBrowsable, TypeConverter(typeof(MySignal.MySignalTypeConverter))]
        [YAXSerializableField(DefaultValue = "<none>")]
        public string SignalName { get; set; }

        public override string Description
        {
            get
            {
                return "Alter signal:";
            }
        }

        public override void UpdateMemoryBlocks()
        {
            Output.Count = GetInputSize(0);
            Output.ColumnHint = Input != null ? Input.ColumnHint : 1;

            if (!SignalName.Equals("<none>"))
            {
                ProxySignal.Source = MySignal.CreateSignalByDefaultName(SignalName);

                if (ProxySignal.Source != null)
                {
                    ProxySignal.Source.Owner = this;
                    ProxySignal.Source.Name = MyProject.RemovePostfix(ProxySignal.Source.DefaultName, "Signal");
                }
            }
            else
            {
                ProxySignal.Source = null;
            }
        }

        public MyAlterSignalTask AlterSignal { get; private set; }

        [Description("Alter Signal")]
        public class MyAlterSignalTask : MyTask<MySignalNode>
        {

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 0.5f)]
            public float SignalThreshold { get; set; }

            public override void Init(int nGPU)
            {
                
            }

            public override void Execute()
            {
                Owner.Signal.SafeCopyToHost();

                if (Owner.Signal.Host[0] > SignalThreshold)
                {
                    Owner.ProxySignal.Raise();
                }
                else
                {
                    Owner.ProxySignal.Drop();
                }

                Owner.Input.CopyToMemoryBlock(Owner.Output, 0, 0, Owner.Input.Count);
            }
        }
    }
}
