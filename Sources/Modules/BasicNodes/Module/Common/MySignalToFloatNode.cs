using GoodAI.Core.Memory;
using GoodAI.Core.Signals;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Core.Nodes
{
    /// <author>GoodAI</author>
    /// <meta>jv</meta>
    /// <status>Working</status>
    /// <summary>Reads the signal from the data and converts it to the output float.</summary>
    /// <description>Converts chosen type of the signal to output value: signal IsIncommingRised means 1 on output, 0 otherwise.</description>
    public class MySignalToFloatNode : MyWorkingNode
    {
        public MyProxySignal ProxySignal { get; set; }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
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
                return "Signal to Float";
            }
        }

        public override void UpdateMemoryBlocks()
        {
            Output.Dims = new TensorDimensions(1);

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

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
        }

        public SignalToFloatTask ConvertSignal { get; private set; }

        /// <summary>
        /// Read the signal and convert it to the float value on the output.
        /// </summary>
        [Description("Signal to Float")]
        public class SignalToFloatTask : MyTask<MySignalToFloatNode>
        {
            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                if (Owner.ProxySignal.IsIncomingRised())
                {
                    Owner.Output.Host[0] = 1;
                }
                else
                {
                    Owner.Output.Host[0] = 0;
                }
                Owner.Output.SafeCopyToDevice();
            }
        }
    }
}
