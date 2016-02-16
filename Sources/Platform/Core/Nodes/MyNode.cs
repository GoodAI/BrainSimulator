using GoodAI.Core.Memory;
using GoodAI.Core.Signals;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using YAXLib;

namespace GoodAI.Core.Nodes
{
    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public abstract class MyNode : IDisposable, IValidatable
    {
        #region Common properties
        
        public virtual bool Sequential { get; set; }        

        [MyBrowsable, ReadOnly(true), Category("\t\tGeneral"), DisplayName("ID")]
        [YAXSerializableField, YAXAttributeForClass]
        public int Id { get; set; }

        [MyBrowsable, Category("\t\tGeneral"), DisplayName("\tName")]
        [YAXSerializableField, YAXAttributeForClass]
        public string Name { get; set; }

        public string DefaultName { get { return "Node_" + Id; } }

        public virtual string Description { get { return MyProject.ShortenNodeTypeName(GetType()); } }
                
        [YAXSerializableField]
        public MyLocation Location { get; set; }

        public virtual MyNodeGroup Parent { get; set; }

        public MyProject Owner { get; internal set; }

        [MyBrowsable, Category("\t\tGeneral")]
        public int TopologicalOrder { get; internal set; }        

        private void InitPropertiesDefault()
        {
            foreach (PropertyInfo pInfo in GetInfo().InitiableProperties)
            {
                object defaultValue = pInfo.GetAttributeProperty((YAXSerializableFieldAttribute fa) => fa.DefaultValue);

                if (defaultValue != null)
                {
                    pInfo.SetValue(this, defaultValue);
                }
            }
        }

        protected void CreateMemoryBlocks()
        {
            foreach (PropertyInfo pInfo in GetInfo().OwnedMemoryBlocks)
            {
                MyAbstractMemoryBlock mb = MyMemoryManager.Instance.CreateMemoryBlock(this, pInfo.PropertyType);
                mb.Name = pInfo.Name;
                mb.Persistable = pInfo.GetCustomAttribute<MyPersistableAttribute>(true) != null;
                mb.Unmanaged = pInfo.GetCustomAttribute<MyUnmanagedAttribute>(true) != null;
                mb.IsOutput = pInfo.GetCustomAttribute<MyOutputBlockAttribute>(true) != null;
                mb.IsDynamic = pInfo.GetCustomAttribute<DynamicBlockAttribute>(true) != null;

                pInfo.SetValue(this, mb);
            }
        }

        #endregion

        #region Signals

        public long IncomingSignals { get; protected set; }
        public long OutgoingSignals { get; protected set; }
        internal long RiseSignalMask { get; set; }
        internal long DropSignalMask { get; set; }        

        protected void CreateSignals()
        {
            foreach (PropertyInfo pInfo in GetInfo().RegisteredSignals)
            {
                MySignal signal = (MySignal)Activator.CreateInstance(pInfo.PropertyType, true);
                signal.Owner = this;
                signal.Name = MyProject.RemovePostfix(pInfo.Name, "Signal");

                pInfo.SetValue(this, signal);
            }
        }

        public virtual void ProcessIncomingSignals()
        {            
            IncomingSignals = MySignalStrategy.SumAllIncomingSignals(this);
        }

        public virtual void ProcessOutgoingSignals()
        {
            OutgoingSignals = IncomingSignals;

            OutgoingSignals |= RiseSignalMask;
            OutgoingSignals &= ~DropSignalMask;
        }

        public void ClearSignals()
        {
            IncomingSignals = 0;
            OutgoingSignals = 0;
            RiseSignalMask = 0;
            DropSignalMask = 0;
        }        

        #endregion       

        #region Events

        public event EventHandler NodeUpdated;

        public void Updated()
        {
            if (NodeUpdated != null)
            {
                NodeUpdated(this, EventArgs.Empty);
            }
        }

        #endregion

        #region Node inputs

        public MyConnection[] InputConnections { get; protected set; }        
        public HashSet<MyConnection>[] OutputConnections { get; protected set; }        

        [MyBrowsable, Category("I/O"), ReadOnly(true)]
        public virtual int InputBranches
        {
            get { return InputConnections != null ? InputConnections.Length : 0; }
            set
            {
                int connToCopy = Math.Min(value, InputBranches);
                MyConnection[] oldConns = InputConnections;

                for (int i = connToCopy; i < InputBranches; i++)
                {
                    var connection = InputConnections[i];
                    if (connection != null)
                        connection.Disconnect();
                }

                InputConnections = new MyConnection[value];

                if (oldConns != null)
                    Array.Copy(oldConns, InputConnections, connToCopy);
            }
        }

        public virtual MyMemoryBlock<float> GetInput(int index)
        {
            return 
                InputConnections != null &&
                InputConnections.Length > index && 
                InputConnections[index] != null ?
                InputConnections[index].FetchInput() : null;
        }

        public virtual MyMemoryBlock<T> GetInput<T>(int index) where T : struct
        {
            return
                InputConnections != null &&
                InputConnections.Length > index &&
                InputConnections[index] != null ?
                InputConnections[index].FetchInput<T>() : null;
        }

        public virtual MyAbstractMemoryBlock GetAbstractInput(int index)
        {
            return
                InputConnections != null &&
                InputConnections.Length > index &&
                InputConnections[index] != null ?
                InputConnections[index].FetchAbstractInput() : null;
        }

        public int GetInputSize(int index)
        {
            MyAbstractMemoryBlock input = GetAbstractInput(index);
            return input != null ? input.Count : 0;
        }

        #endregion

        #region Node outputs

        [MyBrowsable, Category("I/O"), ReadOnly(true)]
        public virtual int OutputBranches
        {
            get { return OutputConnections == null ? 0 : OutputConnections.Length; }
            set
            {
                int connToCopy = Math.Min(value, OutputBranches);
                HashSet<MyConnection>[] oldConns = OutputConnections;

                for (int i = connToCopy; i < OutputBranches; i++)
                    foreach (MyConnection connection in OutputConnections[i].ToList())
                        connection.Disconnect();

                OutputConnections = new HashSet<MyConnection>[value];

                if (oldConns != null)
                    Array.Copy(oldConns, OutputConnections, connToCopy);

                for (int i = connToCopy; i < OutputBranches; i++)
                    if (OutputConnections[i] == null)
                        OutputConnections[i] = new HashSet<MyConnection>();
            }
        }

        public abstract MyMemoryBlock<float> GetOutput(int index);
        public abstract MyMemoryBlock<T> GetOutput<T>(int index) where T : struct;
        public abstract MyAbstractMemoryBlock GetAbstractOutput(int index);

        public int GetOutputSize(int index)
        {
            MyAbstractMemoryBlock output = GetAbstractOutput(index);
            return output != null ? output.Count : 0;
        }

        #endregion                     

        #region Memory Blocks

        public abstract void UpdateMemoryBlocks();

        public virtual void ReallocateMemoryBlocks() { }

        private int[] m_outputBlockSizes = new int[0];

        public void PushOutputBlockSizes()
        {
            if (m_outputBlockSizes.Length != OutputBranches)
            {
                m_outputBlockSizes = new int[OutputBranches];
            }

            for (int i = 0; i < OutputBranches; i++)
            {
                m_outputBlockSizes[i] = GetOutputSize(i);
            }
        }

        public bool AnyOutputSizeChanged()
        {
            if (m_outputBlockSizes.Length != OutputBranches)
            {
                return true;
            }

            for (int i = 0; i < OutputBranches; i++)
            {
                if (m_outputBlockSizes[i] != GetOutputSize(i))
                {
                    return true;
                }
            }
            return false;
        }

        #endregion

        protected internal MyNode() 
        {
            MyNodeInfo.CollectNodeInfo(GetType());            

            InputBranches = GetInfo().InputBlocks.Count;
            OutputBranches = GetInfo().OutputBlocks.Count;

            CreateMemoryBlocks();
            CreateSignals();
            InitPropertiesDefault();            
        }

        internal virtual void Init()
        {
            Id = Owner.GenerateNodeId();            

            if (Name == null)
            {
                Name = DefaultName;
            }             
        }

        public virtual void Validate(MyValidator validator)
        {
            for (int i = 0; i < InputBranches; i++)
            {
                validator.AssertError(GetAbstractInput(i) != null, this, "No input available");   
            }          
  
            if (!(this is MyNetwork))
            {
                for (int i = 0; i < InputBranches; i++)
                {
                    if (GetAbstractInput(i) != null)
                        validator.AssertError(GetAbstractInput(i).Count > 0, this, "Input size has to be larger than zero.");
                }
            }
        }      
  
        internal virtual void ValidateMandatory(MyValidator validator) { }

        public virtual void Dispose()
        {
            MyMemoryManager.Instance.RemoveBlocks(this);
        }

        public MyNodeInfo GetInfo()
        {
            return MyNodeInfo.Get(GetType());
        }

        // When node is pinned to any particular GPU - contains an ID of that GPU, 
        //   otherwise contain -1     
        [MyBrowsable, Category("\t\tGeneral"), ReadOnly(true)]
        public Int32 GPU { get; set; }        
               
        public virtual void TransferToDevice() { }
        public virtual void TransferToHost() { }

        public virtual bool AcceptsConnection(MyNode fromNode, int fromIndex, int toIndex)
        {
            MyAbstractMemoryBlock outputBlock = fromNode.GetAbstractOutput(fromIndex);

            if (outputBlock != null && outputBlock.IsDynamic)
            {
                // TODO(HonzaS): Enable this later when variable count of dynamic memblocks is supported.
                if (toIndex >= GetInfo().InputBlocks.Count)
                    return false;

                PropertyInfo inputBlock = GetInfo().InputBlocks[toIndex];
                var dynamicAttribute = inputBlock.GetCustomAttribute<DynamicBlockAttribute>();
                if (dynamicAttribute == null)
                    return false;
            }

            return true;
        }

        public bool CheckForCycle(MyNode to)
        {
            var visited = new HashSet<MyNode> ();
            return CheckForCycle(this, to, visited);
        }

        private static bool CheckForCycle(MyNode node, MyNode target, ISet<MyNode> visited)
        {
            visited.Add(node);

            return node.InputConnections.Where(connection => connection != null && !connection.IsLowPriority)
                .Select(connection => connection.From)
                .Any(source =>
                    source == target || (!visited.Contains(source) && CheckForCycle(source, target, visited)));
        }

        /// <summary>
        /// This method is called after deserialization.
        /// </summary>
        public virtual void UpdateAfterDeserialization() { }
    }
}
