using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using GoodAI.Platform.Core.Utils;
using YAXLib;

namespace GoodAI.Core.Nodes
{
    /// <author>GoodAI</author>
    /// <meta>df</meta>
    /// <status>Working</status>
    /// <summary>Groups several nodes to one entity.</summary>
    /// <description>Enables nodes to be put inside a group, which makes model more structured.</description>
    [YAXSerializeAs("Group")]    
    public class MyNodeGroup : MyWorkingNode
    {
        #region Common

        [MyBrowsable]
        [YAXSerializableField(DefaultValue = false), YAXAttributeForClass]
        public override bool Sequential { get; set; }
        
        [YAXSerializableField]    
        public List<MyNode> Children { get; protected set; }

        [YAXSerializableField]
        public MyLayout LayoutProperties { get; set; }        

        #endregion

        #region IO

        private MyParentInput[] m_groupInputNodes;
        [YAXSerializableField]
        public MyParentInput[] GroupInputNodes 
        {
            get { return m_groupInputNodes; }
            internal set
            {
                m_groupInputNodes = value;
                base.InputBranches = value.Length;
            }
        }

        [ReadOnly(false)]
        public override int InputBranches
        {
            get { return base.InputBranches; }
            set 
            {
                DisconnectInputNodesFromEnd(value);

                int nodesToCopy = Math.Min(value, InputBranches);

                MyParentInput[] oldInputNodes = GroupInputNodes;
                GroupInputNodes = new MyParentInput[value];

                if (oldInputNodes != null)
                {
                    Array.Copy(oldInputNodes, GroupInputNodes, nodesToCopy);
                }

                if (Owner != null)
                {
                    InitInputNodes();
                }                
            }
        }

        [YAXSerializableField]
        public MyOutput[] GroupOutputNodes { get; internal set; }

        [ReadOnly(false)]
        public override int OutputBranches
        {
            get { return base.OutputBranches; }
            set
            {
                // If GroupOutputNodes is set, it has preference over OutputBranches because of deserialization.
                int nodesToCopy = Math.Min(value, GroupOutputNodes == null ? OutputBranches : GroupOutputNodes.Length);

                base.OutputBranches = value;

                MyOutput[] oldOutputs = GroupOutputNodes;
                GroupOutputNodes = new MyOutput[value];

                if (oldOutputs != null && oldOutputs.Length >= nodesToCopy)
                {
                    Array.Copy(oldOutputs, GroupOutputNodes, nodesToCopy);
                }

                if (Owner != null)
                {
                    InitOutputNodes();
                }                
            }
        }

        public override void UpdateAfterDeserialization()
        {
            base.UpdateAfterDeserialization();

            if (GroupOutputNodes == null)
                return;

            OutputBranches = GroupOutputNodes.Length;
        }

        public sealed override MyMemoryBlock<float> GetOutput(int index)
        {
            return GroupOutputNodes.Length > index ? GroupOutputNodes[index].Output : null;
        }

        public sealed override MyMemoryBlock<T> GetOutput<T>(int index)
        {
            return GroupOutputNodes.Length > index ? GroupOutputNodes[index].GetOutput<T>(0) : null;
        }

        public sealed override MyAbstractMemoryBlock GetAbstractOutput(int index)
        {
            return GroupOutputNodes.Length > index ? GroupOutputNodes[index].GetAbstractOutput(0) : null;
        }

        #endregion

        public MyNodeGroup()
        {
            InputBranches = 1;
            OutputBranches = 1;
            Children = new List<MyNode>();        
        }

        internal override void Init()
        {
            base.Init();

            InitInputNodes();
            InitOutputNodes();            
        }

        internal void InitOutputNodes()
        {
            for (int i = 0; i < GroupOutputNodes.Length; i++)
            {
                if (GroupOutputNodes[i] == null)
                {
                    GroupOutputNodes[i] = Owner.CreateNode<MyOutput>();
                    GroupOutputNodes[i].Name = "Output " + (i + 1);
                    GroupOutputNodes[i].Parent = this;
                }
            }
        }

        internal void InitInputNodes()
        {
            for (int i = 0; i < GroupInputNodes.Length; i++)
            {                              
                if (GroupInputNodes[i] == null)
                {
                    MyParentInput inputNode = Owner.CreateNode<MyParentInput>();
                    inputNode.Name = "Input " + (i + 1);
                    inputNode.ParentInputIndex = i;
                    inputNode.Parent = this;

                    GroupInputNodes[i] = inputNode;
                }                
            }
        }

        private void DisconnectInputNodesFromEnd(int fromIndex)
        {
            if (Children != null)
            {
                foreach (MyNode child in Children)
                {
                    for (int i = 0; i < child.InputConnections.Length; i++)
                    {
                        MyConnection c = child.InputConnections[i];

                        if (c != null && 
                            c.From is MyParentInput &&
                            (c.From as MyParentInput).ParentInputIndex >= fromIndex)
                        {
                            c.Disconnect();
                            child.InputConnections[i] = null;
                        }
                    }
                }
            }
            
            if (GroupInputNodes != null)
            {             
                foreach (MyOutput outputNode in GroupOutputNodes)
                {
                    if (outputNode != null &&
                        outputNode.InputConnections[0] != null &&
                        outputNode.InputConnections[0].From is MyParentInput &&
                        (outputNode.InputConnections[0].From as MyParentInput).ParentInputIndex >= fromIndex)
                    {
                        outputNode.InputConnections[0].Disconnect();
                        outputNode.InputConnections[0] = null;
                    }
                }
            }
        }

        public void AddChild(MyNode child)
        {
            Children.Add(child);
            child.Parent = this;
        }

        public void RemoveChild(MyNode child)
        {
            Children.Remove(child);
            child.Parent = null;
            child.Dispose();
        }

        public void AppendGroupContent(MyNodeGroup group)
        {
            foreach (MyNode childNode in group.Children) {

                //take only non-IO nodes
                if ( ! (childNode is MyParentInput || childNode is MyOutput))
                {
                    //generate new ID
                    bool hasDefaultName = childNode.Name.Equals(childNode.DefaultName);                                                                  
                    
                    childNode.Id = Owner.GenerateNodeId();

                    if (hasDefaultName)
                    {
                        childNode.Name = childNode.DefaultName;
                    }

                    //disconnect possible group inputs
                    foreach (MyConnection c in childNode.InputConnections)
                    {
                        if (c != null && c.From is MyParentInput)
                        {
                            c.Disconnect();
                        }  
                    }

                    //dive into child groups and correct IDs
                    if (childNode is MyNodeGroup)
                    {
                        (childNode as MyNodeGroup).Iterate(true, true, delegate(MyNode node)
                        {
                            node.Id = Owner.GenerateNodeId();
                        });
                    }

                    Children.Add(childNode);
                    childNode.Parent = this;
                    childNode.Owner = this.Owner;
                }
            }            
        }
     
        public override void Dispose()
        {
            base.Dispose();
            Children.ForEach(node => node.Dispose());                        
        }

        public override void UpdateMemoryBlocks()
        {
            Children.ForEach(node => node.UpdateMemoryBlocks());
        }                          

        public override void Validate(MyValidator validator)
        {
            base.ValidateMandatory(validator);
            base.Validate(validator);

            Children.ForEach(node => 
            {
                try
                {
                    node.ValidateMandatory(validator);
                    node.Validate(validator);
                }
                catch (Exception e)
                {
                    MyLog.ERROR.WriteLine("Exception occured while validating " + node.Name + ": " + e.Message);
                    validator.AddError(node, "Exception occured while validating: " + e.Message);
                }
            });
        }

        public MyNode GetChildNodeById(int nodeId)
        {
            foreach (MyNode child in Children)
            {
                if (child.Id == nodeId)
                {
                    return child;
                }
                else if (child is MyNodeGroup)
                {
                    MyNode childSearch = (child as MyNodeGroup).GetChildNodeById(nodeId);
                    if (childSearch != null) return childSearch;
                }
            }
            return null;
        }

        public List<MyNode> GetChildNodesByName(String nodeName)
        {
            List<MyNode> nodes = new List<MyNode>();
            foreach (MyNode child in Children)
            {
                if (child.Name == nodeName)
                {
                    nodes.Add(child);
                }

                if (child is MyNodeGroup)
                {
                    nodes.AddRange((child as MyNodeGroup).GetChildNodesByName(nodeName));
                }
            }
            return nodes;
        }

        public delegate void IteratorAction(MyNode node);

        public void Iterate(bool recursive, IteratorAction action)
        {
            Iterate(this, recursive, false, action);
        }

        public void Iterate(bool recursive, bool includeIONode, IteratorAction action)
        {
            Iterate(this, recursive, includeIONode, action);
        }

        private void Iterate(MyNodeGroup group, bool recursive, bool includeIONodes, IteratorAction action)
        {
            if (includeIONodes)
            {
                foreach (MyNode inputNode in group.GroupInputNodes)
                {
                    action(inputNode);
                }

                foreach (MyNode outputNode in group.GroupOutputNodes)
                {
                    action(outputNode);
                }
            }

            foreach (MyNode childNode in group.Children)
            {
                action(childNode);
                  
                if (recursive && childNode is MyNodeGroup)
                {
                    Iterate(childNode as MyNodeGroup, recursive, includeIONodes, action);
                }
            }
        }

        public override bool AcceptsConnection(MyNode fromNode, int fromIndex, int toIndex)
        {
            MyParentInput groupInput = GroupInputNodes[toIndex];

            // Find all nodes connected to the input and check that they all accept the connection.
            // If the input is not connected to anything, it'll accept all connections.
            return Children.Union(GroupOutputNodes).All(child => child.InputConnections
                .Where(connection => connection != null && connection.From == groupInput)
                .WithIndex()
                .All(connection => child.AcceptsConnection(fromNode, fromIndex, connection.Index)));
        }

        public IEnumerable<MyConnection> GetConnections(MyOutput output)
        {
            if (Parent == null)
                return null;

            // Among the siblings of this node group, find connections that originate in the given output.
            int fromIndex = Array.IndexOf(GroupOutputNodes, output);
            if (fromIndex < 0)
                return null;

            return Parent.Children.SelectMany(sibling => sibling.InputConnections
                .Where(connection => connection != null && connection.From == this && connection.FromIndex == fromIndex));
        }
    }
}
