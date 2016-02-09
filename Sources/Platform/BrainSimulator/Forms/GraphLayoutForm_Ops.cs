using GoodAI.BrainSimulator.NodeView;
using GoodAI.BrainSimulator.Nodes;
using GoodAI.Core;
using GoodAI.Core.Configuration;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using Graph;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using GoodAI.BrainSimulator.Utils;
using GoodAI.Core.Execution;
using GoodAI.Core.Task;
using System.Diagnostics;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class GraphLayoutForm
    {
        private bool m_wasProfiling;

        private ToolStripDropDownButton FindTargetMenuButton(string categoryName)
        {
            ToolStripDropDownButton targetMenuButton = null;

            foreach (var item in nodesToolStrip.Items)
            {
                var menuButton = item as ToolStripDropDownButton;
                if (menuButton == null)
                    continue;

                if ((menuButton.Tag as string) == categoryName)
                {
                    targetMenuButton = menuButton;
                    break;
                }
            }

            if (targetMenuButton == null)
            {
                MyLog.WARNING.WriteLine("Unable to find menu drop down button for category " + categoryName);
            }

            return targetMenuButton;
        }

        private void AddNodeButtonToCategoryMenu(MyNodeConfig nodeConfig)
        {
            ToolStripDropDownButton targetMenuButton =
                FindTargetMenuButton(CategorySortingHat.DetectCategoryName(nodeConfig));  // TODO: optimize with HashSet
            if (targetMenuButton == null)
                return;

            ToolStripItem newButton = new ToolStripMenuItem()
            {
                Text = MyProject.ShortenNodeTypeName(nodeConfig.NodeType),
                DisplayStyle = ToolStripItemDisplayStyle.ImageAndText
            };

            ToolStripItemCollection targetItems = targetMenuButton.DropDownItems;

            InnerAddNodeButtonOrMenuItem(newButton, nodeConfig, targetItems, addSeparators: true);
        }

        private void AddNodeButton(MyNodeConfig nodeConfig)
        {
            var newButton = new ToolStripButton
            {
                ToolTipText = MyProject.ShortenNodeTypeName(nodeConfig.NodeType),
                DisplayStyle = ToolStripItemDisplayStyle.Image
            };

            newButton.MouseUp += newButton_MouseUp;

            ToolStripItemCollection targetItems = nodesToolStrip.Items;

            InnerAddNodeButtonOrMenuItem(newButton, nodeConfig, targetItems);
        }

        private void InnerAddNodeButtonOrMenuItem(ToolStripItem newButton, MyNodeConfig nodeConfig,
            ToolStripItemCollection targetItems, bool addSeparators = false)
        {
            newButton.Image = nodeConfig.SmallImage;
            newButton.Name = nodeConfig.NodeType.Name;
            newButton.MouseDown += addNodeButton_MouseDown;
            newButton.Tag = nodeConfig.NodeType;

            newButton.ImageScaling = ToolStripItemImageScaling.None;
            newButton.ImageTransparentColor = System.Drawing.Color.Magenta;

            // separate buttons for nodes from different namespaces
            if (addSeparators && (targetItems.Count > 0))
            {
                var nodeType = targetItems[targetItems.Count - 1].Tag as Type;
                if ((nodeType != null) && (nodeType.Namespace != nodeConfig.NodeType.Namespace))
                {
                    targetItems.Add(new ToolStripSeparator());
                }
            }

            targetItems.Add(newButton);

            // TODO: Add undo here if we also want to undo non-model-related actions
        }

        void newButton_MouseUp(object sender, MouseEventArgs e)
        {
            if (e.Button == System.Windows.Forms.MouseButtons.Right)
            {
                contextMenuStrip.Tag = sender;
                ToolStripItem button = sender as ToolStripItem;
                contextMenuStrip.Show(nodesToolStrip, button.Bounds.Left + e.Location.X + 2, button.Bounds.Top + e.Location.Y + 2);                
            }
        }

        private void RemoveNodeButton(ToolStripItem nodeButton)
        {
            StringCollection quickToolBarNodes = Properties.Settings.Default.QuickToolBarNodes;
            if ((quickToolBarNodes == null) || !(nodeButton.Tag is Type))
                return;

            string typeName = ((Type) nodeButton.Tag).Name;

            if (quickToolBarNodes.Contains(typeName))
            {
                quickToolBarNodes.Remove(typeName);
                nodesToolStrip.Items.Remove(nodeButton);

                // TODO: Add undo here if we also want to undo non-model-related actions
            }
        }

        private void LoadContentIntoDesktop()
        {
            Dictionary<MyNode, MyNodeView> nodeViewTable = new Dictionary<MyNode, MyNodeView>();

            //Global i/o

            for(int i = 0; i < Target.GroupInputNodes.Length; i++)
            {
                MyParentInput inputNode = Target.GroupInputNodes[i];

                if (inputNode.Location == null)
                {
                    inputNode.Location = new MyLocation() { X = 50, Y = 150 * i + 100 };
                }

                MyNodeView inputView = MyNodeView.CreateNodeView(inputNode, Desktop);
                inputView.UpdateView();
                Desktop.AddNode(inputView);
                nodeViewTable[inputNode] = inputView;
            }


            for (int i = 0; i < Target.GroupOutputNodes.Length; i++)
            {
                MyOutput outputNode = Target.GroupOutputNodes[i];

                if (outputNode.Location == null)
                {
                    outputNode.Location = new MyLocation() { X = 800, Y = 150 * i + 100 };
                }

                MyNodeView outputView = MyNodeView.CreateNodeView(outputNode, Desktop);
                outputView.UpdateView();
                Desktop.AddNode(outputView);
                nodeViewTable[outputNode] = outputView;
            }                       

            //other nodes
            foreach (MyNode node in Target.Children)
            {
                MyNodeView newNodeView = MyNodeView.CreateNodeView(node, Desktop);
                newNodeView.UpdateView();

                Desktop.AddNode(newNodeView);
                nodeViewTable[node] = newNodeView;
            }

            foreach (MyNode outputNode in Target.GroupOutputNodes)
            {             
                RestoreConnections(outputNode, nodeViewTable);
            }

            //other connections
            foreach (MyNode node in Target.Children)
            {
                RestoreConnections(node, nodeViewTable);
            }         

            RefreshProfiling();
        }

        private void RestoreConnections(MyNode node, Dictionary<MyNode, MyNodeView> nodeViewTable) 
        {
            MyNodeView toNodeView = nodeViewTable[node];
   
            for (int i = 0; i < node.InputBranches; i++)
            {
                MyConnection connection = node.InputConnections[i];

                if (connection != null)
                {
                    MyNodeView fromNodeView = nodeViewTable[connection.From];
                    NodeItem fromNodeViewItem = fromNodeView.GetOuputBranchItem(connection.FromIndex);

                    MyNodeViewConnection c = Desktop.Connect(fromNodeViewItem, toNodeView.GetInputBranchItem(connection.ToIndex)) as MyNodeViewConnection;
                    Debug.Assert(c != null, "Invalid connection factory delegate");

                    c.Tag = connection;
                    c.Hidden = connection.IsHidden;
                }
            }       
        }

        private void StoreLayoutProperties() 
        {
            Target.LayoutProperties = new MyLayout();
            Target.LayoutProperties.Zoom = Desktop.Zoom;
            Target.LayoutProperties.Translation.X = Desktop.Translation.X;
            Target.LayoutProperties.Translation.Y = Desktop.Translation.Y;
        }

        public void SelectNodeView(MyNode node)
        {
            Node nodeView = Desktop.Nodes.First(nw => (nw as MyNodeView).Node == node);

            if (nodeView != null)
                Desktop.FocusElement = nodeView;
        }

        public void SelectNodeView(int nodeId)
        {
            Node nodeView = Desktop.Nodes.FirstOrDefault(nw => (nw as MyNodeView).Node.Id == nodeId);

            if (nodeView != null)
                Desktop.FocusElement = nodeView;
        }

        void SimulationHandler_StateChanged(object sender, MySimulationHandler.StateEventArgs e)
        {
            nodesToolStrip.Enabled = e.NewState == MySimulationHandler.SimulationState.STOPPED;
            updateModelButton.Enabled = nodesToolStrip.Enabled;

            if (e.NewState == MySimulationHandler.SimulationState.STOPPED)
                ResetNodeColours();
        }

        private void SimulationHandler_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            RefreshProfiling();
        }

        private void RefreshProfiling()
        {
            if (MyExecutionBlock.IsProfiling)
            {
                m_wasProfiling = true;

                // Maps IMyExecutable to an object that holds both node and its view.
                Dictionary<IMyExecutable, MyNodeView> nodes = Desktop.Nodes.Cast<MyNodeView>()
                    .Select(view => new {View = view, Node = view.Node as MyWorkingNode})
                    .Where(nodeInfo => nodeInfo.Node != null)
                    .ToDictionary(nodeInfo => nodeInfo.Node.ExecutionBlock as IMyExecutable, nodeInfo => nodeInfo.View);

                IDictionary<IMyExecutable, TimeSpan> profilingInfo = Target.ExecutionBlock.ProfilingInfo;
                Dictionary<IMyExecutable, TimeSpan> profilingInfoWithTasks = GetProfilingInfoWithTasks(profilingInfo);

                // The total duration of the displayed nodes.
                double sum = profilingInfo.Values.Sum(value => value.TotalMilliseconds);

                foreach (KeyValuePair<IMyExecutable, TimeSpan> profiling in profilingInfoWithTasks)
                {
                    // Find the node that corresponds to the executable.
                    MyNodeView nodeView;
                    if (!nodes.TryGetValue(profiling.Key, out nodeView))
                        continue;

                    // Calculate and assign the color to the node.
                    double factor = profiling.Value.TotalMilliseconds/sum;

                    nodeView.BackgroundColor = Profiling.ItemColor(factor); 
                }
            }
            else if (m_wasProfiling)
            {
                m_wasProfiling = false;
                ResetNodeColours();
            }
        }

        /// <summary>
        /// Merge tasks directly below the Target node into the nodes in this level of the execution tree.
        /// This adds the signal values but mostly is used for custom execution plans.
        /// </summary>
        /// <param name="profilingInfo">The target's </param>
        /// <returns></returns>
        private static Dictionary<IMyExecutable, TimeSpan> GetProfilingInfoWithTasks(IDictionary<IMyExecutable, TimeSpan> profilingInfo)
        {
            var profilingInfoWithTasks = new Dictionary<IMyExecutable, TimeSpan>();
            foreach (KeyValuePair<IMyExecutable, TimeSpan> profiling in profilingInfo)
            {
                var task = profiling.Key as MyTask;
                if (task != null)
                {
                    // Tasks belong to a node, their time should be added to the node's.
                    MyWorkingNode node = task.GenericOwner;
                    if (profilingInfoWithTasks.ContainsKey(node.ExecutionBlock))
                    {
                        profilingInfoWithTasks[node.ExecutionBlock] =
                            profilingInfoWithTasks[node.ExecutionBlock].Add(profiling.Value);
                    }
                    else
                    {
                        profilingInfoWithTasks[node.ExecutionBlock] = profiling.Value;
                    }
                }
                else
                {
                    profilingInfoWithTasks[profiling.Key] = profiling.Value;
                }
            }
            return profilingInfoWithTasks;
        }

        private void ResetNodeColours()
        {
            foreach (var node in Desktop.Nodes.Cast<MyNodeView>())
                node.SetDefaultBackground();

            Desktop.Invalidate();
        }
    }
}
