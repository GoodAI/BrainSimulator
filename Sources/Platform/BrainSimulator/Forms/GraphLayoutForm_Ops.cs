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

            ToolStripItem newButton = new ToolStripMenuItem();
            ToolStripItemCollection items = targetMenuButton.DropDownItems;

            newButton.Image = nodeConfig.SmallImage;
            newButton.Name = nodeConfig.NodeType.Name;
            newButton.Text = MyProject.ShortenNodeTypeName(nodeConfig.NodeType);
            newButton.MouseDown += addNodeButton_MouseDown;
            newButton.Tag = nodeConfig.NodeType;

            newButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.ImageAndText;
            newButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            newButton.ImageTransparentColor = System.Drawing.Color.Magenta;

            if (items.Count > 0 && (items[items.Count - 1].Tag as Type).Namespace != nodeConfig.NodeType.Namespace)
            {
                items.Add(new ToolStripSeparator());
            }

            items.Add(newButton);
        }


        // TODO(P): delete/reuse this func
        /*
        private void AddNodeButton(MyNodeConfig nodeInfo, bool isTransform)
        {            
            ToolStripItem newButton = isTransform ? new ToolStripMenuItem() : newButton = new ToolStripButton();
            ToolStripItemCollection items;

            newButton.Image = nodeInfo.SmallImage;
            newButton.Name = nodeInfo.NodeType.Name;
            newButton.ToolTipText =  MyProject.ShortenNodeTypeName(nodeInfo.NodeType);
            newButton.MouseDown += addNodeButton_MouseDown;
            newButton.Tag = nodeInfo.NodeType;

            newButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            newButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            newButton.ImageTransparentColor = System.Drawing.Color.Magenta;

            if (isTransform)
            {
                newButton.DisplayStyle = ToolStripItemDisplayStyle.ImageAndText;
                newButton.Text = newButton.ToolTipText;
                // items = transformMenu.DropDownItems;  // TODO(P)
                return;
            }
            else
            {
                items = nodesToolStrip.Items;
                newButton.MouseUp += newButton_MouseUp;          
            }

            if (items.Count > 0 && (items[items.Count - 1].Tag as Type).Namespace != nodeInfo.NodeType.Namespace)
            {
                items.Add(new ToolStripSeparator());
            }
            items.Add(newButton);
        }
        */

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
            StringCollection toolBarNodes = Properties.Settings.Default.ToolBarNodes;
            string typeName = (nodeButton.Tag as Type).Name;
            if (toolBarNodes != null && toolBarNodes.Contains(typeName))
            {
                toolBarNodes.Remove(typeName);
                nodesToolStrip.Items.Remove(nodeButton);
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

                    NodeConnection c = Desktop.Connect(fromNodeViewItem, toNodeView.GetInputBranchItem(connection.ToIndex));
                    c.Tag = connection;
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
            {
                Desktop.FocusElement = nodeView;
            }
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

                IDictionary<IMyExecutable, TimeSpan> profilingInfo = Target.ExecutionBlock.ProfilingInfo;

                // Maps IMyExecutable to an object that holds both node and its view.
                Dictionary<IMyExecutable, MyNodeView> nodes = Desktop.Nodes.Cast<MyNodeView>()
                    .Select(view => new {View = view, Node = view.Node as MyWorkingNode})
                    .Where(nodeInfo => nodeInfo.Node != null)
                    .ToDictionary(nodeInfo => nodeInfo.Node.ExecutionBlock as IMyExecutable, nodeInfo => nodeInfo.View);

                // The total duration of the displayed nodes.
                double sum = profilingInfo.Values.Sum(value => value.TotalMilliseconds);

                foreach (KeyValuePair<IMyExecutable, TimeSpan> profiling in profilingInfo)
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

        private void ResetNodeColours()
        {
            foreach (var node in Desktop.Nodes.Cast<MyNodeView>())
                node.SetDefaultBackground();

            Desktop.Invalidate();
        }
    }
}
