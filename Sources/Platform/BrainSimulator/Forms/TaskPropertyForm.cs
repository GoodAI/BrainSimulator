using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.BrainSimulator.NodeView;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class TaskPropertyForm : DockContent
    {
        private MainForm m_mainForm;

        public MyTask Target
        {
            get { return propertyGrid.SelectedObject as MyTask; }
            set { propertyGrid.SelectedObject = value; }
        }

        public TaskPropertyForm(MainForm mainForm)
        {
            m_mainForm = mainForm;
            InitializeComponent();                   
            propertyGrid.BrowsableAttributes = new AttributeCollection(new MyBrowsableAttribute());
        }

        private void propertyGrid_PropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            //TODO: rewrite this, no need to loop all, just topmost
            foreach (GraphLayoutForm graphView in m_mainForm.GraphViews.Values)
            {
                if (graphView.Desktop.FocusElement is MyNodeView)
                {
                    MyNodeView nodeView = graphView.Desktop.FocusElement as MyNodeView;
                    nodeView.UpdateView();
                }                
                graphView.Desktop.Invalidate();
            }
        }
    }
}
