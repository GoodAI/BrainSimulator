using GoodAI.Core.Nodes;
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
    public partial class TextEditForm : DockContent
    {
        private MainForm m_mainForm;        

        public MyScriptableNode Target { get; private set; } 

        public TextEditForm(MainForm mainForm, MyScriptableNode target)
        {
            InitializeComponent();

            m_mainForm = mainForm;
            Target = target;
            Text = target.Name;
            textBox.Text = Target.Script;
        }

        private void textBox_TextChanged(object sender, EventArgs e)
        {
            Target.Script = textBox.Text;
        }
    }
}
