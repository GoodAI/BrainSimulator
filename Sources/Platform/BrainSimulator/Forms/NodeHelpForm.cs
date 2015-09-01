using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using System;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class NodeHelpForm : DockContent
    {
        private MainForm m_mainForm;
        private MyNode m_target;
        private string m_style;

        public MyNode Target
        {
            get { return m_target; }
            set
            {
                m_target = value;
                if (Visible)
                {
                    UpdateWebBrowser();
                }
            }
        }

        public NodeHelpForm(MainForm mainForm)
        {
            InitializeComponent();
            m_mainForm = mainForm;

            //m_style = MyResources.GetTextFromAssembly(Assembly.GetExecutingAssembly(), "help_page.css", "res");
            m_style = Properties.Resources.help_page;
        }

        private void CheckText(ref string text) 
        {
            if (text == null)
            {
                text = "<span class=\"red\">DOCUMENTATION IS MISSING</span>";
            }
        }

        private void UpdateWebBrowser() 
        {                            
            if (Target != null)
            {
                string author;
                string status;
                string summary;
                string description;
                string include;

                m_mainForm.Documentation.HasAuthor(Target.GetType(), out author);
                CheckText(ref author);
                m_mainForm.Documentation.HasStatus(Target.GetType(), out status);
                CheckText(ref status);
                m_mainForm.Documentation.HasSummary(Target.GetType(), out summary);
                CheckText(ref summary);
                m_mainForm.Documentation.HasDescription(Target.GetType(), out description);
                CheckText(ref description);

                if (m_mainForm.Documentation.HasDocElement(Target.GetType(), "externdoc", out include))
                {
                    webBrowser.Navigate(include);
                }
                else
                {
                    string html = "<html><head><meta http-equiv=\"X-UA-Compatible\" content=\"IE=11\"/>";
                    html += "<style type=\"text/css\">" + m_style + "</style></head><body>";

                    html += "<h1>" + Target.GetType().Name + "</h1>";
                    html += "<em>Author: " + author + "<br/> Status: " + status + "</em>";
                    html += "<p>" + summary + "</p>";
                    html += "<h2>Node description</h2>";
                    html += "<p>" + description + "</p>";

                    MyNodeInfo nodeInfo = Target.GetInfo();

                    if (Target is MyWorkingNode)
                    {
                        foreach (string taskName in nodeInfo.KnownTasks.Keys)
                        {
                            MyTask task = (Target as MyWorkingNode).GetTaskByPropertyName(taskName);
                            Type taskType = task.GetType();
                            m_mainForm.Documentation.HasSummary(taskType, out summary);
                            CheckText(ref summary);                            

                            if (taskType.Name != task.Name)
                            {
                                html += "<h3>" + taskType.Name + " \u2014 " + task.Name + "</h3>";
                            }
                            else
                            {
                                html += "<h3>" + taskType.Name + "</h3>";
                            }
                            html += "<p>" + summary + "</p>";
                        }

                        html += "</body></html>";
                        webBrowser.DocumentText = html;
                    }
                }
            }
        }

        private void NodeHelpForm_VisibleChanged(object sender, EventArgs e)
        {
            if (Visible)
            {
                UpdateWebBrowser();
            }
        }      
    }
}
