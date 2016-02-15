using GoodAI.BrainSimulator.Forms;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.School.GUI
{
    public partial class SchoolRunForm : DockContent
    {
        public List<LearningTaskNode> Data;
        public PlanDesign Design;

        private readonly MainForm m_mainForm;
        private string m_runName;
        private SchoolWorld m_school;
        private bool m_showVisual;
        private ObserverForm m_observer;

        public string RunName
        {
            get { return m_runName; }
            set
            {
                m_runName = value;

                string result = "School run";
                if (!String.IsNullOrEmpty(m_runName))
                    result += " - " + RunName;
                this.Text = result;
            }
        }

        public SchoolRunForm(MainForm mainForm)
        {
            m_mainForm = mainForm;
            InitializeComponent();

            m_mainForm.SimulationHandler.StateChanged += UpdateButtons;
            UpdateButtons(null, null);
        }

        private void UpdateButtons(object sender, Core.Execution.MySimulationHandler.StateEventArgs e)
        {
            btnRun.Enabled = m_mainForm.runToolButton.Enabled;
            btnPause.Enabled = m_mainForm.pauseToolButton.Enabled;
            btnStop.Enabled = m_mainForm.stopToolButton.Enabled;
        }

        public void UpdateData()
        {
            dataGridView1.DataSource = Data;
            PrepareSimulation();
            m_showVisual = true;
            SetObserver();

            if (Properties.School.Default.AutorunEnabled)
                btnRun.PerformClick();
        }

        private void SetObserver()
        {
            if (m_showVisual) {
                if (m_observer == null)
                {
                    try
                    {
                        MyMemoryBlockObserver observer = new MyMemoryBlockObserver();
                        observer.Target = m_school.Visual;

                        if (observer == null)
                            throw new InvalidOperationException("No observer was initialized");

                        m_observer = new ObserverForm(m_mainForm, observer, m_school);
                        m_mainForm.ObserverViews.Add(m_observer);
                        m_observer.TopLevel = false;
                        observerDockPanel.Controls.Add(m_observer);
                        m_observer.Dock = DockStyle.Fill;
                        m_observer.Show();
                        m_observer.Size = new System.Drawing.Size(300, 300);
                        m_observer.CloseButtonVisible = false;
                        //
                        //Form f = observerDockPanel.FindForm();
                        //newView.MdiParent = f;
                        //f.Show();
                    }
                    catch (Exception e)
                    {
                        MyLog.ERROR.WriteLine("Error creating observer: " + e.Message);
                    }
                }
                else if (m_observer.IsHidden)
                {
                    m_observer.Show();
                }
                else
                {
                    observerDockPanel.Show();
                }
            }
            else
            {
                if (m_observer != null)
                {
                    observerDockPanel.Hide();
                }
            }
        }

        private void SelectSchoolWorld()
        {
            m_mainForm.SelectWorldInWorldList(typeof(SchoolWorld));
            m_school = (SchoolWorld)m_mainForm.Project.World;
        }

        private void CreateCurriculum()
        {
            m_school.Curriculum = Design.AsSchoolCurriculum(m_school);
            foreach (ILearningTask task in m_school.Curriculum)
                task.SchoolWorld = m_school;
        }

        private void PrepareSimulation()
        {
            SelectSchoolWorld();
            CreateCurriculum();
        }

        private void dataGridView1_CellFormatting(object sender, DataGridViewCellFormattingEventArgs e)
        {
            string columnName = dataGridView1.Columns[e.ColumnIndex].Name;
            if (columnName.Equals(TaskType.Name) || columnName.Equals(WorldType.Name))
            {
                // I am not sure about how bad this approach is, but it get things done
                if (e.Value != null)
                {
                    Type typeValue = e.Value as Type;

                    DisplayNameAttribute displayNameAtt = typeValue.GetCustomAttributes(typeof(DisplayNameAttribute), true).FirstOrDefault() as DisplayNameAttribute;
                    if (displayNameAtt != null)
                        e.Value = displayNameAtt.DisplayName;
                    else
                        e.Value = typeValue.Name;
                }
            }
        }

        private void SchoolRunForm_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.KeyCode)
            {
                case Keys.F5:
                    {
                        btnRun.PerformClick();
                        break;
                    }
                case Keys.F7:
                    {
                        btnPause.PerformClick();
                        break;
                    }
                case Keys.F8:
                    {
                        btnStop.PerformClick();
                        break;
                    }
            }
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            CheckBox c = (CheckBox)sender;
            m_showVisual = c.Checked;
            SetObserver();
        }
    }
}
