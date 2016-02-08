using GoodAI.BrainSimulator.Forms;
using GoodAI.Core.Configuration;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.School.GUI
{
    public partial class SchoolRunForm : DockContent
    {
        public List<LearningTaskNode> Data;
        public PlanDesign Design;
        //private BindingSource m_source;
        private readonly MainForm m_mainForm;
        private SchoolWorld m_school;

        public SchoolRunForm(MainForm mainForm)
        {
            m_mainForm = mainForm;
            InitializeComponent();

            dataGridView1.DataSource = Data;
            // using BindingSource is probably better but it wasn't updating; don't know why - postponed
            // m_source = new BindingSource();
            // m_source.DataSource = Data;
            // dataGridView1.DataSource = m_source;
        }

        private void SelectSchoolWorld()
        {
            // Enable SchoolWorld
            // Add SchoolWorld to World combolist
            // -- lets assume we have this (TODO but some of this is checked by SelectWorldInWorldList)
            // select it
            m_school = new SchoolWorld();
            m_mainForm.SelectWorldInWorldList(m_school);
            //m_mainForm.Project.World = m_school;
        }

        private void CreateCurriculum()
        {
            m_school.Curriculum = (SchoolCurriculum)Design;
        }

        private void PrepareSimulation()
        {
            SelectSchoolWorld();
            CreateCurriculum();
        }

        public void UpdateData()
        {
            //m_source.ResetBindings(true);
            dataGridView1.DataSource = Data;
            PrepareSimulation();
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
                    if (typeValue != null)
                        e.Value = typeValue.Name;
                }
            }
        }

        private void btnPlay_Click(object sender, EventArgs e)
        {
            m_mainForm.runToolButton.PerformClick();
        }

        private void btnPause_Click(object sender, EventArgs e)
        {
            m_mainForm.pauseToolButton.PerformClick();
        }

        private void btnStop_Click(object sender, EventArgs e)
        {
            m_mainForm.stopToolButton.PerformClick();
        }

        private void btnClose_Click(object sender, EventArgs e)
        {
            Close();
        }
    }
}
