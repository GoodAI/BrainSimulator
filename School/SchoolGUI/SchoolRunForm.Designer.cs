namespace GoodAI.School.GUI
{
    partial class SchoolRunForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.dataGridView1 = new System.Windows.Forms.DataGridView();
            this.TaskType = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.WorldType = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.dataGridViewTextBoxColumn1 = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.btnRun = new System.Windows.Forms.ToolStripButton();
            this.btnStop = new System.Windows.Forms.ToolStripButton();
            this.btnPause = new System.Windows.Forms.ToolStripButton();
            this.btnStepOver = new System.Windows.Forms.ToolStripButton();
            this.btnDebug = new System.Windows.Forms.ToolStripButton();
            this.stepsDataGridViewTextBoxColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.timeDataGridViewTextBoxColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.statusDataGridViewTextBoxColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.isHiddenDataGridViewCheckBoxColumn = new System.Windows.Forms.DataGridViewCheckBoxColumn();
            this.checkStateDataGridViewCheckBoxColumn = new System.Windows.Forms.DataGridViewCheckBoxColumn();
            this.imageDataGridViewImageColumn = new System.Windows.Forms.DataGridViewImageColumn();
            this.isCheckedDataGridViewCheckBoxColumn = new System.Windows.Forms.DataGridViewCheckBoxColumn();
            this.isLeafDataGridViewCheckBoxColumn = new System.Windows.Forms.DataGridViewCheckBoxColumn();
            this.tagDataGridViewTextBoxColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.learningTaskNodeBindingSource = new System.Windows.Forms.BindingSource(this.components);
            ((System.ComponentModel.ISupportInitialize)(this.dataGridView1)).BeginInit();
            this.toolStrip1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.learningTaskNodeBindingSource)).BeginInit();
            this.SuspendLayout();
            // 
            // dataGridView1
            // 
            this.dataGridView1.AutoGenerateColumns = false;
            this.dataGridView1.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.dataGridView1.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {
            this.TaskType,
            this.WorldType,
            this.stepsDataGridViewTextBoxColumn,
            this.timeDataGridViewTextBoxColumn,
            this.statusDataGridViewTextBoxColumn,
            this.isHiddenDataGridViewCheckBoxColumn,
            this.checkStateDataGridViewCheckBoxColumn,
            this.imageDataGridViewImageColumn,
            this.isCheckedDataGridViewCheckBoxColumn,
            this.isLeafDataGridViewCheckBoxColumn,
            this.tagDataGridViewTextBoxColumn});
            this.dataGridView1.DataSource = this.learningTaskNodeBindingSource;
            this.dataGridView1.Location = new System.Drawing.Point(12, 28);
            this.dataGridView1.Name = "dataGridView1";
            this.dataGridView1.RowHeadersVisible = false;
            this.dataGridView1.Size = new System.Drawing.Size(807, 421);
            this.dataGridView1.TabIndex = 0;
            this.dataGridView1.CellFormatting += new System.Windows.Forms.DataGridViewCellFormattingEventHandler(this.dataGridView1_CellFormatting);
            this.dataGridView1.KeyDown += new System.Windows.Forms.KeyEventHandler(this.SchoolRunForm_KeyDown);
            // 
            // TaskType
            // 
            this.TaskType.DataPropertyName = "TaskType";
            this.TaskType.HeaderText = "TaskType";
            this.TaskType.Name = "TaskType";
            this.TaskType.ReadOnly = true;
            // 
            // WorldType
            // 
            this.WorldType.DataPropertyName = "WorldType";
            this.WorldType.HeaderText = "WorldType";
            this.WorldType.Name = "WorldType";
            this.WorldType.ReadOnly = true;
            // 
            // dataGridViewTextBoxColumn1
            // 
            this.dataGridViewTextBoxColumn1.DataPropertyName = "Tag";
            this.dataGridViewTextBoxColumn1.HeaderText = "Tag";
            this.dataGridViewTextBoxColumn1.Name = "dataGridViewTextBoxColumn1";
            // 
            // toolStrip1
            // 
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.btnRun,
            this.btnStop,
            this.btnPause,
            this.btnStepOver,
            this.btnDebug});
            this.toolStrip1.Location = new System.Drawing.Point(0, 0);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(831, 25);
            this.toolStrip1.TabIndex = 5;
            this.toolStrip1.Text = "toolStrip1";
            // 
            // btnRun
            // 
            this.btnRun.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnRun.Image = global::GoodAI.School.GUI.Properties.Resources.StatusAnnotations_Play_16xLG_color;
            this.btnRun.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnRun.Name = "btnRun";
            this.btnRun.Size = new System.Drawing.Size(23, 22);
            this.btnRun.Text = "Run Simulation";
            this.btnRun.Click += new System.EventHandler(m_mainForm.runToolButton_Click);
            // 
            // btnStop
            // 
            this.btnStop.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnStop.Image = global::GoodAI.School.GUI.Properties.Resources.StatusAnnotations_Stop_16xLG_color;
            this.btnStop.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnStop.Name = "btnStop";
            this.btnStop.Size = new System.Drawing.Size(23, 22);
            this.btnStop.Text = "Stop Simulation";
            this.btnStop.Click += new System.EventHandler(m_mainForm.stopToolButton_Click);
            // 
            // btnPause
            // 
            this.btnPause.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnPause.Image = global::GoodAI.School.GUI.Properties.Resources.StatusAnnotations_Pause_16xLG_color;
            this.btnPause.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnPause.Name = "btnPause";
            this.btnPause.Size = new System.Drawing.Size(23, 22);
            this.btnPause.Text = "Pause Simulation";
            this.btnPause.Click += new System.EventHandler(m_mainForm.pauseToolButton_Click);
            // 
            // btnStepOver
            // 
            this.btnStepOver.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnStepOver.Image = global::GoodAI.School.GUI.Properties.Resources.StepOver;
            this.btnStepOver.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnStepOver.Name = "btnStepOver";
            this.btnStepOver.Size = new System.Drawing.Size(23, 22);
            this.btnStepOver.Text = "Step Over";
            this.btnStepOver.Click += new System.EventHandler(m_mainForm.stepOverToolButton_Click);
            // 
            // btnDebug
            // 
            this.btnDebug.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnDebug.Image = global::GoodAI.School.GUI.Properties.Resources.debug;
            this.btnDebug.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnDebug.Name = "btnDebug";
            this.btnDebug.Size = new System.Drawing.Size(23, 22);
            this.btnDebug.Text = "Debug Simulation";
            this.btnDebug.Click += new System.EventHandler(m_mainForm.debugToolButton_Click);
            // 
            // stepsDataGridViewTextBoxColumn
            // 
            this.stepsDataGridViewTextBoxColumn.DataPropertyName = "Steps";
            this.stepsDataGridViewTextBoxColumn.HeaderText = "Steps";
            this.stepsDataGridViewTextBoxColumn.Name = "stepsDataGridViewTextBoxColumn";
            // 
            // timeDataGridViewTextBoxColumn
            // 
            this.timeDataGridViewTextBoxColumn.DataPropertyName = "Time";
            this.timeDataGridViewTextBoxColumn.HeaderText = "Time";
            this.timeDataGridViewTextBoxColumn.Name = "timeDataGridViewTextBoxColumn";
            // 
            // statusDataGridViewTextBoxColumn
            // 
            this.statusDataGridViewTextBoxColumn.DataPropertyName = "Status";
            this.statusDataGridViewTextBoxColumn.HeaderText = "Status";
            this.statusDataGridViewTextBoxColumn.Name = "statusDataGridViewTextBoxColumn";
            // 
            // isHiddenDataGridViewCheckBoxColumn
            // 
            this.isHiddenDataGridViewCheckBoxColumn.DataPropertyName = "IsHidden";
            this.isHiddenDataGridViewCheckBoxColumn.HeaderText = "IsHidden";
            this.isHiddenDataGridViewCheckBoxColumn.Name = "isHiddenDataGridViewCheckBoxColumn";
            this.isHiddenDataGridViewCheckBoxColumn.Visible = false;
            // 
            // checkStateDataGridViewCheckBoxColumn
            // 
            this.checkStateDataGridViewCheckBoxColumn.DataPropertyName = "CheckState";
            this.checkStateDataGridViewCheckBoxColumn.HeaderText = "CheckState";
            this.checkStateDataGridViewCheckBoxColumn.Name = "checkStateDataGridViewCheckBoxColumn";
            this.checkStateDataGridViewCheckBoxColumn.Visible = false;
            // 
            // imageDataGridViewImageColumn
            // 
            this.imageDataGridViewImageColumn.DataPropertyName = "Image";
            this.imageDataGridViewImageColumn.HeaderText = "Image";
            this.imageDataGridViewImageColumn.Name = "imageDataGridViewImageColumn";
            this.imageDataGridViewImageColumn.Visible = false;
            // 
            // isCheckedDataGridViewCheckBoxColumn
            // 
            this.isCheckedDataGridViewCheckBoxColumn.DataPropertyName = "IsChecked";
            this.isCheckedDataGridViewCheckBoxColumn.HeaderText = "IsChecked";
            this.isCheckedDataGridViewCheckBoxColumn.Name = "isCheckedDataGridViewCheckBoxColumn";
            this.isCheckedDataGridViewCheckBoxColumn.Visible = false;
            // 
            // isLeafDataGridViewCheckBoxColumn
            // 
            this.isLeafDataGridViewCheckBoxColumn.DataPropertyName = "IsLeaf";
            this.isLeafDataGridViewCheckBoxColumn.HeaderText = "IsLeaf";
            this.isLeafDataGridViewCheckBoxColumn.Name = "isLeafDataGridViewCheckBoxColumn";
            this.isLeafDataGridViewCheckBoxColumn.ReadOnly = true;
            this.isLeafDataGridViewCheckBoxColumn.Visible = false;
            // 
            // tagDataGridViewTextBoxColumn
            // 
            this.tagDataGridViewTextBoxColumn.DataPropertyName = "Tag";
            this.tagDataGridViewTextBoxColumn.HeaderText = "Tag";
            this.tagDataGridViewTextBoxColumn.Name = "tagDataGridViewTextBoxColumn";
            this.tagDataGridViewTextBoxColumn.Visible = false;
            // 
            // learningTaskNodeBindingSource
            // 
            this.learningTaskNodeBindingSource.DataSource = typeof(GoodAI.School.GUI.LearningTaskNode);
            // 
            // SchoolRunForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(831, 461);
            this.Controls.Add(this.toolStrip1);
            this.Controls.Add(this.dataGridView1);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.HideOnClose = true;
            this.Name = "SchoolRunForm";
            this.Text = "SchoolRunForm";
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.SchoolRunForm_KeyDown);
            ((System.ComponentModel.ISupportInitialize)(this.dataGridView1)).EndInit();
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.learningTaskNodeBindingSource)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.DataGridView dataGridView1;
        private System.Windows.Forms.BindingSource learningTaskNodeBindingSource;
        private System.Windows.Forms.DataGridViewTextBoxColumn nameDataGridViewTextBoxColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn worldDataGridViewTextBoxColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn dataGridViewTextBoxColumn1;
        private System.Windows.Forms.DataGridViewTextBoxColumn TaskType;
        private System.Windows.Forms.DataGridViewTextBoxColumn WorldType;
        private System.Windows.Forms.DataGridViewTextBoxColumn stepsDataGridViewTextBoxColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn timeDataGridViewTextBoxColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn statusDataGridViewTextBoxColumn;
        private System.Windows.Forms.DataGridViewCheckBoxColumn isHiddenDataGridViewCheckBoxColumn;
        private System.Windows.Forms.DataGridViewCheckBoxColumn checkStateDataGridViewCheckBoxColumn;
        private System.Windows.Forms.DataGridViewImageColumn imageDataGridViewImageColumn;
        private System.Windows.Forms.DataGridViewCheckBoxColumn isCheckedDataGridViewCheckBoxColumn;
        private System.Windows.Forms.DataGridViewCheckBoxColumn isLeafDataGridViewCheckBoxColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn tagDataGridViewTextBoxColumn;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.ToolStripButton btnRun;
        private System.Windows.Forms.ToolStripButton btnPause;
        private System.Windows.Forms.ToolStripButton btnStop;
        private System.Windows.Forms.ToolStripButton btnStepOver;
        private System.Windows.Forms.ToolStripButton btnDebug;
    }
}