namespace GoodAI.School.GUI
{
    partial class SchoolMainForm
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
            this.tree = new Aga.Controls.Tree.TreeViewAdv();
            this.nodeCheckBox1 = new Aga.Controls.Tree.NodeControls.NodeCheckBox();
            this.nodeTextBox1 = new Aga.Controls.Tree.NodeControls.NodeTextBox();
            this.groupBoxCurr = new System.Windows.Forms.GroupBox();
            this.btnDetailsCurr = new System.Windows.Forms.Button();
            this.btnDeleteCurr = new System.Windows.Forms.Button();
            this.btnNewCurr = new System.Windows.Forms.Button();
            this.btnOpen = new System.Windows.Forms.Button();
            this.btnSave = new System.Windows.Forms.Button();
            this.checkBoxAutosave = new System.Windows.Forms.CheckBox();
            this.btnImport = new System.Windows.Forms.Button();
            this.btnSaveAs = new System.Windows.Forms.Button();
            this.btnRun = new System.Windows.Forms.Button();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.folderBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.groupBoxTask = new System.Windows.Forms.GroupBox();
            this.btnDetailsTask = new System.Windows.Forms.Button();
            this.btnDeleteTask = new System.Windows.Forms.Button();
            this.btnNewTask = new System.Windows.Forms.Button();
            this.groupBoxCurr.SuspendLayout();
            this.groupBoxTask.SuspendLayout();
            this.SuspendLayout();
            // 
            // tree
            // 
            this.tree.AllowDrop = true;
            this.tree.BackColor = System.Drawing.SystemColors.Window;
            this.tree.ColumnHeaderHeight = 0;
            this.tree.DefaultToolTipProvider = null;
            this.tree.DragDropMarkColor = System.Drawing.Color.Black;
            this.tree.FullRowSelectActiveColor = System.Drawing.Color.Empty;
            this.tree.FullRowSelectInactiveColor = System.Drawing.Color.Empty;
            this.tree.LineColor = System.Drawing.SystemColors.ControlDark;
            this.tree.Location = new System.Drawing.Point(12, 31);
            this.tree.Model = null;
            this.tree.Name = "tree";
            this.tree.NodeControls.Add(this.nodeCheckBox1);
            this.tree.NodeControls.Add(this.nodeTextBox1);
            this.tree.NodeFilter = null;
            this.tree.SelectedNode = null;
            this.tree.Size = new System.Drawing.Size(405, 399);
            this.tree.TabIndex = 0;
            this.tree.ItemDrag += new System.Windows.Forms.ItemDragEventHandler(this.tree_ItemDrag);
            this.tree.SelectionChanged += new System.EventHandler(this.tree_SelectionChanged);
            this.tree.DragDrop += new System.Windows.Forms.DragEventHandler(this.tree_DragDrop);
            this.tree.DragOver += new System.Windows.Forms.DragEventHandler(this.tree_DragOver);
            this.tree.KeyDown += new System.Windows.Forms.KeyEventHandler(this.SchoolMainForm_KeyDown);
            // 
            // nodeCheckBox1
            // 
            this.nodeCheckBox1.DataPropertyName = "Enabled";
            this.nodeCheckBox1.EditEnabled = true;
            this.nodeCheckBox1.LeftMargin = 0;
            this.nodeCheckBox1.ParentColumn = null;
            // 
            // nodeTextBox1
            // 
            this.nodeTextBox1.DataPropertyName = "Text";
            this.nodeTextBox1.EditEnabled = true;
            this.nodeTextBox1.IncrementalSearchEnabled = true;
            this.nodeTextBox1.LeftMargin = 3;
            this.nodeTextBox1.ParentColumn = null;
            this.nodeTextBox1.DrawText += new System.EventHandler<Aga.Controls.Tree.NodeControls.DrawTextEventArgs>(this.nodeTextBox1_DrawText);
            // 
            // groupBoxCurr
            // 
            this.groupBoxCurr.Controls.Add(this.btnDetailsCurr);
            this.groupBoxCurr.Controls.Add(this.btnDeleteCurr);
            this.groupBoxCurr.Controls.Add(this.btnNewCurr);
            this.groupBoxCurr.Location = new System.Drawing.Point(423, 12);
            this.groupBoxCurr.Name = "groupBoxCurr";
            this.groupBoxCurr.Size = new System.Drawing.Size(92, 110);
            this.groupBoxCurr.TabIndex = 1;
            this.groupBoxCurr.TabStop = false;
            this.groupBoxCurr.Text = "Curriculum";
            // 
            // btnDetailsCurr
            // 
            this.btnDetailsCurr.Location = new System.Drawing.Point(6, 77);
            this.btnDetailsCurr.Name = "btnDetailsCurr";
            this.btnDetailsCurr.Size = new System.Drawing.Size(75, 23);
            this.btnDetailsCurr.TabIndex = 2;
            this.btnDetailsCurr.Text = "Details";
            this.btnDetailsCurr.UseVisualStyleBackColor = true;
            // 
            // btnDeleteCurr
            // 
            this.btnDeleteCurr.Location = new System.Drawing.Point(6, 48);
            this.btnDeleteCurr.Name = "btnDeleteCurr";
            this.btnDeleteCurr.Size = new System.Drawing.Size(75, 23);
            this.btnDeleteCurr.TabIndex = 1;
            this.btnDeleteCurr.Text = "Delete";
            this.btnDeleteCurr.UseVisualStyleBackColor = true;
            this.btnDeleteCurr.Click += new System.EventHandler(this.btnDeleteCurr_Click);
            // 
            // btnNewCurr
            // 
            this.btnNewCurr.Location = new System.Drawing.Point(6, 19);
            this.btnNewCurr.Name = "btnNewCurr";
            this.btnNewCurr.Size = new System.Drawing.Size(75, 23);
            this.btnNewCurr.TabIndex = 0;
            this.btnNewCurr.Text = "New";
            this.btnNewCurr.UseVisualStyleBackColor = true;
            this.btnNewCurr.Click += new System.EventHandler(this.btnNewCurr_Click);
            // 
            // btnOpen
            // 
            this.btnOpen.Location = new System.Drawing.Point(174, 2);
            this.btnOpen.Name = "btnOpen";
            this.btnOpen.Size = new System.Drawing.Size(75, 23);
            this.btnOpen.TabIndex = 11;
            this.btnOpen.Text = "Open";
            this.btnOpen.UseVisualStyleBackColor = true;
            this.btnOpen.Click += new System.EventHandler(this.btnOpen_Click);
            // 
            // btnSave
            // 
            this.btnSave.Location = new System.Drawing.Point(12, 2);
            this.btnSave.Name = "btnSave";
            this.btnSave.Size = new System.Drawing.Size(75, 23);
            this.btnSave.TabIndex = 10;
            this.btnSave.Text = "Save";
            this.btnSave.UseVisualStyleBackColor = true;
            this.btnSave.Click += new System.EventHandler(this.btnSave_Click);
            // 
            // checkBoxAutosave
            // 
            this.checkBoxAutosave.AutoSize = true;
            this.checkBoxAutosave.Location = new System.Drawing.Point(429, 413);
            this.checkBoxAutosave.Name = "checkBoxAutosave";
            this.checkBoxAutosave.Size = new System.Drawing.Size(104, 17);
            this.checkBoxAutosave.TabIndex = 5;
            this.checkBoxAutosave.Text = "Autosave results";
            this.checkBoxAutosave.UseVisualStyleBackColor = true;
            this.checkBoxAutosave.CheckedChanged += new System.EventHandler(this.checkBoxAutosave_CheckedChanged);
            // 
            // btnImport
            // 
            this.btnImport.Location = new System.Drawing.Point(255, 2);
            this.btnImport.Name = "btnImport";
            this.btnImport.Size = new System.Drawing.Size(75, 23);
            this.btnImport.TabIndex = 4;
            this.btnImport.Text = "Import";
            this.btnImport.UseVisualStyleBackColor = true;
            this.btnImport.Click += new System.EventHandler(this.btnImport_Click);
            // 
            // btnSaveAs
            // 
            this.btnSaveAs.Location = new System.Drawing.Point(93, 2);
            this.btnSaveAs.Name = "btnSaveAs";
            this.btnSaveAs.Size = new System.Drawing.Size(75, 23);
            this.btnSaveAs.TabIndex = 3;
            this.btnSaveAs.Text = "Save As...";
            this.btnSaveAs.UseVisualStyleBackColor = true;
            this.btnSaveAs.Click += new System.EventHandler(this.SaveProjectAs);
            // 
            // btnRun
            // 
            this.btnRun.Location = new System.Drawing.Point(429, 384);
            this.btnRun.Name = "btnRun";
            this.btnRun.Size = new System.Drawing.Size(75, 23);
            this.btnRun.TabIndex = 2;
            this.btnRun.Text = "Run...";
            this.btnRun.UseVisualStyleBackColor = true;
            this.btnRun.Click += new System.EventHandler(this.btnRun_Click);
            // 
            // saveFileDialog1
            // 
            this.saveFileDialog1.DefaultExt = "xml";
            this.saveFileDialog1.Filter = "Curriculum files|*.xml|All files|*.*";
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // groupBoxTask
            // 
            this.groupBoxTask.Controls.Add(this.btnDetailsTask);
            this.groupBoxTask.Controls.Add(this.btnDeleteTask);
            this.groupBoxTask.Controls.Add(this.btnNewTask);
            this.groupBoxTask.Location = new System.Drawing.Point(423, 128);
            this.groupBoxTask.Name = "groupBoxTask";
            this.groupBoxTask.Size = new System.Drawing.Size(92, 111);
            this.groupBoxTask.TabIndex = 10;
            this.groupBoxTask.TabStop = false;
            this.groupBoxTask.Text = "Learning task";
            // 
            // btnDetailsTask
            // 
            this.btnDetailsTask.Location = new System.Drawing.Point(5, 77);
            this.btnDetailsTask.Name = "btnDetailsTask";
            this.btnDetailsTask.Size = new System.Drawing.Size(75, 23);
            this.btnDetailsTask.TabIndex = 2;
            this.btnDetailsTask.Text = "Details";
            this.btnDetailsTask.UseVisualStyleBackColor = true;
            this.btnDetailsTask.Click += new System.EventHandler(this.btnDetailsTask_Click);
            // 
            // btnDeleteTask
            // 
            this.btnDeleteTask.Location = new System.Drawing.Point(5, 48);
            this.btnDeleteTask.Name = "btnDeleteTask";
            this.btnDeleteTask.Size = new System.Drawing.Size(75, 23);
            this.btnDeleteTask.TabIndex = 1;
            this.btnDeleteTask.Text = "Delete";
            this.btnDeleteTask.UseVisualStyleBackColor = true;
            this.btnDeleteTask.Click += new System.EventHandler(this.DeleteNode);
            // 
            // btnNewTask
            // 
            this.btnNewTask.Location = new System.Drawing.Point(6, 19);
            this.btnNewTask.Name = "btnNewTask";
            this.btnNewTask.Size = new System.Drawing.Size(75, 23);
            this.btnNewTask.TabIndex = 0;
            this.btnNewTask.Text = "New...";
            this.btnNewTask.UseVisualStyleBackColor = true;
            this.btnNewTask.Click += new System.EventHandler(this.btnNewTask_Click);
            // 
            // SchoolMainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(532, 436);
            this.Controls.Add(this.groupBoxTask);
            this.Controls.Add(this.btnSave);
            this.Controls.Add(this.btnOpen);
            this.Controls.Add(this.btnRun);
            this.Controls.Add(this.btnSaveAs);
            this.Controls.Add(this.groupBoxCurr);
            this.Controls.Add(this.tree);
            this.Controls.Add(this.btnImport);
            this.Controls.Add(this.checkBoxAutosave);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.HideOnClose = true;
            this.Name = "SchoolMainForm";
            this.Text = "SchoolMainForm";
            this.Load += new System.EventHandler(this.SchoolMainForm_Load);
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.SchoolMainForm_KeyDown);
            this.groupBoxCurr.ResumeLayout(false);
            this.groupBoxTask.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private Aga.Controls.Tree.TreeViewAdv tree;
        private System.Windows.Forms.GroupBox groupBoxCurr;
        private System.Windows.Forms.CheckBox checkBoxAutosave;
        private System.Windows.Forms.Button btnImport;
        private System.Windows.Forms.Button btnSaveAs;
        private System.Windows.Forms.Button btnDetailsCurr;
        private System.Windows.Forms.Button btnDeleteCurr;
        private System.Windows.Forms.Button btnNewCurr;
        private System.Windows.Forms.Button btnRun;
        private Aga.Controls.Tree.NodeControls.NodeTextBox nodeTextBox1;
        private System.Windows.Forms.SaveFileDialog saveFileDialog1;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog1;
        private Aga.Controls.Tree.NodeControls.NodeCheckBox nodeCheckBox1;
        private System.Windows.Forms.Button btnOpen;
        private System.Windows.Forms.Button btnSave;
        private System.Windows.Forms.GroupBox groupBoxTask;
        private System.Windows.Forms.Button btnDetailsTask;
        private System.Windows.Forms.Button btnDeleteTask;
        private System.Windows.Forms.Button btnNewTask;
    }
}