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
            this.nodeTextBox1 = new Aga.Controls.Tree.NodeControls.NodeTextBox();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.btnNewTask = new System.Windows.Forms.Button();
            this.btnDetailsTask = new System.Windows.Forms.Button();
            this.checkBoxAutosave = new System.Windows.Forms.CheckBox();
            this.btnImportCurr = new System.Windows.Forms.Button();
            this.btnExportCurr = new System.Windows.Forms.Button();
            this.btnDetailsCurr = new System.Windows.Forms.Button();
            this.btnDelete = new System.Windows.Forms.Button();
            this.btnNewCurr = new System.Windows.Forms.Button();
            this.btnRun = new System.Windows.Forms.Button();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.btnCurrFolder = new System.Windows.Forms.Button();
            this.folderBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.groupBox1.SuspendLayout();
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
            this.tree.Location = new System.Drawing.Point(12, 12);
            this.tree.Model = null;
            this.tree.Name = "tree";
            this.tree.NodeControls.Add(this.nodeTextBox1);
            this.tree.NodeFilter = null;
            this.tree.SelectedNode = null;
            this.tree.Size = new System.Drawing.Size(405, 401);
            this.tree.TabIndex = 0;
            this.tree.ItemDrag += new System.Windows.Forms.ItemDragEventHandler(this.tree_ItemDrag);
            this.tree.SelectionChanged += new System.EventHandler(this.tree_SelectionChanged);
            this.tree.DragDrop += new System.Windows.Forms.DragEventHandler(this.tree_DragDrop);
            this.tree.DragOver += new System.Windows.Forms.DragEventHandler(this.tree_DragOver);
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
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.btnNewTask);
            this.groupBox1.Controls.Add(this.btnDetailsTask);
            this.groupBox1.Controls.Add(this.checkBoxAutosave);
            this.groupBox1.Controls.Add(this.btnImportCurr);
            this.groupBox1.Controls.Add(this.btnExportCurr);
            this.groupBox1.Controls.Add(this.btnDetailsCurr);
            this.groupBox1.Controls.Add(this.btnDelete);
            this.groupBox1.Controls.Add(this.btnNewCurr);
            this.groupBox1.Location = new System.Drawing.Point(423, 12);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(118, 249);
            this.groupBox1.TabIndex = 1;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Curriculum";
            // 
            // btnNewTask
            // 
            this.btnNewTask.Location = new System.Drawing.Point(6, 48);
            this.btnNewTask.Name = "btnNewTask";
            this.btnNewTask.Size = new System.Drawing.Size(75, 23);
            this.btnNewTask.TabIndex = 9;
            this.btnNewTask.Text = "New task";
            this.btnNewTask.UseVisualStyleBackColor = true;
            this.btnNewTask.Click += new System.EventHandler(this.btnNewTask_Click);
            // 
            // btnDetailsTask
            // 
            this.btnDetailsTask.Location = new System.Drawing.Point(6, 195);
            this.btnDetailsTask.Name = "btnDetailsTask";
            this.btnDetailsTask.Size = new System.Drawing.Size(75, 23);
            this.btnDetailsTask.TabIndex = 8;
            this.btnDetailsTask.Text = "Details";
            this.btnDetailsTask.UseVisualStyleBackColor = true;
            // 
            // checkBoxAutosave
            // 
            this.checkBoxAutosave.AutoSize = true;
            this.checkBoxAutosave.Location = new System.Drawing.Point(5, 224);
            this.checkBoxAutosave.Name = "checkBoxAutosave";
            this.checkBoxAutosave.Size = new System.Drawing.Size(104, 17);
            this.checkBoxAutosave.TabIndex = 5;
            this.checkBoxAutosave.Text = "Autosave results";
            this.checkBoxAutosave.UseVisualStyleBackColor = true;
            this.checkBoxAutosave.CheckedChanged += new System.EventHandler(this.checkBoxAutosave_CheckedChanged);
            // 
            // btnImportCurr
            // 
            this.btnImportCurr.Location = new System.Drawing.Point(6, 166);
            this.btnImportCurr.Name = "btnImportCurr";
            this.btnImportCurr.Size = new System.Drawing.Size(75, 23);
            this.btnImportCurr.TabIndex = 4;
            this.btnImportCurr.Text = "Import";
            this.btnImportCurr.UseVisualStyleBackColor = true;
            this.btnImportCurr.Click += new System.EventHandler(this.btnImportCurr_Click);
            // 
            // btnExportCurr
            // 
            this.btnExportCurr.Location = new System.Drawing.Point(6, 136);
            this.btnExportCurr.Name = "btnExportCurr";
            this.btnExportCurr.Size = new System.Drawing.Size(75, 23);
            this.btnExportCurr.TabIndex = 3;
            this.btnExportCurr.Text = "Export";
            this.btnExportCurr.UseVisualStyleBackColor = true;
            this.btnExportCurr.Click += new System.EventHandler(this.btnExportCurr_Click);
            // 
            // btnDetailsCurr
            // 
            this.btnDetailsCurr.Location = new System.Drawing.Point(5, 106);
            this.btnDetailsCurr.Name = "btnDetailsCurr";
            this.btnDetailsCurr.Size = new System.Drawing.Size(75, 23);
            this.btnDetailsCurr.TabIndex = 2;
            this.btnDetailsCurr.Text = "Details";
            this.btnDetailsCurr.UseVisualStyleBackColor = true;
            // 
            // btnDelete
            // 
            this.btnDelete.Location = new System.Drawing.Point(5, 77);
            this.btnDelete.Name = "btnDelete";
            this.btnDelete.Size = new System.Drawing.Size(75, 23);
            this.btnDelete.TabIndex = 1;
            this.btnDelete.Text = "Delete";
            this.btnDelete.UseVisualStyleBackColor = true;
            this.btnDelete.Click += new System.EventHandler(this.btnDelete_Click);
            // 
            // btnNewCurr
            // 
            this.btnNewCurr.Location = new System.Drawing.Point(6, 19);
            this.btnNewCurr.Name = "btnNewCurr";
            this.btnNewCurr.Size = new System.Drawing.Size(75, 23);
            this.btnNewCurr.TabIndex = 0;
            this.btnNewCurr.Text = "New curr.";
            this.btnNewCurr.UseVisualStyleBackColor = true;
            this.btnNewCurr.Click += new System.EventHandler(this.btnNewCurr_Click);
            // 
            // btnRun
            // 
            this.btnRun.Location = new System.Drawing.Point(429, 267);
            this.btnRun.Name = "btnRun";
            this.btnRun.Size = new System.Drawing.Size(75, 23);
            this.btnRun.TabIndex = 2;
            this.btnRun.Text = "Run";
            this.btnRun.UseVisualStyleBackColor = true;
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // btnCurrFolder
            // 
            this.btnCurrFolder.Location = new System.Drawing.Point(429, 296);
            this.btnCurrFolder.Name = "btnCurrFolder";
            this.btnCurrFolder.Size = new System.Drawing.Size(75, 23);
            this.btnCurrFolder.TabIndex = 10;
            this.btnCurrFolder.Text = "Curr. folder";
            this.btnCurrFolder.UseVisualStyleBackColor = true;
            this.btnCurrFolder.Click += new System.EventHandler(this.btnCurrFolder_Click);
            // 
            // SchoolMainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(553, 425);
            this.Controls.Add(this.btnCurrFolder);
            this.Controls.Add(this.btnRun);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.tree);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Name = "SchoolMainForm";
            this.Text = "SchoolMainForm";
            this.Load += new System.EventHandler(this.SchoolMainForm_Load);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private Aga.Controls.Tree.TreeViewAdv tree;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Button btnDetailsTask;
        private System.Windows.Forms.CheckBox checkBoxAutosave;
        private System.Windows.Forms.Button btnImportCurr;
        private System.Windows.Forms.Button btnExportCurr;
        private System.Windows.Forms.Button btnDetailsCurr;
        private System.Windows.Forms.Button btnDelete;
        private System.Windows.Forms.Button btnNewCurr;
        private System.Windows.Forms.Button btnRun;
        private Aga.Controls.Tree.NodeControls.NodeTextBox nodeTextBox1;
        private System.Windows.Forms.Button btnNewTask;
        private System.Windows.Forms.SaveFileDialog saveFileDialog1;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.Button btnCurrFolder;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog1;
    }
}