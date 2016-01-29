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
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.btnNewTask = new System.Windows.Forms.Button();
            this.checkBoxAutosave = new System.Windows.Forms.CheckBox();
            this.btnImport = new System.Windows.Forms.Button();
            this.btnSaveAs = new System.Windows.Forms.Button();
            this.btnDetails = new System.Windows.Forms.Button();
            this.btnDelete = new System.Windows.Forms.Button();
            this.btnNewCurr = new System.Windows.Forms.Button();
            this.btnRun = new System.Windows.Forms.Button();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.btnCurrFolder = new System.Windows.Forms.Button();
            this.folderBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.btnSave = new System.Windows.Forms.Button();
            this.btnLoad = new System.Windows.Forms.Button();
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
            this.tree.NodeControls.Add(this.nodeCheckBox1);
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
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.btnLoad);
            this.groupBox1.Controls.Add(this.btnSave);
            this.groupBox1.Controls.Add(this.btnNewTask);
            this.groupBox1.Controls.Add(this.checkBoxAutosave);
            this.groupBox1.Controls.Add(this.btnImport);
            this.groupBox1.Controls.Add(this.btnSaveAs);
            this.groupBox1.Controls.Add(this.btnDetails);
            this.groupBox1.Controls.Add(this.btnDelete);
            this.groupBox1.Controls.Add(this.btnNewCurr);
            this.groupBox1.Location = new System.Drawing.Point(423, 12);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(118, 279);
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
            // checkBoxAutosave
            // 
            this.checkBoxAutosave.AutoSize = true;
            this.checkBoxAutosave.Location = new System.Drawing.Point(6, 256);
            this.checkBoxAutosave.Name = "checkBoxAutosave";
            this.checkBoxAutosave.Size = new System.Drawing.Size(104, 17);
            this.checkBoxAutosave.TabIndex = 5;
            this.checkBoxAutosave.Text = "Autosave results";
            this.checkBoxAutosave.UseVisualStyleBackColor = true;
            this.checkBoxAutosave.CheckedChanged += new System.EventHandler(this.checkBoxAutosave_CheckedChanged);
            // 
            // btnImport
            // 
            this.btnImport.Location = new System.Drawing.Point(5, 227);
            this.btnImport.Name = "btnImport";
            this.btnImport.Size = new System.Drawing.Size(75, 23);
            this.btnImport.TabIndex = 4;
            this.btnImport.Text = "Import";
            this.btnImport.UseVisualStyleBackColor = true;
            this.btnImport.Click += new System.EventHandler(this.btnImportCurr_Click);
            // 
            // btnSaveAs
            // 
            this.btnSaveAs.Location = new System.Drawing.Point(5, 169);
            this.btnSaveAs.Name = "btnSaveAs";
            this.btnSaveAs.Size = new System.Drawing.Size(75, 23);
            this.btnSaveAs.TabIndex = 3;
            this.btnSaveAs.Text = "Save As...";
            this.btnSaveAs.UseVisualStyleBackColor = true;
            this.btnSaveAs.Click += new System.EventHandler(this.btnExportCurr_Click);
            // 
            // btnDetails
            // 
            this.btnDetails.Location = new System.Drawing.Point(5, 106);
            this.btnDetails.Name = "btnDetails";
            this.btnDetails.Size = new System.Drawing.Size(75, 23);
            this.btnDetails.TabIndex = 2;
            this.btnDetails.Text = "Details";
            this.btnDetails.UseVisualStyleBackColor = true;
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
            this.btnRun.Location = new System.Drawing.Point(428, 297);
            this.btnRun.Name = "btnRun";
            this.btnRun.Size = new System.Drawing.Size(75, 23);
            this.btnRun.TabIndex = 2;
            this.btnRun.Text = "Run";
            this.btnRun.UseVisualStyleBackColor = true;
            this.btnRun.Click += new System.EventHandler(this.btnRun_Click);
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // btnCurrFolder
            // 
            this.btnCurrFolder.Location = new System.Drawing.Point(428, 326);
            this.btnCurrFolder.Name = "btnCurrFolder";
            this.btnCurrFolder.Size = new System.Drawing.Size(75, 23);
            this.btnCurrFolder.TabIndex = 10;
            this.btnCurrFolder.Text = "Curr. folder";
            this.btnCurrFolder.UseVisualStyleBackColor = true;
            this.btnCurrFolder.Click += new System.EventHandler(this.btnCurrFolder_Click);
            // 
            // btnSave
            // 
            this.btnSave.Location = new System.Drawing.Point(5, 135);
            this.btnSave.Name = "btnSave";
            this.btnSave.Size = new System.Drawing.Size(75, 23);
            this.btnSave.TabIndex = 10;
            this.btnSave.Text = "Save";
            this.btnSave.UseVisualStyleBackColor = true;
            // 
            // btnLoad
            // 
            this.btnLoad.Location = new System.Drawing.Point(5, 198);
            this.btnLoad.Name = "btnLoad";
            this.btnLoad.Size = new System.Drawing.Size(75, 23);
            this.btnLoad.TabIndex = 11;
            this.btnLoad.Text = "Load";
            this.btnLoad.UseVisualStyleBackColor = true;
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
        private System.Windows.Forms.CheckBox checkBoxAutosave;
        private System.Windows.Forms.Button btnImport;
        private System.Windows.Forms.Button btnSaveAs;
        private System.Windows.Forms.Button btnDetails;
        private System.Windows.Forms.Button btnDelete;
        private System.Windows.Forms.Button btnNewCurr;
        private System.Windows.Forms.Button btnRun;
        private Aga.Controls.Tree.NodeControls.NodeTextBox nodeTextBox1;
        private System.Windows.Forms.Button btnNewTask;
        private System.Windows.Forms.SaveFileDialog saveFileDialog1;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.Button btnCurrFolder;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog1;
        private Aga.Controls.Tree.NodeControls.NodeCheckBox nodeCheckBox1;
        private System.Windows.Forms.Button btnLoad;
        private System.Windows.Forms.Button btnSave;
    }
}