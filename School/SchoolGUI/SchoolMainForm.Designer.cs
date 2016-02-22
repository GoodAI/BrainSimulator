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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(SchoolMainForm));
            this.nodeCheckBox1 = new Aga.Controls.Tree.NodeControls.NodeCheckBox();
            this.nodeTextBox1 = new Aga.Controls.Tree.NodeControls.NodeTextBox();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.folderBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.tree = new Aga.Controls.Tree.TreeViewAdv();
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.btnNew = new System.Windows.Forms.ToolStripButton();
            this.btnSave = new System.Windows.Forms.ToolStripButton();
            this.btnSaveAs = new System.Windows.Forms.ToolStripButton();
            this.btnOpen = new System.Windows.Forms.ToolStripButton();
            this.btnImport = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.toolStripLabel1 = new System.Windows.Forms.ToolStripLabel();
            this.btnNewCurr = new System.Windows.Forms.ToolStripButton();
            this.btnDetailsCurr = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
            this.toolStripLabel2 = new System.Windows.Forms.ToolStripLabel();
            this.btnNewTask = new System.Windows.Forms.ToolStripButton();
            this.btnDetailsTask = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator3 = new System.Windows.Forms.ToolStripSeparator();
            this.btnDelete = new System.Windows.Forms.ToolStripButton();
            this.btnAutorun = new System.Windows.Forms.ToolStripButton();
            this.btnAutosave = new System.Windows.Forms.ToolStripButton();
            this.btnUpload = new System.Windows.Forms.ToolStripButton();
            this.btnRun = new System.Windows.Forms.ToolStripButton();
            this.toolStrip1.SuspendLayout();
            this.SuspendLayout();
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
            // saveFileDialog1
            // 
            this.saveFileDialog1.DefaultExt = "xml";
            this.saveFileDialog1.Filter = "Curriculum files|*.xml|All files|*.*";
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // tree
            // 
            this.tree.AllowDrop = true;
            this.tree.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tree.BackColor = System.Drawing.SystemColors.Window;
            this.tree.ColumnHeaderHeight = 0;
            this.tree.DefaultToolTipProvider = null;
            this.tree.DisplayDraggingNodes = true;
            this.tree.DragDropMarkColor = System.Drawing.Color.Black;
            this.tree.FullRowSelectActiveColor = System.Drawing.Color.Empty;
            this.tree.FullRowSelectInactiveColor = System.Drawing.Color.Empty;
            this.tree.LineColor = System.Drawing.SystemColors.ControlDark;
            this.tree.Location = new System.Drawing.Point(3, 32);
            this.tree.Model = null;
            this.tree.Name = "tree";
            this.tree.NodeControls.Add(this.nodeCheckBox1);
            this.tree.NodeControls.Add(this.nodeTextBox1);
            this.tree.NodeFilter = null;
            this.tree.SelectedNode = null;
            this.tree.SelectionMode = Aga.Controls.Tree.TreeSelectionMode.MultiSameParent;
            this.tree.Size = new System.Drawing.Size(630, 387);
            this.tree.TabIndex = 0;
            this.tree.ItemDrag += new System.Windows.Forms.ItemDragEventHandler(this.tree_ItemDrag);
            this.tree.SelectionChanged += new System.EventHandler(this.tree_SelectionChanged);
            this.tree.DragDrop += new System.Windows.Forms.DragEventHandler(this.tree_DragDrop);
            this.tree.DragOver += new System.Windows.Forms.DragEventHandler(this.tree_DragOver);
            this.tree.KeyDown += new System.Windows.Forms.KeyEventHandler(this.SchoolMainForm_KeyDown);
            // 
            // toolStrip1
            // 
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.btnNew,
            this.btnSave,
            this.btnSaveAs,
            this.btnOpen,
            this.btnImport,
            this.toolStripSeparator1,
            this.toolStripLabel1,
            this.btnNewCurr,
            this.btnDetailsCurr,
            this.toolStripSeparator2,
            this.toolStripLabel2,
            this.btnNewTask,
            this.btnDetailsTask,
            this.toolStripSeparator3,
            this.btnDelete,
            this.btnAutorun,
            this.btnAutosave,
            this.btnUpload,
            this.btnRun});
            this.toolStrip1.Location = new System.Drawing.Point(0, 0);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(645, 25);
            this.toolStrip1.TabIndex = 14;
            this.toolStrip1.Text = "toolStrip1";
            // 
            // btnNew
            // 
            this.btnNew.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnNew.Image = global::GoodAI.School.GUI.Properties.Resources.AddNewItem_6273;
            this.btnNew.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnNew.Name = "btnNew";
            this.btnNew.Size = new System.Drawing.Size(23, 22);
            this.btnNew.Text = "New Project";
            this.btnNew.Click += new System.EventHandler(this.btnNew_Click);
            // 
            // btnSave
            // 
            this.btnSave.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnSave.Image = global::GoodAI.School.GUI.Properties.Resources.save_16xLG;
            this.btnSave.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnSave.Name = "btnSave";
            this.btnSave.Size = new System.Drawing.Size(23, 22);
            this.btnSave.Text = "Save Project";
            this.btnSave.Click += new System.EventHandler(this.btnSave_Click);
            // 
            // btnSaveAs
            // 
            this.btnSaveAs.Image = global::GoodAI.School.GUI.Properties.Resources.saveas_16xLG;
            this.btnSaveAs.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnSaveAs.Name = "btnSaveAs";
            this.btnSaveAs.Size = new System.Drawing.Size(23, 22);
            this.btnSaveAs.ToolTipText = "Save Project As";
            this.btnSaveAs.Click += new System.EventHandler(this.SaveProjectAs);
            // 
            // btnOpen
            // 
            this.btnOpen.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnOpen.Image = global::GoodAI.School.GUI.Properties.Resources.Open_6529;
            this.btnOpen.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnOpen.Name = "btnOpen";
            this.btnOpen.Size = new System.Drawing.Size(23, 22);
            this.btnOpen.Text = "Open Project";
            this.btnOpen.Click += new System.EventHandler(this.btnOpen_Click);
            // 
            // btnImport
            // 
            this.btnImport.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnImport.Image = global::GoodAI.School.GUI.Properties.Resources.AddExistingItem_6269;
            this.btnImport.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnImport.Name = "btnImport";
            this.btnImport.Size = new System.Drawing.Size(23, 22);
            this.btnImport.Text = "Import Project";
            this.btnImport.Click += new System.EventHandler(this.btnImport_Click);
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            this.toolStripSeparator1.Size = new System.Drawing.Size(6, 25);
            // 
            // toolStripLabel1
            // 
            this.toolStripLabel1.Name = "toolStripLabel1";
            this.toolStripLabel1.Size = new System.Drawing.Size(67, 22);
            this.toolStripLabel1.Text = "Curriculum";
            // 
            // btnNewCurr
            // 
            this.btnNewCurr.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnNewCurr.Image = global::GoodAI.School.GUI.Properties.Resources.action_add_16xMD;
            this.btnNewCurr.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnNewCurr.Name = "btnNewCurr";
            this.btnNewCurr.Size = new System.Drawing.Size(23, 22);
            this.btnNewCurr.Text = "New Curriculum";
            this.btnNewCurr.Click += new System.EventHandler(this.btnNewCurr_Click);
            // 
            // btnDetailsCurr
            // 
            this.btnDetailsCurr.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnDetailsCurr.Image = global::GoodAI.School.GUI.Properties.Resources.Symbols_Information_16xLG;
            this.btnDetailsCurr.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnDetailsCurr.Name = "btnDetailsCurr";
            this.btnDetailsCurr.Size = new System.Drawing.Size(23, 22);
            this.btnDetailsCurr.Text = "Curriculum Details";
            this.btnDetailsCurr.Click += new System.EventHandler(this.btnDetailsCurr_Click);
            // 
            // toolStripSeparator2
            // 
            this.toolStripSeparator2.Name = "toolStripSeparator2";
            this.toolStripSeparator2.Size = new System.Drawing.Size(6, 25);
            // 
            // toolStripLabel2
            // 
            this.toolStripLabel2.Name = "toolStripLabel2";
            this.toolStripLabel2.Size = new System.Drawing.Size(31, 22);
            this.toolStripLabel2.Text = "Task";
            // 
            // btnNewTask
            // 
            this.btnNewTask.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnNewTask.Image = global::GoodAI.School.GUI.Properties.Resources.action_add_16xMD;
            this.btnNewTask.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnNewTask.Name = "btnNewTask";
            this.btnNewTask.Size = new System.Drawing.Size(23, 22);
            this.btnNewTask.Text = "New Learning Task";
            this.btnNewTask.Click += new System.EventHandler(this.btnNewTask_Click);
            // 
            // btnDetailsTask
            // 
            this.btnDetailsTask.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnDetailsTask.Image = global::GoodAI.School.GUI.Properties.Resources.Symbols_Information_16xLG;
            this.btnDetailsTask.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnDetailsTask.Name = "btnDetailsTask";
            this.btnDetailsTask.Size = new System.Drawing.Size(23, 22);
            this.btnDetailsTask.Text = "Learning Task Details";
            this.btnDetailsTask.Click += new System.EventHandler(this.btnDetailsTask_Click);
            // 
            // toolStripSeparator3
            // 
            this.toolStripSeparator3.Name = "toolStripSeparator3";
            this.toolStripSeparator3.Size = new System.Drawing.Size(6, 25);
            // 
            // btnDelete
            // 
            this.btnDelete.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnDelete.Image = global::GoodAI.School.GUI.Properties.Resources.action_Cancel_16xMD;
            this.btnDelete.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnDelete.Name = "btnDelete";
            this.btnDelete.Size = new System.Drawing.Size(23, 22);
            this.btnDelete.Text = "Delete";
            this.btnDelete.Click += new System.EventHandler(this.DeleteNodes);
            // 
            // btnAutorun
            // 
            this.btnAutorun.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.btnAutorun.Image = ((System.Drawing.Image)(resources.GetObject("btnAutorun.Image")));
            this.btnAutorun.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnAutorun.Name = "btnAutorun";
            this.btnAutorun.Size = new System.Drawing.Size(55, 22);
            this.btnAutorun.Text = "Autorun";
            this.btnAutorun.CheckedChanged += new System.EventHandler(this.btnAutorun_CheckedChanged);
            this.btnAutorun.Click += new System.EventHandler(this.btnToggleCheck);
            // 
            // btnAutosave
            // 
            this.btnAutosave.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnAutosave.Image = global::GoodAI.School.GUI.Properties.Resources.autosave;
            this.btnAutosave.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnAutosave.Name = "btnAutosave";
            this.btnAutosave.Size = new System.Drawing.Size(23, 22);
            this.btnAutosave.Text = "Autosave Results";
            this.btnAutosave.CheckedChanged += new System.EventHandler(this.btnAutosave_CheckedChanged);
            this.btnAutosave.Click += new System.EventHandler(this.btnToggleCheck);
            // 
            // btnUpload
            // 
            this.btnUpload.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.btnUpload.Enabled = false;
            this.btnUpload.Image = ((System.Drawing.Image)(resources.GetObject("btnUpload.Image")));
            this.btnUpload.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnUpload.Name = "btnUpload";
            this.btnUpload.Size = new System.Drawing.Size(103, 22);
            this.btnUpload.Text = "Upload to Project";
            this.btnUpload.Click += new System.EventHandler(this.btnUpload_Click);
            // 
            // btnRun
            // 
            this.btnRun.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnRun.Image = global::GoodAI.School.GUI.Properties.Resources.StatusAnnotations_Play_16xLG_color;
            this.btnRun.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnRun.Name = "btnRun";
            this.btnRun.Size = new System.Drawing.Size(23, 22);
            this.btnRun.Text = "Run Simulation";
            this.btnRun.Click += new System.EventHandler(this.btnRun_Click);
            // 
            // SchoolMainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(645, 419);
            this.Controls.Add(this.toolStrip1);
            this.Controls.Add(this.tree);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.HideOnClose = true;
            this.Name = "SchoolMainForm";
            this.Text = "School for AI";
            this.Load += new System.EventHandler(this.SchoolMainForm_Load);
            this.VisibleChanged += new System.EventHandler(this.UpdateWindowName);
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.SchoolMainForm_KeyDown);
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private Aga.Controls.Tree.NodeControls.NodeTextBox nodeTextBox1;
        private System.Windows.Forms.SaveFileDialog saveFileDialog1;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog1;
        private Aga.Controls.Tree.NodeControls.NodeCheckBox nodeCheckBox1;
        private Aga.Controls.Tree.TreeViewAdv tree;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.ToolStripButton btnNew;
        private System.Windows.Forms.ToolStripButton btnSave;
        private System.Windows.Forms.ToolStripButton btnSaveAs;
        private System.Windows.Forms.ToolStripButton btnOpen;
        private System.Windows.Forms.ToolStripButton btnImport;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripLabel toolStripLabel1;
        private System.Windows.Forms.ToolStripButton btnNewCurr;
        private System.Windows.Forms.ToolStripButton btnDetailsCurr;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator2;
        private System.Windows.Forms.ToolStripLabel toolStripLabel2;
        private System.Windows.Forms.ToolStripButton btnNewTask;
        private System.Windows.Forms.ToolStripButton btnDetailsTask;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator3;
        private System.Windows.Forms.ToolStripButton btnDelete;
        private System.Windows.Forms.ToolStripButton btnAutorun;
        private System.Windows.Forms.ToolStripButton btnAutosave;
        private System.Windows.Forms.ToolStripButton btnUpload;
        private System.Windows.Forms.ToolStripButton btnRun;
    }
}