namespace BrainSimulatorGUI.Forms
{
    partial class DebugForm
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
            this.debugTreeView = new Aga.Controls.Tree.TreeViewAdv();
            this.treeColumn2 = new Aga.Controls.Tree.TreeColumn();
            this.treeColumn3 = new Aga.Controls.Tree.TreeColumn();
            this.treeColumn1 = new Aga.Controls.Tree.TreeColumn();
            this.nodeCheckBox1 = new Aga.Controls.Tree.NodeControls.NodeCheckBox();
            this.nodeStateIcon1 = new Aga.Controls.Tree.NodeControls.NodeStateIcon();
            this.nodeTextBox1 = new Aga.Controls.Tree.NodeControls.NodeTextBox();
            this.nodeTextBox2 = new Aga.Controls.Tree.NodeControls.NodeTextBox();
            this.toolStrip = new System.Windows.Forms.ToolStrip();
            this.showSignalsButton = new System.Windows.Forms.ToolStripButton();
            this.showDisabledTasksButton = new System.Windows.Forms.ToolStripButton();
            this.runToolButton = new System.Windows.Forms.ToolStripButton();
            this.pauseToolButton = new System.Windows.Forms.ToolStripButton();
            this.stopToolButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
            this.stepInButton = new System.Windows.Forms.ToolStripButton();
            this.stepOverButton = new System.Windows.Forms.ToolStripButton();
            this.stepOutButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.noDebugLabel = new System.Windows.Forms.ToolStripLabel();
            this.toolStrip.SuspendLayout();
            this.SuspendLayout();
            // 
            // debugTreeView
            // 
            this.debugTreeView.AllowColumnReorder = true;
            this.debugTreeView.BackColor = System.Drawing.SystemColors.Window;
            this.debugTreeView.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.debugTreeView.Columns.Add(this.treeColumn2);
            this.debugTreeView.Columns.Add(this.treeColumn3);
            this.debugTreeView.Columns.Add(this.treeColumn1);
            this.debugTreeView.DefaultToolTipProvider = null;
            this.debugTreeView.Dock = System.Windows.Forms.DockStyle.Fill;
            this.debugTreeView.DragDropMarkColor = System.Drawing.Color.Black;
            this.debugTreeView.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.debugTreeView.FullRowSelect = true;
            this.debugTreeView.GridLineStyle = Aga.Controls.Tree.GridLineStyle.Vertical;
            this.debugTreeView.Indent = 10;
            this.debugTreeView.LineColor = System.Drawing.Color.DarkGray;
            this.debugTreeView.Location = new System.Drawing.Point(3, 28);
            this.debugTreeView.Model = null;
            this.debugTreeView.Name = "debugTreeView";
            this.debugTreeView.NodeControls.Add(this.nodeCheckBox1);
            this.debugTreeView.NodeControls.Add(this.nodeStateIcon1);
            this.debugTreeView.NodeControls.Add(this.nodeTextBox1);
            this.debugTreeView.NodeControls.Add(this.nodeTextBox2);
            this.debugTreeView.RowHeight = 19;
            this.debugTreeView.SelectedNode = null;
            this.debugTreeView.ShowNodeToolTips = true;
            this.debugTreeView.Size = new System.Drawing.Size(623, 820);
            this.debugTreeView.TabIndex = 2;
            this.debugTreeView.Text = "Scheduled Tasks";
            this.debugTreeView.UseColumns = true;
            // 
            // treeColumn2
            // 
            this.treeColumn2.Header = "Task Name";
            this.treeColumn2.SortOrder = System.Windows.Forms.SortOrder.None;
            this.treeColumn2.TooltipText = null;
            this.treeColumn2.Width = 250;
            // 
            // treeColumn3
            // 
            this.treeColumn3.Header = "Task Type";
            this.treeColumn3.SortOrder = System.Windows.Forms.SortOrder.None;
            this.treeColumn3.TooltipText = null;
            this.treeColumn3.Width = 200;
            // 
            // treeColumn1
            // 
            this.treeColumn1.Header = "Enabled";
            this.treeColumn1.SortOrder = System.Windows.Forms.SortOrder.None;
            this.treeColumn1.TooltipText = null;
            this.treeColumn1.Width = 60;
            // 
            // nodeCheckBox1
            // 
            this.nodeCheckBox1.DataPropertyName = "Checked";
            this.nodeCheckBox1.LeftMargin = 20;
            this.nodeCheckBox1.ParentColumn = this.treeColumn1;
            this.nodeCheckBox1.IsVisibleValueNeeded += new System.EventHandler<Aga.Controls.Tree.NodeControls.NodeControlValueEventArgs>(this.nodeCheckBox1_IsVisibleValueNeeded);
            // 
            // nodeStateIcon1
            // 
            this.nodeStateIcon1.DataPropertyName = "Icon";
            this.nodeStateIcon1.LeftMargin = 1;
            this.nodeStateIcon1.ParentColumn = this.treeColumn2;
            this.nodeStateIcon1.ScaleMode = Aga.Controls.Tree.ImageScaleMode.Clip;
            // 
            // nodeTextBox1
            // 
            this.nodeTextBox1.DataPropertyName = "Text";
            this.nodeTextBox1.IncrementalSearchEnabled = true;
            this.nodeTextBox1.LeftMargin = 3;
            this.nodeTextBox1.ParentColumn = this.treeColumn2;
            this.nodeTextBox1.DrawText += new System.EventHandler<Aga.Controls.Tree.NodeControls.DrawEventArgs>(this.nodeTextBox1_DrawText);
            // 
            // nodeTextBox2
            // 
            this.nodeTextBox2.DataPropertyName = "OwnerName";
            this.nodeTextBox2.IncrementalSearchEnabled = true;
            this.nodeTextBox2.LeftMargin = 3;
            this.nodeTextBox2.ParentColumn = this.treeColumn3;
            this.nodeTextBox2.DrawText += new System.EventHandler<Aga.Controls.Tree.NodeControls.DrawEventArgs>(this.nodeTextBox1_DrawText);
            // 
            // toolStrip
            // 
            this.toolStrip.Enabled = false;
            this.toolStrip.GripStyle = System.Windows.Forms.ToolStripGripStyle.Hidden;
            this.toolStrip.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.showSignalsButton,
            this.showDisabledTasksButton,
            this.runToolButton,
            this.pauseToolButton,
            this.stopToolButton,
            this.toolStripSeparator2,
            this.stepInButton,
            this.stepOverButton,
            this.stepOutButton,
            this.toolStripSeparator1,
            this.noDebugLabel});
            this.toolStrip.Location = new System.Drawing.Point(3, 3);
            this.toolStrip.Name = "toolStrip";
            this.toolStrip.Size = new System.Drawing.Size(623, 25);
            this.toolStrip.TabIndex = 3;
            this.toolStrip.Text = "toolStrip";
            // 
            // showSignalsButton
            // 
            this.showSignalsButton.Alignment = System.Windows.Forms.ToolStripItemAlignment.Right;
            this.showSignalsButton.Checked = true;
            this.showSignalsButton.CheckOnClick = true;
            this.showSignalsButton.CheckState = System.Windows.Forms.CheckState.Checked;
            this.showSignalsButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.showSignalsButton.Image = global::BrainSimulatorGUI.Properties.Resources.signal_in;
            this.showSignalsButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.showSignalsButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.showSignalsButton.Name = "showSignalsButton";
            this.showSignalsButton.Size = new System.Drawing.Size(23, 22);
            this.showSignalsButton.Text = "Show Signal Tasks";
            this.showSignalsButton.CheckedChanged += new System.EventHandler(this.showSignalsButton_CheckedChanged);
            // 
            // showDisabledTasksButton
            // 
            this.showDisabledTasksButton.Alignment = System.Windows.Forms.ToolStripItemAlignment.Right;
            this.showDisabledTasksButton.Checked = true;
            this.showDisabledTasksButton.CheckOnClick = true;
            this.showDisabledTasksButton.CheckState = System.Windows.Forms.CheckState.Checked;
            this.showDisabledTasksButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.showDisabledTasksButton.Image = global::BrainSimulatorGUI.Properties.Resources.tasks_disabled;
            this.showDisabledTasksButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.showDisabledTasksButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.showDisabledTasksButton.Margin = new System.Windows.Forms.Padding(0, 1, 3, 2);
            this.showDisabledTasksButton.Name = "showDisabledTasksButton";
            this.showDisabledTasksButton.Size = new System.Drawing.Size(23, 22);
            this.showDisabledTasksButton.Text = "Show Disabled Tasks";
            this.showDisabledTasksButton.CheckedChanged += new System.EventHandler(this.showDisabledTasksButton_CheckedChanged);
            // 
            // runToolButton
            // 
            this.runToolButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.runToolButton.Image = global::BrainSimulatorGUI.Properties.Resources.StatusAnnotations_Play_16xLG_color;
            this.runToolButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.runToolButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.runToolButton.Name = "runToolButton";
            this.runToolButton.Size = new System.Drawing.Size(23, 22);
            this.runToolButton.Text = "Run Simulation";
            this.runToolButton.Click += new System.EventHandler(this.runToolButton_Click);
            // 
            // pauseToolButton
            // 
            this.pauseToolButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.pauseToolButton.Enabled = false;
            this.pauseToolButton.Image = global::BrainSimulatorGUI.Properties.Resources.StatusAnnotations_Pause_16xLG_color;
            this.pauseToolButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.pauseToolButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.pauseToolButton.Name = "pauseToolButton";
            this.pauseToolButton.Size = new System.Drawing.Size(23, 22);
            this.pauseToolButton.Text = "Pause Simulation";
            this.pauseToolButton.Click += new System.EventHandler(this.pauseToolButton_Click);
            // 
            // stopToolButton
            // 
            this.stopToolButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.stopToolButton.Image = global::BrainSimulatorGUI.Properties.Resources.StatusAnnotations_Stop_16xLG_color;
            this.stopToolButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.stopToolButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.stopToolButton.Name = "stopToolButton";
            this.stopToolButton.Size = new System.Drawing.Size(23, 22);
            this.stopToolButton.Text = "Stop Simulation";
            this.stopToolButton.Click += new System.EventHandler(this.stopToolButton_Click);
            // 
            // toolStripSeparator2
            // 
            this.toolStripSeparator2.Name = "toolStripSeparator2";
            this.toolStripSeparator2.Size = new System.Drawing.Size(6, 25);
            // 
            // stepInButton
            // 
            this.stepInButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.stepInButton.Image = global::BrainSimulatorGUI.Properties.Resources.StepIn;
            this.stepInButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.stepInButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.stepInButton.Name = "stepInButton";
            this.stepInButton.Size = new System.Drawing.Size(23, 22);
            this.stepInButton.Text = "Step Into";
            this.stepInButton.Click += new System.EventHandler(this.stepInButton_Click);
            // 
            // stepOverButton
            // 
            this.stepOverButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.stepOverButton.Image = global::BrainSimulatorGUI.Properties.Resources.StepOver;
            this.stepOverButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.stepOverButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.stepOverButton.Name = "stepOverButton";
            this.stepOverButton.Size = new System.Drawing.Size(23, 22);
            this.stepOverButton.Text = "Step Over";
            this.stepOverButton.Click += new System.EventHandler(this.stepOverButton_Click);
            // 
            // stepOutButton
            // 
            this.stepOutButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.stepOutButton.Image = global::BrainSimulatorGUI.Properties.Resources.Stepout;
            this.stepOutButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.stepOutButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.stepOutButton.Name = "stepOutButton";
            this.stepOutButton.Size = new System.Drawing.Size(23, 22);
            this.stepOutButton.Text = "Step Out";
            this.stepOutButton.Click += new System.EventHandler(this.stepOutButton_Click);
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            this.toolStripSeparator1.Size = new System.Drawing.Size(6, 25);
            // 
            // noDebugLabel
            // 
            this.noDebugLabel.Name = "noDebugLabel";
            this.noDebugLabel.Size = new System.Drawing.Size(110, 22);
            this.noDebugLabel.Text = "Debugging Inactive";
            // 
            // DebugForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(629, 851);
            this.Controls.Add(this.debugTreeView);
            this.Controls.Add(this.toolStrip);
            this.DockAreas = ((WeifenLuo.WinFormsUI.Docking.DockAreas)(((((WeifenLuo.WinFormsUI.Docking.DockAreas.Float | WeifenLuo.WinFormsUI.Docking.DockAreas.DockLeft) 
            | WeifenLuo.WinFormsUI.Docking.DockAreas.DockRight) 
            | WeifenLuo.WinFormsUI.Docking.DockAreas.DockTop) 
            | WeifenLuo.WinFormsUI.Docking.DockAreas.DockBottom)));
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(238)));
            this.HideOnClose = true;
            this.Name = "DebugForm";
            this.Padding = new System.Windows.Forms.Padding(3);
            this.Text = "Debugging";
            this.toolStrip.ResumeLayout(false);
            this.toolStrip.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private Aga.Controls.Tree.TreeViewAdv debugTreeView;
        private Aga.Controls.Tree.NodeControls.NodeStateIcon nodeStateIcon1;
        private Aga.Controls.Tree.NodeControls.NodeTextBox nodeTextBox1;
        public Aga.Controls.Tree.NodeControls.NodeCheckBox nodeCheckBox1;
        private Aga.Controls.Tree.TreeColumn treeColumn2;
        private Aga.Controls.Tree.TreeColumn treeColumn3;
        private Aga.Controls.Tree.TreeColumn treeColumn1;
        private Aga.Controls.Tree.NodeControls.NodeTextBox nodeTextBox2;
        private System.Windows.Forms.ToolStrip toolStrip;
        private System.Windows.Forms.ToolStripButton showDisabledTasksButton;
        private System.Windows.Forms.ToolStripButton showSignalsButton;
        private System.Windows.Forms.ToolStripButton stepInButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripLabel noDebugLabel;
        private System.Windows.Forms.ToolStripButton stepOverButton;
        private System.Windows.Forms.ToolStripButton stepOutButton;
        public System.Windows.Forms.ToolStripButton runToolButton;
        public System.Windows.Forms.ToolStripButton pauseToolButton;
        public System.Windows.Forms.ToolStripButton stopToolButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator2;

    }
}