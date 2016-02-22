namespace GoodAI.BrainSimulator.Forms
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
            this.taskName = new Aga.Controls.Tree.TreeColumn();
            this.taskType = new Aga.Controls.Tree.TreeColumn();
            this.enabled = new Aga.Controls.Tree.TreeColumn();
            this.breakpoint = new Aga.Controls.Tree.TreeColumn();
            this.profilerTime = new Aga.Controls.Tree.TreeColumn();
            this.checkEnabled = new Aga.Controls.Tree.NodeControls.NodeCheckBox();
            this.stateIcon = new Aga.Controls.Tree.NodeControls.NodeStateIcon();
            this.nodeTextBox1 = new Aga.Controls.Tree.NodeControls.NodeTextBox();
            this.nodeTextBox2 = new Aga.Controls.Tree.NodeControls.NodeTextBox();
            this.breakpointCheckBox = new Aga.Controls.Tree.NodeControls.NodeCheckBox();
            this.profilerTimeValue = new Aga.Controls.Tree.NodeControls.NodeTextBox();
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
            this.collapseAllButton = new System.Windows.Forms.ToolStripButton();
            this.expandAllButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator3 = new System.Windows.Forms.ToolStripSeparator();
            this.noDebugLabel = new System.Windows.Forms.ToolStripLabel();
            this.toolStrip.SuspendLayout();
            this.SuspendLayout();
            // 
            // debugTreeView
            // 
            this.debugTreeView.AllowColumnReorder = true;
            this.debugTreeView.BackColor = System.Drawing.SystemColors.Window;
            this.debugTreeView.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.debugTreeView.Columns.Add(this.taskName);
            this.debugTreeView.Columns.Add(this.taskType);
            this.debugTreeView.Columns.Add(this.enabled);
            this.debugTreeView.Columns.Add(this.breakpoint);
            this.debugTreeView.Columns.Add(this.profilerTime);
            this.debugTreeView.DefaultToolTipProvider = null;
            this.debugTreeView.Dock = System.Windows.Forms.DockStyle.Fill;
            this.debugTreeView.DragDropMarkColor = System.Drawing.Color.Black;
            this.debugTreeView.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.debugTreeView.FullRowSelect = true;
            this.debugTreeView.FullRowSelectActiveColor = System.Drawing.Color.Empty;
            this.debugTreeView.FullRowSelectInactiveColor = System.Drawing.Color.Empty;
            this.debugTreeView.GridLineStyle = Aga.Controls.Tree.GridLineStyle.Vertical;
            this.debugTreeView.Indent = 10;
            this.debugTreeView.LineColor = System.Drawing.Color.DarkGray;
            this.debugTreeView.Location = new System.Drawing.Point(3, 28);
            this.debugTreeView.Model = null;
            this.debugTreeView.Name = "debugTreeView";
            this.debugTreeView.NodeControls.Add(this.checkEnabled);
            this.debugTreeView.NodeControls.Add(this.stateIcon);
            this.debugTreeView.NodeControls.Add(this.nodeTextBox1);
            this.debugTreeView.NodeControls.Add(this.nodeTextBox2);
            this.debugTreeView.NodeControls.Add(this.breakpointCheckBox);
            this.debugTreeView.NodeControls.Add(this.profilerTimeValue);
            this.debugTreeView.NodeFilter = null;
            this.debugTreeView.RowHeight = 19;
            this.debugTreeView.SelectedNode = null;
            this.debugTreeView.ShowNodeToolTips = true;
            this.debugTreeView.Size = new System.Drawing.Size(648, 820);
            this.debugTreeView.TabIndex = 2;
            this.debugTreeView.Text = "Scheduled Tasks";
            this.debugTreeView.UseColumns = true;
            this.debugTreeView.SelectionChanged += new System.EventHandler(this.debugTreeView_SelectionChanged);
            // 
            // taskName
            // 
            this.taskName.Header = "Task Name";
            this.taskName.SortOrder = System.Windows.Forms.SortOrder.None;
            this.taskName.TooltipText = null;
            this.taskName.Width = 250;
            // 
            // taskType
            // 
            this.taskType.Header = "Task Type";
            this.taskType.SortOrder = System.Windows.Forms.SortOrder.None;
            this.taskType.TooltipText = null;
            this.taskType.Width = 200;
            // 
            // enabled
            // 
            this.enabled.Header = "Enabled";
            this.enabled.SortOrder = System.Windows.Forms.SortOrder.None;
            this.enabled.TooltipText = null;
            this.enabled.Width = 60;
            // 
            // breakpoint
            // 
            this.breakpoint.Header = "Breakpoint";
            this.breakpoint.SortOrder = System.Windows.Forms.SortOrder.None;
            this.breakpoint.TooltipText = null;
            this.breakpoint.Width = 25;
            // 
            // profilerTime
            // 
            this.profilerTime.Header = "Profiling Time";
            this.profilerTime.SortOrder = System.Windows.Forms.SortOrder.None;
            this.profilerTime.TooltipText = "";
            this.profilerTime.Width = 70;
            // 
            // checkEnabled
            // 
            this.checkEnabled.DataPropertyName = "Checked";
            this.checkEnabled.EditEnabled = true;
            this.checkEnabled.LeftMargin = 20;
            this.checkEnabled.ParentColumn = this.enabled;
            this.checkEnabled.IsVisibleValueNeeded += new System.EventHandler<Aga.Controls.Tree.NodeControls.NodeControlValueEventArgs>(this.nodeCheckBox1_IsVisibleValueNeeded);
            // 
            // stateIcon
            // 
            this.stateIcon.DataPropertyName = "Icon";
            this.stateIcon.LeftMargin = 1;
            this.stateIcon.ParentColumn = this.taskName;
            this.stateIcon.ScaleMode = Aga.Controls.Tree.ImageScaleMode.Clip;
            // 
            // nodeTextBox1
            // 
            this.nodeTextBox1.DataPropertyName = "Text";
            this.nodeTextBox1.IncrementalSearchEnabled = true;
            this.nodeTextBox1.LeftMargin = 3;
            this.nodeTextBox1.ParentColumn = this.taskName;
            this.nodeTextBox1.DrawText += new System.EventHandler<Aga.Controls.Tree.NodeControls.DrawTextEventArgs>(this.nodeTextBox1_DrawText);
            // 
            // nodeTextBox2
            // 
            this.nodeTextBox2.DataPropertyName = "OwnerName";
            this.nodeTextBox2.IncrementalSearchEnabled = true;
            this.nodeTextBox2.LeftMargin = 3;
            this.nodeTextBox2.ParentColumn = this.taskType;
            this.nodeTextBox2.DrawText += new System.EventHandler<Aga.Controls.Tree.NodeControls.DrawTextEventArgs>(this.nodeTextBox1_DrawText);
            // 
            // breakpointCheckBox
            // 
            this.breakpointCheckBox.DataPropertyName = "Breakpoint";
            this.breakpointCheckBox.EditEnabled = true;
            this.breakpointCheckBox.LeftMargin = 0;
            this.breakpointCheckBox.ParentColumn = this.breakpoint;
            // 
            // profilerTimeValue
            // 
            this.profilerTimeValue.DataPropertyName = "ProfilerTimeFormatted";
            this.profilerTimeValue.IncrementalSearchEnabled = true;
            this.profilerTimeValue.LeftMargin = 3;
            this.profilerTimeValue.ParentColumn = this.profilerTime;
            this.profilerTimeValue.DrawText += new System.EventHandler<Aga.Controls.Tree.NodeControls.DrawTextEventArgs>(this.profilerTimeValue_DrawText);
            // 
            // toolStrip
            // 
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
            this.collapseAllButton,
            this.expandAllButton,
            this.toolStripSeparator3,
            this.noDebugLabel});
            this.toolStrip.Location = new System.Drawing.Point(3, 3);
            this.toolStrip.Name = "toolStrip";
            this.toolStrip.Size = new System.Drawing.Size(648, 25);
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
            this.showSignalsButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.signal_in;
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
            this.showDisabledTasksButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.tasks_disabled;
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
            this.runToolButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.StatusAnnotations_Play_16xLG_color;
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
            this.pauseToolButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.StatusAnnotations_Pause_16xLG_color;
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
            this.stopToolButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.StatusAnnotations_Stop_16xLG_color;
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
            this.stepInButton.Enabled = false;
            this.stepInButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.StepIn;
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
            this.stepOverButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.StepOver;
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
            this.stepOutButton.Enabled = false;
            this.stepOutButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.Stepout;
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
            // collapseAllButton
            // 
            this.collapseAllButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.collapseAllButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.collapse;
            this.collapseAllButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.collapseAllButton.Name = "collapseAllButton";
            this.collapseAllButton.Size = new System.Drawing.Size(23, 22);
            this.collapseAllButton.Text = "Collapse all";
            this.collapseAllButton.ToolTipText = "Collapse all tree nodes";
            this.collapseAllButton.Click += new System.EventHandler(this.collapseAllButton_Click);
            // 
            // expandAllButton
            // 
            this.expandAllButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.expandAllButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.expand;
            this.expandAllButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.expandAllButton.Name = "expandAllButton";
            this.expandAllButton.Size = new System.Drawing.Size(23, 22);
            this.expandAllButton.Text = "Expand all";
            this.expandAllButton.ToolTipText = "Expand all tree nodes";
            this.expandAllButton.Click += new System.EventHandler(this.expandAllButton_Click);
            // 
            // toolStripSeparator3
            // 
            this.toolStripSeparator3.Name = "toolStripSeparator3";
            this.toolStripSeparator3.Size = new System.Drawing.Size(6, 25);
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
            this.ClientSize = new System.Drawing.Size(654, 851);
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
            this.Text = "Debugging / Profiling";
            this.toolStrip.ResumeLayout(false);
            this.toolStrip.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private Aga.Controls.Tree.TreeViewAdv debugTreeView;
        private Aga.Controls.Tree.NodeControls.NodeStateIcon stateIcon;
        private Aga.Controls.Tree.NodeControls.NodeTextBox nodeTextBox1;
        public Aga.Controls.Tree.NodeControls.NodeCheckBox checkEnabled;
        private Aga.Controls.Tree.TreeColumn taskName;
        private Aga.Controls.Tree.TreeColumn taskType;
        private Aga.Controls.Tree.TreeColumn enabled;
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
        private Aga.Controls.Tree.TreeColumn breakpoint;
        private Aga.Controls.Tree.TreeColumn profilerTime;
        private Aga.Controls.Tree.NodeControls.NodeTextBox profilerTimeValue;
        public Aga.Controls.Tree.NodeControls.NodeCheckBox breakpointCheckBox;
        private System.Windows.Forms.ToolStripButton collapseAllButton;
        private System.Windows.Forms.ToolStripButton expandAllButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator3;

    }
}