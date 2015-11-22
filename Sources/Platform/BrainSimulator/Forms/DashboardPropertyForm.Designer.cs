namespace GoodAI.BrainSimulator.Forms
{
    partial class DashboardPropertyForm
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
            this.propertyGrid = new System.Windows.Forms.PropertyGrid();
            this.directorySearcher1 = new System.DirectoryServices.DirectorySearcher();
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.removeButton = new System.Windows.Forms.ToolStripButton();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.toolStrip2 = new System.Windows.Forms.ToolStrip();
            this.addGroupButton = new System.Windows.Forms.ToolStripButton();
            this.removeGroupButton = new System.Windows.Forms.ToolStripButton();
            this.editGroupButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.addToGroupButton = new System.Windows.Forms.ToolStripButton();
            this.propertyGridGrouped = new System.Windows.Forms.PropertyGrid();
            this.toolStrip1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            this.toolStrip2.SuspendLayout();
            this.SuspendLayout();
            // 
            // propertyGrid
            // 
            this.propertyGrid.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.propertyGrid.CategoryForeColor = System.Drawing.SystemColors.InactiveCaptionText;
            this.propertyGrid.Location = new System.Drawing.Point(3, 25);
            this.propertyGrid.Margin = new System.Windows.Forms.Padding(0);
            this.propertyGrid.Name = "propertyGrid";
            this.propertyGrid.Size = new System.Drawing.Size(422, 202);
            this.propertyGrid.TabIndex = 0;
            this.propertyGrid.ToolbarVisible = false;
            this.propertyGrid.SelectedGridItemChanged += new System.Windows.Forms.SelectedGridItemChangedEventHandler(this.propertyGrid_SelectedGridItemChanged);
            // 
            // directorySearcher1
            // 
            this.directorySearcher1.ClientTimeout = System.TimeSpan.Parse("-00:00:01");
            this.directorySearcher1.ServerPageTimeLimit = System.TimeSpan.Parse("-00:00:01");
            this.directorySearcher1.ServerTimeLimit = System.TimeSpan.Parse("-00:00:01");
            // 
            // toolStrip1
            // 
            this.toolStrip1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.toolStrip1.AutoSize = false;
            this.toolStrip1.Dock = System.Windows.Forms.DockStyle.None;
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.removeButton});
            this.toolStrip1.Location = new System.Drawing.Point(3, 0);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(425, 25);
            this.toolStrip1.TabIndex = 1;
            this.toolStrip1.Text = "toolStrip1";
            // 
            // removeButton
            // 
            this.removeButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.removeButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.action_Cancel_16xMD;
            this.removeButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.removeButton.Name = "removeButton";
            this.removeButton.Size = new System.Drawing.Size(23, 22);
            this.removeButton.Text = "Remove the property";
            this.removeButton.ToolTipText = "Remove the property from the dashboard";
            this.removeButton.Click += new System.EventHandler(this.removeButton_Click);
            // 
            // splitContainer1
            // 
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.Location = new System.Drawing.Point(0, 0);
            this.splitContainer1.Name = "splitContainer1";
            this.splitContainer1.Orientation = System.Windows.Forms.Orientation.Horizontal;
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.Controls.Add(this.toolStrip1);
            this.splitContainer1.Panel1.Controls.Add(this.propertyGrid);
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.Controls.Add(this.toolStrip2);
            this.splitContainer1.Panel2.Controls.Add(this.propertyGridGrouped);
            this.splitContainer1.Size = new System.Drawing.Size(428, 561);
            this.splitContainer1.SplitterDistance = 227;
            this.splitContainer1.TabIndex = 2;
            // 
            // toolStrip2
            // 
            this.toolStrip2.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.toolStrip2.AutoSize = false;
            this.toolStrip2.Dock = System.Windows.Forms.DockStyle.None;
            this.toolStrip2.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.addGroupButton,
            this.removeGroupButton,
            this.editGroupButton,
            this.toolStripSeparator1,
            this.addToGroupButton});
            this.toolStrip2.Location = new System.Drawing.Point(0, 0);
            this.toolStrip2.Name = "toolStrip2";
            this.toolStrip2.Size = new System.Drawing.Size(425, 25);
            this.toolStrip2.TabIndex = 1;
            this.toolStrip2.Text = "toolStrip2";
            // 
            // addGroupButton
            // 
            this.addGroupButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.addGroupButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.action_add_16xMD;
            this.addGroupButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.addGroupButton.Name = "addGroupButton";
            this.addGroupButton.Size = new System.Drawing.Size(23, 22);
            this.addGroupButton.Text = "Add a group";
            this.addGroupButton.ToolTipText = "Add a new property group";
            this.addGroupButton.Click += new System.EventHandler(this.addGroupButton_Click);
            // 
            // removeGroupButton
            // 
            this.removeGroupButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.removeGroupButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.action_Cancel_16xMD;
            this.removeGroupButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.removeGroupButton.Name = "removeGroupButton";
            this.removeGroupButton.Size = new System.Drawing.Size(23, 22);
            this.removeGroupButton.Text = "Remove group";
            this.removeGroupButton.ToolTipText = "Remove selected property group";
            this.removeGroupButton.Click += new System.EventHandler(this.removeGroupButton_Click);
            // 
            // editGroupButton
            // 
            this.editGroupButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.editGroupButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.pencil_003_16xMD;
            this.editGroupButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.editGroupButton.Name = "editGroupButton";
            this.editGroupButton.Size = new System.Drawing.Size(23, 22);
            this.editGroupButton.Text = "Edit group";
            this.editGroupButton.ToolTipText = "Edit the name of the selected group name";
            this.editGroupButton.Click += new System.EventHandler(this.editGroupButton_Click);
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            this.toolStripSeparator1.Size = new System.Drawing.Size(6, 25);
            // 
            // addToGroupButton
            // 
            this.addToGroupButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.addToGroupButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.arrow_Down_16xLG;
            this.addToGroupButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.addToGroupButton.Name = "addToGroupButton";
            this.addToGroupButton.Size = new System.Drawing.Size(23, 22);
            this.addToGroupButton.Text = "Add to group";
            this.addToGroupButton.ToolTipText = "Add the selected property from top into the selected group in the bottom";
            this.addToGroupButton.Click += new System.EventHandler(this.addToGroupButton_Click);
            // 
            // propertyGridGrouped
            // 
            this.propertyGridGrouped.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.propertyGridGrouped.CategoryForeColor = System.Drawing.SystemColors.InactiveCaptionText;
            this.propertyGridGrouped.Location = new System.Drawing.Point(3, 28);
            this.propertyGridGrouped.Name = "propertyGridGrouped";
            this.propertyGridGrouped.PropertySort = System.Windows.Forms.PropertySort.Alphabetical;
            this.propertyGridGrouped.Size = new System.Drawing.Size(422, 299);
            this.propertyGridGrouped.TabIndex = 0;
            this.propertyGridGrouped.ToolbarVisible = false;
            this.propertyGridGrouped.SelectedGridItemChanged += new System.Windows.Forms.SelectedGridItemChangedEventHandler(this.propertyGridGrouped_SelectedGridItemChanged);
            // 
            // DashboardPropertyForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(428, 561);
            this.Controls.Add(this.splitContainer1);
            this.DockAreas = ((WeifenLuo.WinFormsUI.Docking.DockAreas)(((((WeifenLuo.WinFormsUI.Docking.DockAreas.Float | WeifenLuo.WinFormsUI.Docking.DockAreas.DockLeft) 
            | WeifenLuo.WinFormsUI.Docking.DockAreas.DockRight) 
            | WeifenLuo.WinFormsUI.Docking.DockAreas.DockTop) 
            | WeifenLuo.WinFormsUI.Docking.DockAreas.DockBottom)));
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Name = "DashboardPropertyForm";
            this.Text = "Dashboard Properties";
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            this.toolStrip2.ResumeLayout(false);
            this.toolStrip2.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.DirectoryServices.DirectorySearcher directorySearcher1;
        private System.Windows.Forms.PropertyGrid propertyGrid;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.ToolStripButton removeButton;
        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.PropertyGrid propertyGridGrouped;
        private System.Windows.Forms.ToolStrip toolStrip2;
        private System.Windows.Forms.ToolStripButton addGroupButton;
        private System.Windows.Forms.ToolStripButton removeGroupButton;
        private System.Windows.Forms.ToolStripButton editGroupButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripButton addToGroupButton;




    }
}