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
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.removeButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
            this.goToNodeButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator4 = new System.Windows.Forms.ToolStripSeparator();
            this.showMembersButton = new System.Windows.Forms.ToolStripButton();
            this.showGroupsButton = new System.Windows.Forms.ToolStripButton();
            this.splitContainerProperties = new System.Windows.Forms.SplitContainer();
            this.globalLabel = new System.Windows.Forms.Label();
            this.splitContainerGroups = new System.Windows.Forms.SplitContainer();
            this.errorText = new System.Windows.Forms.Label();
            this.groupedLabel = new System.Windows.Forms.Label();
            this.toolStrip2 = new System.Windows.Forms.ToolStrip();
            this.addGroupButton = new System.Windows.Forms.ToolStripButton();
            this.removeGroupButton = new System.Windows.Forms.ToolStripButton();
            this.editGroupButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.addToGroupButton = new System.Windows.Forms.ToolStripButton();
            this.propertyGridGrouped = new System.Windows.Forms.PropertyGrid();
            this.memberListBox = new System.Windows.Forms.ListBox();
            this.membersLabel = new System.Windows.Forms.Label();
            this.toolStrip3 = new System.Windows.Forms.ToolStrip();
            this.removeFromGroupButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator3 = new System.Windows.Forms.ToolStripSeparator();
            this.goToNodeFromMemberButton = new System.Windows.Forms.ToolStripButton();
            this.toolStrip1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainerProperties)).BeginInit();
            this.splitContainerProperties.Panel1.SuspendLayout();
            this.splitContainerProperties.Panel2.SuspendLayout();
            this.splitContainerProperties.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainerGroups)).BeginInit();
            this.splitContainerGroups.Panel1.SuspendLayout();
            this.splitContainerGroups.Panel2.SuspendLayout();
            this.splitContainerGroups.SuspendLayout();
            this.toolStrip2.SuspendLayout();
            this.toolStrip3.SuspendLayout();
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
            this.propertyGrid.Size = new System.Drawing.Size(422, 241);
            this.propertyGrid.TabIndex = 0;
            this.propertyGrid.ToolbarVisible = false;
            this.propertyGrid.SelectedGridItemChanged += new System.Windows.Forms.SelectedGridItemChangedEventHandler(this.propertyGrid_SelectedGridItemChanged);
            // 
            // toolStrip1
            // 
            this.toolStrip1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.toolStrip1.AutoSize = false;
            this.toolStrip1.Dock = System.Windows.Forms.DockStyle.None;
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.removeButton,
            this.toolStripSeparator2,
            this.goToNodeButton,
            this.toolStripSeparator4,
            this.showMembersButton,
            this.showGroupsButton});
            this.toolStrip1.Location = new System.Drawing.Point(3, 0);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(425, 25);
            this.toolStrip1.TabIndex = 1;
            this.toolStrip1.Text = "toolStrip1";
            // 
            // removeButton
            // 
            this.removeButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.removeButton.Enabled = false;
            this.removeButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.action_Cancel_16xMD;
            this.removeButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.removeButton.Name = "removeButton";
            this.removeButton.Size = new System.Drawing.Size(23, 22);
            this.removeButton.Text = "Remove the property";
            this.removeButton.ToolTipText = "Remove the property from the dashboard";
            this.removeButton.Click += new System.EventHandler(this.removeButton_Click);
            // 
            // toolStripSeparator2
            // 
            this.toolStripSeparator2.Name = "toolStripSeparator2";
            this.toolStripSeparator2.Size = new System.Drawing.Size(6, 25);
            // 
            // goToNodeButton
            // 
            this.goToNodeButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.goToNodeButton.Enabled = false;
            this.goToNodeButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.view_16xLG;
            this.goToNodeButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.goToNodeButton.Name = "goToNodeButton";
            this.goToNodeButton.Size = new System.Drawing.Size(23, 22);
            this.goToNodeButton.Text = "Go to Node";
            this.goToNodeButton.Click += new System.EventHandler(this.goToNodeButton_Click);
            // 
            // toolStripSeparator4
            // 
            this.toolStripSeparator4.Alignment = System.Windows.Forms.ToolStripItemAlignment.Right;
            this.toolStripSeparator4.Margin = new System.Windows.Forms.Padding(0, 0, 100, 0);
            this.toolStripSeparator4.Name = "toolStripSeparator4";
            this.toolStripSeparator4.Size = new System.Drawing.Size(6, 25);
            // 
            // showMembersButton
            // 
            this.showMembersButton.Alignment = System.Windows.Forms.ToolStripItemAlignment.Right;
            this.showMembersButton.Checked = true;
            this.showMembersButton.CheckOnClick = true;
            this.showMembersButton.CheckState = System.Windows.Forms.CheckState.Checked;
            this.showMembersButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.showMembersButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.list_16xLG;
            this.showMembersButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.showMembersButton.Name = "showMembersButton";
            this.showMembersButton.Size = new System.Drawing.Size(23, 22);
            this.showMembersButton.Text = "Show/hide members";
            this.showMembersButton.ToolTipText = "Show/hide group members";
            this.showMembersButton.CheckedChanged += new System.EventHandler(this.showMembersButton_CheckedChanged);
            // 
            // showGroupsButton
            // 
            this.showGroupsButton.Alignment = System.Windows.Forms.ToolStripItemAlignment.Right;
            this.showGroupsButton.Checked = true;
            this.showGroupsButton.CheckOnClick = true;
            this.showGroupsButton.CheckState = System.Windows.Forms.CheckState.Checked;
            this.showGroupsButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.showGroupsButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.folder_Closed_16xLG;
            this.showGroupsButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.showGroupsButton.Name = "showGroupsButton";
            this.showGroupsButton.Size = new System.Drawing.Size(23, 22);
            this.showGroupsButton.Text = "Show/hide groups";
            this.showGroupsButton.TextImageRelation = System.Windows.Forms.TextImageRelation.TextBeforeImage;
            this.showGroupsButton.ToolTipText = "Show/hide property groups";
            this.showGroupsButton.CheckedChanged += new System.EventHandler(this.showGroupsButton_CheckedChanged);
            // 
            // splitContainerProperties
            // 
            this.splitContainerProperties.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.splitContainerProperties.Location = new System.Drawing.Point(0, 0);
            this.splitContainerProperties.Name = "splitContainerProperties";
            this.splitContainerProperties.Orientation = System.Windows.Forms.Orientation.Horizontal;
            // 
            // splitContainerProperties.Panel1
            // 
            this.splitContainerProperties.Panel1.Controls.Add(this.globalLabel);
            this.splitContainerProperties.Panel1.Controls.Add(this.toolStrip1);
            this.splitContainerProperties.Panel1.Controls.Add(this.propertyGrid);
            // 
            // splitContainerProperties.Panel2
            // 
            this.splitContainerProperties.Panel2.Controls.Add(this.splitContainerGroups);
            this.splitContainerProperties.Size = new System.Drawing.Size(428, 658);
            this.splitContainerProperties.SplitterDistance = 266;
            this.splitContainerProperties.TabIndex = 2;
            // 
            // globalLabel
            // 
            this.globalLabel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.globalLabel.AutoSize = true;
            this.globalLabel.Location = new System.Drawing.Point(338, 4);
            this.globalLabel.Name = "globalLabel";
            this.globalLabel.Size = new System.Drawing.Size(87, 13);
            this.globalLabel.TabIndex = 2;
            this.globalLabel.Text = "Global Properties";
            // 
            // splitContainerGroups
            // 
            this.splitContainerGroups.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.splitContainerGroups.Location = new System.Drawing.Point(0, -1);
            this.splitContainerGroups.Name = "splitContainerGroups";
            this.splitContainerGroups.Orientation = System.Windows.Forms.Orientation.Horizontal;
            // 
            // splitContainerGroups.Panel1
            // 
            this.splitContainerGroups.Panel1.Controls.Add(this.errorText);
            this.splitContainerGroups.Panel1.Controls.Add(this.groupedLabel);
            this.splitContainerGroups.Panel1.Controls.Add(this.toolStrip2);
            this.splitContainerGroups.Panel1.Controls.Add(this.propertyGridGrouped);
            // 
            // splitContainerGroups.Panel2
            // 
            this.splitContainerGroups.Panel2.Controls.Add(this.memberListBox);
            this.splitContainerGroups.Panel2.Controls.Add(this.membersLabel);
            this.splitContainerGroups.Panel2.Controls.Add(this.toolStrip3);
            this.splitContainerGroups.Size = new System.Drawing.Size(428, 389);
            this.splitContainerGroups.SplitterDistance = 194;
            this.splitContainerGroups.TabIndex = 2;
            // 
            // errorText
            // 
            this.errorText.AutoSize = true;
            this.errorText.ForeColor = System.Drawing.Color.Firebrick;
            this.errorText.Location = new System.Drawing.Point(114, 6);
            this.errorText.Name = "errorText";
            this.errorText.Size = new System.Drawing.Size(0, 13);
            this.errorText.TabIndex = 3;
            // 
            // groupedLabel
            // 
            this.groupedLabel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.groupedLabel.AutoSize = true;
            this.groupedLabel.Location = new System.Drawing.Point(384, 6);
            this.groupedLabel.Name = "groupedLabel";
            this.groupedLabel.Size = new System.Drawing.Size(41, 13);
            this.groupedLabel.TabIndex = 2;
            this.groupedLabel.Text = "Groups";
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
            this.toolStrip2.Location = new System.Drawing.Point(3, 0);
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
            this.removeGroupButton.Enabled = false;
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
            this.editGroupButton.Enabled = false;
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
            this.addToGroupButton.Enabled = false;
            this.addToGroupButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.arrow_Down_16xLG;
            this.addToGroupButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.addToGroupButton.Name = "addToGroupButton";
            this.addToGroupButton.Size = new System.Drawing.Size(23, 22);
            this.addToGroupButton.Text = "Add to group";
            this.addToGroupButton.ToolTipText = "Add the selected property from top into the selected group";
            this.addToGroupButton.Click += new System.EventHandler(this.addToGroupButton_Click);
            // 
            // propertyGridGrouped
            // 
            this.propertyGridGrouped.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.propertyGridGrouped.CategoryForeColor = System.Drawing.SystemColors.InactiveCaptionText;
            this.propertyGridGrouped.HelpVisible = false;
            this.propertyGridGrouped.Location = new System.Drawing.Point(3, 28);
            this.propertyGridGrouped.Name = "propertyGridGrouped";
            this.propertyGridGrouped.PropertySort = System.Windows.Forms.PropertySort.Alphabetical;
            this.propertyGridGrouped.Size = new System.Drawing.Size(422, 163);
            this.propertyGridGrouped.TabIndex = 0;
            this.propertyGridGrouped.ToolbarVisible = false;
            this.propertyGridGrouped.SelectedGridItemChanged += new System.Windows.Forms.SelectedGridItemChangedEventHandler(this.propertyGridGrouped_SelectedGridItemChanged);
            // 
            // memberListBox
            // 
            this.memberListBox.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.memberListBox.FormattingEnabled = true;
            this.memberListBox.Location = new System.Drawing.Point(3, 28);
            this.memberListBox.Name = "memberListBox";
            this.memberListBox.Size = new System.Drawing.Size(422, 160);
            this.memberListBox.TabIndex = 3;
            // 
            // membersLabel
            // 
            this.membersLabel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.membersLabel.AutoSize = true;
            this.membersLabel.Location = new System.Drawing.Point(344, 6);
            this.membersLabel.Name = "membersLabel";
            this.membersLabel.Size = new System.Drawing.Size(81, 13);
            this.membersLabel.TabIndex = 1;
            this.membersLabel.Text = "Group members";
            // 
            // toolStrip3
            // 
            this.toolStrip3.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.toolStrip3.AutoSize = false;
            this.toolStrip3.Dock = System.Windows.Forms.DockStyle.None;
            this.toolStrip3.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.removeFromGroupButton,
            this.toolStripSeparator3,
            this.goToNodeFromMemberButton});
            this.toolStrip3.Location = new System.Drawing.Point(3, 0);
            this.toolStrip3.Name = "toolStrip3";
            this.toolStrip3.Size = new System.Drawing.Size(425, 25);
            this.toolStrip3.TabIndex = 2;
            this.toolStrip3.Text = "toolStrip3";
            // 
            // removeFromGroupButton
            // 
            this.removeFromGroupButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.removeFromGroupButton.Enabled = false;
            this.removeFromGroupButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.arrow_Up_16xLG;
            this.removeFromGroupButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.removeFromGroupButton.Name = "removeFromGroupButton";
            this.removeFromGroupButton.Size = new System.Drawing.Size(23, 22);
            this.removeFromGroupButton.Text = "Remove from group";
            this.removeFromGroupButton.ToolTipText = "Remove the selected member property from the group";
            this.removeFromGroupButton.Click += new System.EventHandler(this.removeFromGroupButton_Click);
            // 
            // toolStripSeparator3
            // 
            this.toolStripSeparator3.Name = "toolStripSeparator3";
            this.toolStripSeparator3.Size = new System.Drawing.Size(6, 25);
            // 
            // goToNodeFromMemberButton
            // 
            this.goToNodeFromMemberButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.goToNodeFromMemberButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.view_16xLG;
            this.goToNodeFromMemberButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.goToNodeFromMemberButton.Name = "goToNodeFromMemberButton";
            this.goToNodeFromMemberButton.Size = new System.Drawing.Size(23, 22);
            this.goToNodeFromMemberButton.Text = "Go to Node";
            this.goToNodeFromMemberButton.Click += new System.EventHandler(this.goToNodeFromMemberButton_Click);
            // 
            // DashboardPropertyForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(428, 658);
            this.Controls.Add(this.splitContainerProperties);
            this.DockAreas = ((WeifenLuo.WinFormsUI.Docking.DockAreas)(((((WeifenLuo.WinFormsUI.Docking.DockAreas.Float | WeifenLuo.WinFormsUI.Docking.DockAreas.DockLeft) 
            | WeifenLuo.WinFormsUI.Docking.DockAreas.DockRight) 
            | WeifenLuo.WinFormsUI.Docking.DockAreas.DockTop) 
            | WeifenLuo.WinFormsUI.Docking.DockAreas.DockBottom)));
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.HideOnClose = true;
            this.Name = "DashboardPropertyForm";
            this.Text = "Dashboard Properties";
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            this.splitContainerProperties.Panel1.ResumeLayout(false);
            this.splitContainerProperties.Panel1.PerformLayout();
            this.splitContainerProperties.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainerProperties)).EndInit();
            this.splitContainerProperties.ResumeLayout(false);
            this.splitContainerGroups.Panel1.ResumeLayout(false);
            this.splitContainerGroups.Panel1.PerformLayout();
            this.splitContainerGroups.Panel2.ResumeLayout(false);
            this.splitContainerGroups.Panel2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainerGroups)).EndInit();
            this.splitContainerGroups.ResumeLayout(false);
            this.toolStrip2.ResumeLayout(false);
            this.toolStrip2.PerformLayout();
            this.toolStrip3.ResumeLayout(false);
            this.toolStrip3.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PropertyGrid propertyGrid;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.ToolStripButton removeButton;
        private System.Windows.Forms.SplitContainer splitContainerProperties;
        private System.Windows.Forms.PropertyGrid propertyGridGrouped;
        private System.Windows.Forms.ToolStrip toolStrip2;
        private System.Windows.Forms.ToolStripButton addGroupButton;
        private System.Windows.Forms.ToolStripButton removeGroupButton;
        private System.Windows.Forms.ToolStripButton editGroupButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripButton addToGroupButton;
        private System.Windows.Forms.SplitContainer splitContainerGroups;
        private System.Windows.Forms.Label globalLabel;
        private System.Windows.Forms.Label groupedLabel;
        private System.Windows.Forms.Label membersLabel;
        private System.Windows.Forms.ToolStrip toolStrip3;
        private System.Windows.Forms.ToolStripButton removeFromGroupButton;
        private System.Windows.Forms.ListBox memberListBox;
        private System.Windows.Forms.Label errorText;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator2;
        private System.Windows.Forms.ToolStripButton goToNodeButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator3;
        private System.Windows.Forms.ToolStripButton goToNodeFromMemberButton;
        private System.Windows.Forms.ToolStripButton showGroupsButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator4;
        private System.Windows.Forms.ToolStripButton showMembersButton;




    }
}