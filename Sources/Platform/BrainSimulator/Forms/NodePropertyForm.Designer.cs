namespace GoodAI.BrainSimulator.Forms
{
    partial class NodePropertyForm
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
            this.nodeNameTextBox = new System.Windows.Forms.RichTextBox();
            this.panel1 = new System.Windows.Forms.Panel();
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.observerDropDownButton = new System.Windows.Forms.ToolStripDropDownButton();
            this.helpButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.loadNodeDataButton = new System.Windows.Forms.ToolStripButton();
            this.saveNodeDataButton = new System.Windows.Forms.ToolStripButton();
            this.clearDataButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
            this.snapshotButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator3 = new System.Windows.Forms.ToolStripSeparator();
            this.dashboardButton = new System.Windows.Forms.ToolStripButton();
            this.folderBrowserDialog = new System.Windows.Forms.FolderBrowserDialog();
            this.panel1.SuspendLayout();
            this.toolStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // propertyGrid
            // 
            this.propertyGrid.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.propertyGrid.Location = new System.Drawing.Point(3, 56);
            this.propertyGrid.Name = "propertyGrid";
            this.propertyGrid.Size = new System.Drawing.Size(236, 594);
            this.propertyGrid.TabIndex = 8;
            this.propertyGrid.ToolbarVisible = false;
            this.propertyGrid.PropertyValueChanged += new System.Windows.Forms.PropertyValueChangedEventHandler(this.propertyGrid_PropertyValueChanged);
            this.propertyGrid.SelectedGridItemChanged += new System.Windows.Forms.SelectedGridItemChangedEventHandler(this.propertyGrid_SelectedGridItemChanged);
            this.propertyGrid.Enter += new System.EventHandler(this.propertyGrid_Enter);
            // 
            // nodeNameTextBox
            // 
            this.nodeNameTextBox.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.nodeNameTextBox.BackColor = System.Drawing.SystemColors.Window;
            this.nodeNameTextBox.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.nodeNameTextBox.Location = new System.Drawing.Point(4, 4);
            this.nodeNameTextBox.Margin = new System.Windows.Forms.Padding(0);
            this.nodeNameTextBox.Multiline = false;
            this.nodeNameTextBox.Name = "nodeNameTextBox";
            this.nodeNameTextBox.ReadOnly = true;
            this.nodeNameTextBox.ScrollBars = System.Windows.Forms.RichTextBoxScrollBars.None;
            this.nodeNameTextBox.Size = new System.Drawing.Size(232, 17);
            this.nodeNameTextBox.TabIndex = 9;
            this.nodeNameTextBox.Text = "";
            // 
            // panel1
            // 
            this.panel1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel1.BackColor = System.Drawing.SystemColors.Window;
            this.panel1.Controls.Add(this.nodeNameTextBox);
            this.panel1.Location = new System.Drawing.Point(3, 6);
            this.panel1.Name = "panel1";
            this.panel1.Padding = new System.Windows.Forms.Padding(4, 4, 0, 0);
            this.panel1.Size = new System.Drawing.Size(236, 21);
            this.panel1.TabIndex = 10;
            // 
            // toolStrip1
            // 
            this.toolStrip1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.toolStrip1.AutoSize = false;
            this.toolStrip1.BackColor = System.Drawing.SystemColors.Control;
            this.toolStrip1.Dock = System.Windows.Forms.DockStyle.None;
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.observerDropDownButton,
            this.helpButton,
            this.toolStripSeparator1,
            this.loadNodeDataButton,
            this.saveNodeDataButton,
            this.clearDataButton,
            this.toolStripSeparator2,
            this.snapshotButton,
            this.toolStripSeparator3,
            this.dashboardButton});
            this.toolStrip1.Location = new System.Drawing.Point(3, 30);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(236, 23);
            this.toolStrip1.TabIndex = 11;
            this.toolStrip1.Text = "toolStrip1";
            // 
            // observerDropDownButton
            // 
            this.observerDropDownButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.observerDropDownButton.Enabled = false;
            this.observerDropDownButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.add_observer;
            this.observerDropDownButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.observerDropDownButton.Name = "observerDropDownButton";
            this.observerDropDownButton.Size = new System.Drawing.Size(29, 20);
            this.observerDropDownButton.Text = "Add Node Observer";
            // 
            // helpButton
            // 
            this.helpButton.Alignment = System.Windows.Forms.ToolStripItemAlignment.Right;
            this.helpButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.helpButton.Enabled = false;
            this.helpButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.StatusAnnotations_Help;
            this.helpButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.helpButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.helpButton.Name = "helpButton";
            this.helpButton.Size = new System.Drawing.Size(23, 20);
            this.helpButton.Text = "Help";
            this.helpButton.Click += new System.EventHandler(this.helpButton_Click);
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            this.toolStripSeparator1.Size = new System.Drawing.Size(6, 23);
            // 
            // loadNodeDataButton
            // 
            this.loadNodeDataButton.CheckOnClick = true;
            this.loadNodeDataButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.loadNodeDataButton.Enabled = false;
            this.loadNodeDataButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.open_mb;
            this.loadNodeDataButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.loadNodeDataButton.Margin = new System.Windows.Forms.Padding(0, 1, 1, 2);
            this.loadNodeDataButton.Name = "loadNodeDataButton";
            this.loadNodeDataButton.Size = new System.Drawing.Size(23, 20);
            this.loadNodeDataButton.Text = "Load Data on Startup";
            this.loadNodeDataButton.Click += new System.EventHandler(this.loadNodeDataButton_Click);
            // 
            // saveNodeDataButton
            // 
            this.saveNodeDataButton.CheckOnClick = true;
            this.saveNodeDataButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.saveNodeDataButton.Enabled = false;
            this.saveNodeDataButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.save_mb;
            this.saveNodeDataButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.saveNodeDataButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.saveNodeDataButton.Name = "saveNodeDataButton";
            this.saveNodeDataButton.Size = new System.Drawing.Size(23, 20);
            this.saveNodeDataButton.Text = "Save Data on Stop";
            this.saveNodeDataButton.Click += new System.EventHandler(this.saveDataNodeButton_Click);
            // 
            // clearDataButton
            // 
            this.clearDataButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.clearDataButton.Enabled = false;
            this.clearDataButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.clear_mb;
            this.clearDataButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.clearDataButton.Name = "clearDataButton";
            this.clearDataButton.Size = new System.Drawing.Size(23, 20);
            this.clearDataButton.Text = "Clear Saved Data";
            this.clearDataButton.Click += new System.EventHandler(this.clearDataButton_Click);
            // 
            // toolStripSeparator2
            // 
            this.toolStripSeparator2.Name = "toolStripSeparator2";
            this.toolStripSeparator2.Size = new System.Drawing.Size(6, 23);
            // 
            // snapshotButton
            // 
            this.snapshotButton.CheckOnClick = true;
            this.snapshotButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.snapshotButton.Enabled = false;
            this.snapshotButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.Snapshot;
            this.snapshotButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.snapshotButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.snapshotButton.Name = "snapshotButton";
            this.snapshotButton.Size = new System.Drawing.Size(23, 20);
            this.snapshotButton.Text = "Autosave Snapshot";
            this.snapshotButton.Click += new System.EventHandler(this.snapshotButton_Click);
            // 
            // toolStripSeparator3
            // 
            this.toolStripSeparator3.Name = "toolStripSeparator3";
            this.toolStripSeparator3.Size = new System.Drawing.Size(6, 23);
            // 
            // dashboardButton
            // 
            this.dashboardButton.CheckOnClick = true;
            this.dashboardButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.dashboardButton.Enabled = false;
            this.dashboardButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.pushpin_16xMD;
            this.dashboardButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.dashboardButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.dashboardButton.Name = "dashboardButton";
            this.dashboardButton.Size = new System.Drawing.Size(23, 20);
            this.dashboardButton.Text = "Show in Dashboard";
            this.dashboardButton.CheckedChanged += new System.EventHandler(this.dashboardButton_CheckedChanged);
            // 
            // folderBrowserDialog
            // 
            this.folderBrowserDialog.ShowNewFolderButton = false;
            // 
            // NodePropertyForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(242, 653);
            this.Controls.Add(this.toolStrip1);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.propertyGrid);
            this.DockAreas = ((WeifenLuo.WinFormsUI.Docking.DockAreas)(((((WeifenLuo.WinFormsUI.Docking.DockAreas.Float | WeifenLuo.WinFormsUI.Docking.DockAreas.DockLeft) 
            | WeifenLuo.WinFormsUI.Docking.DockAreas.DockRight) 
            | WeifenLuo.WinFormsUI.Docking.DockAreas.DockTop) 
            | WeifenLuo.WinFormsUI.Docking.DockAreas.DockBottom)));
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(238)));
            this.HideOnClose = true;
            this.Name = "NodePropertyForm";
            this.Padding = new System.Windows.Forms.Padding(3);
            this.Text = "Node Properties";
            this.panel1.ResumeLayout(false);
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PropertyGrid propertyGrid;
        private System.Windows.Forms.RichTextBox nodeNameTextBox;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.ToolStripDropDownButton observerDropDownButton;
        private System.Windows.Forms.ToolStripButton saveNodeDataButton;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog;
        private System.Windows.Forms.ToolStripButton helpButton;
        private System.Windows.Forms.ToolStripButton loadNodeDataButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripButton clearDataButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator2;
        private System.Windows.Forms.ToolStripButton snapshotButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator3;
        private System.Windows.Forms.ToolStripButton dashboardButton;

    }
}