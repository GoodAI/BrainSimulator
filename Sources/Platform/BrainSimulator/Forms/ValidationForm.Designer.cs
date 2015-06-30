namespace GoodAI.BrainSimulator.Forms
{
    partial class ValidationForm
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
            this.listView = new System.Windows.Forms.ListView();
            this.nodeColumn = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.descColumn = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.toolStrip = new System.Windows.Forms.ToolStrip();
            this.errorStripButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.warningStripButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
            this.infoStripButton = new System.Windows.Forms.ToolStripButton();
            this.imageList = new System.Windows.Forms.ImageList(this.components);
            this.toolStrip.SuspendLayout();
            this.SuspendLayout();
            // 
            // listView
            // 
            this.listView.Activation = System.Windows.Forms.ItemActivation.OneClick;
            this.listView.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.nodeColumn,
            this.descColumn});
            this.listView.Dock = System.Windows.Forms.DockStyle.Fill;
            this.listView.GridLines = true;
            this.listView.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.None;
            this.listView.Location = new System.Drawing.Point(0, 25);
            this.listView.MultiSelect = false;
            this.listView.Name = "listView";
            this.listView.Size = new System.Drawing.Size(864, 237);
            this.listView.TabIndex = 0;
            this.listView.UseCompatibleStateImageBehavior = false;
            this.listView.View = System.Windows.Forms.View.Details;
            this.listView.MouseDoubleClick += new System.Windows.Forms.MouseEventHandler(this.listView_MouseDoubleClick);
            // 
            // nodeColumn
            // 
            this.nodeColumn.Text = "Node";
            this.nodeColumn.Width = 150;
            // 
            // descColumn
            // 
            this.descColumn.Text = "Description";
            this.descColumn.Width = 700;
            // 
            // toolStrip
            // 
            this.toolStrip.BackColor = System.Drawing.SystemColors.Control;
            this.toolStrip.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.errorStripButton,
            this.toolStripSeparator1,
            this.warningStripButton,
            this.toolStripSeparator2,
            this.infoStripButton});
            this.toolStrip.Location = new System.Drawing.Point(0, 0);
            this.toolStrip.Name = "toolStrip";
            this.toolStrip.Size = new System.Drawing.Size(864, 25);
            this.toolStrip.TabIndex = 1;
            this.toolStrip.Text = "toolStrip1";
            // 
            // errorStripButton
            // 
            this.errorStripButton.Checked = true;
            this.errorStripButton.CheckOnClick = true;
            this.errorStripButton.CheckState = System.Windows.Forms.CheckState.Checked;
            this.errorStripButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.StatusAnnotations_Critical_16xMD;
            this.errorStripButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.errorStripButton.Name = "errorStripButton";
            this.errorStripButton.Size = new System.Drawing.Size(57, 22);
            this.errorStripButton.Text = "Errors";
            this.errorStripButton.CheckedChanged += new System.EventHandler(this.stripButton_CheckedChanged);
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            this.toolStripSeparator1.Size = new System.Drawing.Size(6, 25);
            // 
            // warningStripButton
            // 
            this.warningStripButton.Checked = true;
            this.warningStripButton.CheckOnClick = true;
            this.warningStripButton.CheckState = System.Windows.Forms.CheckState.Checked;
            this.warningStripButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.StatusAnnotations_Warning_16xMD;
            this.warningStripButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.warningStripButton.Name = "warningStripButton";
            this.warningStripButton.Size = new System.Drawing.Size(77, 22);
            this.warningStripButton.Text = "Warnings";
            this.warningStripButton.CheckedChanged += new System.EventHandler(this.stripButton_CheckedChanged);
            // 
            // toolStripSeparator2
            // 
            this.toolStripSeparator2.Name = "toolStripSeparator2";
            this.toolStripSeparator2.Size = new System.Drawing.Size(6, 25);
            // 
            // infoStripButton
            // 
            this.infoStripButton.Checked = true;
            this.infoStripButton.CheckOnClick = true;
            this.infoStripButton.CheckState = System.Windows.Forms.CheckState.Checked;
            this.infoStripButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.StatusAnnotations_Information_16xMD;
            this.infoStripButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.infoStripButton.Name = "infoStripButton";
            this.infoStripButton.Size = new System.Drawing.Size(48, 22);
            this.infoStripButton.Text = "Info";
            this.infoStripButton.CheckedChanged += new System.EventHandler(this.stripButton_CheckedChanged);
            // 
            // imageList
            // 
            this.imageList.ColorDepth = System.Windows.Forms.ColorDepth.Depth24Bit;
            this.imageList.ImageSize = new System.Drawing.Size(16, 16);
            this.imageList.TransparentColor = System.Drawing.Color.Transparent;
            // 
            // ValidationForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(864, 262);
            this.Controls.Add(this.listView);
            this.Controls.Add(this.toolStrip);
            this.HideOnClose = true;
            this.Name = "ValidationForm";
            this.Text = "Validation";
            this.Load += new System.EventHandler(this.ValidationForm_Load);
            this.toolStrip.ResumeLayout(false);
            this.toolStrip.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ListView listView;
        private System.Windows.Forms.ColumnHeader nodeColumn;
        private System.Windows.Forms.ColumnHeader descColumn;
        private System.Windows.Forms.ToolStrip toolStrip;
        private System.Windows.Forms.ToolStripButton errorStripButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripButton warningStripButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator2;
        private System.Windows.Forms.ToolStripButton infoStripButton;
        private System.Windows.Forms.ImageList imageList;
    }
}