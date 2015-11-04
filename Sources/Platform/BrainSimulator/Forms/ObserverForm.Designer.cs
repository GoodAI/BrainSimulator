namespace GoodAI.BrainSimulator.Forms
{
    partial class ObserverForm
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
            this.glControl = new OpenTK.GLControl();
            this.contextMenuStrip = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.updateViewToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.snapshotToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.goToNodeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.peekLabel = new System.Windows.Forms.Label();
            this.contextMenuStrip.SuspendLayout();
            this.SuspendLayout();
            // 
            // glControl
            // 
            this.glControl.BackColor = System.Drawing.Color.Black;
            this.glControl.Dock = System.Windows.Forms.DockStyle.Fill;
            this.glControl.Location = new System.Drawing.Point(0, 0);
            this.glControl.Margin = new System.Windows.Forms.Padding(8, 7, 8, 7);
            this.glControl.Name = "glControl";
            this.glControl.Size = new System.Drawing.Size(308, 262);
            this.glControl.TabIndex = 0;
            this.glControl.VSync = false;
            this.glControl.Paint += new System.Windows.Forms.PaintEventHandler(this.glControl_Paint);
            this.glControl.MouseDown += new System.Windows.Forms.MouseEventHandler(this.glControl_MouseDown);
            this.glControl.MouseUp += new System.Windows.Forms.MouseEventHandler(this.glControl_MouseUp);
            this.glControl.Resize += new System.EventHandler(this.glControl_Resize);
            // 
            // contextMenuStrip
            // 
            this.contextMenuStrip.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.updateViewToolStripMenuItem,
            this.snapshotToolStripMenuItem,
            this.goToNodeToolStripMenuItem});
            this.contextMenuStrip.Name = "contextMenuStrip";
            this.contextMenuStrip.Size = new System.Drawing.Size(160, 70);
            // 
            // updateViewToolStripMenuItem
            // 
            this.updateViewToolStripMenuItem.Enabled = false;
            this.updateViewToolStripMenuItem.Image = global::GoodAI.BrainSimulator.Properties.Resources.refresh_16xLG;
            this.updateViewToolStripMenuItem.Name = "updateViewToolStripMenuItem";
            this.updateViewToolStripMenuItem.Size = new System.Drawing.Size(183, 46);
            this.updateViewToolStripMenuItem.Text = "Update View";
            this.updateViewToolStripMenuItem.Click += new System.EventHandler(this.updateViewToolStripMenuItem_Click);
            // 
            // snapshotToolStripMenuItem
            // 
            this.snapshotToolStripMenuItem.Enabled = false;
            this.snapshotToolStripMenuItem.Image = global::GoodAI.BrainSimulator.Properties.Resources.Snapshot;
            this.snapshotToolStripMenuItem.Name = "snapshotToolStripMenuItem";
            this.snapshotToolStripMenuItem.Size = new System.Drawing.Size(183, 46);
            this.snapshotToolStripMenuItem.Text = "Save Snapshot...";
            this.snapshotToolStripMenuItem.Click += new System.EventHandler(this.snapshotToolStripMenuItem_Click);
            // 
            // goToNodeToolStripMenuItem
            // 
            this.goToNodeToolStripMenuItem.Name = "goToNodeToolStripMenuItem";
            this.goToNodeToolStripMenuItem.Size = new System.Drawing.Size(183, 46);
            this.goToNodeToolStripMenuItem.Text = "Go to node";
            this.goToNodeToolStripMenuItem.Click += new System.EventHandler(this.goToNodeToolStripMenuItem_Click);
            // 
            // peekLabel
            // 
            this.peekLabel.AutoSize = true;
            this.peekLabel.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.peekLabel.Dock = System.Windows.Forms.DockStyle.Right;
            this.peekLabel.Font = new System.Drawing.Font("Consolas", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(238)));
            this.peekLabel.Location = new System.Drawing.Point(292, 0);
            this.peekLabel.Margin = new System.Windows.Forms.Padding(3);
            this.peekLabel.Name = "peekLabel";
            this.peekLabel.Padding = new System.Windows.Forms.Padding(0, 0, 0, 1);
            this.peekLabel.Size = new System.Drawing.Size(16, 18);
            this.peekLabel.TabIndex = 1;
            this.peekLabel.Text = "0";
            this.peekLabel.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            this.peekLabel.Visible = false;
            // 
            // ObserverForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(308, 262);
            this.Controls.Add(this.peekLabel);
            this.Controls.Add(this.glControl);
            this.DockAreas = WeifenLuo.WinFormsUI.Docking.DockAreas.Float;
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(238)));
            this.KeyPreview = true;
            this.Name = "ObserverForm";
            this.Text = "Observer";
            this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.ObserverForm_FormClosed);
            this.Load += new System.EventHandler(this.ObserverForm_Load);
            this.contextMenuStrip.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        protected OpenTK.GLControl glControl;
        private System.Windows.Forms.ContextMenuStrip contextMenuStrip;
        private System.Windows.Forms.ToolStripMenuItem snapshotToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem updateViewToolStripMenuItem;
        private System.Windows.Forms.Label peekLabel;
        private System.Windows.Forms.ToolStripMenuItem goToNodeToolStripMenuItem;
    }
}