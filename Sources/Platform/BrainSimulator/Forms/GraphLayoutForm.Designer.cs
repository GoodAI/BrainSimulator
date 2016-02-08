using System;
using System.Windows.Forms;
using GoodAI.BrainSimulator.Nodes;
using GoodAI.BrainSimulator.Utils;
using Graph;

namespace GoodAI.BrainSimulator.Forms
{
    partial class GraphLayoutForm
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
            Graph.Compatibility.AlwaysCompatible alwaysCompatible1 = new Graph.Compatibility.AlwaysCompatible();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(GraphLayoutForm));
            this.nodesToolStrip = new System.Windows.Forms.ToolStrip();
            this.worldButtonPanel = new System.Windows.Forms.Panel();
            this.worldButton = new System.Windows.Forms.PictureBox();
            this.contextMenuStrip = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.removeNodeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.desktopContextMenuStrip = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.searchTextBox = new GoodAI.BrainSimulator.Utils.CueToolStripTextBox();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.itemToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.topToolStrip = new System.Windows.Forms.ToolStrip();
            this.updateModelButton = new System.Windows.Forms.ToolStripButton();
            this.zoomToFitButton = new System.Windows.Forms.ToolStripButton();
            this.groupButtonPanel = new System.Windows.Forms.Panel();
            this.groupButton = new System.Windows.Forms.PictureBox();
            this.Desktop = new Graph.GraphControl();
            this.nodeContextMenuStrip = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.openEditorToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openGroupToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.worldButtonPanel.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.worldButton)).BeginInit();
            this.contextMenuStrip.SuspendLayout();
            this.desktopContextMenuStrip.SuspendLayout();
            this.topToolStrip.SuspendLayout();
            this.groupButtonPanel.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.groupButton)).BeginInit();
            this.nodeContextMenuStrip.SuspendLayout();
            this.SuspendLayout();
            // 
            // nodesToolStrip
            // 
            this.nodesToolStrip.AllowDrop = true;
            this.nodesToolStrip.AutoSize = false;
            this.nodesToolStrip.BackColor = System.Drawing.SystemColors.ControlLight;
            this.nodesToolStrip.Dock = System.Windows.Forms.DockStyle.Left;
            this.nodesToolStrip.GripStyle = System.Windows.Forms.ToolStripGripStyle.Hidden;
            this.nodesToolStrip.ImageScalingSize = new System.Drawing.Size(32, 32);
            this.nodesToolStrip.Location = new System.Drawing.Point(0, 0);
            this.nodesToolStrip.Name = "nodesToolStrip";
            this.nodesToolStrip.Size = new System.Drawing.Size(41, 455);
            this.nodesToolStrip.TabIndex = 0;
            this.nodesToolStrip.Text = "toolStrip1";
            this.nodesToolStrip.DragDrop += new System.Windows.Forms.DragEventHandler(this.nodesToolStrip_DragDrop);
            this.nodesToolStrip.DragEnter += new System.Windows.Forms.DragEventHandler(this.nodesToolStrip_DragEnter);
            // 
            // worldButtonPanel
            // 
            this.worldButtonPanel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.worldButtonPanel.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.worldButtonPanel.Controls.Add(this.worldButton);
            this.worldButtonPanel.Location = new System.Drawing.Point(502, 411);
            this.worldButtonPanel.Name = "worldButtonPanel";
            this.worldButtonPanel.Size = new System.Drawing.Size(36, 35);
            this.worldButtonPanel.TabIndex = 5;
            // 
            // worldButton
            // 
            this.worldButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.world;
            this.worldButton.Location = new System.Drawing.Point(3, 3);
            this.worldButton.Name = "worldButton";
            this.worldButton.Size = new System.Drawing.Size(32, 32);
            this.worldButton.TabIndex = 0;
            this.worldButton.TabStop = false;
            this.worldButton.Click += new System.EventHandler(this.worldButton_Click);
            // 
            // contextMenuStrip
            // 
            this.contextMenuStrip.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.removeNodeToolStripMenuItem});
            this.contextMenuStrip.Name = "contextMenuStrip";
            this.contextMenuStrip.Size = new System.Drawing.Size(148, 26);
            // 
            // removeNodeToolStripMenuItem
            // 
            this.removeNodeToolStripMenuItem.Name = "removeNodeToolStripMenuItem";
            this.removeNodeToolStripMenuItem.Size = new System.Drawing.Size(147, 22);
            this.removeNodeToolStripMenuItem.Text = "Remove node";
            this.removeNodeToolStripMenuItem.Click += new System.EventHandler(this.removeNodeToolStripMenuItem_Click);
            // 
            // desktopContextMenuStrip
            // 
            this.desktopContextMenuStrip.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.searchTextBox,
            this.toolStripSeparator1,
            this.itemToolStripMenuItem});
            this.desktopContextMenuStrip.Name = "desktopContextMenuStrip";
            this.desktopContextMenuStrip.Size = new System.Drawing.Size(161, 57);
            this.desktopContextMenuStrip.Opened += new System.EventHandler(this.desktopContextMenuStrip_Opened);
            // 
            // searchTextBox
            // 
            this.searchTextBox.CueText = "Search...";
            this.searchTextBox.Name = "searchTextBox";
            this.searchTextBox.ShowCueTextWithFocus = true;
            this.searchTextBox.Size = new System.Drawing.Size(100, 23);
            this.searchTextBox.ToolTipText = "Search...";
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            this.toolStripSeparator1.Size = new System.Drawing.Size(157, 6);
            // 
            // itemToolStripMenuItem
            // 
            this.itemToolStripMenuItem.Name = "itemToolStripMenuItem";
            this.itemToolStripMenuItem.Size = new System.Drawing.Size(160, 22);
            this.itemToolStripMenuItem.Text = "Item";
            // 
            // topToolStrip
            // 
            this.topToolStrip.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.topToolStrip.AutoSize = false;
            this.topToolStrip.Dock = System.Windows.Forms.DockStyle.None;
            this.topToolStrip.GripStyle = System.Windows.Forms.ToolStripGripStyle.Hidden;
            this.topToolStrip.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.updateModelButton,
            this.zoomToFitButton});
            this.topToolStrip.Location = new System.Drawing.Point(494, 4);
            this.topToolStrip.Name = "topToolStrip";
            this.topToolStrip.Size = new System.Drawing.Size(45, 22);
            this.topToolStrip.TabIndex = 6;
            // 
            // updateModelButton
            // 
            this.updateModelButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.updateModelButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.Restart_6322;
            this.updateModelButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.updateModelButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.updateModelButton.Margin = new System.Windows.Forms.Padding(2, 1, 0, 2);
            this.updateModelButton.Name = "updateModelButton";
            this.updateModelButton.Size = new System.Drawing.Size(23, 19);
            this.updateModelButton.Text = "Update Memory Blocks";
            this.updateModelButton.Click += new System.EventHandler(this.updateModelButton_Click);
            // 
            // zoomToFitButton
            // 
            this.zoomToFitButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.zoomToFitButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.ZoomToFit;
            this.zoomToFitButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.zoomToFitButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.zoomToFitButton.Name = "zoomToFitButton";
            this.zoomToFitButton.Size = new System.Drawing.Size(23, 20);
            this.zoomToFitButton.Text = "Zoom To Fit";
            this.zoomToFitButton.Click += new System.EventHandler(this.zoomButton_Click);
            // 
            // groupButtonPanel
            // 
            this.groupButtonPanel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.groupButtonPanel.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.groupButtonPanel.Controls.Add(this.groupButton);
            this.groupButtonPanel.Location = new System.Drawing.Point(460, 411);
            this.groupButtonPanel.Name = "groupButtonPanel";
            this.groupButtonPanel.Size = new System.Drawing.Size(36, 35);
            this.groupButtonPanel.TabIndex = 6;
            // 
            // groupButton
            // 
            this.groupButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.group;
            this.groupButton.Location = new System.Drawing.Point(3, 3);
            this.groupButton.Name = "groupButton";
            this.groupButton.Size = new System.Drawing.Size(32, 32);
            this.groupButton.TabIndex = 0;
            this.groupButton.TabStop = false;
            this.groupButton.Click += new System.EventHandler(this.groupButton_Click);
            // 
            // Desktop
            // 
            this.Desktop.AllowDrop = true;
            this.Desktop.BackColor = System.Drawing.SystemColors.ControlDark;
            this.Desktop.CompatibilityStrategy = alwaysCompatible1;
            this.Desktop.ConnectorSafeBounds = 4;
            this.Desktop.Dock = System.Windows.Forms.DockStyle.Fill;
            this.Desktop.FocusElement = null;
            this.Desktop.HighlightCompatible = true;
            this.Desktop.LargeGridStep = 128F;
            this.Desktop.LargeStepGridColor = System.Drawing.Color.FromArgb(((int)(((byte)(180)))), ((int)(((byte)(180)))), ((int)(((byte)(180)))));
            this.Desktop.Location = new System.Drawing.Point(41, 0);
            this.Desktop.Name = "Desktop";
            this.Desktop.ShowLabels = true;
            this.Desktop.Size = new System.Drawing.Size(502, 455);
            this.Desktop.SmallGridStep = 16F;
            this.Desktop.SmallStepGridColor = System.Drawing.Color.DarkGray;
            this.Desktop.TabIndex = 1;
            this.Desktop.Text = "desktop";
            this.Desktop.FocusChanged += new System.EventHandler<Graph.ElementEventArgs>(this.desktop_FocusChanged);
            this.Desktop.NodeRemoving += new System.EventHandler<Graph.AcceptNodeEventArgs>(this.Desktop_NodeRemoving);
            this.Desktop.NodeRemoved += new System.EventHandler<Graph.NodeEventArgs>(this.Desktop_NodeRemoved);
            this.Desktop.ShowElementMenu += new System.EventHandler<Graph.AcceptElementLocationEventArgs>(this.Desktop_ShowElementMenu);
            this.Desktop.ConnectionAdding += new System.EventHandler<Graph.AcceptNodeConnectionEventArgs>(this.Desktop_ConnectionAdding);
            this.Desktop.ConnectionRemoving += new System.EventHandler<Graph.AcceptNodeConnectionEventArgs>(this.Desktop_ConnectionRemoving);
            this.Desktop.PositionChanged += new System.EventHandler<Graph.PositionChangedEventArgs>(this.Desktop_PositionChanged);
            this.Desktop.DoubleClick += new System.EventHandler(this.Desktop_DoubleClick);
            this.Desktop.MouseDown += new System.Windows.Forms.MouseEventHandler(this.Desktop_MouseDown);
            this.Desktop.MouseEnter += new System.EventHandler(this.desktop_MouseEnter);
            this.Desktop.MouseLeave += new System.EventHandler(this.desktop_MouseLeave);
            // 
            // nodeContextMenuStrip
            // 
            this.nodeContextMenuStrip.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.openEditorToolStripMenuItem,
            this.openGroupToolStripMenuItem});
            this.nodeContextMenuStrip.Name = "nodeContextMenuStrip";
            this.nodeContextMenuStrip.Size = new System.Drawing.Size(153, 70);
            // 
            // openEditorToolStripMenuItem
            // 
            this.openEditorToolStripMenuItem.Image = global::GoodAI.BrainSimulator.Properties.Resources.text_12xMD;
            this.openEditorToolStripMenuItem.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.openEditorToolStripMenuItem.Name = "openEditorToolStripMenuItem";
            this.openEditorToolStripMenuItem.Size = new System.Drawing.Size(152, 22);
            this.openEditorToolStripMenuItem.Text = "Open Editor";
            this.openEditorToolStripMenuItem.Click += new System.EventHandler(this.openEditorToolStripMenuItem_Click);
            // 
            // openGroupToolStripMenuItem
            // 
            this.openGroupToolStripMenuItem.Image = global::GoodAI.BrainSimulator.Properties.Resources.Diagram_16XMD;
            this.openGroupToolStripMenuItem.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.openGroupToolStripMenuItem.Name = "openGroupToolStripMenuItem";
            this.openGroupToolStripMenuItem.Size = new System.Drawing.Size(152, 22);
            this.openGroupToolStripMenuItem.Text = "Open Group";
            this.openGroupToolStripMenuItem.Click += new System.EventHandler(this.openGroupToolStripMenuItem_Click);
            // 
            // GraphLayoutForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(543, 455);
            this.CloseButtonVisible = false;
            this.Controls.Add(this.groupButtonPanel);
            this.Controls.Add(this.topToolStrip);
            this.Controls.Add(this.worldButtonPanel);
            this.Controls.Add(this.Desktop);
            this.Controls.Add(this.nodesToolStrip);
            this.DockAreas = WeifenLuo.WinFormsUI.Docking.DockAreas.Document;
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "GraphLayoutForm";
            this.Text = "Desktop";
            this.Activated += new System.EventHandler(this.GraphLayoutForm_Enter);
            this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.GraphLayoutForm_FormClosed);
            this.Load += new System.EventHandler(this.GraphLayoutForm_Load);
            this.Shown += new System.EventHandler(this.GraphLayoutForm_Shown);
            this.Enter += new System.EventHandler(this.GraphLayoutForm_Enter);
            this.worldButtonPanel.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.worldButton)).EndInit();
            this.contextMenuStrip.ResumeLayout(false);
            this.desktopContextMenuStrip.ResumeLayout(false);
            this.desktopContextMenuStrip.PerformLayout();
            this.topToolStrip.ResumeLayout(false);
            this.topToolStrip.PerformLayout();
            this.groupButtonPanel.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.groupButton)).EndInit();
            this.nodeContextMenuStrip.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        #endregion

        public Graph.GraphControl Desktop;
        private System.Windows.Forms.ToolStrip nodesToolStrip;
        private System.Windows.Forms.Panel worldButtonPanel;
        private System.Windows.Forms.PictureBox worldButton;
        private System.Windows.Forms.ContextMenuStrip contextMenuStrip;
        private System.Windows.Forms.ToolStripMenuItem removeNodeToolStripMenuItem;
        private System.Windows.Forms.ToolStripButton updateModelButton;
        private System.Windows.Forms.ToolStripButton zoomToFitButton;
        private System.Windows.Forms.ToolStrip topToolStrip;
        private System.Windows.Forms.Panel groupButtonPanel;
        private System.Windows.Forms.PictureBox groupButton;
        private System.Windows.Forms.ContextMenuStrip desktopContextMenuStrip;
        private CueToolStripTextBox searchTextBox;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripMenuItem itemToolStripMenuItem;
        private ContextMenuStrip nodeContextMenuStrip;
        private ToolStripMenuItem openEditorToolStripMenuItem;
        private ToolStripMenuItem openGroupToolStripMenuItem;

    }
}