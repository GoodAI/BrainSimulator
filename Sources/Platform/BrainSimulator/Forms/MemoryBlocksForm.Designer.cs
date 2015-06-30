namespace GoodAI.BrainSimulator.Forms
{
    partial class MemoryBlocksForm
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
            this.listView = new System.Windows.Forms.ListView();
            this.nameColumn = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.sizeColumn = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.typeColumn = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.toolStrip = new System.Windows.Forms.ToolStrip();
            this.addObserverButton = new System.Windows.Forms.ToolStripButton();
            this.addPlotButton = new System.Windows.Forms.ToolStripButton();
            this.addMatrixObserver = new System.Windows.Forms.ToolStripButton();
            this.addSpikeObserver = new System.Windows.Forms.ToolStripButton();
            this.addHistogramObserver = new System.Windows.Forms.ToolStripButton();
            this.toolStrip.SuspendLayout();
            this.SuspendLayout();
            // 
            // listView
            // 
            this.listView.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.nameColumn,
            this.sizeColumn,
            this.typeColumn});
            this.listView.Dock = System.Windows.Forms.DockStyle.Fill;
            this.listView.FullRowSelect = true;
            this.listView.GridLines = true;
            this.listView.HideSelection = false;
            this.listView.Location = new System.Drawing.Point(0, 25);
            this.listView.MultiSelect = false;
            this.listView.Name = "listView";
            this.listView.ShowItemToolTips = true;
            this.listView.Size = new System.Drawing.Size(284, 237);
            this.listView.TabIndex = 0;
            this.listView.UseCompatibleStateImageBehavior = false;
            this.listView.View = System.Windows.Forms.View.Details;
            this.listView.SelectedIndexChanged += new System.EventHandler(this.listView_SelectedIndexChanged);
            this.listView.MouseDoubleClick += new System.Windows.Forms.MouseEventHandler(this.listView_MouseDoubleClick);
            // 
            // nameColumn
            // 
            this.nameColumn.Text = "Name";
            this.nameColumn.Width = 100;
            // 
            // sizeColumn
            // 
            this.sizeColumn.Text = "Size";
            this.sizeColumn.Width = 130;
            // 
            // typeColumn
            // 
            this.typeColumn.Text = "Type";
            // 
            // toolStrip
            // 
            this.toolStrip.BackColor = System.Drawing.SystemColors.Control;
            this.toolStrip.Enabled = false;
            this.toolStrip.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.addObserverButton,
            this.addPlotButton,
            this.addMatrixObserver,
            this.addSpikeObserver,
            this.addHistogramObserver});
            this.toolStrip.Location = new System.Drawing.Point(0, 0);
            this.toolStrip.Name = "toolStrip";
            this.toolStrip.Size = new System.Drawing.Size(284, 25);
            this.toolStrip.TabIndex = 1;
            this.toolStrip.Text = "toolStrip1";
            // 
            // addObserverButton
            // 
            this.addObserverButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.addObserverButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.add_observer;
            this.addObserverButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.addObserverButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.addObserverButton.Name = "addObserverButton";
            this.addObserverButton.Size = new System.Drawing.Size(23, 22);
            this.addObserverButton.Text = "Add Observer to Block";
            this.addObserverButton.Click += new System.EventHandler(this.addObserverButton_Click);
            // 
            // addPlotButton
            // 
            this.addPlotButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.addPlotButton.Image = global::GoodAI.BrainSimulator.Properties.Resources.add_plot;
            this.addPlotButton.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.addPlotButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.addPlotButton.Name = "addPlotButton";
            this.addPlotButton.Size = new System.Drawing.Size(23, 22);
            this.addPlotButton.Text = "Add Plot Observer to Block";
            this.addPlotButton.Click += new System.EventHandler(this.addPlotButton_Click);
            // 
            // addMatrixObserver
            // 
            this.addMatrixObserver.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.addMatrixObserver.Image = global::GoodAI.BrainSimulator.Properties.Resources.matrix;
            this.addMatrixObserver.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.addMatrixObserver.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.addMatrixObserver.Name = "addMatrixObserver";
            this.addMatrixObserver.Size = new System.Drawing.Size(23, 22);
            this.addMatrixObserver.Text = "Add Matrix Observer to Block";
            this.addMatrixObserver.Click += new System.EventHandler(this.addMatrixObserver_Click);
            // 
            // addSpikeObserver
            // 
            this.addSpikeObserver.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.addSpikeObserver.Image = global::GoodAI.BrainSimulator.Properties.Resources.spike;
            this.addSpikeObserver.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.addSpikeObserver.Name = "addSpikeObserver";
            this.addSpikeObserver.Size = new System.Drawing.Size(23, 22);
            this.addSpikeObserver.Text = "Spike observer";
            this.addSpikeObserver.Click += new System.EventHandler(this.addSpikeObserver_Click);
            // 
            // addHistogramObserver
            // 
            this.addHistogramObserver.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.addHistogramObserver.Image = global::GoodAI.BrainSimulator.Properties.Resources.histo;
            this.addHistogramObserver.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.addHistogramObserver.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.addHistogramObserver.Name = "addHistogramObserver";
            this.addHistogramObserver.Size = new System.Drawing.Size(23, 22);
            this.addHistogramObserver.Text = "Add Histogram to Block";
            this.addHistogramObserver.Click += new System.EventHandler(this.addHistogramObserver_Click);
            // 
            // MemoryBlocksForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(284, 262);
            this.Controls.Add(this.listView);
            this.Controls.Add(this.toolStrip);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.HideOnClose = true;
            this.Name = "MemoryBlocksForm";
            this.Text = "Memory Blocks";
            this.toolStrip.ResumeLayout(false);
            this.toolStrip.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ListView listView;
        private System.Windows.Forms.ColumnHeader nameColumn;
        private System.Windows.Forms.ColumnHeader sizeColumn;
        private System.Windows.Forms.ColumnHeader typeColumn;
        private System.Windows.Forms.ToolStrip toolStrip;
        private System.Windows.Forms.ToolStripButton addObserverButton;
        private System.Windows.Forms.ToolStripButton addPlotButton;
        private System.Windows.Forms.ToolStripButton addMatrixObserver;
        private System.Windows.Forms.ToolStripButton addSpikeObserver;
        private System.Windows.Forms.ToolStripButton addHistogramObserver;

    }
}