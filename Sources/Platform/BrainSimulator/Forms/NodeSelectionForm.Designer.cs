namespace GoodAI.BrainSimulator.Forms
{
    partial class NodeSelectionForm
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
            this.nodeImageList = new System.Windows.Forms.ImageList(this.components);
            this.acceptButton = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.cancelButton = new System.Windows.Forms.Button();
            this.nodeListView = new GoodAI.BrainSimulator.Forms.NodeSelectionForm.MyListView();
            this.nameColumn = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.authorcolumn = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.statusColumn = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.summaryColumn = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.searchTextBox = new GoodAI.BrainSimulator.Forms.NodeSelectionForm.CueTextBox();
            this.nodesSplitContainer = new System.Windows.Forms.SplitContainer();
            this.filterList = new System.Windows.Forms.ListView();
            this.categoryHeader = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.filterImageList = new System.Windows.Forms.ImageList(this.components);
            ((System.ComponentModel.ISupportInitialize)(this.nodesSplitContainer)).BeginInit();
            this.nodesSplitContainer.Panel1.SuspendLayout();
            this.nodesSplitContainer.Panel2.SuspendLayout();
            this.nodesSplitContainer.SuspendLayout();
            this.SuspendLayout();
            // 
            // nodeImageList
            // 
            this.nodeImageList.ColorDepth = System.Windows.Forms.ColorDepth.Depth32Bit;
            this.nodeImageList.ImageSize = new System.Drawing.Size(36, 32);
            this.nodeImageList.TransparentColor = System.Drawing.Color.Transparent;
            // 
            // acceptButton
            // 
            this.acceptButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.acceptButton.Location = new System.Drawing.Point(930, 681);
            this.acceptButton.Name = "acceptButton";
            this.acceptButton.Size = new System.Drawing.Size(116, 23);
            this.acceptButton.TabIndex = 2;
            this.acceptButton.Text = "OK";
            this.acceptButton.UseVisualStyleBackColor = true;
            this.acceptButton.Click += new System.EventHandler(this.acceptButton_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(6, 16);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(144, 13);
            this.label1.TabIndex = 4;
            this.label1.Text = "Select nodes for side toolbar:";
            // 
            // cancelButton
            // 
            this.cancelButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.cancelButton.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.cancelButton.Location = new System.Drawing.Point(808, 681);
            this.cancelButton.Name = "cancelButton";
            this.cancelButton.Size = new System.Drawing.Size(116, 23);
            this.cancelButton.TabIndex = 3;
            this.cancelButton.Text = "Cancel";
            this.cancelButton.UseVisualStyleBackColor = true;
            this.cancelButton.Click += new System.EventHandler(this.cancelButton_Click);
            // 
            // nodeListView
            // 
            this.nodeListView.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.nodeListView.CheckBoxes = true;
            this.nodeListView.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.nameColumn,
            this.authorcolumn,
            this.statusColumn,
            this.summaryColumn});
            this.nodeListView.FullRowSelect = true;
            this.nodeListView.GridLines = true;
            this.nodeListView.Location = new System.Drawing.Point(0, 0);
            this.nodeListView.Margin = new System.Windows.Forms.Padding(0);
            this.nodeListView.MultiSelect = false;
            this.nodeListView.Name = "nodeListView";
            this.nodeListView.OwnerDraw = true;
            this.nodeListView.ShowItemToolTips = true;
            this.nodeListView.Size = new System.Drawing.Size(781, 640);
            this.nodeListView.SmallImageList = this.nodeImageList;
            this.nodeListView.TabIndex = 1;
            this.nodeListView.UseCompatibleStateImageBehavior = false;
            this.nodeListView.View = System.Windows.Forms.View.Details;
            this.nodeListView.DrawColumnHeader += new System.Windows.Forms.DrawListViewColumnHeaderEventHandler(this.nodeListView_DrawColumnHeader);
            this.nodeListView.DrawItem += new System.Windows.Forms.DrawListViewItemEventHandler(this.nodeListView_DrawItem);
            this.nodeListView.DrawSubItem += new System.Windows.Forms.DrawListViewSubItemEventHandler(this.nodeListView_DrawSubItem);
            this.nodeListView.ItemCheck += new System.Windows.Forms.ItemCheckEventHandler(this.nodeListView_ItemCheck);
            // 
            // nameColumn
            // 
            this.nameColumn.Text = "Name";
            this.nameColumn.Width = 220;
            // 
            // authorcolumn
            // 
            this.authorcolumn.Text = "Author";
            this.authorcolumn.Width = 110;
            // 
            // statusColumn
            // 
            this.statusColumn.Text = "Status";
            this.statusColumn.Width = 150;
            // 
            // summaryColumn
            // 
            this.summaryColumn.Text = "Summary";
            this.summaryColumn.Width = 400;
            // 
            // searchTextBox
            // 
            this.searchTextBox.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.searchTextBox.Cue = "Search...";
            this.searchTextBox.Location = new System.Drawing.Point(766, 9);
            this.searchTextBox.Name = "searchTextBox";
            this.searchTextBox.Size = new System.Drawing.Size(279, 20);
            this.searchTextBox.TabIndex = 0;
            this.searchTextBox.TextChanged += new System.EventHandler(this.searchTextBox_TextChanged);
            // 
            // nodesSplitContainer
            // 
            this.nodesSplitContainer.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.nodesSplitContainer.Location = new System.Drawing.Point(6, 32);
            this.nodesSplitContainer.Margin = new System.Windows.Forms.Padding(0, 3, 0, 3);
            this.nodesSplitContainer.Name = "nodesSplitContainer";
            // 
            // nodesSplitContainer.Panel1
            // 
            this.nodesSplitContainer.Panel1.Controls.Add(this.filterList);
            // 
            // nodesSplitContainer.Panel2
            // 
            this.nodesSplitContainer.Panel2.Controls.Add(this.nodeListView);
            this.nodesSplitContainer.Size = new System.Drawing.Size(1041, 643);
            this.nodesSplitContainer.SplitterDistance = 254;
            this.nodesSplitContainer.TabIndex = 5;
            // 
            // filterList
            // 
            this.filterList.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.filterList.BackColor = System.Drawing.SystemColors.Menu;
            this.filterList.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.filterList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.categoryHeader});
            this.filterList.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.None;
            this.filterList.Location = new System.Drawing.Point(0, 0);
            this.filterList.Margin = new System.Windows.Forms.Padding(0);
            this.filterList.MultiSelect = false;
            this.filterList.Name = "filterList";
            this.filterList.Size = new System.Drawing.Size(253, 640);
            this.filterList.SmallImageList = this.filterImageList;
            this.filterList.TabIndex = 0;
            this.filterList.UseCompatibleStateImageBehavior = false;
            this.filterList.View = System.Windows.Forms.View.Details;
            this.filterList.ItemSelectionChanged += new System.Windows.Forms.ListViewItemSelectionChangedEventHandler(this.nodeFilterList_ItemSelectionChanged);
            // 
            // categoryHeader
            // 
            this.categoryHeader.Text = "Category";
            this.categoryHeader.Width = 200;
            // 
            // filterImageList
            // 
            this.filterImageList.ColorDepth = System.Windows.Forms.ColorDepth.Depth32Bit;
            this.filterImageList.ImageSize = new System.Drawing.Size(36, 32);
            this.filterImageList.TransparentColor = System.Drawing.Color.Transparent;
            // 
            // NodeSelectionForm
            // 
            this.AcceptButton = this.acceptButton;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.cancelButton;
            this.ClientSize = new System.Drawing.Size(1053, 707);
            this.Controls.Add(this.nodesSplitContainer);
            this.Controls.Add(this.searchTextBox);
            this.Controls.Add(this.cancelButton);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.acceptButton);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.SizableToolWindow;
            this.Name = "NodeSelectionForm";
            this.Text = "Configure Node Selection";
            this.Load += new System.EventHandler(this.NodeSelectionForm_Load);
            this.nodesSplitContainer.Panel1.ResumeLayout(false);
            this.nodesSplitContainer.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.nodesSplitContainer)).EndInit();
            this.nodesSplitContainer.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button acceptButton;
        private System.Windows.Forms.ImageList nodeImageList;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button cancelButton;
        private System.Windows.Forms.ColumnHeader nameColumn;
        private System.Windows.Forms.ColumnHeader authorcolumn;
        private System.Windows.Forms.ColumnHeader summaryColumn;
        private System.Windows.Forms.ColumnHeader statusColumn;
        private NodeSelectionForm.MyListView nodeListView;
        private NodeSelectionForm.CueTextBox searchTextBox;
        private System.Windows.Forms.SplitContainer nodesSplitContainer;
        private System.Windows.Forms.ListView filterList;
        private System.Windows.Forms.ColumnHeader categoryHeader;
        private System.Windows.Forms.ImageList filterImageList;
    }
}