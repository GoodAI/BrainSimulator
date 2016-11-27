namespace MapsMerger
{
    partial class Form1
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.openFileDialogLoadComposition = new System.Windows.Forms.OpenFileDialog();
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.toolStripButtonLoadComposition = new System.Windows.Forms.ToolStripButton();
            this.toolStripButtonSaveComposition = new System.Windows.Forms.ToolStripButton();
            this.toolStripButtonMerge = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.toolStripButtonEditCompositionFile = new System.Windows.Forms.ToolStripButton();
            this.richTextBoxLog = new System.Windows.Forms.RichTextBox();
            this.saveFileDialogSaveComposition = new System.Windows.Forms.SaveFileDialog();
            this.saveFileDialogSaveMerged = new System.Windows.Forms.SaveFileDialog();
            this.toolStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // openFileDialogLoadComposition
            // 
            this.openFileDialogLoadComposition.FileName = "openFileDialog1";
            this.openFileDialogLoadComposition.FileOk += new System.ComponentModel.CancelEventHandler(this.openFileDialogLoadComposition_FileOk);
            // 
            // toolStrip1
            // 
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripButtonLoadComposition,
            this.toolStripButtonSaveComposition,
            this.toolStripButtonMerge,
            this.toolStripSeparator1,
            this.toolStripButtonEditCompositionFile});
            this.toolStrip1.Location = new System.Drawing.Point(0, 0);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(619, 25);
            this.toolStrip1.TabIndex = 0;
            this.toolStrip1.Text = "toolStrip1";
            // 
            // toolStripButtonLoadComposition
            // 
            this.toolStripButtonLoadComposition.Image = ((System.Drawing.Image)(resources.GetObject("toolStripButtonLoadComposition.Image")));
            this.toolStripButtonLoadComposition.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.toolStripButtonLoadComposition.Name = "toolStripButtonLoadComposition";
            this.toolStripButtonLoadComposition.Size = new System.Drawing.Size(125, 22);
            this.toolStripButtonLoadComposition.Text = "Load Composition";
            this.toolStripButtonLoadComposition.Click += new System.EventHandler(this.toolStripButton1_Click);
            // 
            // toolStripButtonSaveComposition
            // 
            this.toolStripButtonSaveComposition.Image = ((System.Drawing.Image)(resources.GetObject("toolStripButtonSaveComposition.Image")));
            this.toolStripButtonSaveComposition.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.toolStripButtonSaveComposition.Name = "toolStripButtonSaveComposition";
            this.toolStripButtonSaveComposition.Size = new System.Drawing.Size(120, 22);
            this.toolStripButtonSaveComposition.Text = "SaveComposition";
            this.toolStripButtonSaveComposition.Click += new System.EventHandler(this.toolStripButtonSaveComposition_Click);
            // 
            // toolStripButtonMerge
            // 
            this.toolStripButtonMerge.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.toolStripButtonMerge.Image = ((System.Drawing.Image)(resources.GetObject("toolStripButtonMerge.Image")));
            this.toolStripButtonMerge.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.toolStripButtonMerge.Name = "toolStripButtonMerge";
            this.toolStripButtonMerge.Size = new System.Drawing.Size(45, 22);
            this.toolStripButtonMerge.Text = "Merge";
            this.toolStripButtonMerge.Click += new System.EventHandler(this.toolStripButtonMerge_Click);
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            this.toolStripSeparator1.Size = new System.Drawing.Size(6, 25);
            // 
            // toolStripButtonEditCompositionFile
            // 
            this.toolStripButtonEditCompositionFile.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.toolStripButtonEditCompositionFile.Image = ((System.Drawing.Image)(resources.GetObject("toolStripButtonEditCompositionFile.Image")));
            this.toolStripButtonEditCompositionFile.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.toolStripButtonEditCompositionFile.Name = "toolStripButtonEditCompositionFile";
            this.toolStripButtonEditCompositionFile.Size = new System.Drawing.Size(135, 22);
            this.toolStripButtonEditCompositionFile.Text = "Composition File Editor";
            // 
            // richTextBoxLog
            // 
            this.richTextBoxLog.Location = new System.Drawing.Point(0, 29);
            this.richTextBoxLog.Name = "richTextBoxLog";
            this.richTextBoxLog.ReadOnly = true;
            this.richTextBoxLog.Size = new System.Drawing.Size(619, 503);
            this.richTextBoxLog.TabIndex = 1;
            this.richTextBoxLog.Text = "";
            // 
            // saveFileDialogSaveComposition
            // 
            this.saveFileDialogSaveComposition.DefaultExt = "xml";
            this.saveFileDialogSaveComposition.FileName = "composition";
            this.saveFileDialogSaveComposition.Filter = "XML files (*.xml)|*.xml";
            this.saveFileDialogSaveComposition.FileOk += new System.ComponentModel.CancelEventHandler(this.saveFileDialogSaveComposition_FileOk);
            // 
            // saveFileDialogSaveMerged
            // 
            this.saveFileDialogSaveMerged.FileOk += new System.ComponentModel.CancelEventHandler(this.saveFileDialogSaveMerged_FileOk);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(619, 544);
            this.Controls.Add(this.richTextBoxLog);
            this.Controls.Add(this.toolStrip1);
            this.Name = "Form1";
            this.Text = "Form1";
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.OpenFileDialog openFileDialogLoadComposition;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.ToolStripButton toolStripButtonLoadComposition;
        private System.Windows.Forms.RichTextBox richTextBoxLog;
        private System.Windows.Forms.ToolStripButton toolStripButtonMerge;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripButton toolStripButtonEditCompositionFile;
        private System.Windows.Forms.ToolStripButton toolStripButtonSaveComposition;
        private System.Windows.Forms.SaveFileDialog saveFileDialogSaveComposition;
        private System.Windows.Forms.SaveFileDialog saveFileDialogSaveMerged;
    }
}

