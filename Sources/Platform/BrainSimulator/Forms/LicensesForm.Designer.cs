namespace GoodAI.BrainSimulator.Forms
{
    partial class LicensesForm
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
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.licenseList = new System.Windows.Forms.ListBox();
            this.licenseText = new System.Windows.Forms.RichTextBox();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            this.SuspendLayout();
            // 
            // splitContainer1
            // 
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.Location = new System.Drawing.Point(0, 0);
            this.splitContainer1.Name = "splitContainer1";
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.Controls.Add(this.licenseList);
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.Controls.Add(this.licenseText);
            this.splitContainer1.Size = new System.Drawing.Size(979, 499);
            this.splitContainer1.SplitterDistance = 326;
            this.splitContainer1.TabIndex = 0;
            // 
            // licenseList
            // 
            this.licenseList.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.licenseList.FormattingEnabled = true;
            this.licenseList.Location = new System.Drawing.Point(3, 3);
            this.licenseList.Name = "licenseList";
            this.licenseList.Size = new System.Drawing.Size(320, 485);
            this.licenseList.TabIndex = 0;
            this.licenseList.SelectedIndexChanged += new System.EventHandler(this.licenseList_SelectedIndexChanged);
            // 
            // licenseText
            // 
            this.licenseText.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.licenseText.Location = new System.Drawing.Point(3, 3);
            this.licenseText.Name = "licenseText";
            this.licenseText.ReadOnly = true;
            this.licenseText.Size = new System.Drawing.Size(643, 484);
            this.licenseText.TabIndex = 0;
            this.licenseText.Text = "";
            // 
            // LicensesForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(979, 499);
            this.Controls.Add(this.splitContainer1);
            this.Name = "LicensesForm";
            this.ShowIcon = false;
            this.Text = "Licensing Information";
            this.Load += new System.EventHandler(this.LicensesForm_Load);
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.ListBox licenseList;
        private System.Windows.Forms.RichTextBox licenseText;
    }
}