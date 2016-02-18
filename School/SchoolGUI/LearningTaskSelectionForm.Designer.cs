namespace GoodAI.School.GUI
{
    partial class LearningTaskSelectionForm
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
            this.button1 = new System.Windows.Forms.Button();
            this.button2 = new System.Windows.Forms.Button();
            this.learningTaskList = new System.Windows.Forms.CheckedListBox();
            this.learningTaskDescription = new System.Windows.Forms.WebBrowser();
            this.label1 = new System.Windows.Forms.Label();
            this.worldList = new System.Windows.Forms.ListBox();
            this.SuspendLayout();
            // 
            // button1
            // 
            this.button1.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.button1.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.button1.Location = new System.Drawing.Point(753, 434);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(75, 23);
            this.button1.TabIndex = 0;
            this.button1.Text = "OK";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.okButton_Click);
            // 
            // button2
            // 
            this.button2.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.button2.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.button2.Location = new System.Drawing.Point(650, 434);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(75, 23);
            this.button2.TabIndex = 1;
            this.button2.Text = "Cancel";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.cancelButton_Click);
            // 
            // learningTaskList
            // 
            this.learningTaskList.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.learningTaskList.FormattingEnabled = true;
            this.learningTaskList.Location = new System.Drawing.Point(-1, 105);
            this.learningTaskList.Name = "learningTaskList";
            this.learningTaskList.Size = new System.Drawing.Size(224, 304);
            this.learningTaskList.TabIndex = 2;
            this.learningTaskList.SelectedValueChanged += new System.EventHandler(this.learningTaskList_SelectedValueChanged);
            // 
            // learningTaskDescription
            // 
            this.learningTaskDescription.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.learningTaskDescription.Location = new System.Drawing.Point(223, 105);
            this.learningTaskDescription.MinimumSize = new System.Drawing.Size(20, 20);
            this.learningTaskDescription.Name = "learningTaskDescription";
            this.learningTaskDescription.Size = new System.Drawing.Size(625, 311);
            this.learningTaskDescription.TabIndex = 3;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 26);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(38, 13);
            this.label1.TabIndex = 5;
            this.label1.Text = "World:";
            // 
            // worldList
            // 
            this.worldList.DisplayMember = "DisplayName";
            this.worldList.FormattingEnabled = true;
            this.worldList.Location = new System.Drawing.Point(56, 26);
            this.worldList.Name = "worldList";
            this.worldList.Size = new System.Drawing.Size(185, 69);
            this.worldList.TabIndex = 6;
            this.worldList.SelectedIndexChanged += new System.EventHandler(this.worldList_SelectedIndexChanged);
            // 
            // LearningTaskSelectionForm
            // 
            this.AcceptButton = this.button1;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.button2;
            this.ClientSize = new System.Drawing.Size(849, 469);
            this.Controls.Add(this.worldList);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.learningTaskDescription);
            this.Controls.Add(this.learningTaskList);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.button1);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.HideOnClose = true;
            this.Name = "LearningTaskSelectionForm";
            this.Text = "Learning Task Selection Form";
            this.Load += new System.EventHandler(this.LearningTaskSelectionForm_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.CheckedListBox learningTaskList;
        private System.Windows.Forms.WebBrowser learningTaskDescription;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.ListBox worldList;
    }
}