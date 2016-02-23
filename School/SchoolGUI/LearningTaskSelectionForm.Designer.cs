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
            this.okButton = new System.Windows.Forms.Button();
            this.cancelButton = new System.Windows.Forms.Button();
            this.learningTaskList = new System.Windows.Forms.CheckedListBox();
            this.learningTaskDescription = new System.Windows.Forms.WebBrowser();
            this.label1 = new System.Windows.Forms.Label();
            this.worldList = new System.Windows.Forms.ComboBox();
            this.SuspendLayout();
            // 
            // okButton
            // 
            this.okButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.okButton.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.okButton.Location = new System.Drawing.Point(753, 434);
            this.okButton.Name = "okButton";
            this.okButton.Size = new System.Drawing.Size(75, 23);
            this.okButton.TabIndex = 0;
            this.okButton.Text = "Add";
            this.okButton.UseVisualStyleBackColor = true;
            this.okButton.Click += new System.EventHandler(this.okButton_Click);
            // 
            // cancelButton
            // 
            this.cancelButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.cancelButton.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.cancelButton.Location = new System.Drawing.Point(650, 434);
            this.cancelButton.Name = "cancelButton";
            this.cancelButton.Size = new System.Drawing.Size(75, 23);
            this.cancelButton.TabIndex = 1;
            this.cancelButton.Text = "Cancel";
            this.cancelButton.UseVisualStyleBackColor = true;
            this.cancelButton.Click += new System.EventHandler(this.cancelButton_Click);
            // 
            // learningTaskList
            // 
            this.learningTaskList.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.learningTaskList.CheckOnClick = true;
            this.learningTaskList.FormattingEnabled = true;
            this.learningTaskList.Location = new System.Drawing.Point(-1, 45);
            this.learningTaskList.Name = "learningTaskList";
            this.learningTaskList.Size = new System.Drawing.Size(224, 364);
            this.learningTaskList.TabIndex = 2;
            this.learningTaskList.ItemCheck += new System.Windows.Forms.ItemCheckEventHandler(this.learningTaskList_ItemCheck);
            this.learningTaskList.MouseClick += new System.Windows.Forms.MouseEventHandler(this.learningTaskList_MouseClick);
            this.learningTaskList.SelectedValueChanged += new System.EventHandler(this.learningTaskList_SelectedValueChanged);
            // 
            // learningTaskDescription
            // 
            this.learningTaskDescription.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.learningTaskDescription.Location = new System.Drawing.Point(223, 45);
            this.learningTaskDescription.MinimumSize = new System.Drawing.Size(20, 20);
            this.learningTaskDescription.Name = "learningTaskDescription";
            this.learningTaskDescription.Size = new System.Drawing.Size(625, 371);
            this.learningTaskDescription.TabIndex = 3;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(13, 12);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(38, 13);
            this.label1.TabIndex = 5;
            this.label1.Text = "World:";
            // 
            // worldList
            // 
            this.worldList.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.worldList.FormattingEnabled = true;
            this.worldList.Location = new System.Drawing.Point(57, 8);
            this.worldList.Name = "worldList";
            this.worldList.Size = new System.Drawing.Size(174, 21);
            this.worldList.TabIndex = 7;
            this.worldList.SelectedIndexChanged += new System.EventHandler(this.worldList_SelectedIndexChanged);
            // 
            // LearningTaskSelectionForm
            // 
            this.AcceptButton = this.okButton;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.cancelButton;
            this.ClientSize = new System.Drawing.Size(849, 469);
            this.Controls.Add(this.worldList);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.learningTaskDescription);
            this.Controls.Add(this.learningTaskList);
            this.Controls.Add(this.cancelButton);
            this.Controls.Add(this.okButton);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.HideOnClose = true;
            this.Name = "LearningTaskSelectionForm";
            this.Text = "Learning Task Selection Form";
            this.Load += new System.EventHandler(this.LearningTaskSelectionForm_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button okButton;
        private System.Windows.Forms.Button cancelButton;
        private System.Windows.Forms.CheckedListBox learningTaskList;
        private System.Windows.Forms.WebBrowser learningTaskDescription;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.ComboBox worldList;
    }
}