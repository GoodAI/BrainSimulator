namespace GoodAI.BrainSimulator.Forms
{
    partial class DashboardGroupNameDialog
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
            this.groupNameText = new System.Windows.Forms.TextBox();
            this.groupNameLabel = new System.Windows.Forms.Label();
            this.okButton = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // groupNameText
            // 
            this.groupNameText.Location = new System.Drawing.Point(86, 6);
            this.groupNameText.Name = "groupNameText";
            this.groupNameText.Size = new System.Drawing.Size(127, 20);
            this.groupNameText.TabIndex = 0;
            this.groupNameText.TextChanged += new System.EventHandler(this.groupNameText_TextChanged);
            this.groupNameText.KeyUp += new System.Windows.Forms.KeyEventHandler(this.groupNameText_KeyUp);
            // 
            // groupNameLabel
            // 
            this.groupNameLabel.AutoSize = true;
            this.groupNameLabel.Location = new System.Drawing.Point(12, 9);
            this.groupNameLabel.Name = "groupNameLabel";
            this.groupNameLabel.Size = new System.Drawing.Size(68, 13);
            this.groupNameLabel.TabIndex = 1;
            this.groupNameLabel.Text = "Group name:";
            // 
            // okButton
            // 
            this.okButton.Location = new System.Drawing.Point(219, 4);
            this.okButton.Name = "okButton";
            this.okButton.Size = new System.Drawing.Size(75, 23);
            this.okButton.TabIndex = 2;
            this.okButton.Text = "Ok";
            this.okButton.UseVisualStyleBackColor = true;
            this.okButton.Click += new System.EventHandler(this.okButton_Click);
            // 
            // DashboardGroupNameDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(304, 37);
            this.Controls.Add(this.okButton);
            this.Controls.Add(this.groupNameLabel);
            this.Controls.Add(this.groupNameText);
            this.MaximumSize = new System.Drawing.Size(320, 75);
            this.MinimumSize = new System.Drawing.Size(320, 75);
            this.Name = "DashboardGroupNameDialog";
            this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
            this.Text = "Group name";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox groupNameText;
        private System.Windows.Forms.Label groupNameLabel;
        private System.Windows.Forms.Button okButton;
    }
}