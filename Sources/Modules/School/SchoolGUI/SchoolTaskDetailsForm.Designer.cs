namespace GoodAI.School.GUI
{
    partial class SchoolTaskDetailsForm
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
            this.learningTaskDescription = new GoodAI.School.GUI.LearningTaskBrowser();
            this.SuspendLayout();
            // 
            // learningTaskDescription
            // 
            this.learningTaskDescription.Dock = System.Windows.Forms.DockStyle.Fill;
            this.learningTaskDescription.LearningTaskType = null;
            this.learningTaskDescription.Location = new System.Drawing.Point(0, 0);
            this.learningTaskDescription.MinimumSize = new System.Drawing.Size(20, 20);
            this.learningTaskDescription.Name = "learningTaskDescription";
            this.learningTaskDescription.Size = new System.Drawing.Size(819, 506);
            this.learningTaskDescription.TabIndex = 0;
            // 
            // SchoolTaskDetailsForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(819, 506);
            this.Controls.Add(this.learningTaskDescription);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.HideOnClose = true;
            this.Name = "SchoolTaskDetailsForm";
            this.Text = "SchoolTaskDetailsForm";
            this.ResumeLayout(false);

        }

        #endregion

        private LearningTaskBrowser learningTaskDescription;



    }
}