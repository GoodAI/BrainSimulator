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
            this.learningTaskDetailsControl1 = new GoodAI.School.GUI.LearningTaskDetailsControl();
            this.SuspendLayout();
            //
            // learningTaskDetailsControl1
            //
            this.learningTaskDetailsControl1.Location = new System.Drawing.Point(12, 12);
            this.learningTaskDetailsControl1.Name = "learningTaskDetailsControl1";
            this.learningTaskDetailsControl1.Size = new System.Drawing.Size(232, 266);
            this.learningTaskDetailsControl1.TabIndex = 0;
            //
            // SchoolTaskDetailsForm
            //
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(310, 390);
            this.Controls.Add(this.learningTaskDetailsControl1);
            this.HideOnClose = true;
            this.Name = "SchoolTaskDetailsForm";
            this.Text = "SchoolTaskDetailsForm";
            this.ResumeLayout(false);

        }

        #endregion

        private LearningTaskDetailsControl learningTaskDetailsControl1;

    }
}