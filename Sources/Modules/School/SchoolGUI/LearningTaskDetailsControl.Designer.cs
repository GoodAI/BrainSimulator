namespace GoodAI.School.GUI
{
    partial class LearningTaskDetailsControl
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

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.labelDescriptionControl1 = new GoodAI.School.GUI.LabelDescriptionControl();
            this.SuspendLayout();
            // 
            // labelDescriptionControl1
            // 
            this.labelDescriptionControl1.Location = new System.Drawing.Point(4, 4);
            this.labelDescriptionControl1.Name = "labelDescriptionControl1";
            this.labelDescriptionControl1.Size = new System.Drawing.Size(200, 187);
            this.labelDescriptionControl1.TabIndex = 0;
            this.labelDescriptionControl1.Type = null;
            // 
            // LearningTaskDetailsControl
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.labelDescriptionControl1);
            this.Name = "LearningTaskDetailsControl";
            this.Size = new System.Drawing.Size(212, 342);
            this.ResumeLayout(false);

        }

        #endregion

        private LabelDescriptionControl labelDescriptionControl1;


    }
}
