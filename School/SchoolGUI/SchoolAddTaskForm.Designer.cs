namespace GoodAI.School.GUI
{
    partial class SchoolAddTaskForm
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
            this.label1 = new System.Windows.Forms.Label();
            this.comboTasks = new System.Windows.Forms.ComboBox();
            this.label2 = new System.Windows.Forms.Label();
            this.comboWorlds = new System.Windows.Forms.ComboBox();
            this.btnAdd = new System.Windows.Forms.Button();
            this.SuspendLayout();
            //
            // label1
            //
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 9);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(66, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Choose task";
            //
            // comboTasks
            //
            this.comboTasks.FormattingEnabled = true;
            this.comboTasks.Location = new System.Drawing.Point(12, 25);
            this.comboTasks.Name = "comboTasks";
            this.comboTasks.Size = new System.Drawing.Size(123, 21);
            this.comboTasks.TabIndex = 1;
            this.comboTasks.SelectedIndexChanged += new System.EventHandler(this.comboTasks_SelectedIndexChanged);
            //
            // label2
            //
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(13, 53);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(71, 13);
            this.label2.TabIndex = 2;
            this.label2.Text = "Choose world";
            //
            // comboWorlds
            //
            this.comboWorlds.FormattingEnabled = true;
            this.comboWorlds.Location = new System.Drawing.Point(12, 69);
            this.comboWorlds.Name = "comboWorlds";
            this.comboWorlds.Size = new System.Drawing.Size(123, 21);
            this.comboWorlds.TabIndex = 3;
            //
            // btnAdd
            //
            this.btnAdd.Location = new System.Drawing.Point(37, 103);
            this.btnAdd.Name = "btnAdd";
            this.btnAdd.Size = new System.Drawing.Size(75, 23);
            this.btnAdd.TabIndex = 4;
            this.btnAdd.Text = "Add";
            this.btnAdd.UseVisualStyleBackColor = true;
            this.btnAdd.Click += new System.EventHandler(this.btnAdd_Click);
            //
            // SchoolAddTaskForm
            //
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(146, 138);
            this.Controls.Add(this.btnAdd);
            this.Controls.Add(this.comboWorlds);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.comboTasks);
            this.Controls.Add(this.label1);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Name = "SchoolAddTaskForm";
            this.Text = "SchoolAddTaskForm";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.ComboBox comboTasks;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.ComboBox comboWorlds;
        private System.Windows.Forms.Button btnAdd;
    }
}