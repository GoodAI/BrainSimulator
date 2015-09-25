namespace GoodAI.BrainSimulator.Forms
{
    partial class TextEditForm
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(TextEditForm));
            this.scintilla = new ScintillaNET.Scintilla();
            this.SuspendLayout();
            // 
            // scintilla
            // 
            this.scintilla.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.scintilla.Dock = System.Windows.Forms.DockStyle.Fill;
            this.scintilla.IndentWidth = 4;
            this.scintilla.Location = new System.Drawing.Point(0, 0);
            this.scintilla.Name = "scintilla";
            this.scintilla.Size = new System.Drawing.Size(284, 262);
            this.scintilla.TabIndex = 1;
            this.scintilla.Text = "scintilla editor";
            this.scintilla.UseTabs = false;
            this.scintilla.CharAdded += new System.EventHandler<ScintillaNET.CharAddedEventArgs>(this.scintilla_CharAdded);
            // 
            // TextEditForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(284, 262);
            this.CloseButtonVisible = false;
            this.Controls.Add(this.scintilla);
            this.DockAreas = ((WeifenLuo.WinFormsUI.Docking.DockAreas)((WeifenLuo.WinFormsUI.Docking.DockAreas.Float | WeifenLuo.WinFormsUI.Docking.DockAreas.Document)));
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "TextEditForm";
            this.Text = "Code Editor";
            this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.TextEditForm_FormClosed);
            this.Load += new System.EventHandler(this.TextEditForm_Load);
            this.Enter += new System.EventHandler(this.TextEditForm_Enter);
            this.ResumeLayout(false);

        }

        #endregion

        private ScintillaNET.Scintilla scintilla;

    }
}