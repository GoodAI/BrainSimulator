namespace DemoApplication
{
    partial class FormMain
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
            this.btnSerialize = new System.Windows.Forms.Button();
            this.rtbXMLOutput = new System.Windows.Forms.RichTextBox();
            this.btnDeserialize = new System.Windows.Forms.Button();
            this.lstSampleClasses = new System.Windows.Forms.ListBox();
            this.rtbParsingErrors = new System.Windows.Forms.RichTextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.btnSerializeToFile = new System.Windows.Forms.Button();
            this.btnDeserializeFromFile = new System.Windows.Forms.Button();
            this.label3 = new System.Windows.Forms.Label();
            this.comboPolicy = new System.Windows.Forms.ComboBox();
            this.label4 = new System.Windows.Forms.Label();
            this.comboErrorType = new System.Windows.Forms.ComboBox();
            this.label5 = new System.Windows.Forms.Label();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.label7 = new System.Windows.Forms.Label();
            this.comboOptions = new System.Windows.Forms.ComboBox();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.rtbDeserializeOutput = new System.Windows.Forms.RichTextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.panel1 = new System.Windows.Forms.Panel();
            this.numMaxRecursion = new System.Windows.Forms.NumericUpDown();
            this.label8 = new System.Windows.Forms.Label();
            this.groupBox1.SuspendLayout();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            this.panel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numMaxRecursion)).BeginInit();
            this.SuspendLayout();
            // 
            // btnSerialize
            // 
            this.btnSerialize.Location = new System.Drawing.Point(6, 19);
            this.btnSerialize.Name = "btnSerialize";
            this.btnSerialize.Size = new System.Drawing.Size(75, 23);
            this.btnSerialize.TabIndex = 0;
            this.btnSerialize.Text = "Serialize";
            this.btnSerialize.UseVisualStyleBackColor = true;
            this.btnSerialize.Click += new System.EventHandler(this.btnSerialize_Click);
            // 
            // rtbXMLOutput
            // 
            this.rtbXMLOutput.AcceptsTab = true;
            this.rtbXMLOutput.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.rtbXMLOutput.Font = new System.Drawing.Font("Courier New", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.rtbXMLOutput.Location = new System.Drawing.Point(216, 19);
            this.rtbXMLOutput.Name = "rtbXMLOutput";
            this.rtbXMLOutput.Size = new System.Drawing.Size(353, 310);
            this.rtbXMLOutput.TabIndex = 1;
            this.rtbXMLOutput.Text = "";
            this.rtbXMLOutput.WordWrap = false;
            // 
            // btnDeserialize
            // 
            this.btnDeserialize.Location = new System.Drawing.Point(87, 19);
            this.btnDeserialize.Name = "btnDeserialize";
            this.btnDeserialize.Size = new System.Drawing.Size(75, 23);
            this.btnDeserialize.TabIndex = 2;
            this.btnDeserialize.Text = "Deserialize";
            this.btnDeserialize.UseVisualStyleBackColor = true;
            this.btnDeserialize.Click += new System.EventHandler(this.btnDeserialize_Click);
            // 
            // lstSampleClasses
            // 
            this.lstSampleClasses.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.lstSampleClasses.FormattingEnabled = true;
            this.lstSampleClasses.Location = new System.Drawing.Point(9, 19);
            this.lstSampleClasses.Name = "lstSampleClasses";
            this.lstSampleClasses.Size = new System.Drawing.Size(201, 472);
            this.lstSampleClasses.TabIndex = 3;
            this.lstSampleClasses.MouseDoubleClick += new System.Windows.Forms.MouseEventHandler(this.lstSampleClasses_MouseDoubleClick);
            // 
            // rtbParsingErrors
            // 
            this.rtbParsingErrors.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.rtbParsingErrors.Font = new System.Drawing.Font("Courier New", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.rtbParsingErrors.Location = new System.Drawing.Point(216, 348);
            this.rtbParsingErrors.Name = "rtbParsingErrors";
            this.rtbParsingErrors.ReadOnly = true;
            this.rtbParsingErrors.Size = new System.Drawing.Size(353, 144);
            this.rtbParsingErrors.TabIndex = 4;
            this.rtbParsingErrors.Text = "";
            this.rtbParsingErrors.WordWrap = false;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(216, 3);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(322, 13);
            this.label1.TabIndex = 5;
            this.label1.Text = "XML Serialized (Modify and press Deserialize to see what happens)";
            // 
            // label2
            // 
            this.label2.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(216, 332);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(75, 13);
            this.label2.TabIndex = 6;
            this.label2.Text = "Parsing Errors:";
            // 
            // btnSerializeToFile
            // 
            this.btnSerializeToFile.Location = new System.Drawing.Point(168, 19);
            this.btnSerializeToFile.Name = "btnSerializeToFile";
            this.btnSerializeToFile.Size = new System.Drawing.Size(110, 23);
            this.btnSerializeToFile.TabIndex = 7;
            this.btnSerializeToFile.Text = "Serialize To File";
            this.btnSerializeToFile.UseVisualStyleBackColor = true;
            this.btnSerializeToFile.Click += new System.EventHandler(this.btnSerializeToFile_Click);
            // 
            // btnDeserializeFromFile
            // 
            this.btnDeserializeFromFile.Location = new System.Drawing.Point(284, 19);
            this.btnDeserializeFromFile.Name = "btnDeserializeFromFile";
            this.btnDeserializeFromFile.Size = new System.Drawing.Size(121, 23);
            this.btnDeserializeFromFile.TabIndex = 8;
            this.btnDeserializeFromFile.Text = "Deserialize From File";
            this.btnDeserializeFromFile.UseVisualStyleBackColor = true;
            this.btnDeserializeFromFile.Click += new System.EventHandler(this.btnDeserializeFromFile_Click);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(9, 3);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(170, 13);
            this.label3.TabIndex = 9;
            this.label3.Text = "Choose a type to test YAXLib with:";
            // 
            // comboPolicy
            // 
            this.comboPolicy.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboPolicy.FormattingEnabled = true;
            this.comboPolicy.Location = new System.Drawing.Point(120, 52);
            this.comboPolicy.Name = "comboPolicy";
            this.comboPolicy.Size = new System.Drawing.Size(155, 21);
            this.comboPolicy.TabIndex = 10;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(6, 55);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(108, 13);
            this.label4.TabIndex = 12;
            this.label4.Text = "Error Handling Policy:";
            // 
            // comboErrorType
            // 
            this.comboErrorType.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboErrorType.FormattingEnabled = true;
            this.comboErrorType.Location = new System.Drawing.Point(396, 52);
            this.comboErrorType.Name = "comboErrorType";
            this.comboErrorType.Size = new System.Drawing.Size(80, 21);
            this.comboErrorType.TabIndex = 13;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(292, 55);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(98, 13);
            this.label5.TabIndex = 14;
            this.label5.Text = "Defaullt Error Type:";
            // 
            // groupBox1
            // 
            this.groupBox1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBox1.Controls.Add(this.label8);
            this.groupBox1.Controls.Add(this.numMaxRecursion);
            this.groupBox1.Controls.Add(this.label7);
            this.groupBox1.Controls.Add(this.comboOptions);
            this.groupBox1.Controls.Add(this.btnSerialize);
            this.groupBox1.Controls.Add(this.label5);
            this.groupBox1.Controls.Add(this.btnDeserialize);
            this.groupBox1.Controls.Add(this.comboErrorType);
            this.groupBox1.Controls.Add(this.btnSerializeToFile);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.btnDeserializeFromFile);
            this.groupBox1.Controls.Add(this.comboPolicy);
            this.groupBox1.Location = new System.Drawing.Point(12, 12);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(856, 88);
            this.groupBox1.TabIndex = 15;
            this.groupBox1.TabStop = false;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(486, 55);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(41, 13);
            this.label7.TabIndex = 16;
            this.label7.Text = "Option:";
            // 
            // comboOptions
            // 
            this.comboOptions.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboOptions.FormattingEnabled = true;
            this.comboOptions.Location = new System.Drawing.Point(533, 52);
            this.comboOptions.Name = "comboOptions";
            this.comboOptions.Size = new System.Drawing.Size(223, 21);
            this.comboOptions.TabIndex = 15;
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            this.openFileDialog1.Filter = "XML Files|*.xml|All files|*.*";
            // 
            // saveFileDialog1
            // 
            this.saveFileDialog1.Filter = "XML Files|*.xml|All files|*.*";
            // 
            // rtbDeserializeOutput
            // 
            this.rtbDeserializeOutput.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.rtbDeserializeOutput.Font = new System.Drawing.Font("Courier New", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.rtbDeserializeOutput.Location = new System.Drawing.Point(6, 22);
            this.rtbDeserializeOutput.Name = "rtbDeserializeOutput";
            this.rtbDeserializeOutput.ReadOnly = true;
            this.rtbDeserializeOutput.Size = new System.Drawing.Size(291, 470);
            this.rtbDeserializeOutput.TabIndex = 16;
            this.rtbDeserializeOutput.Text = "";
            this.rtbDeserializeOutput.WordWrap = false;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(3, 3);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(149, 13);
            this.label6.TabIndex = 17;
            this.label6.Text = "Deserialized object\'s ToString:";
            // 
            // splitContainer1
            // 
            this.splitContainer1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.Location = new System.Drawing.Point(0, 107);
            this.splitContainer1.Name = "splitContainer1";
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.Controls.Add(this.rtbXMLOutput);
            this.splitContainer1.Panel1.Controls.Add(this.lstSampleClasses);
            this.splitContainer1.Panel1.Controls.Add(this.rtbParsingErrors);
            this.splitContainer1.Panel1.Controls.Add(this.label1);
            this.splitContainer1.Panel1.Controls.Add(this.label2);
            this.splitContainer1.Panel1.Controls.Add(this.label3);
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.Controls.Add(this.rtbDeserializeOutput);
            this.splitContainer1.Panel2.Controls.Add(this.label6);
            this.splitContainer1.Size = new System.Drawing.Size(880, 497);
            this.splitContainer1.SplitterDistance = 574;
            this.splitContainer1.TabIndex = 18;
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.groupBox1);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Top;
            this.panel1.Location = new System.Drawing.Point(0, 0);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(880, 107);
            this.panel1.TabIndex = 19;
            // 
            // numMaxRecursion
            // 
            this.numMaxRecursion.Location = new System.Drawing.Point(659, 19);
            this.numMaxRecursion.Maximum = new decimal(new int[] {
            999999999,
            0,
            0,
            0});
            this.numMaxRecursion.Name = "numMaxRecursion";
            this.numMaxRecursion.Size = new System.Drawing.Size(97, 20);
            this.numMaxRecursion.TabIndex = 17;
            this.numMaxRecursion.Value = new decimal(new int[] {
            300,
            0,
            0,
            0});
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(572, 21);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(81, 13);
            this.label8.TabIndex = 18;
            this.label8.Text = "Max Recursion:";
            // 
            // FormMain
            // 
            this.AcceptButton = this.btnSerialize;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(880, 604);
            this.Controls.Add(this.splitContainer1);
            this.Controls.Add(this.panel1);
            this.MinimumSize = new System.Drawing.Size(506, 393);
            this.Name = "FormMain";
            this.Text = "YAXLib Demo Application";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel1.PerformLayout();
            this.splitContainer1.Panel2.ResumeLayout(false);
            this.splitContainer1.Panel2.PerformLayout();
            this.splitContainer1.ResumeLayout(false);
            this.panel1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.numMaxRecursion)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Button btnSerialize;
        private System.Windows.Forms.RichTextBox rtbXMLOutput;
        private System.Windows.Forms.Button btnDeserialize;
        private System.Windows.Forms.ListBox lstSampleClasses;
        private System.Windows.Forms.RichTextBox rtbParsingErrors;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Button btnSerializeToFile;
        private System.Windows.Forms.Button btnDeserializeFromFile;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.ComboBox comboPolicy;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.ComboBox comboErrorType;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.SaveFileDialog saveFileDialog1;
        private System.Windows.Forms.RichTextBox rtbDeserializeOutput;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.ComboBox comboOptions;
        private System.Windows.Forms.NumericUpDown numMaxRecursion;
        private System.Windows.Forms.Label label8;
    }
}

