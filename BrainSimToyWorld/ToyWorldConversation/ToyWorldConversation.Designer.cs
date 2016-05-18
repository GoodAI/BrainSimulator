namespace ToyWorldConversation
{
    partial class ToyWorldConversation
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
            this.textBox_send = new System.Windows.Forms.TextBox();
            this.label_show = new System.Windows.Forms.Label();
            this.checkBox_show_message = new System.Windows.Forms.CheckBox();
            this.checkBox_show_string = new System.Windows.Forms.CheckBox();
            this.richTextBox_messages = new System.Windows.Forms.RichTextBox();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            this.SuspendLayout();
            // 
            // textBox_send
            // 
            this.textBox_send.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.textBox_send.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.textBox_send.Location = new System.Drawing.Point(3, 26);
            this.textBox_send.Name = "textBox_send";
            this.textBox_send.Size = new System.Drawing.Size(230, 20);
            this.textBox_send.TabIndex = 0;
            this.textBox_send.KeyDown += new System.Windows.Forms.KeyEventHandler(this.textBox_send_KeyDown);
            // 
            // label_show
            // 
            this.label_show.AutoSize = true;
            this.label_show.Location = new System.Drawing.Point(3, 6);
            this.label_show.Name = "label_show";
            this.label_show.Size = new System.Drawing.Size(34, 13);
            this.label_show.TabIndex = 1;
            this.label_show.Text = "Show";
            // 
            // checkBox_show_message
            // 
            this.checkBox_show_message.AutoSize = true;
            this.checkBox_show_message.Checked = true;
            this.checkBox_show_message.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBox_show_message.Location = new System.Drawing.Point(43, 6);
            this.checkBox_show_message.Name = "checkBox_show_message";
            this.checkBox_show_message.Size = new System.Drawing.Size(69, 17);
            this.checkBox_show_message.TabIndex = 2;
            this.checkBox_show_message.Text = "Message";
            this.checkBox_show_message.UseVisualStyleBackColor = true;
            // 
            // checkBox_show_string
            // 
            this.checkBox_show_string.AutoSize = true;
            this.checkBox_show_string.Checked = true;
            this.checkBox_show_string.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBox_show_string.Location = new System.Drawing.Point(118, 6);
            this.checkBox_show_string.Name = "checkBox_show_string";
            this.checkBox_show_string.Size = new System.Drawing.Size(53, 17);
            this.checkBox_show_string.TabIndex = 3;
            this.checkBox_show_string.Text = "String";
            this.checkBox_show_string.UseVisualStyleBackColor = true;
            // 
            // richTextBox_messages
            // 
            this.richTextBox_messages.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.richTextBox_messages.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.richTextBox_messages.Location = new System.Drawing.Point(3, 3);
            this.richTextBox_messages.Name = "richTextBox_messages";
            this.richTextBox_messages.ReadOnly = true;
            this.richTextBox_messages.Size = new System.Drawing.Size(230, 245);
            this.richTextBox_messages.TabIndex = 4;
            this.richTextBox_messages.Text = "";
            // 
            // splitContainer1
            // 
            this.splitContainer1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.splitContainer1.FixedPanel = System.Windows.Forms.FixedPanel.Panel2;
            this.splitContainer1.Location = new System.Drawing.Point(11, 12);
            this.splitContainer1.Name = "splitContainer1";
            this.splitContainer1.Orientation = System.Windows.Forms.Orientation.Horizontal;
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.Controls.Add(this.richTextBox_messages);
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.Controls.Add(this.checkBox_show_string);
            this.splitContainer1.Panel2.Controls.Add(this.label_show);
            this.splitContainer1.Panel2.Controls.Add(this.checkBox_show_message);
            this.splitContainer1.Panel2.Controls.Add(this.textBox_send);
            this.splitContainer1.Size = new System.Drawing.Size(236, 304);
            this.splitContainer1.SplitterDistance = 251;
            this.splitContainer1.TabIndex = 6;
            // 
            // ToyWorldConversation
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(262, 327);
            this.Controls.Add(this.splitContainer1);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.HideOnClose = true;
            this.Name = "ToyWorldConversation";
            this.Text = "ToyWorld - Conversation";
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel2.ResumeLayout(false);
            this.splitContainer1.Panel2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TextBox textBox_send;
        private System.Windows.Forms.Label label_show;
        private System.Windows.Forms.CheckBox checkBox_show_message;
        private System.Windows.Forms.CheckBox checkBox_show_string;
        private System.Windows.Forms.RichTextBox richTextBox_messages;
        private System.Windows.Forms.SplitContainer splitContainer1;

    }
}