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
            this.checkBox_show_agent = new System.Windows.Forms.CheckBox();
            this.checkBox_show_world = new System.Windows.Forms.CheckBox();
            this.richTextBox_messages = new System.Windows.Forms.RichTextBox();
            this.SuspendLayout();
            // 
            // textBox_send
            // 
            this.textBox_send.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.textBox_send.Location = new System.Drawing.Point(15, 202);
            this.textBox_send.Name = "textBox_send";
            this.textBox_send.Size = new System.Drawing.Size(260, 20);
            this.textBox_send.TabIndex = 0;
            // 
            // label_show
            // 
            this.label_show.AutoSize = true;
            this.label_show.Location = new System.Drawing.Point(12, 186);
            this.label_show.Name = "label_show";
            this.label_show.Size = new System.Drawing.Size(34, 13);
            this.label_show.TabIndex = 1;
            this.label_show.Text = "Show";
            // 
            // checkBox_show_agent
            // 
            this.checkBox_show_agent.AutoSize = true;
            this.checkBox_show_agent.Checked = true;
            this.checkBox_show_agent.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBox_show_agent.Location = new System.Drawing.Point(73, 185);
            this.checkBox_show_agent.Name = "checkBox_show_agent";
            this.checkBox_show_agent.Size = new System.Drawing.Size(54, 17);
            this.checkBox_show_agent.TabIndex = 2;
            this.checkBox_show_agent.Text = "Agent";
            this.checkBox_show_agent.UseVisualStyleBackColor = true;
            // 
            // checkBox_show_world
            // 
            this.checkBox_show_world.AutoSize = true;
            this.checkBox_show_world.Checked = true;
            this.checkBox_show_world.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBox_show_world.Location = new System.Drawing.Point(133, 185);
            this.checkBox_show_world.Name = "checkBox_show_world";
            this.checkBox_show_world.Size = new System.Drawing.Size(54, 17);
            this.checkBox_show_world.TabIndex = 3;
            this.checkBox_show_world.Text = "World";
            this.checkBox_show_world.UseVisualStyleBackColor = true;
            // 
            // richTextBox_messages
            // 
            this.richTextBox_messages.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.richTextBox_messages.Location = new System.Drawing.Point(12, 13);
            this.richTextBox_messages.Name = "richTextBox_messages";
            this.richTextBox_messages.ReadOnly = true;
            this.richTextBox_messages.Size = new System.Drawing.Size(260, 166);
            this.richTextBox_messages.TabIndex = 4;
            this.richTextBox_messages.Text = "";
            // 
            // ToyWorldConversation
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(284, 232);
            this.Controls.Add(this.richTextBox_messages);
            this.Controls.Add(this.checkBox_show_world);
            this.Controls.Add(this.checkBox_show_agent);
            this.Controls.Add(this.label_show);
            this.Controls.Add(this.textBox_send);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Name = "ToyWorldConversation";
            this.Text = "ToyWorld - Conversation";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox textBox_send;
        private System.Windows.Forms.Label label_show;
        private System.Windows.Forms.CheckBox checkBox_show_agent;
        private System.Windows.Forms.CheckBox checkBox_show_world;
        private System.Windows.Forms.RichTextBox richTextBox_messages;

    }
}