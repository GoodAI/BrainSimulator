namespace GoodAI.School.GUI
{
    partial class SchoolRunForm
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
            this.components = new System.ComponentModel.Container();
            WeifenLuo.WinFormsUI.Docking.DockPanelSkin dockPanelSkin2 = new WeifenLuo.WinFormsUI.Docking.DockPanelSkin();
            WeifenLuo.WinFormsUI.Docking.AutoHideStripSkin autoHideStripSkin2 = new WeifenLuo.WinFormsUI.Docking.AutoHideStripSkin();
            WeifenLuo.WinFormsUI.Docking.DockPanelGradient dockPanelGradient4 = new WeifenLuo.WinFormsUI.Docking.DockPanelGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient8 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.DockPaneStripSkin dockPaneStripSkin2 = new WeifenLuo.WinFormsUI.Docking.DockPaneStripSkin();
            WeifenLuo.WinFormsUI.Docking.DockPaneStripGradient dockPaneStripGradient2 = new WeifenLuo.WinFormsUI.Docking.DockPaneStripGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient9 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.DockPanelGradient dockPanelGradient5 = new WeifenLuo.WinFormsUI.Docking.DockPanelGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient10 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.DockPaneStripToolWindowGradient dockPaneStripToolWindowGradient2 = new WeifenLuo.WinFormsUI.Docking.DockPaneStripToolWindowGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient11 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient12 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.DockPanelGradient dockPanelGradient6 = new WeifenLuo.WinFormsUI.Docking.DockPanelGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient13 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            WeifenLuo.WinFormsUI.Docking.TabGradient tabGradient14 = new WeifenLuo.WinFormsUI.Docking.TabGradient();
            this.dataGridView1 = new System.Windows.Forms.DataGridView();
            this.TaskType = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.WorldType = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.stepsDataGridViewTextBoxColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.timeDataGridViewTextBoxColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.ProgressColumn = new GoodAI.BrainSimulator.Utils.DataGridViewProgressColumn();
            this.statusDataGridViewTextBoxColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.isHiddenDataGridViewCheckBoxColumn = new System.Windows.Forms.DataGridViewCheckBoxColumn();
            this.checkStateDataGridViewCheckBoxColumn = new System.Windows.Forms.DataGridViewCheckBoxColumn();
            this.imageDataGridViewImageColumn = new System.Windows.Forms.DataGridViewImageColumn();
            this.isCheckedDataGridViewCheckBoxColumn = new System.Windows.Forms.DataGridViewCheckBoxColumn();
            this.isLeafDataGridViewCheckBoxColumn = new System.Windows.Forms.DataGridViewCheckBoxColumn();
            this.tagDataGridViewTextBoxColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.learningTaskNodeBindingSource = new System.Windows.Forms.BindingSource(this.components);
            this.dataGridViewTextBoxColumn1 = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.btnRun = new System.Windows.Forms.ToolStripButton();
            this.btnStop = new System.Windows.Forms.ToolStripButton();
            this.btnPause = new System.Windows.Forms.ToolStripButton();
            this.btnStepOver = new System.Windows.Forms.ToolStripButton();
            this.btnDebug = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.btnEmulateSuccess = new System.Windows.Forms.ToolStripButton();
            this.btnObserver = new System.Windows.Forms.ToolStripButton();
            this.observerDockPanel = new WeifenLuo.WinFormsUI.Docking.DockPanel();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.panel1 = new System.Windows.Forms.Panel();
            this.actualRewardTextBox = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.unitNumberTextBox = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.currentLevelTextBox = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.dataGridView1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.learningTaskNodeBindingSource)).BeginInit();
            this.toolStrip1.SuspendLayout();
            this.panel1.SuspendLayout();
            this.SuspendLayout();
            // 
            // dataGridView1
            // 
            this.dataGridView1.AllowUserToAddRows = false;
            this.dataGridView1.AllowUserToDeleteRows = false;
            this.dataGridView1.AllowUserToResizeRows = false;
            this.dataGridView1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.dataGridView1.AutoGenerateColumns = false;
            this.dataGridView1.AutoSizeColumnsMode = System.Windows.Forms.DataGridViewAutoSizeColumnsMode.Fill;
            this.dataGridView1.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.dataGridView1.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {
            this.TaskType,
            this.WorldType,
            this.stepsDataGridViewTextBoxColumn,
            this.timeDataGridViewTextBoxColumn,
            this.ProgressColumn,
            this.statusDataGridViewTextBoxColumn,
            this.isHiddenDataGridViewCheckBoxColumn,
            this.checkStateDataGridViewCheckBoxColumn,
            this.imageDataGridViewImageColumn,
            this.isCheckedDataGridViewCheckBoxColumn,
            this.isLeafDataGridViewCheckBoxColumn,
            this.tagDataGridViewTextBoxColumn});
            this.dataGridView1.DataSource = this.learningTaskNodeBindingSource;
            this.dataGridView1.Location = new System.Drawing.Point(12, 28);
            this.dataGridView1.MultiSelect = false;
            this.dataGridView1.Name = "dataGridView1";
            this.dataGridView1.RowHeadersVisible = false;
            this.dataGridView1.SelectionMode = System.Windows.Forms.DataGridViewSelectionMode.FullRowSelect;
            this.dataGridView1.Size = new System.Drawing.Size(370, 552);
            this.dataGridView1.TabIndex = 0;
            this.dataGridView1.CellFormatting += new System.Windows.Forms.DataGridViewCellFormattingEventHandler(this.dataGridView1_CellFormatting);
            this.dataGridView1.SelectionChanged += new System.EventHandler(this.dataGridView1_SelectionChanged);
            this.dataGridView1.KeyDown += new System.Windows.Forms.KeyEventHandler(this.SchoolRunForm_KeyDown);
            // 
            // TaskType
            // 
            this.TaskType.DataPropertyName = "TaskType";
            this.TaskType.HeaderText = "Task";
            this.TaskType.Name = "TaskType";
            this.TaskType.ReadOnly = true;
            // 
            // WorldType
            // 
            this.WorldType.DataPropertyName = "WorldType";
            this.WorldType.HeaderText = "World";
            this.WorldType.Name = "WorldType";
            this.WorldType.ReadOnly = true;
            // 
            // stepsDataGridViewTextBoxColumn
            // 
            this.stepsDataGridViewTextBoxColumn.DataPropertyName = "Steps";
            this.stepsDataGridViewTextBoxColumn.HeaderText = "Steps";
            this.stepsDataGridViewTextBoxColumn.Name = "stepsDataGridViewTextBoxColumn";
            this.stepsDataGridViewTextBoxColumn.ReadOnly = true;
            // 
            // timeDataGridViewTextBoxColumn
            // 
            this.timeDataGridViewTextBoxColumn.DataPropertyName = "Time";
            this.timeDataGridViewTextBoxColumn.HeaderText = "Time";
            this.timeDataGridViewTextBoxColumn.Name = "timeDataGridViewTextBoxColumn";
            // 
            // ProgressColumn
            // 
            this.ProgressColumn.DataPropertyName = "Progress";
            this.ProgressColumn.HeaderText = "Progress";
            this.ProgressColumn.Name = "ProgressColumn";
            // 
            // statusDataGridViewTextBoxColumn
            // 
            this.statusDataGridViewTextBoxColumn.DataPropertyName = "Status";
            this.statusDataGridViewTextBoxColumn.HeaderText = "Status";
            this.statusDataGridViewTextBoxColumn.Name = "statusDataGridViewTextBoxColumn";
            // 
            // isHiddenDataGridViewCheckBoxColumn
            // 
            this.isHiddenDataGridViewCheckBoxColumn.DataPropertyName = "IsHidden";
            this.isHiddenDataGridViewCheckBoxColumn.HeaderText = "IsHidden";
            this.isHiddenDataGridViewCheckBoxColumn.Name = "isHiddenDataGridViewCheckBoxColumn";
            this.isHiddenDataGridViewCheckBoxColumn.Visible = false;
            // 
            // checkStateDataGridViewCheckBoxColumn
            // 
            this.checkStateDataGridViewCheckBoxColumn.DataPropertyName = "CheckState";
            this.checkStateDataGridViewCheckBoxColumn.HeaderText = "CheckState";
            this.checkStateDataGridViewCheckBoxColumn.Name = "checkStateDataGridViewCheckBoxColumn";
            this.checkStateDataGridViewCheckBoxColumn.Visible = false;
            // 
            // imageDataGridViewImageColumn
            // 
            this.imageDataGridViewImageColumn.DataPropertyName = "Image";
            this.imageDataGridViewImageColumn.HeaderText = "Image";
            this.imageDataGridViewImageColumn.Name = "imageDataGridViewImageColumn";
            this.imageDataGridViewImageColumn.Visible = false;
            // 
            // isCheckedDataGridViewCheckBoxColumn
            // 
            this.isCheckedDataGridViewCheckBoxColumn.DataPropertyName = "IsChecked";
            this.isCheckedDataGridViewCheckBoxColumn.HeaderText = "IsChecked";
            this.isCheckedDataGridViewCheckBoxColumn.Name = "isCheckedDataGridViewCheckBoxColumn";
            this.isCheckedDataGridViewCheckBoxColumn.Visible = false;
            // 
            // isLeafDataGridViewCheckBoxColumn
            // 
            this.isLeafDataGridViewCheckBoxColumn.DataPropertyName = "IsLeaf";
            this.isLeafDataGridViewCheckBoxColumn.HeaderText = "IsLeaf";
            this.isLeafDataGridViewCheckBoxColumn.Name = "isLeafDataGridViewCheckBoxColumn";
            this.isLeafDataGridViewCheckBoxColumn.ReadOnly = true;
            this.isLeafDataGridViewCheckBoxColumn.Visible = false;
            // 
            // tagDataGridViewTextBoxColumn
            // 
            this.tagDataGridViewTextBoxColumn.DataPropertyName = "Tag";
            this.tagDataGridViewTextBoxColumn.HeaderText = "Tag";
            this.tagDataGridViewTextBoxColumn.Name = "tagDataGridViewTextBoxColumn";
            this.tagDataGridViewTextBoxColumn.Visible = false;
            // 
            // learningTaskNodeBindingSource
            // 
            this.learningTaskNodeBindingSource.DataSource = typeof(GoodAI.School.GUI.LearningTaskNode);
            // 
            // dataGridViewTextBoxColumn1
            // 
            this.dataGridViewTextBoxColumn1.DataPropertyName = "Tag";
            this.dataGridViewTextBoxColumn1.HeaderText = "Tag";
            this.dataGridViewTextBoxColumn1.Name = "dataGridViewTextBoxColumn1";
            // 
            // toolStrip1
            // 
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.btnRun,
            this.btnStop,
            this.btnPause,
            this.btnStepOver,
            this.btnDebug,
            this.toolStripSeparator1,
            this.btnEmulateSuccess,
            this.btnObserver});
            this.toolStrip1.Location = new System.Drawing.Point(0, 0);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(1147, 25);
            this.toolStrip1.TabIndex = 5;
            this.toolStrip1.Text = "toolStrip1";
            // 
            // btnRun
            // 
            this.btnRun.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnRun.Image = global::GoodAI.School.GUI.Properties.Resources.StatusAnnotations_Play_16xLG_color;
            this.btnRun.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnRun.Name = "btnRun";
            this.btnRun.Size = new System.Drawing.Size(23, 22);
            this.btnRun.Text = "Run Simulation";
            // 
            // btnStop
            // 
            this.btnStop.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnStop.Image = global::GoodAI.School.GUI.Properties.Resources.StatusAnnotations_Stop_16xLG_color;
            this.btnStop.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnStop.Name = "btnStop";
            this.btnStop.Size = new System.Drawing.Size(23, 22);
            this.btnStop.Text = "Stop Simulation";
            // 
            // btnPause
            // 
            this.btnPause.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnPause.Image = global::GoodAI.School.GUI.Properties.Resources.StatusAnnotations_Pause_16xLG_color;
            this.btnPause.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnPause.Name = "btnPause";
            this.btnPause.Size = new System.Drawing.Size(23, 22);
            this.btnPause.Text = "Pause Simulation";
            // 
            // btnStepOver
            // 
            this.btnStepOver.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnStepOver.Image = global::GoodAI.School.GUI.Properties.Resources.StepOver;
            this.btnStepOver.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnStepOver.Name = "btnStepOver";
            this.btnStepOver.Size = new System.Drawing.Size(23, 22);
            this.btnStepOver.Text = "Step Over";
            // 
            // btnDebug
            // 
            this.btnDebug.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnDebug.Image = global::GoodAI.School.GUI.Properties.Resources.debug;
            this.btnDebug.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnDebug.Name = "btnDebug";
            this.btnDebug.Size = new System.Drawing.Size(23, 22);
            this.btnDebug.Text = "Debug Simulation";
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            this.toolStripSeparator1.Size = new System.Drawing.Size(6, 25);
            // 
            // btnEmulateSuccess
            // 
            this.btnEmulateSuccess.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnEmulateSuccess.Image = global::GoodAI.School.GUI.Properties.Resources.Symbols_Complete_and_ok_16xLG;
            this.btnEmulateSuccess.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnEmulateSuccess.Name = "btnEmulateSuccess";
            this.btnEmulateSuccess.Size = new System.Drawing.Size(23, 22);
            this.btnEmulateSuccess.Text = "Emulate Success";
            this.btnEmulateSuccess.Click += new System.EventHandler(this.btnEmulateSuccess_Click);
            // 
            // btnObserver
            // 
            this.btnObserver.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnObserver.Image = global::GoodAI.School.GUI.Properties.Resources.observer_icon;
            this.btnObserver.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnObserver.Name = "btnObserver";
            this.btnObserver.Size = new System.Drawing.Size(23, 22);
            this.btnObserver.Text = "Show Observer";
            this.btnObserver.Click += new System.EventHandler(this.btnObserver_Click);
            // 
            // observerDockPanel
            // 
            this.observerDockPanel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.observerDockPanel.BackColor = System.Drawing.SystemColors.ButtonShadow;
            this.observerDockPanel.Location = new System.Drawing.Point(755, 28);
            this.observerDockPanel.Name = "observerDockPanel";
            this.observerDockPanel.Size = new System.Drawing.Size(380, 380);
            dockPanelGradient4.EndColor = System.Drawing.SystemColors.ControlLight;
            dockPanelGradient4.StartColor = System.Drawing.SystemColors.ControlLight;
            autoHideStripSkin2.DockStripGradient = dockPanelGradient4;
            tabGradient8.EndColor = System.Drawing.SystemColors.Control;
            tabGradient8.StartColor = System.Drawing.SystemColors.Control;
            tabGradient8.TextColor = System.Drawing.SystemColors.ControlDarkDark;
            autoHideStripSkin2.TabGradient = tabGradient8;
            autoHideStripSkin2.TextFont = new System.Drawing.Font("Segoe UI", 9F);
            dockPanelSkin2.AutoHideStripSkin = autoHideStripSkin2;
            tabGradient9.EndColor = System.Drawing.SystemColors.ControlLightLight;
            tabGradient9.StartColor = System.Drawing.SystemColors.ControlLightLight;
            tabGradient9.TextColor = System.Drawing.SystemColors.ControlText;
            dockPaneStripGradient2.ActiveTabGradient = tabGradient9;
            dockPanelGradient5.EndColor = System.Drawing.SystemColors.Control;
            dockPanelGradient5.StartColor = System.Drawing.SystemColors.Control;
            dockPaneStripGradient2.DockStripGradient = dockPanelGradient5;
            tabGradient10.EndColor = System.Drawing.SystemColors.ControlLight;
            tabGradient10.StartColor = System.Drawing.SystemColors.ControlLight;
            tabGradient10.TextColor = System.Drawing.SystemColors.ControlText;
            dockPaneStripGradient2.InactiveTabGradient = tabGradient10;
            dockPaneStripSkin2.DocumentGradient = dockPaneStripGradient2;
            dockPaneStripSkin2.TextFont = new System.Drawing.Font("Segoe UI", 9F);
            tabGradient11.EndColor = System.Drawing.SystemColors.ActiveCaption;
            tabGradient11.LinearGradientMode = System.Drawing.Drawing2D.LinearGradientMode.Vertical;
            tabGradient11.StartColor = System.Drawing.SystemColors.GradientActiveCaption;
            tabGradient11.TextColor = System.Drawing.SystemColors.ActiveCaptionText;
            dockPaneStripToolWindowGradient2.ActiveCaptionGradient = tabGradient11;
            tabGradient12.EndColor = System.Drawing.SystemColors.Control;
            tabGradient12.StartColor = System.Drawing.SystemColors.Control;
            tabGradient12.TextColor = System.Drawing.SystemColors.ControlText;
            dockPaneStripToolWindowGradient2.ActiveTabGradient = tabGradient12;
            dockPanelGradient6.EndColor = System.Drawing.SystemColors.ControlLight;
            dockPanelGradient6.StartColor = System.Drawing.SystemColors.ControlLight;
            dockPaneStripToolWindowGradient2.DockStripGradient = dockPanelGradient6;
            tabGradient13.EndColor = System.Drawing.SystemColors.InactiveCaption;
            tabGradient13.LinearGradientMode = System.Drawing.Drawing2D.LinearGradientMode.Vertical;
            tabGradient13.StartColor = System.Drawing.SystemColors.GradientInactiveCaption;
            tabGradient13.TextColor = System.Drawing.SystemColors.InactiveCaptionText;
            dockPaneStripToolWindowGradient2.InactiveCaptionGradient = tabGradient13;
            tabGradient14.EndColor = System.Drawing.Color.Transparent;
            tabGradient14.StartColor = System.Drawing.Color.Transparent;
            tabGradient14.TextColor = System.Drawing.SystemColors.ControlDarkDark;
            dockPaneStripToolWindowGradient2.InactiveTabGradient = tabGradient14;
            dockPaneStripSkin2.ToolWindowGradient = dockPaneStripToolWindowGradient2;
            dockPanelSkin2.DockPaneStripSkin = dockPaneStripSkin2;
            this.observerDockPanel.Skin = dockPanelSkin2;
            this.observerDockPanel.TabIndex = 6;
            // 
            // tabControl1
            // 
            this.tabControl1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tabControl1.Location = new System.Drawing.Point(389, 29);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(360, 551);
            this.tabControl1.TabIndex = 10;
            // 
            // panel1
            // 
            this.panel1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel1.Controls.Add(this.currentLevelTextBox);
            this.panel1.Controls.Add(this.label4);
            this.panel1.Controls.Add(this.actualRewardTextBox);
            this.panel1.Controls.Add(this.label3);
            this.panel1.Controls.Add(this.unitNumberTextBox);
            this.panel1.Controls.Add(this.label2);
            this.panel1.Controls.Add(this.label1);
            this.panel1.Location = new System.Drawing.Point(755, 414);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(380, 166);
            this.panel1.TabIndex = 13;
            // 
            // actualRewardTextBox
            // 
            this.actualRewardTextBox.Location = new System.Drawing.Point(118, 80);
            this.actualRewardTextBox.Name = "actualRewardTextBox";
            this.actualRewardTextBox.ReadOnly = true;
            this.actualRewardTextBox.Size = new System.Drawing.Size(57, 20);
            this.actualRewardTextBox.TabIndex = 17;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(3, 83);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(103, 13);
            this.label3.TabIndex = 16;
            this.label3.Text = "Previus step reward:";
            // 
            // unitNumberTextBox
            // 
            this.unitNumberTextBox.Location = new System.Drawing.Point(88, 28);
            this.unitNumberTextBox.Name = "unitNumberTextBox";
            this.unitNumberTextBox.ReadOnly = true;
            this.unitNumberTextBox.Size = new System.Drawing.Size(57, 20);
            this.unitNumberTextBox.TabIndex = 15;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(4, 31);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(78, 13);
            this.label2.TabIndex = 14;
            this.label2.Text = "Training unit #:";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Location = new System.Drawing.Point(3, 5);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(37, 13);
            this.label1.TabIndex = 13;
            this.label1.Text = "Status";
            // 
            // currentLevelTextBox
            // 
            this.currentLevelTextBox.Location = new System.Drawing.Point(79, 54);
            this.currentLevelTextBox.Name = "currentLevelTextBox";
            this.currentLevelTextBox.ReadOnly = true;
            this.currentLevelTextBox.Size = new System.Drawing.Size(57, 20);
            this.currentLevelTextBox.TabIndex = 19;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(4, 57);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(69, 13);
            this.label4.TabIndex = 18;
            this.label4.Text = "Current level:";
            this.label4.Click += new System.EventHandler(this.label4_Click);
            // 
            // SchoolRunForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1147, 592);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.tabControl1);
            this.Controls.Add(this.observerDockPanel);
            this.Controls.Add(this.toolStrip1);
            this.Controls.Add(this.dataGridView1);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.HideOnClose = true;
            this.MinimumSize = new System.Drawing.Size(410, 410);
            this.Name = "SchoolRunForm";
            this.Text = "School Run Controller";
            this.VisibleChanged += new System.EventHandler(this.SchoolRunForm_VisibleChanged);
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.SchoolRunForm_KeyDown);
            ((System.ComponentModel.ISupportInitialize)(this.dataGridView1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.learningTaskNodeBindingSource)).EndInit();
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.DataGridView dataGridView1;
        private System.Windows.Forms.BindingSource learningTaskNodeBindingSource;
        private System.Windows.Forms.DataGridViewTextBoxColumn nameDataGridViewTextBoxColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn worldDataGridViewTextBoxColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn dataGridViewTextBoxColumn1;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.ToolStripButton btnRun;
        private System.Windows.Forms.ToolStripButton btnPause;
        private System.Windows.Forms.ToolStripButton btnStop;
        private System.Windows.Forms.ToolStripButton btnStepOver;
        private System.Windows.Forms.ToolStripButton btnDebug;
        private WeifenLuo.WinFormsUI.Docking.DockPanel observerDockPanel;
        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.DataGridViewTextBoxColumn taskTypeDataGridViewTextBoxColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn worldTypeDataGridViewTextBoxColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn stepsDataGridViewTextBoxColumn1;
        private System.Windows.Forms.DataGridViewTextBoxColumn timeDataGridViewTextBoxColumn1;
        private System.Windows.Forms.DataGridViewTextBoxColumn statusDataGridViewTextBoxColumn1;
        private System.Windows.Forms.DataGridViewCheckBoxColumn enabledDataGridViewCheckBoxColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn dataGridViewTextBoxColumn4;
        private System.Windows.Forms.DataGridViewTextBoxColumn dataGridViewTextBoxColumn5;
        private System.Windows.Forms.DataGridViewTextBoxColumn dataGridViewTextBoxColumn6;
        private System.Windows.Forms.DataGridViewTextBoxColumn dataGridViewTextBoxColumn7;
        private System.Windows.Forms.DataGridViewTextBoxColumn dataGridViewTextBoxColumn8;
        private System.Windows.Forms.DataGridViewCheckBoxColumn dataGridViewCheckBoxColumn1;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripButton btnObserver;
        private System.Windows.Forms.DataGridViewTextBoxColumn TaskType;
        private System.Windows.Forms.DataGridViewTextBoxColumn WorldType;
        private System.Windows.Forms.DataGridViewTextBoxColumn stepsDataGridViewTextBoxColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn timeDataGridViewTextBoxColumn;
        private BrainSimulator.Utils.DataGridViewProgressColumn ProgressColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn statusDataGridViewTextBoxColumn;
        private System.Windows.Forms.DataGridViewCheckBoxColumn isHiddenDataGridViewCheckBoxColumn;
        private System.Windows.Forms.DataGridViewCheckBoxColumn checkStateDataGridViewCheckBoxColumn;
        private System.Windows.Forms.DataGridViewImageColumn imageDataGridViewImageColumn;
        private System.Windows.Forms.DataGridViewCheckBoxColumn isCheckedDataGridViewCheckBoxColumn;
        private System.Windows.Forms.DataGridViewCheckBoxColumn isLeafDataGridViewCheckBoxColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn tagDataGridViewTextBoxColumn;
        private System.Windows.Forms.ToolStripButton btnEmulateSuccess;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox unitNumberTextBox;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox actualRewardTextBox;
        private System.Windows.Forms.TextBox currentLevelTextBox;
        private System.Windows.Forms.Label label4;
    }
}