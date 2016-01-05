using GoodAI.BrainSimulator.UserSettings;
using GoodAI.Core.Execution;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class ConsoleForm : DockContent
    {
        private MainForm m_mainForm;

        private static int MAX_LINES = 1000;
        private static int ADDITIONAL_LINES_REMOVED_PER_CHECK = 50;

        private class TextBoxCache : MyLogWriter
        {
            private ConsoleForm m_consoleForm;
            
            private StringBuilder[] m_builders;
            private static Color[] COLORS = { Color.SteelBlue, Color.Black, Color.Chocolate, Color.Red};

            public TextBoxCache(ConsoleForm consoleForm)
            {
                m_consoleForm = consoleForm;
                m_builders = new StringBuilder[Enum.GetValues(typeof(MyLogLevel)).Length];

                for (int i = 0; i < m_builders.Length; i++)
                {
                    m_builders[i] = new StringBuilder();
                }
            }

            public void WriteLine(MyLogLevel level, string message)
            {
                lock (m_builders)
                {
                    StringBuilder builder = m_builders[(int)level];
                    if (level > MyLogLevel.INFO)
                    {
                        builder.Append("[" + level + "] ");
                    }
                    builder.Append(message);
                    builder.Append("\n");
                }
                FlushIfInsideSameThread();
            }

            public void Write(MyLogLevel level, string message)
            {
                lock (m_builders)
                {
                    m_builders[(int)level].Append(message);
                }
                FlushIfInsideSameThread();
            }

            public void Write(MyLogLevel level, char message)
            {
                lock (m_builders)
                {
                    m_builders[(int)level].Append(message);
                }
                FlushIfInsideSameThread();
            }

            public void FlushCache()
            {
                lock (m_builders)
                {
                    int caretPosition = 0;
                    int currentTextLength = 0;

                    if (!m_consoleForm.InvokeRequired)
                    {
                        caretPosition = m_consoleForm.CaretPosition;
                        currentTextLength = m_consoleForm.TextLength;
                    }

                    bool written = false;
                    for (int i = 0; i < m_builders.Length; i++)
                    {
                        string text = m_builders[i].ToString();
                        if (string.IsNullOrEmpty(text))
                            continue;

                        written = true;

                        m_builders[i].Clear();

                        if (m_consoleForm.InvokeRequired)
                        {
                            m_consoleForm.Invoke((MethodInvoker)(() => m_consoleForm.AppendText(text, COLORS[i])));                            
                        }
                        else
                        {
                            m_consoleForm.AppendText(text, COLORS[i]);
                        }
                    }

                    if (!m_consoleForm.InvokeRequired && written && caretPosition == currentTextLength)
                        m_consoleForm.ScrollToCaret();
                }
            }

            private void FlushIfInsideSameThread()
            {
                if (!m_consoleForm.InvokeRequired)
                {
                    FlushCache();
                }
            }
        }

        private int TextLength
        {
            get { return textBox.TextLength; }
        }

        private int CaretPosition
        {
            get { return textBox.SelectionStart; }
            set {
                textBox.SelectionStart = value;
                textBox.SelectionLength = 0;
            }
        }

        private class StringBuilderCache : MyLogWriter
        {            
            private StringBuilder m_builder;
            private ConsoleForm m_consoleForm;

            public StringBuilderCache(ConsoleForm consoleForm)
            {
                m_builder = new StringBuilder();
                m_consoleForm = consoleForm;
            }

            public void WriteLine(MyLogLevel level, string message)
            {
                lock (m_builder)
                {
                    if (level > MyLogLevel.INFO)
                    {
                        m_builder.Append("[" + level + "] ");
                    }
                    m_builder.Append(message);
                    m_builder.Append("\n");
                }
                FlushIfInsideSameThread();
            }

            public void Write(MyLogLevel level, string message)
            {
                lock (m_builder)
                {
                    m_builder.Append(message);
                }
                FlushIfInsideSameThread();
            }

            public void Write(MyLogLevel level, char message)
            {
                lock (m_builder)
                {
                    m_builder.Append(message);
                }
                FlushIfInsideSameThread();
            }

            public void FlushCache()
            {
                lock (m_builder)
                {
                    string text = m_builder.ToString();
                    m_builder.Clear();
                    m_consoleForm.textBox.AppendText(text);
                }
            }

            private void FlushIfInsideSameThread() 
            {
                if (!m_consoleForm.InvokeRequired)
                {
                    FlushCache();
                }
            }
        };

        public ConsoleForm(MainForm mainForm)
        {            
            InitializeComponent();
            m_mainForm = mainForm;            
           
            MyLog.Writer = new TextBoxCache(this);
            MyLog.GrabConsole();

            m_mainForm.SimulationHandler.ProgressChanged += SimulationHandler_ProgressChanged;
            m_mainForm.SimulationHandler.StateChanged += SimulationHandler_StateChanged;

            logLevelStripComboBox.Items.AddRange(Enum.GetNames(typeof(MyLogLevel)));            
            logLevelStripComboBox.SelectedIndexChanged += logLevelStripComboBox_SelectedIndexChanged;

            logLevelStripComboBox.SelectedIndex = (int) AppSettings.GetInitialLogLevel();
        }

        void logLevelStripComboBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            MyLog.Level = (MyLogLevel)logLevelStripComboBox.SelectedIndex;
            AppSettings.SaveSettings(settings => settings.LogLevel = (int)MyLog.Level);
        }

        void SimulationHandler_StateChanged(object sender, MySimulationHandler.StateEventArgs e)
        {
            //UpdateConsole();
        }

        void SimulationHandler_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            this.BeginInvoke((MethodInvoker) UpdateConsole);
        }

        private void UpdateConsole()
        {
            MyLog.Writer.FlushCache();
        }

        public void AppendText(string text, Color textcolor)
        {                      
            textBox.SelectionStart = textBox.Text.Length;
            textBox.SelectionLength = 0;
            textBox.SelectionColor = textcolor;
            textBox.AppendText(text);

            if (textBox.Lines.Length > MAX_LINES)
            {
                textBox.Lines = textBox.Lines.Skip(
                        textBox.Lines.Length - MAX_LINES + ADDITIONAL_LINES_REMOVED_PER_CHECK
                        ).ToArray();
            }
        }

        private bool IsCaretAtTheEnd()
        {
            return textBox.SelectionStart == textBox.TextLength;
        }

        private void ScrollToCaret()
        {
            textBox.ScrollToCaret();
        } 
    }
}
