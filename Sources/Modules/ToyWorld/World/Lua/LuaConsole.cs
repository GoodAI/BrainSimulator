using System;
using System.Collections.Generic;
using System.Threading;
using System.Windows.Forms;
using World.Atlas;

namespace World.Lua
{
    public partial class LuaConsole : Form
    {
        private readonly LuaExecutor m_lex;
        private readonly List<string> m_inputOutputList = new List<string>();
        private readonly List<string> m_inputList = new List<string>();
        private int m_historyPointer = -1;
        private Thread m_currentlyExecutedChunk;

        private int HistoryPointer
        {
            get { return m_historyPointer; }
            set
            {
                if (value >= m_inputList.Count)
                {
                    m_historyPointer = 0;
                    return;
                }
                if (value < 0)
                {
                    m_historyPointer = m_inputList.Count - 1;
                    return;
                }
                m_historyPointer = value;
            }
        }

        public LuaConsole(IAtlas atlas, AutoResetEvent luaSynch)
        {
            InitializeComponent();
            outputListBox.DataSource = m_inputOutputList;

            m_lex = new LuaExecutor(atlas, luaSynch);
        }

        private void LuaConsole_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Enter)
            {
                e.Handled = true;

                string command = inputTextBox.Text;
                m_inputOutputList.Add("I: " + command);
                m_inputList.Add(command);
                HistoryPointer = -1;

                ResetBox();

                inputTextBox.Clear();
                //inputTextBox.Enabled = false;

                if (command.StartsWith("Help")) command = "return " + command;
                if (command.StartsWith("help ")) command = "return Help(" + command.Substring(5) + ")";

                m_currentlyExecutedChunk = m_lex.ExecuteChunk(command, PrintResultAndActivateInput);
            }

            if (e.KeyCode == Keys.Q || e.KeyCode == Keys.Escape)
            {
                m_currentlyExecutedChunk?.Abort();
            }

            if (HistoryPointer < 0) return;
            if (e.KeyCode == Keys.Up)
            {
                if (HistoryPointer < 0) return;
                inputTextBox.Text = m_inputList[HistoryPointer];
                HistoryPointer--;
            }
            if (e.KeyCode == Keys.Down)
            {
                if (HistoryPointer < 0) return;
                inputTextBox.Text = m_inputList[HistoryPointer];
                HistoryPointer++;
            }
        }

        private void PrintResultAndActivateInput(string result)
        {
            Invoke(new Action(() =>
            {
                foreach (string s in result.Split('\n'))
                {
                    m_inputOutputList.Add(s);
                }
                ResetBox();
                //inputTextBox.Enabled = true;
                inputTextBox.Focus();
            }));
        }

        private void ResetBox()
        {
            outputListBox.DataSource = null;
            outputListBox.DataSource = m_inputOutputList;
            outputListBox.SetSelected(m_inputOutputList.Count - 1, true);
        }

        private void inputTextBox_KeyPress(object sender, KeyPressEventArgs e)
        {
            if (e.KeyChar.Equals((char)13))
            {
                e.Handled = true;
            }
        }
    }
}
