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
                inputTextBox.Enabled = false;

                m_lex.ExecuteChunk(command, PrintResultAndActivateInput);
            }
            if (e.KeyCode == Keys.Up)
            {
                inputTextBox.Text = m_inputList[HistoryPointer];
                HistoryPointer--;
            }
            if (e.KeyCode == Keys.Down)
            {
                inputTextBox.Text = m_inputList[HistoryPointer];
                HistoryPointer++;

            }
        }

        private void PrintResultAndActivateInput(string result)
        {
            Invoke(new Action(() =>
            {
                m_inputOutputList.Add("O: " + result);
                ResetBox();
                inputTextBox.Enabled = true;
                inputTextBox.Focus();
            }));
        }

        private void ResetBox()
        {
            outputListBox.DataSource = null;
            outputListBox.DataSource = m_inputOutputList;
            outputListBox.SetSelected(m_inputOutputList.Count - 1, true);
        }

        private static string NewLine => "\r\n>";

        private static string LastCommand(string s)
        {
            int lastIndexOfG = s.LastIndexOf(">", StringComparison.Ordinal);
            return s.Substring(lastIndexOfG + 1);
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
