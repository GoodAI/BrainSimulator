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
        private int m_historyPointer = 0;
        private Thread m_currentlyExecutedChunk;

        private const string INVITATION_MESSAGE = "Lua-console for ToyWorld. Type 'help' for basic usage examples.";

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

            m_inputOutputList.Add(INVITATION_MESSAGE);
            ResetBox();
        }

        private void LuaConsole_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Enter)
            {
                e.Handled = true;

                string command = inputTextBox.Text;
                Print("I: " + command);
                m_inputList.Add(command);
                HistoryPointer = int.MaxValue;

                inputTextBox.Clear();

                if (command.Trim() == "help")
                {
                    PrintHelp();
                    return;
                }

                inputTextBox.Clear();
                inputTextBox.Enabled = false;

                if (command.StartsWith("Help")) command = "return " + command;
                if (command.StartsWith("help ")) command = "return Help(" + command.Substring(5) + ")";

                m_currentlyExecutedChunk = m_lex.ExecuteChunk(command, PrintResultAndActivateInput);
            }

            if (e.KeyCode == Keys.Escape)
            {
                m_lex.StopScript();
            }

            if (HistoryPointer < 0) return;
            if (e.KeyCode == Keys.Up)
            {
                if (m_inputList.Count == 0) return;
                HistoryPointer--;
                inputTextBox.Text = m_inputList[HistoryPointer];
            }
            if (e.KeyCode == Keys.Down)
            {
                if (m_inputList.Count == 0) return;
                HistoryPointer++;
                inputTextBox.Text = m_inputList[HistoryPointer];
            }
        }

        private void PrintHelp()
        {
            Print("Type 'help [object]' for list of accessible methods. \n\n" +
                  "If you want to stop a method, press Esc key.\n\n" +
                  "Useful objects: \n" +
                  "\tac - AvatarControl\n" +
                  "\tavatar - current Avatar\n" +
                  "\tle - LuaExecutor\n" +
                  "\tatlas - Atlas\n\n" +
                  "To acces a property, type '[object].[propery]'.\n" +
                  "To run a method, type '[object]:[method]([arguments])'\n\n" +
                  "You can use assignments and standard Lua control mechanisms.");
        }

        private void PrintResultAndActivateInput(string result)
        {
            Invoke(new Action(() =>
            {
                Print(result);
                inputTextBox.Enabled = true;
                inputTextBox.Focus();
            }));
        }

        private void Print(string toPrint)
        {
            foreach (string s in toPrint.Split('\n'))
            {
                m_inputOutputList.Add(s);
            }
            ResetBox();
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
