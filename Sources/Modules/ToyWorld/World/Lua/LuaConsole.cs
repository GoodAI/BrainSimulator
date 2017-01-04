using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Windows.Forms;
using World.Atlas;
using World.ToyWorldCore;

namespace World.Lua
{
    public partial class LuaConsole : Form
    {
        private readonly LuaExecutor m_lex;
        private readonly List<string> m_inputOutputList = new List<string>();
        private readonly List<string> m_inputList = new List<string>();
        private int m_historyPointer;

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

        public LuaConsole(ToyWorld toyWorld, IAtlas atlas, AutoResetEvent luaSynch)
        {
            InitializeComponent();

            toyWorld.ToyWorldDisposed += CloseConsole;

            outputListBox.DataSource = m_inputOutputList;

            m_lex = new LuaExecutor(atlas, luaSynch, this);

            m_inputOutputList.Add(INVITATION_MESSAGE);
            ResetBox();
        }

        private void CloseConsole(object sender)
        {
            Invoke(new Action(Close));
        }

        private void LuaConsole_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Escape)
            {
                m_lex.StopScript();
            }

            if (!inputTextBox.Enabled) return;

            if (e.KeyCode == Keys.Enter)
            {
                e.Handled = true;

                string command = inputTextBox.Text;
                PrintLines(">>" + command);
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

                if (command.StartsWith("execute "))
                {
                    try
                    {
                        string substring = command.Substring(8);
                        StreamReader sr = new StreamReader(new FileStream(substring, FileMode.Open));
                        string readToEnd = sr.ReadToEnd();
                        command = readToEnd;
                    }
                    catch (Exception)
                    {
                        PrintLines(e.ToString());
                    }
                }
                else if (command.StartsWith("Help")) command = "return " + command;
                else if (command.StartsWith("help ")) command = "return Help(" + command.Substring(5) + ")";

                m_lex.ExecuteChunk(command, PrintResultAndActivateInput);
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
            PrintLines("Type 'help [object]' for list of accessible methods. \n\n" +
                  "If you want to stop a method, press Esc key.\n\n" +
                  "Useful objects: \n" +
                  "\tle - LuaExecutor\n" +
                  "\tlc - LuaConsole (lc:PrintLines(\"toPrint\"))\n" +
                  "\tac - AvatarControl\n" +
                  "\tam - Atlas manipulator\n" +
                  "\tavatar - current Avatar\n" +
                  "\tatlas - Atlas\n\n" +
                  "To acces a property, type '[object].[propery]'.\n" +
                  "To run a method, type '[object]:[method]([arguments])'\n\n" +
                  "You can use assignments and standard Lua control mechanisms.");
        }

        private void PrintResultAndActivateInput(string result)
        {
            Invoke(new Action(() =>
            {
                PrintLines(result);
                inputTextBox.Enabled = true;
                inputTextBox.Focus();
            }));
        }

        private void PrintLines(object o)
        {
            if (o.ToString() == "") return;
            foreach (string s in o.ToString().Split('\n'))
            {
                m_inputOutputList.Add(s);
            }
            ResetBox();
        }

        public void Print(object o)
        {
            Invoke(new Action(() =>
            {
                PrintLines(o);
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
