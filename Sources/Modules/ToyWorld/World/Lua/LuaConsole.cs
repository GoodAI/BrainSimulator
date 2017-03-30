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
        private const string INPUT_LINE_START = ">>";

        /// <summary>
        /// Points to current position in list of all previous inputs.
        /// </summary>
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

        public LuaConsole(ToyWorld toyWorld, IAtlas atlas)
        {
            InitializeComponent();

            toyWorld.ToyWorldDisposed += CloseConsole;

            outputListBox.DataSource = m_inputOutputList;

            m_lex = new LuaExecutor(atlas, this);

            m_inputOutputList.Add(INVITATION_MESSAGE);
            ResetBox();
        }

        private void CloseConsole(object sender)
        {
            if (!Visible) return;
            Invoke(new Action(Close));
        }

        public void RunScript(string scriptPath)
        {
            string script = File.ReadAllText(scriptPath);
            inputTextBox.BeginInvoke((Action)(() => { inputTextBox.Enabled = false; }));
            m_lex.ExecuteChunk(script, PrintResultAndActivateInput);
        }

        public void NotifyAndWait()
        {
            m_lex.NotifyAndWait();
        }

        private void LuaConsole_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Escape)
            {
                m_lex.ShouldStopScript = true;
            }

            if (!inputTextBox.Enabled) return;

            if (e.KeyCode == Keys.Enter)
            {
                e.Handled = true;

                string command = inputTextBox.Text;

                if (command.Trim() == "")
                {
                    PrintLines(INPUT_LINE_START);
                    return;
                }

                PrintLines(INPUT_LINE_START + command);
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

                const string executeCommand = "execute ";
                if (command.StartsWith(executeCommand))
                {
                    try
                    {
                        string substring = command.Substring(executeCommand.Length);
                        command = File.ReadAllText(substring);
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
            PrintLines(@"Type 'help [object]' for list of accessible methods.
For lua script execution (.lua) type 'execute D:\script.lua'.
If you want to stop an execution, press <Esc> key

Useful objects:
    le - LuaExecutor
    lc - LuaConsole (lc:Print('toPrint'))
    ac - AvatarControl
    am - Atlas manipulator
    avatar - current Avatar
    atlas - Atlas

To access a property, type '[object].[property]'.
To execute a method, type '[object]:[method]([arguments])'

You can use assignments and standard Lua control mechanisms.");
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
            if (e.KeyChar.Equals((char)Keys.Return))
            {
                e.Handled = true;
            }
        }
    }
}
