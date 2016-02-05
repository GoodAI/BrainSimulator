using GoodAI.Core.Execution;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using ScintillaNET;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.BrainSimulator.Forms
{
    public sealed partial class TextEditForm : DockContent
    {
        private readonly MainForm m_mainForm;        

        public IScriptableNode Target { get; private set; } 

        public TextEditForm(MainForm mainForm, IScriptableNode target)
        {
            InitializeComponent();

            //Icon = Properties.Resources.TextDoc;                        

            m_mainForm = mainForm;
            Target = target;
            Text = target.Name;

            SetupEditorStyle();

            scintilla.Text = Target.Script;
            
            scintilla.TextChanged += scintilla_TextChanged;
            scintilla.HandleDestroyed += scintilla_HandleDestroyed;
            scintilla.HandleCreated += scintilla_HandleCreated;            
        }

        public void PasteText()
        {
            scintilla.Paste();
        }

        public void CopyText()
        {
            scintilla.Copy();
        }

        public void CutText()
        {
            scintilla.Cut();
        }

        private void scintilla_HandleDestroyed(object sender, EventArgs e)
        {         
            scintilla.TextChanged -= scintilla_TextChanged;            
        }

        private void scintilla_HandleCreated(object sender, EventArgs e)
        {         
            scintilla.Text = Target.Script;
            scintilla.TextChanged += scintilla_TextChanged;
            SetupEditorStyle();                      
        }

        private void scintilla_TextChanged(object sender, EventArgs e)
        {            
            Target.Script = scintilla.Text;            
        }

        private void TextEditForm_Enter(object sender, EventArgs e)
        {
            if (Target != null)
            {
                m_mainForm.NodePropertyView.Target = Target;
                m_mainForm.MemoryBlocksView.Target = Target as MyNode;
                m_mainForm.HelpView.Target = Target as MyNode;
                m_mainForm.TaskView.Target = Target as MyWorkingNode;                
            }
            else
            {
                m_mainForm.NodePropertyView.Target = null;
                m_mainForm.TaskView.Target = null;
                m_mainForm.MemoryBlocksView.Target = null;
                m_mainForm.HelpView.Target = null;
            }
        }

        private void scintilla_CharAdded(object sender, CharAddedEventArgs e)
        {
            // Find the word start
            var currentPos = scintilla.CurrentPosition;
            var wordStartPos = scintilla.WordStartPosition(currentPos, true);

            if (e.Char == '\n')
            {
                if (scintilla.Lines.Count > 1)
                {
                    Line lastLine = scintilla.Lines[scintilla.CurrentLine - 1];                    
                    string lastIndent = lastLine.Text.Substring(0, lastLine.Indentation);
                    scintilla.AddText(lastIndent);
                }
            }

            // Display the autocompletion list
            var lenEntered = currentPos - wordStartPos;
            if (lenEntered > 0)
            {
                scintilla.AutoCShow(lenEntered, Target.NameExpressions);
            }
        }

        private void SetupEditorStyle()
        {
            // Reset the styles
            scintilla.StyleResetDefault();
            scintilla.Styles[Style.Default].Font = "Consolas";
            scintilla.Styles[Style.Default].Size = 11;
            scintilla.Styles[Style.Default].BackColor = scintilla.Enabled ? Color.FromArgb(250, 250, 250) : Color.WhiteSmoke;
            scintilla.Styles[Style.Default].ForeColor = scintilla.Enabled ? Color.Black : Color.DarkGray;
            scintilla.StyleClearAll(); // i.e. Apply to all

            if (scintilla.Enabled) 
            {
                if (Target.Language == "Python")
                {
                    ApplyPythonRecipe();
                }
                else if (Target.Language == "CSharp")
                {
                    ApplyCSharpRecipe();
                }
            }

            // Important for Python
            scintilla.ViewWhitespace = WhitespaceMode.VisibleAlways;

            if (!string.IsNullOrEmpty(Target.Keywords))
            {
                scintilla.SetKeywords(0, Target.Keywords);
            }            

            if (!string.IsNullOrEmpty(Target.NameExpressions)) 
            {
                scintilla.SetKeywords(1, Target.NameExpressions);
            }            
        }

        private void ApplyCSharpRecipe()
        {
            //enable basic line numbering
            scintilla.Margins[0].Width = 30;

            scintilla.Lexer = Lexer.Cpp;
            
            scintilla.Styles[Style.Cpp.Default].ForeColor = Color.Silver;
            scintilla.Styles[Style.Cpp.Comment].ForeColor = Color.FromArgb(0, 128, 0); // Green
            scintilla.Styles[Style.Cpp.CommentLine].ForeColor = Color.FromArgb(0, 128, 0); // Green
            scintilla.Styles[Style.Cpp.CommentLineDoc].ForeColor = Color.FromArgb(128, 128, 128); // Gray            
            scintilla.Styles[Style.Cpp.Word].ForeColor = Color.Blue;
            scintilla.Styles[Style.Cpp.Word2].ForeColor = Color.Maroon;
            scintilla.Styles[Style.Cpp.String].ForeColor = Color.DarkMagenta;
            scintilla.Styles[Style.Cpp.Character].ForeColor = Color.FromArgb(163, 21, 21); // Red
            scintilla.Styles[Style.Cpp.Verbatim].ForeColor = Color.FromArgb(163, 21, 21); // Red                        
        }

        private void ApplyPythonRecipe()
        {
            // Set the lexer
            scintilla.Lexer = Lexer.Python;

            // Known lexer properties:
            // "tab.timmy.whinge.level",
            // "lexer.python.literals.binary",
            // "lexer.python.strings.u",
            // "lexer.python.strings.b",
            // "lexer.python.strings.over.newline",
            // "lexer.python.keywords2.no.sub.identifiers",
            // "fold.quotes.python",
            // "fold.compact",
            // "fold"

            // Some properties we like
            scintilla.SetProperty("tab.timmy.whinge.level", "1");
            scintilla.SetProperty("fold", "1");

            //enable basic line numbering
            scintilla.Margins[0].Width = 20;

            // Use margin 2 for fold markers
            scintilla.Margins[2].Type = MarginType.Symbol;
            scintilla.Margins[2].Mask = Marker.MaskFolders;
            scintilla.Margins[2].Sensitive = true;
            scintilla.Margins[2].Width = 20;

            // Reset folder markers
            for (int i = Marker.FolderEnd; i <= Marker.FolderOpen; i++)
            {
                scintilla.Markers[i].SetForeColor(SystemColors.ControlLightLight);
                scintilla.Markers[i].SetBackColor(SystemColors.ControlDark);
            }

            // Style the folder markers
            scintilla.Markers[Marker.Folder].Symbol = MarkerSymbol.BoxPlus;
            scintilla.Markers[Marker.Folder].SetBackColor(SystemColors.ControlText);
            scintilla.Markers[Marker.FolderOpen].Symbol = MarkerSymbol.BoxMinus;
            scintilla.Markers[Marker.FolderEnd].Symbol = MarkerSymbol.BoxPlusConnected;
            scintilla.Markers[Marker.FolderEnd].SetBackColor(SystemColors.ControlText);
            scintilla.Markers[Marker.FolderMidTail].Symbol = MarkerSymbol.TCorner;
            scintilla.Markers[Marker.FolderOpenMid].Symbol = MarkerSymbol.BoxMinusConnected;
            scintilla.Markers[Marker.FolderSub].Symbol = MarkerSymbol.VLine;
            scintilla.Markers[Marker.FolderTail].Symbol = MarkerSymbol.LCorner;

            // Enable automatic folding
            scintilla.AutomaticFold = (AutomaticFold.Show | AutomaticFold.Click | AutomaticFold.Change);

            // Set the styles
            scintilla.Styles[Style.Python.Default].ForeColor = Color.FromArgb(0x80, 0x80, 0x80);
            scintilla.Styles[Style.Python.CommentLine].ForeColor = Color.FromArgb(0x00, 0x7F, 0x00);
            scintilla.Styles[Style.Python.CommentLine].Italic = true;
            scintilla.Styles[Style.Python.Number].ForeColor = Color.FromArgb(0x00, 0x7F, 0x7F);
            scintilla.Styles[Style.Python.String].ForeColor = Color.FromArgb(0x7F, 0x00, 0x7F);
            scintilla.Styles[Style.Python.Character].ForeColor = Color.FromArgb(0x7F, 0x00, 0x7F);
            scintilla.Styles[Style.Python.Word].ForeColor = Color.Blue;
            scintilla.Styles[Style.Python.Triple].ForeColor = Color.FromArgb(0x7F, 0x00, 0x00);
            scintilla.Styles[Style.Python.TripleDouble].ForeColor = Color.FromArgb(0x7F, 0x00, 0x00);
            scintilla.Styles[Style.Python.ClassName].ForeColor = Color.FromArgb(0x00, 0x00, 0xFF);
            scintilla.Styles[Style.Python.ClassName].Bold = true;
            scintilla.Styles[Style.Python.DefName].ForeColor = Color.FromArgb(0x00, 0x00, 0x7F);
            scintilla.Styles[Style.Python.DefName].Bold = true;
            // scintilla.Styles[Style.Python.Operator].Bold = true;
            // scintilla.Styles[Style.Python.Identifier] ... your keywords styled here
            scintilla.Styles[Style.Python.CommentBlock].ForeColor = Color.FromArgb(0x7F, 0x7F, 0x7F);
            scintilla.Styles[Style.Python.CommentBlock].Italic = true;
            scintilla.Styles[Style.Python.StringEol].ForeColor = Color.FromArgb(0x00, 0x00, 0x00);
            scintilla.Styles[Style.Python.StringEol].BackColor = Color.FromArgb(0xE0, 0xC0, 0xE0);
            scintilla.Styles[Style.Python.StringEol].FillLine = true;

            scintilla.Styles[Style.Python.Word2].ForeColor = Color.DarkRed;
            scintilla.Styles[Style.Python.Decorator].ForeColor = Color.FromArgb(0x80, 0x50, 0x00);
        }

        private void TextEditForm_Load(object sender, EventArgs e)
        {
            m_mainForm.SimulationHandler.StateChanged += SimulationHandler_StateChanged;

            SimulationHandler_StateChanged(this,
                new MySimulationHandler.StateEventArgs(m_mainForm.SimulationHandler.State, m_mainForm.SimulationHandler.State));
        }

        void SimulationHandler_StateChanged(object sender, MySimulationHandler.StateEventArgs e)
        {
            scintilla.Enabled = e.NewState == MySimulationHandler.SimulationState.STOPPED;

            SetupEditorStyle();
        }

        private void TextEditForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            m_mainForm.SimulationHandler.StateChanged -= SimulationHandler_StateChanged;
        }
    }
}
