using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Windows.Forms;
using System.Xml;
using System.Xml.Serialization;
using System.IO;
using TmxMapSerializer.Elements;
using System.Text;

namespace MapsMerger
{
    public partial class Form1 : Form
    {
        private List<List<string>> m_filenames;
        private List<List<Map>> m_mergeList;
        private XmlSerializer m_compositionXmlSerializer;
        private TmxMapSerializer.Serializer.TmxMapSerializer m_mapSerializer;

        public Form1()
        {
            InitializeComponent();

            InitializeSerializer();
        }

        private void InitializeSerializer()
        {
            XmlAttributes attrs = new XmlAttributes();
            XmlAttributeOverrides attrOverrides = new XmlAttributeOverrides();
            attrOverrides.Add(typeof(string), "File", attrs);
            attrOverrides.Add(typeof(List<string>), "Columns", attrs);
            attrOverrides.Add(typeof(List<List<string>>), "Rows", attrs);

            m_compositionXmlSerializer = new XmlSerializer(typeof(List<List<string>>), attrOverrides);

            m_mapSerializer = new TmxMapSerializer.Serializer.TmxMapSerializer();
        }

        private void openFileDialogLoadComposition_FileOk(object sender, CancelEventArgs e)
        {
            m_mergeList = LoadCompositionFile(openFileDialogLoadComposition.FileName);
            if (m_mergeList != null)
            {
                richTextBoxLog.AppendText(openFileDialogLoadComposition.FileName + " loaded successfully\n");
            }
        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            openFileDialogLoadComposition.ShowDialog();
        }

        private void toolStripButtonSaveComposition_Click(object sender, EventArgs e)
        {
            m_filenames = new List<List<string>>();
            for (int i = 0; i < 5; i++)
            {
                m_filenames.Add(new List<string>());
                for (int j = 0; j < 5; j++)
                {
                    m_filenames.Last().Add(@"C:\Program files\" + i + j);
                }
            }

            saveFileDialogSaveComposition.ShowDialog();
        }

        private List<List<Map>> LoadCompositionFile(string fileName)
        {
            FileStream compositionFile = new FileStream(fileName, FileMode.Open);
            XmlReader xmlReader = XmlReader.Create(compositionFile);
            m_filenames = m_compositionXmlSerializer.Deserialize(xmlReader) as List<List<string>>;
            xmlReader.Close();
            List<List<Map>> f = new List<List<Map>>();

            if (m_filenames != null)
                foreach (var row in m_filenames)
                {
                    f.Add(new List<Map>());
                    foreach (var filename in row)
                    {
                        try
                        {
                            f.Last().Add(m_mapSerializer.Deserialize(XmlReader.Create(filename)) as Map);
                        }
                        catch (FileNotFoundException e)
                        {
                            richTextBoxLog.AppendText("Can't find file " + e.FileName + "\n");
                            return null;
                        }
                    }
                }
            else
                richTextBoxLog.AppendText("No filenames found in composition file " + fileName);

            return f;
        }

        private void saveFileDialogSaveComposition_FileOk(object sender, CancelEventArgs e)
        {
            XmlWriterSettings xmlSettings = new XmlWriterSettings();
            xmlSettings.Indent = true;
            xmlSettings.NewLineOnAttributes = true;

            XmlWriter xw = XmlWriter.Create(saveFileDialogSaveComposition.FileName, xmlSettings);

            m_compositionXmlSerializer.Serialize(xw, m_filenames);
            xw.Flush();
            xw.Close();
        }

        private void saveFileDialogSaveMerged_FileOk(object sender, CancelEventArgs e)
        {
            StreamWriter sw = new StreamWriter(saveFileDialogSaveMerged.FileName);
            m_mapSerializer.Serialize(sw, m_mergeList[0][0]);
            sw.Close();
        }

        private void toolStripButtonMerge_Click(object sender, EventArgs e)
        {
            richTextBoxLog.AppendText("Start merging...\n");
            var mapsMerger = new MapsMerger();
            mapsMerger.Merge(m_mergeList);
            richTextBoxLog.AppendText("...End merging\n");
            saveFileDialogSaveMerged.ShowDialog();
        }

    }
}
