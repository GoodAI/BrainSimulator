using System;
using System.Collections.Generic;
using System.Text;
using System.Windows.Forms;
using YAXLib;
using YAXLibTests.SampleClasses;
using System.Linq;
using YAXLibTests;

namespace DemoApplication
{
    public partial class FormMain : Form
    {
        public FormMain()
        {
            InitializeComponent();
            InitListOfClasses();
            InitComboBoxes();
        }

        private void InitComboBoxes()
        {
            comboPolicy.Items.AddRange(Enum.GetNames(typeof(YAXExceptionHandlingPolicies)));
            comboErrorType.Items.AddRange(Enum.GetNames(typeof(YAXExceptionTypes)));
            comboOptions.Items.AddRange(Enum.GetNames(typeof(YAXSerializationOptions)));

            if (comboPolicy.Items.Count > 0)
                comboPolicy.Text = YAXExceptionHandlingPolicies.DoNotThrow.ToString();
            if(comboErrorType.Items.Count > 0)
                comboErrorType.Text = YAXExceptionTypes.Error.ToString();
            if (comboOptions.Items.Count > 0)
                comboOptions.Text = YAXSerializationOptions.SerializeNullObjects.ToString();
        }

        private YAXExceptionTypes GetSelectedDefaultExceptionType()
        {
            return (YAXExceptionTypes)Enum.Parse(typeof(YAXExceptionTypes), comboErrorType.Text);
        }

        private YAXExceptionHandlingPolicies GetSelectedExceptionHandlingPolicy()
        {
            return (YAXExceptionHandlingPolicies)Enum.Parse(typeof(YAXExceptionHandlingPolicies), comboPolicy.Text);
        }

        private YAXSerializationOptions GetSelectedSerializationOption()
        {
            return (YAXSerializationOptions)Enum.Parse(typeof(YAXSerializationOptions), comboOptions.Text);
        }

        private void InitListOfClasses()
        {
            var autoLoadTypes = typeof (Book).Assembly.GetTypes()
                .Where(t => t.GetCustomAttributes(typeof (ShowInDemoApplicationAttribute), false).Any())
                .Select(t => new
                             {
                                 Type = t,
                                 Attr = t.GetCustomAttributes(typeof (ShowInDemoApplicationAttribute), false)
                                     .FirstOrDefault()
                                     as ShowInDemoApplicationAttribute
                             })
                .Select(pair =>
                        {
                            string sortKey = pair.Type.Name;
                            var attr = pair.Attr;
                            if (attr != null && !String.IsNullOrEmpty(attr.SortKey))
                                sortKey = attr.SortKey;
                            string sampleInstanceMethod = "GetSampleInstance";
                            if (attr != null && !String.IsNullOrEmpty(attr.GetSampleInstanceMethodName))
                                sampleInstanceMethod = attr.GetSampleInstanceMethodName;

                            return
                                new {Type = pair.Type, SortKey = sortKey, SampleInstanceMethod = sampleInstanceMethod};
                        }).OrderBy(pair => pair.SortKey);

            var sb = new StringBuilder();
            foreach (var tuple in autoLoadTypes)
            {
                try
                {
                    var type = tuple.Type;
                    var method = type.GetMethod(tuple.SampleInstanceMethod, new Type[0]);
                    var instance = method.Invoke(null, null);
                    lstSampleClasses.Items.Add(new ClassInfoListItem(type, instance));
                }
                catch
                {
                    sb.AppendLine(tuple.Type.FullName);
                }
            }

            if (sb.Length > 0)
            {
                MessageBox.Show("Please provide a parameterless public static method called \"GetSampleInstance\" for the following classes:"
                    + Environment.NewLine + sb.ToString());
            }
        }

        private void btnSerialize_Click(object sender, EventArgs e)
        {
            OnSerialize(false);
        }

        private void btnDeserialize_Click(object sender, EventArgs e)
        {
            OnDeserialize(false);
        }

        private void lstSampleClasses_MouseDoubleClick(object sender, MouseEventArgs e)
        {
            OnSerialize(false);
        }

        private void btnSerializeToFile_Click(object sender, EventArgs e)
        {
            OnSerialize(true);
        }

        private void btnDeserializeFromFile_Click(object sender, EventArgs e)
        {
            OnDeserialize(true);
        }

        private void OnDeserialize(bool openFromFile)
        {
            rtbParsingErrors.Text = "";
            object selItem = lstSampleClasses.SelectedItem;
            if (selItem == null || !(selItem is ClassInfoListItem))
                return;

            string fileName = null;
            if (openFromFile)
            {
                if (DialogResult.OK != openFileDialog1.ShowDialog())
                    return;
                fileName = openFileDialog1.FileName;
            }

            var info = selItem as ClassInfoListItem;
            YAXExceptionTypes defaultExType = GetSelectedDefaultExceptionType();
            YAXExceptionHandlingPolicies exPolicy = GetSelectedExceptionHandlingPolicy();
            YAXSerializationOptions serOption = GetSelectedSerializationOption();

            try
            {
                object deserializedObject = null;
                YAXSerializer serializer = new YAXSerializer(info.ClassType, exPolicy, defaultExType, serOption);
                serializer.MaxRecursion = Convert.ToInt32(numMaxRecursion.Value);

                if (openFromFile)
                    deserializedObject = serializer.DeserializeFromFile(fileName);
                else
                    deserializedObject = serializer.Deserialize(rtbXMLOutput.Text);

                rtbParsingErrors.Text = serializer.ParsingErrors.ToString();

                if (deserializedObject != null)
                {
                    rtbDeserializeOutput.Text = deserializedObject.ToString();

                    if (deserializedObject is List<string>)
                    {
                        StringBuilder sb = new StringBuilder();
                        foreach (var item in deserializedObject as List<string>)
                        {
                            sb.AppendLine(item.ToString());
                        }
                        MessageBox.Show(sb.ToString());
                    }
                }
                else
                    rtbDeserializeOutput.Text = "The deserialized object is null";
            }
            catch (YAXException ex)
            {
                rtbDeserializeOutput.Text = "";
                MessageBox.Show("YAXException handled:\r\n\r\n" + ex.ToString());
            }
            catch (Exception ex)
            {
                rtbDeserializeOutput.Text = "";
                MessageBox.Show("Other Exception handled:\r\n\r\n" + ex.ToString());
            }
        }

        private void OnSerialize(bool saveToFile)
        {
            object selItem = lstSampleClasses.SelectedItem;
            if (selItem == null || !(selItem is ClassInfoListItem))
                return;

            string fileName = null;
            if (saveToFile)
            {
                if (DialogResult.OK != saveFileDialog1.ShowDialog()) 
                    return;
                fileName = saveFileDialog1.FileName;
            }

            ClassInfoListItem info = selItem as ClassInfoListItem;
            YAXExceptionTypes defaultExType = GetSelectedDefaultExceptionType();
            YAXExceptionHandlingPolicies exPolicy = GetSelectedExceptionHandlingPolicy();
            YAXSerializationOptions serOption = GetSelectedSerializationOption();

            try
            {
                YAXSerializer serializer = new YAXSerializer(info.ClassType, exPolicy, defaultExType, serOption);
                serializer.MaxRecursion = Convert.ToInt32(numMaxRecursion.Value);

                if (saveToFile)
                    serializer.SerializeToFile(info.SampleObject, fileName);
                else
                    rtbXMLOutput.Text = serializer.Serialize(info.SampleObject);
                rtbParsingErrors.Text = serializer.ParsingErrors.ToString();
            }
            catch (YAXException ex)
            {
                MessageBox.Show("YAXException handled:\r\n\r\n" + ex.ToString());
            }
            catch (Exception ex)
            {
                MessageBox.Show("Other Exception handled:\r\n\r\n" + ex.ToString());
            }
        }
    }
}
