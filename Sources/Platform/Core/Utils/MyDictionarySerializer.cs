using GoodAI.Core.Task;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Linq;
using YAXLib;

namespace GoodAI.Core.Utils
{
    public class MyTaskSerializer : ICustomSerializer<Dictionary<string, MyTask>>
    {
        internal const string NO_PROPERTY_NAME_KEY = "#NO_PROPERTY_NAME#";

        public void SerializeToElement(Dictionary<string, MyTask> tasksToSerialize, XElement elemToFill)
        {
            if (tasksToSerialize == null)
                return;

            foreach (string taskPropName in tasksToSerialize.Keys) {

                MyTask task = tasksToSerialize[taskPropName];
                task.PropertyName = taskPropName;

                YAXSerializer serializer = new YAXSerializer(typeof(MyTask), YAXExceptionHandlingPolicies.ThrowErrorsOnly, YAXExceptionTypes.Warning);           
                
                XElement taskElement = serializer.SerializeToXDocument(task).Root;                
                XAttribute ns = taskElement.Attributes().FirstOrDefault(a => a.Name.LocalName == "yaxlib");
                if (ns != null) ns.Remove();
               
                elemToFill.Add(taskElement);
            }
        }

        public Dictionary<string, MyTask> DeserializeFromElement(XElement element)
        {
            Dictionary<string, MyTask> result = new Dictionary<string, MyTask>();

            foreach (XElement child in element.Elements())
            {
                YAXSerializer serializer = new YAXSerializer(typeof(MyTask), YAXExceptionHandlingPolicies.ThrowErrorsOnly, YAXExceptionTypes.Warning);

                try
                {
                    MyTask task = (MyTask)serializer.Deserialize(child);

                    if (task.PropertyName == null) 
                    {
                        task.PropertyName = NO_PROPERTY_NAME_KEY + task.GetType().FullName;
                    }
                    
                    result[task.PropertyName] = task;                    
                }
                catch
                {
                    MyLog.WARNING.WriteLine("Task deserialization failed for: " + child);
                }                
            }
            return result;
        }

        public Dictionary<string, MyTask> DeserializeFromAttribute(XAttribute attrib)
        {
            throw new NotImplementedException();
        }

        public Dictionary<string, MyTask> DeserializeFromValue(string value)
        {
            throw new NotImplementedException();
        }

        public void SerializeToAttribute(Dictionary<string, MyTask> objectToSerialize, XAttribute attrToFill)
        {
            throw new NotImplementedException();
        }

        public string SerializeToValue(Dictionary<string, MyTask> objectToSerialize)
        {
            throw new NotImplementedException();
        }
    }
}
