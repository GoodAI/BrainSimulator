using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using ManagedCuda.BasicTypes;
using System;
using System.IO;
using System.Collections.Generic;

namespace GoodAI.Core.Memory
{
    public class MyMemoryBlockSerializer
    {        
        private byte[] m_buffer = new byte[8192];

        public static string GetTempStorage(MyProject project)
        {
            return GetTempStorage(project.Name);
        }

        public static string GetTempStorage(String projectName)
        {
            return Path.GetTempPath() + "\\bs_temporal\\" + projectName + ".statedata";
        }

        public static void ClearTempStorage(MyProject project)
        {
            if (TempDataExists(project))
            {
                Directory.Delete(GetTempStorage(project), true);
                MyLog.INFO.WriteLine(project.Name + ": Temporal data deleted.");
            }
            else
            {
                MyLog.WARNING.WriteLine(project.Name + ": No temporal data to delete.");
            }
        }

        public static void ClearTempData(MyNode node)
        {
            if (TempDataExists(node))
            {
                Directory.Delete(GetTempStorage(node.Owner) + "\\" + GetNodeFolder(node), true);
                MyLog.INFO.WriteLine(node.Name + ": Temporal data deleted.");
            }
            else
            {
                MyLog.WARNING.WriteLine(node.Name + ": No temporal data to delete.");
            }
        }

        public static void ExportTempStorage(MyProject project, string outputFolder)
        {
            string tempStorage = GetTempStorage(project);

            foreach (string path in Directory.GetDirectories(tempStorage, "*", SearchOption.AllDirectories))
            {
                Directory.CreateDirectory(path.Replace(tempStorage, outputFolder));
            }
            
            foreach (string filePath in Directory.GetFiles(tempStorage, "*.mb", SearchOption.AllDirectories))
            {
                File.Copy(filePath, filePath.Replace(tempStorage, outputFolder), true);
            }
        }

        public static string GetNodeFolder(MyNode node)
        {
            return node.GetType().Name + "_" + node.Id;
        }

        public static string GetFileName(MyAbstractMemoryBlock memoryBlock)
        {
            return memoryBlock.Name + ".mb";
        }

        public static string GetUniqueName(MyAbstractMemoryBlock memoryBlock)
        {
            return memoryBlock.Owner.Id.ToString() + "#" + memoryBlock.Name;
        }

        public static bool TempDataExists(MyProject project)
        {
            return Directory.Exists(GetTempStorage(project));
        }

        public static bool TempDataExists(MyNode node)
        {
            return Directory.Exists(GetTempStorage(node.Owner) + "\\" + GetNodeFolder(node));            
        }

        private static string ResolveFilePath(MyAbstractMemoryBlock memoryBlock, string globalDataFolder)
        {
            string fileName = GetTempStorage(memoryBlock.Owner.Owner) + "\\" + GetNodeFolder(memoryBlock.Owner) + "\\" + GetFileName(memoryBlock);

            if (!File.Exists(fileName))
            {
                MyWorkingNode node = memoryBlock.Owner as MyWorkingNode;                

                if (!string.IsNullOrEmpty(node.DataFolder))
                {
                    fileName = node.DataFolder;                    
                }
                else if (!string.IsNullOrEmpty(globalDataFolder))
                {
                    fileName = globalDataFolder + "\\" + GetNodeFolder(memoryBlock.Owner);
                }
                else
                {
                    throw new FileNotFoundException(
                        "File not found in temporal folder and no data folder defined: " + fileName);
                }

                fileName += "\\" + GetFileName(memoryBlock);
            }

            if (!File.Exists(fileName))
            {
                throw new FileNotFoundException("Memory block file not found: " + fileName);
            }

            return fileName;
        }

        public void LoadBlock(MyAbstractMemoryBlock memoryBlock, string globalDataFolder)
        {
            int length = m_buffer.Length;
            SizeT size = memoryBlock.GetSize();

            while (size > length)
            {
                length *= 2;
            }

            if (length != m_buffer.Length)
            {
                m_buffer = new byte[length];
            }

            try
            {
                string filePath = ResolveFilePath(memoryBlock, globalDataFolder);
                long fileSize = new FileInfo(filePath).Length;

                if (fileSize != size)
                {
                    throw new InvalidDataException("Different size of a stored memory block (" + fileSize + " B != " + size + " B)");                        
                }

                using (var reader = new BinaryReader(File.OpenRead(filePath)))
                {
                    reader.Read(m_buffer, 0, size);
                }

                memoryBlock.Fill(m_buffer);                    
            }
            catch (Exception e)
            {
                MyLog.WARNING.WriteLine("Memory block loading failed (node: {0} (id: {1}), block: {2}): {3}", memoryBlock.Owner.Name,
                    memoryBlock.Owner.Id, memoryBlock.Name, e.Message);
            }            
        }

        public void SaveBlock(MyAbstractMemoryBlock memoryBlock)
        {
            int length = m_buffer.Length;
            SizeT size = memoryBlock.GetSize();

            while (size > length)
            {
                length *= 2;
            }

            if (length != m_buffer.Length)
            {
                m_buffer = new byte[length];
            }
            memoryBlock.GetBytes(m_buffer);
            
            string tempFolder = GetTempStorage(memoryBlock.Owner.Owner) + "\\" + GetNodeFolder(memoryBlock.Owner);
            Directory.CreateDirectory(tempFolder);

            string filePath = tempFolder + "\\" + GetFileName(memoryBlock);

            try
            {
                using (var writer = new BinaryWriter(File.Open(filePath, FileMode.Create)))
                {
                    writer.Write(m_buffer, 0, size);
                }
            }
            catch (Exception e)
            {
                MyLog.WARNING.WriteLine("Memory block saving failed (node: {0} (id: {1}), block: {2}): {3}", memoryBlock.Owner.Name,
                    memoryBlock.Owner.Id, memoryBlock.Name, e.Message);
            } 
        }
    }
}
