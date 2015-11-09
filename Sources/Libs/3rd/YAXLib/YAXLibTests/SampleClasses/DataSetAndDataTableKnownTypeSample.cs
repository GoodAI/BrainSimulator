using System.Data;
using System.Text;

namespace YAXLibTests.SampleClasses
{
    public class DataSetAndDataTableKnownTypeSample
    {
        public DataTable TheDataTable { get; set; }

        public DataSet TheDataSet { get; set; }

        public static DataSetAndDataTableKnownTypeSample GetSampleInstance()
        {
            var dataTable = new DataTable("TableName", "http://tableNs/");
            dataTable.Columns.Add(new DataColumn("Col1", typeof(string)));
            dataTable.Columns.Add(new DataColumn("Col2", typeof(int)));
            dataTable.Columns.Add(new DataColumn("Col3", typeof(string)));

            dataTable.Rows.Add("1", 2, "3");
            dataTable.Rows.Add("y", 4, "n");

            var dataTable1 = new DataTable("Table1");
            dataTable1.Columns.Add(new DataColumn("Cl1", typeof(string)));
            dataTable1.Columns.Add(new DataColumn("Cl2", typeof(int)));

            dataTable1.Rows.Add("num1", 34);
            dataTable1.Rows.Add("num2", 54);

            var dataTable2 = new DataTable("Table2");
            dataTable2.Columns.Add(new DataColumn("C1", typeof(string)));
            dataTable2.Columns.Add(new DataColumn("C2", typeof(int)));
            dataTable2.Columns.Add(new DataColumn("C3", typeof(double)));

            dataTable2.Rows.Add("one", 1, 1.5);
            dataTable2.Rows.Add("two", 2, 2.5);

            var dataSet = new DataSet("MyDataSet");
            dataSet.Tables.AddRange(new[] { dataTable1, dataTable2 });

            return new DataSetAndDataTableKnownTypeSample
            {
                TheDataTable = dataTable,
                TheDataSet = dataSet
            };
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine(TheDataTable == null
                              ? "TheDataTable: null"
                              : string.Format("TheDataTable: {0} rows", TheDataTable.Rows.Count));

            sb.AppendLine(TheDataSet == null
                              ? "TheDataSet: null"
                              : string.Format("TheDataSet: {0} tables, {1} rows in the 0th table",
                                              TheDataSet.Tables.Count, TheDataSet.Tables[0].Rows.Count));

            return sb.ToString();

        }
    }
}
