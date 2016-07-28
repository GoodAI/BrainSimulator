using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using GoodAI.Core.Utils;
using GoodAI.School.Learning_tasks;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D1 LT1 - 1 shape")]
    public class Ltsct1 : AbstractLearningTask<RoguelikeWorld>
    {
        private readonly Random m_rndGen = new Random();
        protected bool[][] GenerationsCheckTable;
        protected ScFixPositions Positions;
        protected ScFixColors Colors;
        protected StreamWriter StreamWriter;

        public Ltsct1() : this(null) { }
        public virtual string Path { get { return @"D:\summerCampSamples\D1\SCT1\"; } }

        public bool LtWritten = false;

        public const int NUM_SAMPLES = 101;

        public Ltsct1(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000000000 },
                { TSHintAttributes.IMAGE_NOISE, 1},
                { TSHintAttributes.IMAGE_NOISE_BLACK_AND_WHITE, 1}
            };

            TSProgression.Add(TSHints.Clone());
        }

        public override int NumberOfSuccessesRequired
        {
            get { return NUM_SAMPLES; }  // default value
        }

        public virtual void InitCheckTable()
        {
            GenerationsCheckTable = new bool[ScConstants.numPositions][];
            for (int i = 0; i < GenerationsCheckTable.Length; i++)
            {
                GenerationsCheckTable[i] = new bool[ScConstants.numShapes];
            }
        }

        public override void Init()
        {
            base.Init();
            InitCheckTable();
            Positions = new ScFixPositions(WrappedWorld.GetPowGeometry());
            Colors = new ScFixColors(ScConstants.numColors, WrappedWorld.BackgroundColor);
            WrappedWorld.ImageNoiseStandardDeviation = 256.0f / ScConstants.numColors / 2;

            if (Directory.Exists(Path))
            {
                LtWritten = true;
            }
            else
            {
                Directory.CreateDirectory(Path);
                var path = Path + "labels.csv";
                File.Create(path).Close();
                OpenFileStream();
            }
        }

        public override void Fini()
        {
            if (!LtWritten)
            {
                StreamWriter.Close();
                StreamWriter.Dispose();
                LtWritten = true;
            }
        }

        private void OpenFileStream()
        {
            if (LtWritten)
            {
                // nothing
            }
            else
            {
                var path = Path + "labels.csv";
                StreamWriter = new StreamWriter(path, false);
            }
        }


        public override void PresentNewTrainingUnit()
        {
            WrappedWorld.CreateNonVisibleAgent();
            CreateScene();
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            

            wasUnitSuccessful = true;

            if (CheckTableFull())
            {
                MyLog.INFO.WriteLine("Set Is Complete!");
                //wasUnitSuccessful = true;
            }

            return true;
        }

        protected virtual bool CheckTableFull()
        {
            return !GenerationsCheckTable.Any(b => b.Any(b1 => b1 == false));
        }

        public AvatarsActions Actions { get; protected set; }

        protected virtual void CreateScene()
        {
            Actions = new AvatarsActions();

            if (m_rndGen.Next(ScConstants.numShapes + 1) > 0)
            {
                int randomLocationIdx = m_rndGen.Next(ScConstants.numPositions);
                AddShape(randomLocationIdx);

                Actions.Shapes[ShapeIndex] = true;   
            }

            Actions.WriteActions(StreamWriter);string joinedActions = Actions.ToString();MyLog.INFO.WriteLine(joinedActions);
            MyLog.INFO.WriteLine(joinedActions);
        }

        protected int ShapeIndex;
        protected int ColorIndex;

        protected virtual void AddShape(int randomLocationIndex)
        {
            SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width/4, WrappedWorld.GetPowGeometry().Height/4);

            Color color = Colors.GetRandomColor(m_rndGen, out ColorIndex);

            PointF location = Positions.Positions[randomLocationIndex];

            ShapeIndex = m_rndGen.Next(ScConstants.numShapes);
            Shape.Shapes randomShape = (Shape.Shapes)ShapeIndex;

            WrappedWorld.CreateShape(randomShape, color, location, size);

            GenerationsCheckTable[randomLocationIndex][ShapeIndex] = true;
        }

        protected virtual void AddShape()
        {
            SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width / 4, WrappedWorld.GetPowGeometry().Height / 4);

            Color color = Colors.GetRandomColor(m_rndGen, out ColorIndex);

            PointF location = WrappedWorld.RandomPositionInsidePowNonCovering(m_rndGen, size);

            ShapeIndex = m_rndGen.Next(ScConstants.numShapes);
            Shape.Shapes randomShape = (Shape.Shapes)ShapeIndex;

            WrappedWorld.CreateShape(randomShape, color, location, size);
        }

        protected bool[] MoveActionsToTarget(PointF locationCenter, PointF center)
        {
            bool[] moveActions = new bool[4];

            float step = WrappedWorld.GetPowGeometry().Width/16;
            PointF c = locationCenter - new SizeF(center);
            if (c.X < -step) moveActions[2] = true;
            if (step < c.X) moveActions[3] = true;
            if (c.Y < -step) moveActions[0] = true;
            if (step < c.Y) moveActions[1] = true;

            return moveActions;
        }

        protected bool[] MoveActionsToTarget(PointF locationCenter)
        {
            return MoveActionsToTarget(locationCenter, WrappedWorld.GetPowCenter());
        }

        protected static bool[] NegateMoveActions(bool[] actions)
        {
            bool[] actionsN = new bool[4];
            if (actions[0]) actionsN[1] = true;

            if (actions[1]) actionsN[0] = true;

            if (actions[2]) actionsN[3] = true;

            if (actions[3]) actionsN[2] = true;
           
            return actionsN;
        }
    }
}
