using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using GoodAI.ToyWorld;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("Single pixel RL")]
    public class LTSinglePixelRL : AbstractLearningTask<RoguelikeWorld>
    {
        protected enum ObjectType
        {
            Target,
            Empty,
            BeforeTarget,
            None,
        }

        protected Dictionary<ObjectType, int> m_objectColor = new Dictionary<ObjectType, int>();
        protected readonly Random m_rndGen = new Random();
        protected GameObject m_object;
        protected int m_targetsShown { get; set; }
        protected int m_targetsDetected { get; set; }
        protected int m_targetsMisdetected { get; set; }
        protected int m_targetsPerTU { get { return 10; }}
        protected ObjectType m_currentObjectType { get; set; }
        protected ObjectType m_previousObjectType { get; set; }

        public LTSinglePixelRL() : this(null) { }

        public LTSinglePixelRL(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.IMAGE_NOISE, 0 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);

            m_objectColor[ObjectType.Target] = 0;
            m_objectColor[ObjectType.Empty] = 1;
            m_objectColor[ObjectType.BeforeTarget] = 2;
        }

        public override int NumberOfSuccessesRequired
        {
            get { return 1; }  // the training unit is hard enough so that a single success is enough
        }

        public override void PresentNewTrainingUnit()
        {
            m_targetsShown = 0;
            m_targetsDetected = 0;
            m_targetsMisdetected = 0;
            m_previousObjectType = ObjectType.None;
            m_currentObjectType = ObjectType.None;
            m_object = null;

            ExecuteStep();
        }

        public override void Init()
        {
            m_currentObjectType = ObjectType.None;
            base.Init();
        }

        public override void ExecuteStep()
        {
            if (m_object != null)
                WrappedWorld.RemoveGameObject(m_object);
                
            if (LearningTaskHelpers.FlipBiasedCoin(m_rndGen, 0.33f))
            {
                WrappedWorld.CreateNonVisibleAgent();
                CreateObject(ObjectType.Target);
                m_targetsShown++;
            }
            else
            {
                CreateObject(ObjectType.Empty);
            }
        }

        public override TrainingResult EvaluateStep()
        {
            SchoolWorld.ActionInput.SafeCopyToHost();
            bool wasTargetDetected = SchoolWorld.ActionInput.Host[ControlMapper.Idx("forward")] != 0;
            bool wasTargetPresent = m_previousObjectType == ObjectType.Target;
            if (wasTargetDetected && wasTargetPresent)
            {
                m_targetsDetected++;
                WrappedWorld.Reward.Host[0] = 1f;
            }
            else if (m_previousObjectType != ObjectType.None)
            {
                if (wasTargetDetected && !wasTargetPresent)
                {
                    m_targetsMisdetected++;
                    WrappedWorld.Reward.Host[0] = -1f;
                }
                else if (wasTargetPresent && !wasTargetDetected)
                {
                    WrappedWorld.Reward.Host[0] = -1f;
                }
                else
                {
                    WrappedWorld.Reward.Host[0] = 0f;
                }
            }
            else
            {
                WrappedWorld.Reward.Host[0] = 0;
            }

            return base.EvaluateStep();
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            int correction = m_currentObjectType == ObjectType.Target ? 1 : 0;

            wasUnitSuccessful = (m_targetsShown - correction == m_targetsDetected && m_targetsMisdetected == 0);

            return m_targetsShown - correction == m_targetsPerTU;
            // - correction, because when ending, ExecuteStep is done before the last EvaluateStep - the ExecuteStep may prepare a new state which should be ignored by EvaluateStep
        }

        protected void CreateObject(ObjectType objectType)
        {
            m_previousObjectType = m_currentObjectType;
            m_currentObjectType = objectType;

            CreateTarget();

            SetTargetColor(m_objectColor[objectType]);
        }

        protected void CreateTarget1()
        {
            SizeF size = new SizeF(SchoolWorld.WidthFov, SchoolWorld.HeightFov);
            m_object = new Shape(Shape.Shapes.Square, PointF.Empty, size);
            m_object.GetCenter();
            WrappedWorld.AddGameObject(m_object);
        }

        protected void CreateTarget()
        {
            m_object = new Shape(Shape.Shapes.Square, new PointF(0f,0f), new SizeF(WrappedWorld.Scene.Width, WrappedWorld.Scene.Height));
            WrappedWorld.AddGameObject(m_object);
        }

        protected void SetTargetColor(int colorIndex)
        {
            m_object.IsBitmapAsMask = true;

            Color color = LearningTaskHelpers.GetVisibleGrayscaleColor(colorIndex);
            m_object.ColorMask = Color.FromArgb(
                AddRandomColorOffset(color.R),
                AddRandomColorOffset(color.G),
                AddRandomColorOffset(color.B));
        }

        protected byte AddRandomColorOffset(byte colorComponent)
        {
            if (TSHints[TSHintAttributes.IMAGE_NOISE] != 1.0f)
                return colorComponent;
            const int MAX_RANDOM_OFFSET = 10;
            return (byte)Math.Max(0, Math.Min(255, colorComponent + m_rndGen.Next(-MAX_RANDOM_OFFSET, MAX_RANDOM_OFFSET + 1)));
        }
    }
}
