using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    /// <author>GoodAI</author>
    /// <meta>Os</meta>
    /// <status>WIP</status>
    /// <summary>"Moving Target" learning task</summary>
    /// <description>
    /// Ability Name: Efficient navigation in a simple environment + moving target. The class is derived from LTApproach with some additions for the moving target (currently: circular movement).
    /// Current behaviour of the target, the target moves itself towards the trajectory (the ellipse contained inside the rectangle (the screen)) and then it follows the trajectory
    /// </description>
    public class LTMovingTargetD : LTApproach                           // Deriving from LTApproach
    {
        float angle;                                                    // Used for the trajectory of the circular movement of the target
        Point Trajectory = new Point();

        float ellipseRectangleRatio = 0.15f;                            // Ranges from 0 to 1, ratio 1 generates the biggest ellipse that can fit into the rectangle
        int stepsForEllipse;                                            // How many steps will denote one complete cycle of the ellipse
        int movingVelocity;                                             // Defined in pixels
        int avoidingAgent;


        private readonly TSHintAttribute MOVING_VELOCITY = new TSHintAttribute("Moving Velocity","",TypeCode.Single,0,1);                   //check needed;
        private readonly TSHintAttribute ELLIPSE_RECTANGLE_RATIO = new TSHintAttribute("Ellipse Rectangle Ratio","",TypeCode.Single,0,1);   //check needed;
        private readonly TSHintAttribute STEPS_FOR_ELLIPSE = new TSHintAttribute("Steps for ellipse", "", TypeCode.Single, 0, 1);           //check needed;
        private readonly TSHintAttribute AVOIDING_AGENT = new TSHintAttribute("If avoiding agent or not","",TypeCode.Single,0,1);           //check needed;

        public LTMovingTargetD(SchoolWorld w)
            : base(w)
        {

            TSHints[TSHintAttributes.DEGREES_OF_FREEDOM] = 2;                   // Set degrees of freedom to 2: move in 4 directions (1 means move only right-left)
            //TSHints[TSHintAttributes.MAX_TARGET_DISTANCE] = 0.3f;

            //Reusing TSHints from LTApproach with some additions
            TSHints.Add(MOVING_VELOCITY, 1);
            TSHints.Add(ELLIPSE_RECTANGLE_RATIO, 0.15f);
            TSHints.Add(STEPS_FOR_ELLIPSE, 1000);
            TSHints.Add(AVOIDING_AGENT, 0);

            TSProgression.Clear();                                              // Clearing TSProgression that was declared in LTApproach before filling custom one

            TSProgression.Add(TSHints.Clone());

            TSProgression.Add(
                new TrainingSetHints {
                    { MOVING_VELOCITY, 2 },
                    { ELLIPSE_RECTANGLE_RATIO, 0.20f },
                    { STEPS_FOR_ELLIPSE, 1000 },
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { MOVING_VELOCITY, 3 },
                    { ELLIPSE_RECTANGLE_RATIO, 0.30f },
                    { STEPS_FOR_ELLIPSE, 700 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { MOVING_VELOCITY, 4 },
                    { ELLIPSE_RECTANGLE_RATIO, 0.40f },
                    { STEPS_FOR_ELLIPSE, 600 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { MOVING_VELOCITY, 5 },
                    { ELLIPSE_RECTANGLE_RATIO, 0.50f },
                    { STEPS_FOR_ELLIPSE, 500 }
            });


            TSProgression.Add(
                new TrainingSetHints {
                    { MOVING_VELOCITY, 8 },
                    { ELLIPSE_RECTANGLE_RATIO, 0.65f },
                    { STEPS_FOR_ELLIPSE, 400 }
            });


            TSProgression.Add(
                new TrainingSetHints {
                    { MOVING_VELOCITY, 4 },
                    { ELLIPSE_RECTANGLE_RATIO, 0.65f },
                    { STEPS_FOR_ELLIPSE, 1000 },
                    { AVOIDING_AGENT, 1 }
            });

            SetHints(TSHints);
        }

        public override void UpdateState()                                  // UpdateState calls base's equivalent and then its own additional functions
        {
            base.UpdateState();

            movingVelocity = (int)TSHints[MOVING_VELOCITY];
            ellipseRectangleRatio = (float)TSHints[ELLIPSE_RECTANGLE_RATIO];
            stepsForEllipse = (int)TSHints[STEPS_FOR_ELLIPSE];
            avoidingAgent = (int)TSHints[AVOIDING_AGENT];

            // The movement of the moving target is circular, and this circular movement is represented by an ellipse inside a rectangle (the Screen)           

            angle += (float)(1f) / stepsForEllipse;                         // Increase the angle (which will produce the coordinates that will be used to generate the point to move towards to)

            if (angle > 1f)
            {
                angle = 0;
            }

            Trajectory = GetPointInEllipse(angle, ellipseRectangleRatio);   // Find the current coordinates to follow
            MoveTowardsPoint(Trajectory.X, Trajectory.Y, movingVelocity);   // Move target towards the desired Trajectory

            if (avoidingAgent == 1)                                         // If avoidingAgent is required from the TSHints, compute additional moving step trying to avoid the agent
            {
                MoveAwayFromAgentStep(movingVelocity - 1);
            }

        }


        public override bool DidTrainingUnitFail()
        {
            return false;
        }


        public override void CreateTarget()
        {
            m_target = new GameObject(GameObjectType.None, GetTargetImage((int)TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS]), 0, 0);

            Point RandomPoint = new Point();
            float NewNumber = (float)m_rndGen.NextDouble();

            RandomPoint = GetPointInEllipse(NewNumber, ellipseRectangleRatio);  // Find a random point in the ellipse
            angle = NewNumber;                                                  // Adapt the new current angle to the point generated

            m_target.X = RandomPoint.X;
            m_target.Y = RandomPoint.Y;

            WrappedWorld.AddGameObject(m_target);
        }


        // The ellipse's coordinates are calculated using a canonical form in polar coordinates 
        // angle varies from 0 to 360 degrees, but it's in radians, the range is from 0 to 1 (2Pi)
        // For details: https://en.wikipedia.org/wiki/Ellipse#Parametric_form_in_canonical_position

        public Point GetPointInEllipse(float angle, float EllipseRatioSize)               // angle ranges from 0 to 1 (2Pi), EllipseRatioSize ranges from 0 to 1  (1 being the biggest ellipse inside the screen)
        {
            Point ReturnResult = new Point();

            float width = WrappedWorld.FOW_WIDTH * EllipseRatioSize;                             // Width of the rectangle that will contain the ellipse
            float height = WrappedWorld.FOW_HEIGHT * EllipseRatioSize;                           // Height of the rectangle that will contain the ellipse

            float Cx = WrappedWorld.FOW_WIDTH / 2;                                               // X coordinate of the center of the rectangle
            float Cy = WrappedWorld.FOW_HEIGHT / 2;                                              // Y coordinate of the center of the rectangle

            ReturnResult.X = (int)(Cx + (width / 2) * Math.Cos(angle * (Math.PI * 2)));   // Get the x coordinate of a point in the trajectory represented as an ellipse (with variable angle)
            ReturnResult.Y = (int)(Cy + (height / 2) * Math.Sin(angle * (Math.PI * 2)));  // Get the y coordinate of a point in the trajectory represented as an ellipse (with variable angle)

            return ReturnResult;

        }

        public void MoveTowardsPoint(int x, int y, int velocity)        // Given a point, the target moves towards it using steps of settable length in pixels (velocity)
        {
            // At each step, the target can move only on 1 axis, so choose the axis depending on the distances
            if (Math.Abs(x - m_target.X) > Math.Abs(y - m_target.Y))    // Check what axis has the smallest difference between 2 points, the movement can happen only by 1 axis at a time
            {
                if (m_target.X < x)
                {
                    m_target.X += velocity;
                }
                else if (m_target.X > x)
                {
                    m_target.X -= velocity;
                }
            }
            else                                                        // Move on Y axis
            {
                if (m_target.Y < y)
                {
                    m_target.Y += velocity;
                }
                else if (m_target.Y > y)
                {
                    m_target.Y -= velocity;
                }
            }
        }


        public void MoveAwayFromAgentStep(int velocity)
        {
            // At each step, the target can move only on 1 axis, so choose the axis depending on the distances
            if (Math.Abs(m_agent.X - m_target.X) > Math.Abs(m_agent.Y - m_target.Y))    // Check what axis has the smallest difference between 2 points, the movement can happen only by 1 axis at a time
            {
                if (m_target.X < m_agent.X)
                {
                    m_target.X -= velocity;
                }
                else if (m_target.X > m_agent.X)
                {
                    m_target.X += velocity;
                }
            }
            else                                                        // Move on Y axis
            {
                if (m_target.Y < m_agent.Y)
                {
                    m_target.Y -= velocity;
                }
                else if (m_target.Y > m_agent.Y)
                {
                    m_target.Y += velocity;
                }
            }
        }


    }
}
