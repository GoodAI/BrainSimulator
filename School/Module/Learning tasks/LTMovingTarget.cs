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
    public class LTMovingTarget : LTApproach                           // Deriving from LTApproach
    {
        float angle;                                                    // Used for the trajectory of the circular movement of the target
        Point Trajectory = new Point();

        int stepsTakenForOneCircle;                                     // How many steps will denote one complete cycle of the ellipse

        private readonly TSHintAttribute MOVING_VELOCITY = new TSHintAttribute("Moving Velocity", "", typeof(float), 0, 1);                 // The velocity of the Target when it's trying to avoid the agent
        private readonly TSHintAttribute ELLIPSE_SIZE = new TSHintAttribute("Ellipse Rectangle Ratio", "", typeof(float), 0, 1);            // The size of the ellipse inside the screen it ranges from 0 to 1: 1.0f is the biggest ellipse that can fit in the POW screen
        private readonly TSHintAttribute STEPS_TAKEN_FOR_ONE_CIRCLE = new TSHintAttribute("Steps for ellipse", "", typeof(float), 0, 1);    // How many steps it takes for one complete cycle of the ellipse
        private readonly TSHintAttribute AVOIDING_AGENT = new TSHintAttribute("If avoiding agent or not", "", typeof(float), 0, 1);         // Either 0 or 1, If it's 1, the target tries to move away from the agent instead of moving ina  circular way

        public LTMovingTarget() : base(null) { }

        public LTMovingTarget(SchoolWorld w)
            : base(w)
        {
            TSHints[TSHintAttributes.DEGREES_OF_FREEDOM] = 2;                   // Set degrees of freedom to 2: move in 4 directions (1 means move only right-left)
            //TSHints[TSHintAttributes.MAX_TARGET_DISTANCE] = 0.3f;

            TSHints[DISTANCE_BONUS_COEFFICENT] = 15f;                            // Coefficent of 4 means that the available steps to reach the target are "initialDistance (between agent and target) * 4" (for more info check LTApproach)

            //Reusing TSHints from LTApproach with some additions
            TSHints.Add(MOVING_VELOCITY, 1);
            TSHints.Add(ELLIPSE_SIZE, 0.55f);
            TSHints.Add(STEPS_TAKEN_FOR_ONE_CIRCLE, 600);
            TSHints.Add(AVOIDING_AGENT, 0);

            TSProgression.Clear();                                              // Clearing TSProgression that was declared in LTApproach before filling custom one

            TSProgression.Add(TSHints.Clone());

            TSProgression.Add(
                new TrainingSetHints {
                    { ELLIPSE_SIZE, 0.55f },
                    { STEPS_TAKEN_FOR_ONE_CIRCLE, 600 },
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { ELLIPSE_SIZE, 0.55f },
                    { STEPS_TAKEN_FOR_ONE_CIRCLE, 500 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { ELLIPSE_SIZE, 0.6f },
                    { STEPS_TAKEN_FOR_ONE_CIRCLE, 400 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { ELLIPSE_SIZE, 0.7f },
                    { STEPS_TAKEN_FOR_ONE_CIRCLE, 300 }
            });


            TSProgression.Add(
                new TrainingSetHints {
                    { ELLIPSE_SIZE, 0.7f },
                    { STEPS_TAKEN_FOR_ONE_CIRCLE, 200 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { ELLIPSE_SIZE, 0.8f },
                    { STEPS_TAKEN_FOR_ONE_CIRCLE, 100 }
            });


            TSProgression.Add(
                new TrainingSetHints {
                    { MOVING_VELOCITY, 2 },
                    { ELLIPSE_SIZE, 0.65f },
                    { STEPS_TAKEN_FOR_ONE_CIRCLE, 1000 },
                    { AVOIDING_AGENT, 1 },
                    { DISTANCE_BONUS_COEFFICENT, 7f }
            });
        }

        public override void ExecuteStep()                                  // UpdateState calls base's equivalent and then its own additional functions
        {
            stepsTakenForOneCircle = (int)TSHints[STEPS_TAKEN_FOR_ONE_CIRCLE];

            // The movement of the moving target is circular, and this circular movement is represented by an ellipse inside a rectangle (the Screen)

            angle += (float)(1f) / stepsTakenForOneCircle;                  // Increase the angle (which will produce the coordinates that will be used to generate the point to move towards to)

            if (angle > 1f)
            {
                angle = 0;
            }

            Trajectory = GetPointInEllipse(angle, (float)TSHints[ELLIPSE_SIZE]);             // Find the current coordinates to follow

            if ((int)TSHints[AVOIDING_AGENT] == 0)
            {
                m_target.X = Trajectory.X;
                m_target.Y = Trajectory.Y;
            }

            // MoveTowardsPoint(Trajectory.X, Trajectory.Y, movingVelocity);    // Move target towards the desired Trajectory

            if ((int)TSHints[AVOIDING_AGENT] == 1)                              // If avoidingAgent is required from the TSHints, compute additional moving step trying to avoid the agent
            {
                MoveAwayFromAgentStep((int)TSHints[MOVING_VELOCITY]);
            }
        }

        public override void CreateTarget()
        {
            //MyLog.INFO.WriteLine(" called CreateTarget() from LTMovingTarget");

            m_target = WrappedWorld.CreateTarget(new Point(0, 0));

            m_target.Width = 30;
            m_target.Height = 30;

            Point RandomPoint = new Point();
            float NewNumber = (float)m_rndGen.NextDouble();

            RandomPoint = GetPointInEllipse(NewNumber, (float)TSHints[ELLIPSE_SIZE]);            // Find a random point in the ellipse
            angle = NewNumber;                                                  // Adapt the new current angle to the point generated

            m_target.X = RandomPoint.X;
            m_target.Y = RandomPoint.Y;

            WrappedWorld.AddGameObject(m_target);
        }

        // The ellipse's coordinates are calculated using a canonical form in polar coordinates
        // angle varies from 0 to 360 degrees, but it's in radians, the range is from 0 to 1 (2Pi)
        // For details: https://en.wikipedia.org/wiki/Ellipse#Parametric_form_in_canonical_position

        public Point GetPointInEllipse(float angle, float EllipseRatioSize)                     // angle ranges from 0 to 1 (2Pi), EllipseRatioSize ranges from 0 to 1  (1 being the biggest ellipse inside the screen)
        {
            Point ReturnResult = new Point();

            float width = WrappedWorld.POW_WIDTH * EllipseRatioSize;                            // Width of the rectangle that will contain the ellipse
            float height = WrappedWorld.POW_HEIGHT * EllipseRatioSize;                          // Height of the rectangle that will contain the ellipse

            float Cx = (WrappedWorld.FOW_WIDTH / 2) - (m_agent.Width / 2);                      // X coordinate of the center of the rectangle
            float Cy = (WrappedWorld.FOW_HEIGHT / 2) - (m_agent.Height / 2);                    // Y coordinate of the center of the rectangle

            ReturnResult.X = (int)(Cx + (width / 2) * Math.Cos(angle * (Math.PI * 2)));         // Get the x coordinate of a point in the trajectory represented as an ellipse (with variable angle)
            ReturnResult.Y = (int)(Cy + (height / 2) * Math.Sin(angle * (Math.PI * 2)));        // Get the y coordinate of a point in the trajectory represented as an ellipse (with variable angle)

            ReturnResult.X -= (m_target.Width / 2);
            ReturnResult.Y -= (m_target.Height / 2);

            return ReturnResult;
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
