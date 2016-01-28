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

        public LTMovingTargetD(RoguelikeWorld w)
            : base(w)
        {
            TSHints[TSHintAttributes.DEGREES_OF_FREEDOM] = 2;           // Degrees of freedom initialised to 2: move in 4 directions (1 means move only right-left)
            TSHints[TSHintAttributes.DEPRECATED_MAX_TARGET_DISTANCE] = 0.9f;
            angle = 0f;
        }

        public override void UpdateState()                              // UpdateState calls base's equivalent and then its own additional functions
        {
            base.UpdateState();

            // The movement of the moving target is circular, and this circular movement is represented by an ellipse inside a rectangle (the Screen)

            int stepsForEllipse = 1000;                                  // How many steps will denote one complete cycle of the ellipse

            angle += (float)(1f) / stepsForEllipse;                     // Increase the angle (which will produce the coordinates that will be used to generate the point to move towards to)

            if (angle > 1f )
            {
                angle = 0;
            }

            Trajectory = GetPointInEllipse(angle, 1f);                  // Find the current coordinates to follow
            MoveTowardsPoint(Trajectory.X, Trajectory.Y, 2);            // Move target towards the desired Trajectory

             //MyLog.DEBUG.WriteLine("angle : " + angle);
        }

        
        public override void CreateTarget()
        {
            //base.CreateTarget();
            
            m_target = new GameObject(GameObjectType.None, GetTargetImage((int)TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS]), 0, 0);
            Point RandomPoint = new Point();
            float NewNumber = (float)m_rndGen.NextDouble();
            RandomPoint = GetPointInEllipse(NewNumber, 1f);                        // Find a random point in the ellipse
            angle = NewNumber;

            m_target.X = RandomPoint.X;
            m_target.Y = RandomPoint.Y;

            World.AddGameObject(m_target);
        }



        // The ellipse's coordinates are calculated using a canonical form in polar coordinates 
        // angle is the angle, it varies from 0 to 360 degrees, but it's in radians, so the range is from 0 to 2Pi 
        // For details: https://en.wikipedia.org/wiki/Ellipse#Parametric_form_in_canonical_position

        public Point GetPointInEllipse(float angle, float EllipseRatioSize)    // angle ranges from 0 to 2Pi, EllipseRatioSize ranges from 0 to 1 
        {
            Point ReturnResult = new Point();

            float width = World.FOW_WIDTH / 2;                                 // Width of the rectangle that will contain the ellipse
            float height = World.FOW_HEIGHT / 2;                               // Height of the rectangle that will contain the ellipse

            float Cx = World.FOW_WIDTH / 2;                                    // X coordinate of the center of the rectangle
            float Cy = World.FOW_HEIGHT / 2;                                   // Y coordinate of the center of the rectangle

            ReturnResult.X = (int)(Cx + (width / 2) * Math.Cos(angle * (Math.PI * 2)));        // Get the x coordinate of a point in the trajectory represented as an ellipse (with variable angle)
            ReturnResult.Y = (int)(Cy + (height / 2) * Math.Sin(angle * (Math.PI * 2)));       // Get the y coordinate of a point in the trajectory represented as an ellipse (with variable angle)

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


    }
}
