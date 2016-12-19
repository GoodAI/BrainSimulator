using World.Atlas;
using World.GameActors;

namespace World.GameActions
{
    public abstract class GameAction
    {
        protected GameActor Sender { get; private set; }

        protected GameAction(GameActor sender)
        {
            Sender = sender;
        }


        /// <summary>
        /// Resolve implements default action implementation (where applicable)
        /// </summary>
        /// <param name="target">Target of the action</param>
        /// <param name="atlas"></param>
        public abstract void Resolve(GameActorPosition target, IAtlas atlas);
    }
}