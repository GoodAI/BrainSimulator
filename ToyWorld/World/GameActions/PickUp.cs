using VRageMath;
using World.Atlas;
using World.GameActors;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;
using World.Physics;
using World.ToyWorldCore;

namespace World.GameActions
{
    public class PickUp : GameAction
    {
        public PickUp(GameActor sender) : base(sender) { }

        public override void Resolve(GameActorPosition target, IAtlas atlas, ITilesetTable table)
        {
            ICanPickGameObject picker = Sender as ICanPickGameObject;
            IPickableGameActor pickItem = target.Actor as IPickableGameActor;

            if (picker == null || pickItem == null) return;

            if (picker.AddToInventory(pickItem))
                atlas.Remove(target);
        }
    }

    public class LayDown : GameAction
    {
        public LayDown(GameActor sender) : base(sender) { }

        public override void Resolve(GameActorPosition target, IAtlas atlas, ITilesetTable table)
        {
            ICanPickGameObject picker = Sender as ICanPickGameObject;
            ICharacter character = Sender as ICharacter;
            if (picker == null || character == null) return;
            Vector2 positionInFrontOf = target.Position;

            // solving case when positions where character should place tile collides with character's position
            if (target.Actor is Tile)
            {
                Tile tile = (target.Actor as Tile);
                IPhysicalEntity physicalEntity = tile.GetPhysicalEntity(new Vector2I(positionInFrontOf));
                bool collidesWithSource = physicalEntity.CollidesWith(character.PhysicalEntity);
                if (collidesWithSource)
                {
                    /*  <- change to //* to switch
                    // to the center of current tile
                    character.Position = Vector2.Floor(character.Position) + Vector2.One/2;
                    
                    /*/
                    // back WRT his direction
                    do
                    {
                        character.Position = Physics.Utils.Move(character.Position, character.Direction, -0.01f);
                    } while (physicalEntity.CollidesWith(character.PhysicalEntity));
                    // */
                }
            }

            GameActorPosition toLayDown = new GameActorPosition(target.Actor, positionInFrontOf, target.Layer);

            bool added = atlas.Add(toLayDown);
            if (added)
            {
                picker.RemoveFromInventory();
            }
        }
    }
}