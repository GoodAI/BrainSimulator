using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;

namespace World.ToyWorldCore
{
    public class Atlas
    {
        public List<ITileLayer> TileLayers { get; private set; }

        public List<IObjectLayer> ObjectLayers { get; private set; }

        public Dictionary<int, IAvatar> Avatars { get; private set; }

        public List<Character> Characters { get; private set; }

        public Dictionary<int, StaticTile> StaticTilesContainer { get; private set; }

        public Atlas()
        {
            Avatars = new Dictionary<int, IAvatar>();
            Characters = new List<Character>();
            TileLayers = new List<ITileLayer>();
            ObjectLayers = new List<IObjectLayer>();
            StaticTilesContainer = new Dictionary<int, StaticTile>();
        }

        public object GetLayer(LayerType layerType)
        {
            if (layerType == LayerType.Object
                || layerType == LayerType.ForegroundObject)
            {
                return ObjectLayers.FirstOrDefault(x => x.LayerType == layerType);
            }
            return TileLayers.First(x => x.LayerType == layerType);
        }

        public bool AddAvatar(IAvatar avatar)
        {
            Contract.Requires<ArgumentNullException>(avatar != null, "Avatar cannot be null");
            try
            {
                Avatars.Add(avatar.Id, avatar);
            }
            catch (ArgumentException)
            {
                return false;
            }
            return true;
        }

        public List<IAvatar> GetAvatars()
        {
            Contract.Ensures(Contract.Result<List<IAvatar>>() != null);
            return Avatars.Values.ToList();
        }
    }
}
