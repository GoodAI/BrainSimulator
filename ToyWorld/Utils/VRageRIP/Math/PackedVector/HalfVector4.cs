using System;
using System.Runtime.InteropServices;

namespace VRageMath.PackedVector
{
    /// <summary>
    /// Packed vector type containing four 16-bit floating-point values.
    /// </summary>
    [StructLayout(LayoutKind.Explicit, Size = 8)]
    public struct HalfVector4 : IPackedVector<ulong>, IEquatable<HalfVector4>
    {
        public unsafe struct F2
        {
            public fixed ushort RawValues[4];
        }

        [FieldOffset(0)]
        public F2 Values;

        [FieldOffset(0)]
        public ulong Packed;

        public ulong PackedValue { get { return Packed; } set { Packed = value; } }

        [FieldOffset(0)]
        public ushort X;

        [FieldOffset(2)]
        public ushort Y;

        [FieldOffset(4)]
        public ushort Z;

        [FieldOffset(6)]
        public ushort W;


        /// <summary>
        /// Initializes a new instance of the HalfVector4 class.
        /// </summary>
        /// <param name="x">Initial value for the x component.</param><param name="y">Initial value for the y component.</param><param name="z">Initial value for the z component.</param><param name="w">Initial value for the w component.</param>
        public HalfVector4(ushort x, ushort y, ushort z, ushort w)
        {
            Packed = 0;
            X = x;
            Y = y;
            Z = z;
            W = w;
        }

        /// <summary>
        /// Initializes a new instance of the HalfVector4 class.
        /// </summary>
        /// <param name="x">Initial value for the x component.</param><param name="y">Initial value for the y component.</param><param name="z">Initial value for the z component.</param><param name="w">Initial value for the w component.</param>
        public HalfVector4(int x, int y, int z, int w)
            : this((ushort)x, (ushort)y, (ushort)z, (ushort)w)
        { }

        /// <summary>
        /// Initializes a new instance of the HalfVector4 class.
        /// </summary>
        /// <param name="x">Initial value for the x component.</param><param name="y">Initial value for the y component.</param><param name="z">Initial value for the z component.</param><param name="w">Initial value for the w component.</param>
        public HalfVector4(float x, float y, float z, float w)
            : this(
                HalfUtils.Pack(x),
                HalfUtils.Pack(y),
                HalfUtils.Pack(z),
                HalfUtils.Pack(w))
        { }

        /// <summary>
        /// Initializes a new instance of the HalfVector4 structure.
        /// </summary>
        /// <param name="vector">A vector containing the initial values for the components of the HalfVector4 structure.</param>
        public HalfVector4(Vector4 vector)
            : this(vector.X, vector.Y, vector.Z, vector.W)
        { }

        /// <summary>
        /// Compares the current instance of a class to another instance to determine whether they are the same.
        /// </summary>
        /// <param name="a">The object to the left of the equality operator.</param><param name="b">The object to the right of the equality operator.</param>
        public static bool operator ==(HalfVector4 a, HalfVector4 b)
        {
            return a.Equals(b);
        }

        /// <summary>
        /// Compares the current instance of a class to another instance to determine whether they are different.
        /// </summary>
        /// <param name="a">The object to the left of the equality operator.</param><param name="b">The object to the right of the equality operator.</param>
        public static bool operator !=(HalfVector4 a, HalfVector4 b)
        {
            return !a.Equals(b);
        }

        public static HalfVector4 operator +(HalfVector4 a, ushort val)
        {
            return new HalfVector4(
                a.X + val,
                a.Y + val,
                a.Z + val,
                a.W + val);
        }

        void IPackedVector.PackFromVector4(Vector4 vector)
        {
            PackHelper(vector.X, vector.Y, vector.Z, vector.W);
        }

        private void PackHelper(float x, float y, float z, float w)
        {
            X = HalfUtils.Pack(x);
            Y = HalfUtils.Pack(y);
            Z = HalfUtils.Pack(z);
            W = HalfUtils.Pack(w);
        }

        /// <summary>
        /// Expands the packed representation into a Vector4.
        /// </summary>
        public Vector4 ToVector4()
        {
            Vector4 vector4;
            vector4.X = HalfUtils.Unpack(X);
            vector4.Y = HalfUtils.Unpack(Y);
            vector4.Z = HalfUtils.Unpack(Z);
            vector4.W = HalfUtils.Unpack(W);
            return vector4;
        }

        /// <summary>
        /// Returns a string representation of the current instance.
        /// </summary>
        public override string ToString()
        {
            return this.ToVector4().ToString();
        }

        /// <summary>
        /// Gets the hash code for the current instance.
        /// </summary>
        public override int GetHashCode()
        {
            return PackedValue.GetHashCode();
        }

        /// <summary>
        /// Returns a value that indicates whether the current instance is equal to a specified object.
        /// </summary>
        /// <param name="obj">The object with which to make the comparison.</param>
        public override bool Equals(object obj)
        {
            if (obj is HalfVector4)
                return this.Equals((HalfVector4)obj);
            else
                return false;
        }

        /// <summary>
        /// Returns a value that indicates whether the current instance is equal to a specified object.
        /// </summary>
        /// <param name="other">The object with which to make the comparison.</param>
        public bool Equals(HalfVector4 other)
        {
            return this.PackedValue.Equals(other.PackedValue);
        }
    }
}
