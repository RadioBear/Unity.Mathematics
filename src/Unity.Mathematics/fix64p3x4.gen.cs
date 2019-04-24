//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------
using System;
using System.Runtime.CompilerServices;

#pragma warning disable 0660, 0661

namespace Unity.Mathematics
{
    [System.Serializable]
    public partial struct fix64p3x4 : System.IEquatable<fix64p3x4>, IFormattable
    {
        public fix64p3 c0;
        public fix64p3 c1;
        public fix64p3 c2;
        public fix64p3 c3;

        /// <summary>fix64p3x4 zero value.</summary>
        public static readonly fix64p3x4 zero;

        /// <summary>Constructs a fix64p3x4 matrix from four fix64p3 vectors.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x4(fix64p3 c0, fix64p3 c1, fix64p3 c2, fix64p3 c3)
        { 
            this.c0 = c0;
            this.c1 = c1;
            this.c2 = c2;
            this.c3 = c3;
        }

        /// <summary>Constructs a fix64p3x4 matrix from 12 fix64p values given in row-major order.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x4(fix64p m00, fix64p m01, fix64p m02, fix64p m03,
                         fix64p m10, fix64p m11, fix64p m12, fix64p m13,
                         fix64p m20, fix64p m21, fix64p m22, fix64p m23)
        { 
            this.c0 = new fix64p3(m00, m10, m20);
            this.c1 = new fix64p3(m01, m11, m21);
            this.c2 = new fix64p3(m02, m12, m22);
            this.c3 = new fix64p3(m03, m13, m23);
        }

        /// <summary>Constructs a fix64p3x4 matrix from a single fix64p value by assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x4(fix64p v)
        {
            this.c0 = v;
            this.c1 = v;
            this.c2 = v;
            this.c3 = v;
        }

        /// <summary>Constructs a fix64p3x4 matrix from a single bool value by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x4(bool v)
        {
            this.c0 = math.select(new fix64p3(fix64p.zero), new fix64p3(fix64p.One), v);
            this.c1 = math.select(new fix64p3(fix64p.zero), new fix64p3(fix64p.One), v);
            this.c2 = math.select(new fix64p3(fix64p.zero), new fix64p3(fix64p.One), v);
            this.c3 = math.select(new fix64p3(fix64p.zero), new fix64p3(fix64p.One), v);
        }

        /// <summary>Constructs a fix64p3x4 matrix from a bool3x4 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x4(bool3x4 v)
        {
            this.c0 = math.select(new fix64p3(fix64p.zero), new fix64p3(fix64p.One), v.c0);
            this.c1 = math.select(new fix64p3(fix64p.zero), new fix64p3(fix64p.One), v.c1);
            this.c2 = math.select(new fix64p3(fix64p.zero), new fix64p3(fix64p.One), v.c2);
            this.c3 = math.select(new fix64p3(fix64p.zero), new fix64p3(fix64p.One), v.c3);
        }

        /// <summary>Constructs a fix64p3x4 matrix from a single int value by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x4(int v)
        {
            this.c0 = (fix64p3)v;
            this.c1 = (fix64p3)v;
            this.c2 = (fix64p3)v;
            this.c3 = (fix64p3)v;
        }

        /// <summary>Constructs a fix64p3x4 matrix from a int3x4 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x4(int3x4 v)
        {
            this.c0 = (fix64p3)v.c0;
            this.c1 = (fix64p3)v.c1;
            this.c2 = (fix64p3)v.c2;
            this.c3 = (fix64p3)v.c3;
        }

        /// <summary>Constructs a fix64p3x4 matrix from a single float value by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x4(float v)
        {
            this.c0 = (fix64p3)v;
            this.c1 = (fix64p3)v;
            this.c2 = (fix64p3)v;
            this.c3 = (fix64p3)v;
        }

        /// <summary>Constructs a fix64p3x4 matrix from a float3x4 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x4(float3x4 v)
        {
            this.c0 = (fix64p3)v.c0;
            this.c1 = (fix64p3)v.c1;
            this.c2 = (fix64p3)v.c2;
            this.c3 = (fix64p3)v.c3;
        }


        /// <summary>Implicitly converts a single fix64p value to a fix64p3x4 matrix by assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator fix64p3x4(fix64p v) { return new fix64p3x4(v); }

        /// <summary>Explicitly converts a single bool value to a fix64p3x4 matrix by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator fix64p3x4(bool v) { return new fix64p3x4(v); }

        /// <summary>Explicitly converts a bool3x4 matrix to a fix64p3x4 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator fix64p3x4(bool3x4 v) { return new fix64p3x4(v); }

        /// <summary>Explicitly converts a single int value to a fix64p3x4 matrix by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator fix64p3x4(int v) { return new fix64p3x4(v); }

        /// <summary>Explicitly converts a int3x4 matrix to a fix64p3x4 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator fix64p3x4(int3x4 v) { return new fix64p3x4(v); }

        /// <summary>Explicitly converts a single float value to a fix64p3x4 matrix by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator fix64p3x4(float v) { return new fix64p3x4(v); }

        /// <summary>Explicitly converts a float3x4 matrix to a fix64p3x4 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator fix64p3x4(float3x4 v) { return new fix64p3x4(v); }


        /// <summary>Returns the result of a componentwise multiplication operation on two fix64p3x4 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator * (fix64p3x4 lhs, fix64p3x4 rhs) { return new fix64p3x4 (lhs.c0 * rhs.c0, lhs.c1 * rhs.c1, lhs.c2 * rhs.c2, lhs.c3 * rhs.c3); }

        /// <summary>Returns the result of a componentwise multiplication operation on a fix64p3x4 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator * (fix64p3x4 lhs, fix64p rhs) { return new fix64p3x4 (lhs.c0 * rhs, lhs.c1 * rhs, lhs.c2 * rhs, lhs.c3 * rhs); }

        /// <summary>Returns the result of a componentwise multiplication operation on a fix64p value and a fix64p3x4 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator * (fix64p lhs, fix64p3x4 rhs) { return new fix64p3x4 (lhs * rhs.c0, lhs * rhs.c1, lhs * rhs.c2, lhs * rhs.c3); }


        /// <summary>Returns the result of a componentwise addition operation on two fix64p3x4 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator + (fix64p3x4 lhs, fix64p3x4 rhs) { return new fix64p3x4 (lhs.c0 + rhs.c0, lhs.c1 + rhs.c1, lhs.c2 + rhs.c2, lhs.c3 + rhs.c3); }

        /// <summary>Returns the result of a componentwise addition operation on a fix64p3x4 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator + (fix64p3x4 lhs, fix64p rhs) { return new fix64p3x4 (lhs.c0 + rhs, lhs.c1 + rhs, lhs.c2 + rhs, lhs.c3 + rhs); }

        /// <summary>Returns the result of a componentwise addition operation on a fix64p value and a fix64p3x4 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator + (fix64p lhs, fix64p3x4 rhs) { return new fix64p3x4 (lhs + rhs.c0, lhs + rhs.c1, lhs + rhs.c2, lhs + rhs.c3); }


        /// <summary>Returns the result of a componentwise subtraction operation on two fix64p3x4 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator - (fix64p3x4 lhs, fix64p3x4 rhs) { return new fix64p3x4 (lhs.c0 - rhs.c0, lhs.c1 - rhs.c1, lhs.c2 - rhs.c2, lhs.c3 - rhs.c3); }

        /// <summary>Returns the result of a componentwise subtraction operation on a fix64p3x4 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator - (fix64p3x4 lhs, fix64p rhs) { return new fix64p3x4 (lhs.c0 - rhs, lhs.c1 - rhs, lhs.c2 - rhs, lhs.c3 - rhs); }

        /// <summary>Returns the result of a componentwise subtraction operation on a fix64p value and a fix64p3x4 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator - (fix64p lhs, fix64p3x4 rhs) { return new fix64p3x4 (lhs - rhs.c0, lhs - rhs.c1, lhs - rhs.c2, lhs - rhs.c3); }


        /// <summary>Returns the result of a componentwise division operation on two fix64p3x4 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator / (fix64p3x4 lhs, fix64p3x4 rhs) { return new fix64p3x4 (lhs.c0 / rhs.c0, lhs.c1 / rhs.c1, lhs.c2 / rhs.c2, lhs.c3 / rhs.c3); }

        /// <summary>Returns the result of a componentwise division operation on a fix64p3x4 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator / (fix64p3x4 lhs, fix64p rhs) { return new fix64p3x4 (lhs.c0 / rhs, lhs.c1 / rhs, lhs.c2 / rhs, lhs.c3 / rhs); }

        /// <summary>Returns the result of a componentwise division operation on a fix64p value and a fix64p3x4 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator / (fix64p lhs, fix64p3x4 rhs) { return new fix64p3x4 (lhs / rhs.c0, lhs / rhs.c1, lhs / rhs.c2, lhs / rhs.c3); }


        /// <summary>Returns the result of a componentwise modulus operation on two fix64p3x4 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator % (fix64p3x4 lhs, fix64p3x4 rhs) { return new fix64p3x4 (lhs.c0 % rhs.c0, lhs.c1 % rhs.c1, lhs.c2 % rhs.c2, lhs.c3 % rhs.c3); }

        /// <summary>Returns the result of a componentwise modulus operation on a fix64p3x4 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator % (fix64p3x4 lhs, fix64p rhs) { return new fix64p3x4 (lhs.c0 % rhs, lhs.c1 % rhs, lhs.c2 % rhs, lhs.c3 % rhs); }

        /// <summary>Returns the result of a componentwise modulus operation on a fix64p value and a fix64p3x4 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator % (fix64p lhs, fix64p3x4 rhs) { return new fix64p3x4 (lhs % rhs.c0, lhs % rhs.c1, lhs % rhs.c2, lhs % rhs.c3); }


        /// <summary>Returns the result of a componentwise increment operation on a fix64p3x4 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator ++ (fix64p3x4 val) { return new fix64p3x4 (++val.c0, ++val.c1, ++val.c2, ++val.c3); }


        /// <summary>Returns the result of a componentwise decrement operation on a fix64p3x4 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator -- (fix64p3x4 val) { return new fix64p3x4 (--val.c0, --val.c1, --val.c2, --val.c3); }


        /// <summary>Returns the result of a componentwise less than operation on two fix64p3x4 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator < (fix64p3x4 lhs, fix64p3x4 rhs) { return new bool3x4 (lhs.c0 < rhs.c0, lhs.c1 < rhs.c1, lhs.c2 < rhs.c2, lhs.c3 < rhs.c3); }

        /// <summary>Returns the result of a componentwise less than operation on a fix64p3x4 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator < (fix64p3x4 lhs, fix64p rhs) { return new bool3x4 (lhs.c0 < rhs, lhs.c1 < rhs, lhs.c2 < rhs, lhs.c3 < rhs); }

        /// <summary>Returns the result of a componentwise less than operation on a fix64p value and a fix64p3x4 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator < (fix64p lhs, fix64p3x4 rhs) { return new bool3x4 (lhs < rhs.c0, lhs < rhs.c1, lhs < rhs.c2, lhs < rhs.c3); }


        /// <summary>Returns the result of a componentwise less or equal operation on two fix64p3x4 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator <= (fix64p3x4 lhs, fix64p3x4 rhs) { return new bool3x4 (lhs.c0 <= rhs.c0, lhs.c1 <= rhs.c1, lhs.c2 <= rhs.c2, lhs.c3 <= rhs.c3); }

        /// <summary>Returns the result of a componentwise less or equal operation on a fix64p3x4 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator <= (fix64p3x4 lhs, fix64p rhs) { return new bool3x4 (lhs.c0 <= rhs, lhs.c1 <= rhs, lhs.c2 <= rhs, lhs.c3 <= rhs); }

        /// <summary>Returns the result of a componentwise less or equal operation on a fix64p value and a fix64p3x4 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator <= (fix64p lhs, fix64p3x4 rhs) { return new bool3x4 (lhs <= rhs.c0, lhs <= rhs.c1, lhs <= rhs.c2, lhs <= rhs.c3); }


        /// <summary>Returns the result of a componentwise greater than operation on two fix64p3x4 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator > (fix64p3x4 lhs, fix64p3x4 rhs) { return new bool3x4 (lhs.c0 > rhs.c0, lhs.c1 > rhs.c1, lhs.c2 > rhs.c2, lhs.c3 > rhs.c3); }

        /// <summary>Returns the result of a componentwise greater than operation on a fix64p3x4 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator > (fix64p3x4 lhs, fix64p rhs) { return new bool3x4 (lhs.c0 > rhs, lhs.c1 > rhs, lhs.c2 > rhs, lhs.c3 > rhs); }

        /// <summary>Returns the result of a componentwise greater than operation on a fix64p value and a fix64p3x4 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator > (fix64p lhs, fix64p3x4 rhs) { return new bool3x4 (lhs > rhs.c0, lhs > rhs.c1, lhs > rhs.c2, lhs > rhs.c3); }


        /// <summary>Returns the result of a componentwise greater or equal operation on two fix64p3x4 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator >= (fix64p3x4 lhs, fix64p3x4 rhs) { return new bool3x4 (lhs.c0 >= rhs.c0, lhs.c1 >= rhs.c1, lhs.c2 >= rhs.c2, lhs.c3 >= rhs.c3); }

        /// <summary>Returns the result of a componentwise greater or equal operation on a fix64p3x4 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator >= (fix64p3x4 lhs, fix64p rhs) { return new bool3x4 (lhs.c0 >= rhs, lhs.c1 >= rhs, lhs.c2 >= rhs, lhs.c3 >= rhs); }

        /// <summary>Returns the result of a componentwise greater or equal operation on a fix64p value and a fix64p3x4 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator >= (fix64p lhs, fix64p3x4 rhs) { return new bool3x4 (lhs >= rhs.c0, lhs >= rhs.c1, lhs >= rhs.c2, lhs >= rhs.c3); }


        /// <summary>Returns the result of a componentwise unary minus operation on a fix64p3x4 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator - (fix64p3x4 val) { return new fix64p3x4 (-val.c0, -val.c1, -val.c2, -val.c3); }


        /// <summary>Returns the result of a componentwise unary plus operation on a fix64p3x4 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 operator + (fix64p3x4 val) { return new fix64p3x4 (+val.c0, +val.c1, +val.c2, +val.c3); }


        /// <summary>Returns the result of a componentwise equality operation on two fix64p3x4 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator == (fix64p3x4 lhs, fix64p3x4 rhs) { return new bool3x4 (lhs.c0 == rhs.c0, lhs.c1 == rhs.c1, lhs.c2 == rhs.c2, lhs.c3 == rhs.c3); }

        /// <summary>Returns the result of a componentwise equality operation on a fix64p3x4 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator == (fix64p3x4 lhs, fix64p rhs) { return new bool3x4 (lhs.c0 == rhs, lhs.c1 == rhs, lhs.c2 == rhs, lhs.c3 == rhs); }

        /// <summary>Returns the result of a componentwise equality operation on a fix64p value and a fix64p3x4 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator == (fix64p lhs, fix64p3x4 rhs) { return new bool3x4 (lhs == rhs.c0, lhs == rhs.c1, lhs == rhs.c2, lhs == rhs.c3); }


        /// <summary>Returns the result of a componentwise not equal operation on two fix64p3x4 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator != (fix64p3x4 lhs, fix64p3x4 rhs) { return new bool3x4 (lhs.c0 != rhs.c0, lhs.c1 != rhs.c1, lhs.c2 != rhs.c2, lhs.c3 != rhs.c3); }

        /// <summary>Returns the result of a componentwise not equal operation on a fix64p3x4 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator != (fix64p3x4 lhs, fix64p rhs) { return new bool3x4 (lhs.c0 != rhs, lhs.c1 != rhs, lhs.c2 != rhs, lhs.c3 != rhs); }

        /// <summary>Returns the result of a componentwise not equal operation on a fix64p value and a fix64p3x4 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x4 operator != (fix64p lhs, fix64p3x4 rhs) { return new bool3x4 (lhs != rhs.c0, lhs != rhs.c1, lhs != rhs.c2, lhs != rhs.c3); }



        /// <summary>Returns the fix64p3 element at a specified index.</summary>
        unsafe public ref fix64p3 this[int index]
        {
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                if ((uint)index >= 4)
                    throw new System.ArgumentException("index must be between[0...3]");
#endif
                fixed (fix64p3x4* array = &this) { return ref ((fix64p3*)array)[index]; }
            }
        }

        /// <summary>Returns true if the fix64p3x4 is equal to a given fix64p3x4, false otherwise.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Equals(fix64p3x4 rhs) { return c0.Equals(rhs.c0) && c1.Equals(rhs.c1) && c2.Equals(rhs.c2) && c3.Equals(rhs.c3); }

        /// <summary>Returns true if the fix64p3x4 is equal to a given fix64p3x4, false otherwise.</summary>
        public override bool Equals(object o) { return Equals((fix64p3x4)o); }


        /// <summary>Returns a hash code for the fix64p3x4.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override int GetHashCode() { return (int)math.hash(this); }


        /// <summary>Returns a string representation of the fix64p3x4.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override string ToString()
        {
            return string.Format("fix64p3x4({0}, {1}, {2}, {3},  {4}, {5}, {6}, {7},  {8}, {9}, {10}, {11})", c0.x, c1.x, c2.x, c3.x, c0.y, c1.y, c2.y, c3.y, c0.z, c1.z, c2.z, c3.z);
        }

        /// <summary>Returns a string representation of the fix64p3x4 using a specified format and culture-specific format information.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public string ToString(string format, IFormatProvider formatProvider)
        {
            return string.Format("fix64p3x4({0}, {1}, {2}, {3},  {4}, {5}, {6}, {7},  {8}, {9}, {10}, {11})", c0.x.ToString(format, formatProvider), c1.x.ToString(format, formatProvider), c2.x.ToString(format, formatProvider), c3.x.ToString(format, formatProvider), c0.y.ToString(format, formatProvider), c1.y.ToString(format, formatProvider), c2.y.ToString(format, formatProvider), c3.y.ToString(format, formatProvider), c0.z.ToString(format, formatProvider), c1.z.ToString(format, formatProvider), c2.z.ToString(format, formatProvider), c3.z.ToString(format, formatProvider));
        }

    }

    public static partial class math
    {
        /// <summary>Returns a fix64p3x4 matrix constructed from four fix64p3 vectors.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 fix64p3x4(fix64p3 c0, fix64p3 c1, fix64p3 c2, fix64p3 c3) { return new fix64p3x4(c0, c1, c2, c3); }

        /// <summary>Returns a fix64p3x4 matrix constructed from from 12 fix64p values given in row-major order.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 fix64p3x4(fix64p m00, fix64p m01, fix64p m02, fix64p m03,
                                          fix64p m10, fix64p m11, fix64p m12, fix64p m13,
                                          fix64p m20, fix64p m21, fix64p m22, fix64p m23)
        {
            return new fix64p3x4(m00, m01, m02, m03,
                                 m10, m11, m12, m13,
                                 m20, m21, m22, m23);
        }

        /// <summary>Returns a fix64p3x4 matrix constructed from a single fix64p value by assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 fix64p3x4(fix64p v) { return new fix64p3x4(v); }

        /// <summary>Returns a fix64p3x4 matrix constructed from a single bool value by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 fix64p3x4(bool v) { return new fix64p3x4(v); }

        /// <summary>Return a fix64p3x4 matrix constructed from a bool3x4 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 fix64p3x4(bool3x4 v) { return new fix64p3x4(v); }

        /// <summary>Returns a fix64p3x4 matrix constructed from a single int value by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 fix64p3x4(int v) { return new fix64p3x4(v); }

        /// <summary>Return a fix64p3x4 matrix constructed from a int3x4 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 fix64p3x4(int3x4 v) { return new fix64p3x4(v); }

        /// <summary>Returns a fix64p3x4 matrix constructed from a single float value by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 fix64p3x4(float v) { return new fix64p3x4(v); }

        /// <summary>Return a fix64p3x4 matrix constructed from a float3x4 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 fix64p3x4(float3x4 v) { return new fix64p3x4(v); }

        /// <summary>Return the fix64p4x3 transpose of a fix64p3x4 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 transpose(fix64p3x4 v)
        {
            return fix64p4x3(
                v.c0.x, v.c0.y, v.c0.z,
                v.c1.x, v.c1.y, v.c1.z,
                v.c2.x, v.c2.y, v.c2.z,
                v.c3.x, v.c3.y, v.c3.z);
        }

        /// <summary>Returns a uint hash code of a fix64p3x4 vector.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint hash(fix64p3x4 v)
        {
            return csum(fold_to_uint(v.c0) * uint3(0x6E624EB7u, 0x7383ED49u, 0xDD49C23Bu) + 
                        fold_to_uint(v.c1) * uint3(0xEBD0D005u, 0x91475DF7u, 0x55E84827u) + 
                        fold_to_uint(v.c2) * uint3(0x90A285BBu, 0x5D19E1D5u, 0xFAAF07DDu) + 
                        fold_to_uint(v.c3) * uint3(0x625C45BDu, 0xC9F27FCBu, 0x6D2523B1u)) + 0x6E2BF6A9u;
        }

        /// <summary>
        /// Returns a uint3 vector hash code of a fix64p3x4 vector.
        /// When multiple elements are to be hashes together, it can more efficient to calculate and combine wide hash
        /// that are only reduced to a narrow uint hash at the very end instead of at every step.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint3 hashwide(fix64p3x4 v)
        {
            return (fold_to_uint(v.c0) * uint3(0xCC74B3B7u, 0x83B58237u, 0x833E3E29u) + 
                    fold_to_uint(v.c1) * uint3(0xA9D919BFu, 0xC3EC1D97u, 0xB8B208C7u) + 
                    fold_to_uint(v.c2) * uint3(0x5D3ED947u, 0x4473BBB1u, 0xCBA11D5Fu) + 
                    fold_to_uint(v.c3) * uint3(0x685835CFu, 0xC3D32AE1u, 0xB966942Fu)) + 0xFE9856B3u;
        }

    }
}
