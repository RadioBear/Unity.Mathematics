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
    public partial struct fix64p3x2 : System.IEquatable<fix64p3x2>, IFormattable
    {
        public fix64p3 c0;
        public fix64p3 c1;

        /// <summary>fix64p3x2 zero value.</summary>
        public static readonly fix64p3x2 zero;

        /// <summary>Constructs a fix64p3x2 matrix from two fix64p3 vectors.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x2(fix64p3 c0, fix64p3 c1)
        { 
            this.c0 = c0;
            this.c1 = c1;
        }

        /// <summary>Constructs a fix64p3x2 matrix from 6 fix64p values given in row-major order.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x2(fix64p m00, fix64p m01,
                         fix64p m10, fix64p m11,
                         fix64p m20, fix64p m21)
        { 
            this.c0 = new fix64p3(m00, m10, m20);
            this.c1 = new fix64p3(m01, m11, m21);
        }

        /// <summary>Constructs a fix64p3x2 matrix from a single fix64p value by assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x2(fix64p v)
        {
            this.c0 = v;
            this.c1 = v;
        }

        /// <summary>Constructs a fix64p3x2 matrix from a single bool value by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x2(bool v)
        {
            this.c0 = math.select(new fix64p3(fix64p.zero), new fix64p3(fix64p.One), v);
            this.c1 = math.select(new fix64p3(fix64p.zero), new fix64p3(fix64p.One), v);
        }

        /// <summary>Constructs a fix64p3x2 matrix from a bool3x2 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x2(bool3x2 v)
        {
            this.c0 = math.select(new fix64p3(fix64p.zero), new fix64p3(fix64p.One), v.c0);
            this.c1 = math.select(new fix64p3(fix64p.zero), new fix64p3(fix64p.One), v.c1);
        }

        /// <summary>Constructs a fix64p3x2 matrix from a single int value by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x2(int v)
        {
            this.c0 = (fix64p3)v;
            this.c1 = (fix64p3)v;
        }

        /// <summary>Constructs a fix64p3x2 matrix from a int3x2 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x2(int3x2 v)
        {
            this.c0 = (fix64p3)v.c0;
            this.c1 = (fix64p3)v.c1;
        }

        /// <summary>Constructs a fix64p3x2 matrix from a single float value by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x2(float v)
        {
            this.c0 = (fix64p3)v;
            this.c1 = (fix64p3)v;
        }

        /// <summary>Constructs a fix64p3x2 matrix from a float3x2 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p3x2(float3x2 v)
        {
            this.c0 = (fix64p3)v.c0;
            this.c1 = (fix64p3)v.c1;
        }


        /// <summary>Implicitly converts a single fix64p value to a fix64p3x2 matrix by assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator fix64p3x2(fix64p v) { return new fix64p3x2(v); }

        /// <summary>Explicitly converts a single bool value to a fix64p3x2 matrix by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator fix64p3x2(bool v) { return new fix64p3x2(v); }

        /// <summary>Explicitly converts a bool3x2 matrix to a fix64p3x2 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator fix64p3x2(bool3x2 v) { return new fix64p3x2(v); }

        /// <summary>Explicitly converts a single int value to a fix64p3x2 matrix by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator fix64p3x2(int v) { return new fix64p3x2(v); }

        /// <summary>Explicitly converts a int3x2 matrix to a fix64p3x2 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator fix64p3x2(int3x2 v) { return new fix64p3x2(v); }

        /// <summary>Explicitly converts a single float value to a fix64p3x2 matrix by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator fix64p3x2(float v) { return new fix64p3x2(v); }

        /// <summary>Explicitly converts a float3x2 matrix to a fix64p3x2 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator fix64p3x2(float3x2 v) { return new fix64p3x2(v); }


        /// <summary>Returns the result of a componentwise multiplication operation on two fix64p3x2 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator * (fix64p3x2 lhs, fix64p3x2 rhs) { return new fix64p3x2 (lhs.c0 * rhs.c0, lhs.c1 * rhs.c1); }

        /// <summary>Returns the result of a componentwise multiplication operation on a fix64p3x2 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator * (fix64p3x2 lhs, fix64p rhs) { return new fix64p3x2 (lhs.c0 * rhs, lhs.c1 * rhs); }

        /// <summary>Returns the result of a componentwise multiplication operation on a fix64p value and a fix64p3x2 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator * (fix64p lhs, fix64p3x2 rhs) { return new fix64p3x2 (lhs * rhs.c0, lhs * rhs.c1); }


        /// <summary>Returns the result of a componentwise addition operation on two fix64p3x2 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator + (fix64p3x2 lhs, fix64p3x2 rhs) { return new fix64p3x2 (lhs.c0 + rhs.c0, lhs.c1 + rhs.c1); }

        /// <summary>Returns the result of a componentwise addition operation on a fix64p3x2 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator + (fix64p3x2 lhs, fix64p rhs) { return new fix64p3x2 (lhs.c0 + rhs, lhs.c1 + rhs); }

        /// <summary>Returns the result of a componentwise addition operation on a fix64p value and a fix64p3x2 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator + (fix64p lhs, fix64p3x2 rhs) { return new fix64p3x2 (lhs + rhs.c0, lhs + rhs.c1); }


        /// <summary>Returns the result of a componentwise subtraction operation on two fix64p3x2 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator - (fix64p3x2 lhs, fix64p3x2 rhs) { return new fix64p3x2 (lhs.c0 - rhs.c0, lhs.c1 - rhs.c1); }

        /// <summary>Returns the result of a componentwise subtraction operation on a fix64p3x2 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator - (fix64p3x2 lhs, fix64p rhs) { return new fix64p3x2 (lhs.c0 - rhs, lhs.c1 - rhs); }

        /// <summary>Returns the result of a componentwise subtraction operation on a fix64p value and a fix64p3x2 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator - (fix64p lhs, fix64p3x2 rhs) { return new fix64p3x2 (lhs - rhs.c0, lhs - rhs.c1); }


        /// <summary>Returns the result of a componentwise division operation on two fix64p3x2 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator / (fix64p3x2 lhs, fix64p3x2 rhs) { return new fix64p3x2 (lhs.c0 / rhs.c0, lhs.c1 / rhs.c1); }

        /// <summary>Returns the result of a componentwise division operation on a fix64p3x2 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator / (fix64p3x2 lhs, fix64p rhs) { return new fix64p3x2 (lhs.c0 / rhs, lhs.c1 / rhs); }

        /// <summary>Returns the result of a componentwise division operation on a fix64p value and a fix64p3x2 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator / (fix64p lhs, fix64p3x2 rhs) { return new fix64p3x2 (lhs / rhs.c0, lhs / rhs.c1); }


        /// <summary>Returns the result of a componentwise modulus operation on two fix64p3x2 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator % (fix64p3x2 lhs, fix64p3x2 rhs) { return new fix64p3x2 (lhs.c0 % rhs.c0, lhs.c1 % rhs.c1); }

        /// <summary>Returns the result of a componentwise modulus operation on a fix64p3x2 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator % (fix64p3x2 lhs, fix64p rhs) { return new fix64p3x2 (lhs.c0 % rhs, lhs.c1 % rhs); }

        /// <summary>Returns the result of a componentwise modulus operation on a fix64p value and a fix64p3x2 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator % (fix64p lhs, fix64p3x2 rhs) { return new fix64p3x2 (lhs % rhs.c0, lhs % rhs.c1); }


        /// <summary>Returns the result of a componentwise increment operation on a fix64p3x2 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator ++ (fix64p3x2 val) { return new fix64p3x2 (++val.c0, ++val.c1); }


        /// <summary>Returns the result of a componentwise decrement operation on a fix64p3x2 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator -- (fix64p3x2 val) { return new fix64p3x2 (--val.c0, --val.c1); }


        /// <summary>Returns the result of a componentwise less than operation on two fix64p3x2 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator < (fix64p3x2 lhs, fix64p3x2 rhs) { return new bool3x2 (lhs.c0 < rhs.c0, lhs.c1 < rhs.c1); }

        /// <summary>Returns the result of a componentwise less than operation on a fix64p3x2 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator < (fix64p3x2 lhs, fix64p rhs) { return new bool3x2 (lhs.c0 < rhs, lhs.c1 < rhs); }

        /// <summary>Returns the result of a componentwise less than operation on a fix64p value and a fix64p3x2 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator < (fix64p lhs, fix64p3x2 rhs) { return new bool3x2 (lhs < rhs.c0, lhs < rhs.c1); }


        /// <summary>Returns the result of a componentwise less or equal operation on two fix64p3x2 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator <= (fix64p3x2 lhs, fix64p3x2 rhs) { return new bool3x2 (lhs.c0 <= rhs.c0, lhs.c1 <= rhs.c1); }

        /// <summary>Returns the result of a componentwise less or equal operation on a fix64p3x2 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator <= (fix64p3x2 lhs, fix64p rhs) { return new bool3x2 (lhs.c0 <= rhs, lhs.c1 <= rhs); }

        /// <summary>Returns the result of a componentwise less or equal operation on a fix64p value and a fix64p3x2 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator <= (fix64p lhs, fix64p3x2 rhs) { return new bool3x2 (lhs <= rhs.c0, lhs <= rhs.c1); }


        /// <summary>Returns the result of a componentwise greater than operation on two fix64p3x2 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator > (fix64p3x2 lhs, fix64p3x2 rhs) { return new bool3x2 (lhs.c0 > rhs.c0, lhs.c1 > rhs.c1); }

        /// <summary>Returns the result of a componentwise greater than operation on a fix64p3x2 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator > (fix64p3x2 lhs, fix64p rhs) { return new bool3x2 (lhs.c0 > rhs, lhs.c1 > rhs); }

        /// <summary>Returns the result of a componentwise greater than operation on a fix64p value and a fix64p3x2 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator > (fix64p lhs, fix64p3x2 rhs) { return new bool3x2 (lhs > rhs.c0, lhs > rhs.c1); }


        /// <summary>Returns the result of a componentwise greater or equal operation on two fix64p3x2 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator >= (fix64p3x2 lhs, fix64p3x2 rhs) { return new bool3x2 (lhs.c0 >= rhs.c0, lhs.c1 >= rhs.c1); }

        /// <summary>Returns the result of a componentwise greater or equal operation on a fix64p3x2 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator >= (fix64p3x2 lhs, fix64p rhs) { return new bool3x2 (lhs.c0 >= rhs, lhs.c1 >= rhs); }

        /// <summary>Returns the result of a componentwise greater or equal operation on a fix64p value and a fix64p3x2 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator >= (fix64p lhs, fix64p3x2 rhs) { return new bool3x2 (lhs >= rhs.c0, lhs >= rhs.c1); }


        /// <summary>Returns the result of a componentwise unary minus operation on a fix64p3x2 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator - (fix64p3x2 val) { return new fix64p3x2 (-val.c0, -val.c1); }


        /// <summary>Returns the result of a componentwise unary plus operation on a fix64p3x2 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 operator + (fix64p3x2 val) { return new fix64p3x2 (+val.c0, +val.c1); }


        /// <summary>Returns the result of a componentwise equality operation on two fix64p3x2 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator == (fix64p3x2 lhs, fix64p3x2 rhs) { return new bool3x2 (lhs.c0 == rhs.c0, lhs.c1 == rhs.c1); }

        /// <summary>Returns the result of a componentwise equality operation on a fix64p3x2 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator == (fix64p3x2 lhs, fix64p rhs) { return new bool3x2 (lhs.c0 == rhs, lhs.c1 == rhs); }

        /// <summary>Returns the result of a componentwise equality operation on a fix64p value and a fix64p3x2 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator == (fix64p lhs, fix64p3x2 rhs) { return new bool3x2 (lhs == rhs.c0, lhs == rhs.c1); }


        /// <summary>Returns the result of a componentwise not equal operation on two fix64p3x2 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator != (fix64p3x2 lhs, fix64p3x2 rhs) { return new bool3x2 (lhs.c0 != rhs.c0, lhs.c1 != rhs.c1); }

        /// <summary>Returns the result of a componentwise not equal operation on a fix64p3x2 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator != (fix64p3x2 lhs, fix64p rhs) { return new bool3x2 (lhs.c0 != rhs, lhs.c1 != rhs); }

        /// <summary>Returns the result of a componentwise not equal operation on a fix64p value and a fix64p3x2 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool3x2 operator != (fix64p lhs, fix64p3x2 rhs) { return new bool3x2 (lhs != rhs.c0, lhs != rhs.c1); }



        /// <summary>Returns the fix64p3 element at a specified index.</summary>
        unsafe public ref fix64p3 this[int index]
        {
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                if ((uint)index >= 2)
                    throw new System.ArgumentException("index must be between[0...1]");
#endif
                fixed (fix64p3x2* array = &this) { return ref ((fix64p3*)array)[index]; }
            }
        }

        /// <summary>Returns true if the fix64p3x2 is equal to a given fix64p3x2, false otherwise.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Equals(fix64p3x2 rhs) { return c0.Equals(rhs.c0) && c1.Equals(rhs.c1); }

        /// <summary>Returns true if the fix64p3x2 is equal to a given fix64p3x2, false otherwise.</summary>
        public override bool Equals(object o) { return Equals((fix64p3x2)o); }


        /// <summary>Returns a hash code for the fix64p3x2.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override int GetHashCode() { return (int)math.hash(this); }


        /// <summary>Returns a string representation of the fix64p3x2.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override string ToString()
        {
            return string.Format("fix64p3x2({0}, {1},  {2}, {3},  {4}, {5})", c0.x, c1.x, c0.y, c1.y, c0.z, c1.z);
        }

        /// <summary>Returns a string representation of the fix64p3x2 using a specified format and culture-specific format information.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public string ToString(string format, IFormatProvider formatProvider)
        {
            return string.Format("fix64p3x2({0}, {1},  {2}, {3},  {4}, {5})", c0.x.ToString(format, formatProvider), c1.x.ToString(format, formatProvider), c0.y.ToString(format, formatProvider), c1.y.ToString(format, formatProvider), c0.z.ToString(format, formatProvider), c1.z.ToString(format, formatProvider));
        }

    }

    public static partial class math
    {
        /// <summary>Returns a fix64p3x2 matrix constructed from two fix64p3 vectors.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 fix64p3x2(fix64p3 c0, fix64p3 c1) { return new fix64p3x2(c0, c1); }

        /// <summary>Returns a fix64p3x2 matrix constructed from from 6 fix64p values given in row-major order.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 fix64p3x2(fix64p m00, fix64p m01,
                                          fix64p m10, fix64p m11,
                                          fix64p m20, fix64p m21)
        {
            return new fix64p3x2(m00, m01,
                                 m10, m11,
                                 m20, m21);
        }

        /// <summary>Returns a fix64p3x2 matrix constructed from a single fix64p value by assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 fix64p3x2(fix64p v) { return new fix64p3x2(v); }

        /// <summary>Returns a fix64p3x2 matrix constructed from a single bool value by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 fix64p3x2(bool v) { return new fix64p3x2(v); }

        /// <summary>Return a fix64p3x2 matrix constructed from a bool3x2 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 fix64p3x2(bool3x2 v) { return new fix64p3x2(v); }

        /// <summary>Returns a fix64p3x2 matrix constructed from a single int value by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 fix64p3x2(int v) { return new fix64p3x2(v); }

        /// <summary>Return a fix64p3x2 matrix constructed from a int3x2 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 fix64p3x2(int3x2 v) { return new fix64p3x2(v); }

        /// <summary>Returns a fix64p3x2 matrix constructed from a single float value by converting it to fix64p and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 fix64p3x2(float v) { return new fix64p3x2(v); }

        /// <summary>Return a fix64p3x2 matrix constructed from a float3x2 matrix by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x2 fix64p3x2(float3x2 v) { return new fix64p3x2(v); }

        /// <summary>Return the fix64p2x3 transpose of a fix64p3x2 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p2x3 transpose(fix64p3x2 v)
        {
            return fix64p2x3(
                v.c0.x, v.c0.y, v.c0.z,
                v.c1.x, v.c1.y, v.c1.z);
        }

        /// <summary>Returns a uint hash code of a fix64p3x2 vector.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint hash(fix64p3x2 v)
        {
            return csum(fold_to_uint(v.c0) * uint3(0xAC60D0C3u, 0x9263662Fu, 0xE69626FFu) + 
                        fold_to_uint(v.c1) * uint3(0xBD010EEBu, 0x9CEDE1D1u, 0x43BE0B51u)) + 0xAF836EE1u;
        }

        /// <summary>
        /// Returns a uint3 vector hash code of a fix64p3x2 vector.
        /// When multiple elements are to be hashes together, it can more efficient to calculate and combine wide hash
        /// that are only reduced to a narrow uint hash at the very end instead of at every step.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint3 hashwide(fix64p3x2 v)
        {
            return (fold_to_uint(v.c0) * uint3(0xB130C137u, 0x54834775u, 0x7C022221u) + 
                    fold_to_uint(v.c1) * uint3(0xA2D00EDFu, 0xA8977779u, 0x9F1C739Bu)) + 0x4B1BD187u;
        }

    }
}
