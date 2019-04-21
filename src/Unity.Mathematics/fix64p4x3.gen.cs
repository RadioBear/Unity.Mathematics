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
    public partial struct fix64p4x3 : System.IEquatable<fix64p4x3>, IFormattable
    {
        public fix64p4 c0;
        public fix64p4 c1;
        public fix64p4 c2;


        /// <summary>Constructs a fix64p4x3 matrix from three fix64p4 vectors.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p4x3(fix64p4 c0, fix64p4 c1, fix64p4 c2)
        { 
            this.c0 = c0;
            this.c1 = c1;
            this.c2 = c2;
        }

        /// <summary>Constructs a fix64p4x3 matrix from 12 fix64p values given in row-major order.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p4x3(fix64p m00, fix64p m01, fix64p m02,
                         fix64p m10, fix64p m11, fix64p m12,
                         fix64p m20, fix64p m21, fix64p m22,
                         fix64p m30, fix64p m31, fix64p m32)
        { 
            this.c0 = new fix64p4(m00, m10, m20, m30);
            this.c1 = new fix64p4(m01, m11, m21, m31);
            this.c2 = new fix64p4(m02, m12, m22, m32);
        }

        /// <summary>Constructs a fix64p4x3 matrix from a single fix64p value by assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fix64p4x3(fix64p v)
        {
            this.c0 = v;
            this.c1 = v;
            this.c2 = v;
        }


        /// <summary>Implicitly converts a single fix64p value to a fix64p4x3 matrix by assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator fix64p4x3(fix64p v) { return new fix64p4x3(v); }


        /// <summary>Returns the result of a componentwise multiplication operation on two fix64p4x3 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator * (fix64p4x3 lhs, fix64p4x3 rhs) { return new fix64p4x3 (lhs.c0 * rhs.c0, lhs.c1 * rhs.c1, lhs.c2 * rhs.c2); }

        /// <summary>Returns the result of a componentwise multiplication operation on a fix64p4x3 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator * (fix64p4x3 lhs, fix64p rhs) { return new fix64p4x3 (lhs.c0 * rhs, lhs.c1 * rhs, lhs.c2 * rhs); }

        /// <summary>Returns the result of a componentwise multiplication operation on a fix64p value and a fix64p4x3 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator * (fix64p lhs, fix64p4x3 rhs) { return new fix64p4x3 (lhs * rhs.c0, lhs * rhs.c1, lhs * rhs.c2); }


        /// <summary>Returns the result of a componentwise addition operation on two fix64p4x3 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator + (fix64p4x3 lhs, fix64p4x3 rhs) { return new fix64p4x3 (lhs.c0 + rhs.c0, lhs.c1 + rhs.c1, lhs.c2 + rhs.c2); }

        /// <summary>Returns the result of a componentwise addition operation on a fix64p4x3 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator + (fix64p4x3 lhs, fix64p rhs) { return new fix64p4x3 (lhs.c0 + rhs, lhs.c1 + rhs, lhs.c2 + rhs); }

        /// <summary>Returns the result of a componentwise addition operation on a fix64p value and a fix64p4x3 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator + (fix64p lhs, fix64p4x3 rhs) { return new fix64p4x3 (lhs + rhs.c0, lhs + rhs.c1, lhs + rhs.c2); }


        /// <summary>Returns the result of a componentwise subtraction operation on two fix64p4x3 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator - (fix64p4x3 lhs, fix64p4x3 rhs) { return new fix64p4x3 (lhs.c0 - rhs.c0, lhs.c1 - rhs.c1, lhs.c2 - rhs.c2); }

        /// <summary>Returns the result of a componentwise subtraction operation on a fix64p4x3 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator - (fix64p4x3 lhs, fix64p rhs) { return new fix64p4x3 (lhs.c0 - rhs, lhs.c1 - rhs, lhs.c2 - rhs); }

        /// <summary>Returns the result of a componentwise subtraction operation on a fix64p value and a fix64p4x3 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator - (fix64p lhs, fix64p4x3 rhs) { return new fix64p4x3 (lhs - rhs.c0, lhs - rhs.c1, lhs - rhs.c2); }


        /// <summary>Returns the result of a componentwise division operation on two fix64p4x3 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator / (fix64p4x3 lhs, fix64p4x3 rhs) { return new fix64p4x3 (lhs.c0 / rhs.c0, lhs.c1 / rhs.c1, lhs.c2 / rhs.c2); }

        /// <summary>Returns the result of a componentwise division operation on a fix64p4x3 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator / (fix64p4x3 lhs, fix64p rhs) { return new fix64p4x3 (lhs.c0 / rhs, lhs.c1 / rhs, lhs.c2 / rhs); }

        /// <summary>Returns the result of a componentwise division operation on a fix64p value and a fix64p4x3 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator / (fix64p lhs, fix64p4x3 rhs) { return new fix64p4x3 (lhs / rhs.c0, lhs / rhs.c1, lhs / rhs.c2); }


        /// <summary>Returns the result of a componentwise modulus operation on two fix64p4x3 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator % (fix64p4x3 lhs, fix64p4x3 rhs) { return new fix64p4x3 (lhs.c0 % rhs.c0, lhs.c1 % rhs.c1, lhs.c2 % rhs.c2); }

        /// <summary>Returns the result of a componentwise modulus operation on a fix64p4x3 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator % (fix64p4x3 lhs, fix64p rhs) { return new fix64p4x3 (lhs.c0 % rhs, lhs.c1 % rhs, lhs.c2 % rhs); }

        /// <summary>Returns the result of a componentwise modulus operation on a fix64p value and a fix64p4x3 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator % (fix64p lhs, fix64p4x3 rhs) { return new fix64p4x3 (lhs % rhs.c0, lhs % rhs.c1, lhs % rhs.c2); }


        /// <summary>Returns the result of a componentwise increment operation on a fix64p4x3 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator ++ (fix64p4x3 val) { return new fix64p4x3 (++val.c0, ++val.c1, ++val.c2); }


        /// <summary>Returns the result of a componentwise decrement operation on a fix64p4x3 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator -- (fix64p4x3 val) { return new fix64p4x3 (--val.c0, --val.c1, --val.c2); }


        /// <summary>Returns the result of a componentwise less than operation on two fix64p4x3 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator < (fix64p4x3 lhs, fix64p4x3 rhs) { return new bool4x3 (lhs.c0 < rhs.c0, lhs.c1 < rhs.c1, lhs.c2 < rhs.c2); }

        /// <summary>Returns the result of a componentwise less than operation on a fix64p4x3 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator < (fix64p4x3 lhs, fix64p rhs) { return new bool4x3 (lhs.c0 < rhs, lhs.c1 < rhs, lhs.c2 < rhs); }

        /// <summary>Returns the result of a componentwise less than operation on a fix64p value and a fix64p4x3 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator < (fix64p lhs, fix64p4x3 rhs) { return new bool4x3 (lhs < rhs.c0, lhs < rhs.c1, lhs < rhs.c2); }


        /// <summary>Returns the result of a componentwise less or equal operation on two fix64p4x3 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator <= (fix64p4x3 lhs, fix64p4x3 rhs) { return new bool4x3 (lhs.c0 <= rhs.c0, lhs.c1 <= rhs.c1, lhs.c2 <= rhs.c2); }

        /// <summary>Returns the result of a componentwise less or equal operation on a fix64p4x3 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator <= (fix64p4x3 lhs, fix64p rhs) { return new bool4x3 (lhs.c0 <= rhs, lhs.c1 <= rhs, lhs.c2 <= rhs); }

        /// <summary>Returns the result of a componentwise less or equal operation on a fix64p value and a fix64p4x3 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator <= (fix64p lhs, fix64p4x3 rhs) { return new bool4x3 (lhs <= rhs.c0, lhs <= rhs.c1, lhs <= rhs.c2); }


        /// <summary>Returns the result of a componentwise greater than operation on two fix64p4x3 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator > (fix64p4x3 lhs, fix64p4x3 rhs) { return new bool4x3 (lhs.c0 > rhs.c0, lhs.c1 > rhs.c1, lhs.c2 > rhs.c2); }

        /// <summary>Returns the result of a componentwise greater than operation on a fix64p4x3 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator > (fix64p4x3 lhs, fix64p rhs) { return new bool4x3 (lhs.c0 > rhs, lhs.c1 > rhs, lhs.c2 > rhs); }

        /// <summary>Returns the result of a componentwise greater than operation on a fix64p value and a fix64p4x3 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator > (fix64p lhs, fix64p4x3 rhs) { return new bool4x3 (lhs > rhs.c0, lhs > rhs.c1, lhs > rhs.c2); }


        /// <summary>Returns the result of a componentwise greater or equal operation on two fix64p4x3 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator >= (fix64p4x3 lhs, fix64p4x3 rhs) { return new bool4x3 (lhs.c0 >= rhs.c0, lhs.c1 >= rhs.c1, lhs.c2 >= rhs.c2); }

        /// <summary>Returns the result of a componentwise greater or equal operation on a fix64p4x3 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator >= (fix64p4x3 lhs, fix64p rhs) { return new bool4x3 (lhs.c0 >= rhs, lhs.c1 >= rhs, lhs.c2 >= rhs); }

        /// <summary>Returns the result of a componentwise greater or equal operation on a fix64p value and a fix64p4x3 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator >= (fix64p lhs, fix64p4x3 rhs) { return new bool4x3 (lhs >= rhs.c0, lhs >= rhs.c1, lhs >= rhs.c2); }


        /// <summary>Returns the result of a componentwise unary minus operation on a fix64p4x3 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator - (fix64p4x3 val) { return new fix64p4x3 (-val.c0, -val.c1, -val.c2); }


        /// <summary>Returns the result of a componentwise unary plus operation on a fix64p4x3 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 operator + (fix64p4x3 val) { return new fix64p4x3 (+val.c0, +val.c1, +val.c2); }


        /// <summary>Returns the result of a componentwise equality operation on two fix64p4x3 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator == (fix64p4x3 lhs, fix64p4x3 rhs) { return new bool4x3 (lhs.c0 == rhs.c0, lhs.c1 == rhs.c1, lhs.c2 == rhs.c2); }

        /// <summary>Returns the result of a componentwise equality operation on a fix64p4x3 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator == (fix64p4x3 lhs, fix64p rhs) { return new bool4x3 (lhs.c0 == rhs, lhs.c1 == rhs, lhs.c2 == rhs); }

        /// <summary>Returns the result of a componentwise equality operation on a fix64p value and a fix64p4x3 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator == (fix64p lhs, fix64p4x3 rhs) { return new bool4x3 (lhs == rhs.c0, lhs == rhs.c1, lhs == rhs.c2); }


        /// <summary>Returns the result of a componentwise not equal operation on two fix64p4x3 matrices.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator != (fix64p4x3 lhs, fix64p4x3 rhs) { return new bool4x3 (lhs.c0 != rhs.c0, lhs.c1 != rhs.c1, lhs.c2 != rhs.c2); }

        /// <summary>Returns the result of a componentwise not equal operation on a fix64p4x3 matrix and a fix64p value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator != (fix64p4x3 lhs, fix64p rhs) { return new bool4x3 (lhs.c0 != rhs, lhs.c1 != rhs, lhs.c2 != rhs); }

        /// <summary>Returns the result of a componentwise not equal operation on a fix64p value and a fix64p4x3 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool4x3 operator != (fix64p lhs, fix64p4x3 rhs) { return new bool4x3 (lhs != rhs.c0, lhs != rhs.c1, lhs != rhs.c2); }



        /// <summary>Returns the fix64p4 element at a specified index.</summary>
        unsafe public ref fix64p4 this[int index]
        {
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                if ((uint)index >= 3)
                    throw new System.ArgumentException("index must be between[0...2]");
#endif
                fixed (fix64p4x3* array = &this) { return ref ((fix64p4*)array)[index]; }
            }
        }

        /// <summary>Returns true if the fix64p4x3 is equal to a given fix64p4x3, false otherwise.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Equals(fix64p4x3 rhs) { return c0.Equals(rhs.c0) && c1.Equals(rhs.c1) && c2.Equals(rhs.c2); }

        /// <summary>Returns true if the fix64p4x3 is equal to a given fix64p4x3, false otherwise.</summary>
        public override bool Equals(object o) { return Equals((fix64p4x3)o); }


        /// <summary>Returns a hash code for the fix64p4x3.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override int GetHashCode() { return (int)math.hash(this); }


        /// <summary>Returns a string representation of the fix64p4x3.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override string ToString()
        {
            return string.Format("fix64p4x3({0}, {1}, {2},  {3}, {4}, {5},  {6}, {7}, {8},  {9}, {10}, {11})", c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z, c0.w, c1.w, c2.w);
        }

        /// <summary>Returns a string representation of the fix64p4x3 using a specified format and culture-specific format information.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public string ToString(string format, IFormatProvider formatProvider)
        {
            return string.Format("fix64p4x3({0}, {1}, {2},  {3}, {4}, {5},  {6}, {7}, {8},  {9}, {10}, {11})", c0.x.ToString(format, formatProvider), c1.x.ToString(format, formatProvider), c2.x.ToString(format, formatProvider), c0.y.ToString(format, formatProvider), c1.y.ToString(format, formatProvider), c2.y.ToString(format, formatProvider), c0.z.ToString(format, formatProvider), c1.z.ToString(format, formatProvider), c2.z.ToString(format, formatProvider), c0.w.ToString(format, formatProvider), c1.w.ToString(format, formatProvider), c2.w.ToString(format, formatProvider));
        }

    }

    public static partial class math
    {
        /// <summary>Returns a fix64p4x3 matrix constructed from three fix64p4 vectors.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 fix64p4x3(fix64p4 c0, fix64p4 c1, fix64p4 c2) { return new fix64p4x3(c0, c1, c2); }

        /// <summary>Returns a fix64p4x3 matrix constructed from from 12 fix64p values given in row-major order.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 fix64p4x3(fix64p m00, fix64p m01, fix64p m02,
                                          fix64p m10, fix64p m11, fix64p m12,
                                          fix64p m20, fix64p m21, fix64p m22,
                                          fix64p m30, fix64p m31, fix64p m32)
        {
            return new fix64p4x3(m00, m01, m02,
                                 m10, m11, m12,
                                 m20, m21, m22,
                                 m30, m31, m32);
        }

        /// <summary>Returns a fix64p4x3 matrix constructed from a single fix64p value by assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p4x3 fix64p4x3(fix64p v) { return new fix64p4x3(v); }

        /// <summary>Return the fix64p3x4 transpose of a fix64p4x3 matrix.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fix64p3x4 transpose(fix64p4x3 v)
        {
            return fix64p3x4(
                v.c0.x, v.c0.y, v.c0.z, v.c0.w,
                v.c1.x, v.c1.y, v.c1.z, v.c1.w,
                v.c2.x, v.c2.y, v.c2.z, v.c2.w);
        }

        /// <summary>Returns a uint hash code of a fix64p4x3 vector.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint hash(fix64p4x3 v)
        {
            return csum(fold_to_uint(v.c0) * uint4(0xF83BD927u, 0x6A243BCBu, 0x509B84C9u, 0x91D13847u) + 
                        fold_to_uint(v.c1) * uint4(0x52F7230Fu, 0xCF286E83u, 0xE121E6ADu, 0xC9CA1249u) + 
                        fold_to_uint(v.c2) * uint4(0x69B60C81u, 0xE0EB6C25u, 0xF648BEABu, 0x6BDB2B07u)) + 0xEF63C699u;
        }

        /// <summary>
        /// Returns a uint4 vector hash code of a fix64p4x3 vector.
        /// When multiple elements are to be hashes together, it can more efficient to calculate and combine wide hash
        /// that are only reduced to a narrow uint hash at the very end instead of at every step.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint4 hashwide(fix64p4x3 v)
        {
            return (fold_to_uint(v.c0) * uint4(0x9001903Fu, 0xA895B9CDu, 0x9D23B201u, 0x4B01D3E1u) + 
                    fold_to_uint(v.c1) * uint4(0x7461CA0Du, 0x79725379u, 0xD6258E5Bu, 0xEE390C97u) + 
                    fold_to_uint(v.c2) * uint4(0x9C8A2F05u, 0x4DDC6509u, 0x7CF083CBu, 0x5C4D6CEDu)) + 0xF9137117u;
        }

    }
}
