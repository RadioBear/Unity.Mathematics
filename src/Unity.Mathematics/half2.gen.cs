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
using System.Diagnostics;

#pragma warning disable 0660, 0661

namespace Unity.Mathematics
{
    [DebuggerTypeProxy(typeof(half2.DebuggerProxy))]
    public partial struct half2 : System.IEquatable<half2>, IFormattable
    {
        public half x;
        public half y;

        /// <summary>half2 zero value.</summary>
        public static readonly half2 zero;

        /// <summary>Constructs a half2 vector from two half values.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public half2(half x, half y)
        { 
            this.x = x;
            this.y = y;
        }

        /// <summary>Constructs a half2 vector from a half2 vector.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public half2(half2 xy)
        { 
            this.x = xy.x;
            this.y = xy.y;
        }

        /// <summary>Constructs a half2 vector from a single half value by assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public half2(half v)
        {
            this.x = v;
            this.y = v;
        }

        /// <summary>Constructs a half2 vector from a single float value by converting it to half and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public half2(float v)
        {
            this.x = (half)v;
            this.y = (half)v;
        }

        /// <summary>Constructs a half2 vector from a float2 vector by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public half2(float2 v)
        {
            this.x = (half)v.x;
            this.y = (half)v.y;
        }

        /// <summary>Constructs a half2 vector from a single double value by converting it to half and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public half2(double v)
        {
            this.x = (half)v;
            this.y = (half)v;
        }

        /// <summary>Constructs a half2 vector from a double2 vector by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public half2(double2 v)
        {
            this.x = (half)v.x;
            this.y = (half)v.y;
        }


        /// <summary>Implicitly converts a single half value to a half2 vector by assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator half2(half v) { return new half2(v); }

        /// <summary>Explicitly converts a single float value to a half2 vector by converting it to half and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator half2(float v) { return new half2(v); }

        /// <summary>Explicitly converts a float2 vector to a half2 vector by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator half2(float2 v) { return new half2(v); }

        /// <summary>Explicitly converts a single double value to a half2 vector by converting it to half and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator half2(double v) { return new half2(v); }

        /// <summary>Explicitly converts a double2 vector to a half2 vector by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator half2(double2 v) { return new half2(v); }


        /// <summary>Returns the result of a componentwise equality operation on two half2 vectors.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool2 operator == (half2 lhs, half2 rhs) { return new bool2 (lhs.x == rhs.x, lhs.y == rhs.y); }

        /// <summary>Returns the result of a componentwise equality operation on a half2 vector and a half value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool2 operator == (half2 lhs, half rhs) { return new bool2 (lhs.x == rhs, lhs.y == rhs); }

        /// <summary>Returns the result of a componentwise equality operation on a half value and a half2 vector.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool2 operator == (half lhs, half2 rhs) { return new bool2 (lhs == rhs.x, lhs == rhs.y); }


        /// <summary>Returns the result of a componentwise not equal operation on two half2 vectors.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool2 operator != (half2 lhs, half2 rhs) { return new bool2 (lhs.x != rhs.x, lhs.y != rhs.y); }

        /// <summary>Returns the result of a componentwise not equal operation on a half2 vector and a half value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool2 operator != (half2 lhs, half rhs) { return new bool2 (lhs.x != rhs, lhs.y != rhs); }

        /// <summary>Returns the result of a componentwise not equal operation on a half value and a half2 vector.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool2 operator != (half lhs, half2 rhs) { return new bool2 (lhs != rhs.x, lhs != rhs.y); }




        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half4 xxxx
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half4(x, x, x, x); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half4 xxxy
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half4(x, x, x, y); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half4 xxyx
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half4(x, x, y, x); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half4 xxyy
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half4(x, x, y, y); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half4 xyxx
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half4(x, y, x, x); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half4 xyxy
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half4(x, y, x, y); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half4 xyyx
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half4(x, y, y, x); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half4 xyyy
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half4(x, y, y, y); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half4 yxxx
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half4(y, x, x, x); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half4 yxxy
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half4(y, x, x, y); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half4 yxyx
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half4(y, x, y, x); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half4 yxyy
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half4(y, x, y, y); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half4 yyxx
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half4(y, y, x, x); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half4 yyxy
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half4(y, y, x, y); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half4 yyyx
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half4(y, y, y, x); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half4 yyyy
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half4(y, y, y, y); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half3 xxx
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half3(x, x, x); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half3 xxy
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half3(x, x, y); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half3 xyx
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half3(x, y, x); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half3 xyy
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half3(x, y, y); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half3 yxx
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half3(y, x, x); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half3 yxy
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half3(y, x, y); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half3 yyx
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half3(y, y, x); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half3 yyy
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half3(y, y, y); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half2 xx
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half2(x, x); }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half2 xy
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half2(x, y); }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set { x = value.x; y = value.y; }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half2 yx
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half2(y, x); }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set { y = value.x; x = value.y; }
        }


        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
        public half2 yy
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return new half2(y, y); }
        }



        /// <summary>Returns the half element at a specified index.</summary>
        unsafe public half this[int index]
        {
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                if ((uint)index >= 2)
                    throw new System.ArgumentException("index must be between[0...1]");
#endif
                fixed (half2* array = &this) { return ((half*)array)[index]; }
            }
            set
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                if ((uint)index >= 2)
                    throw new System.ArgumentException("index must be between[0...1]");
#endif
                fixed (half* array = &x) { array[index] = value; }
            }
        }

        /// <summary>Returns true if the half2 is equal to a given half2, false otherwise.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Equals(half2 rhs) { return x == rhs.x && y == rhs.y; }

        /// <summary>Returns true if the half2 is equal to a given half2, false otherwise.</summary>
        public override bool Equals(object o) { return Equals((half2)o); }


        /// <summary>Returns a hash code for the half2.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override int GetHashCode() { return (int)math.hash(this); }


        /// <summary>Returns a string representation of the half2.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override string ToString()
        {
            return string.Format("half2({0}, {1})", x, y);
        }

        /// <summary>Returns a string representation of the half2 using a specified format and culture-specific format information.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public string ToString(string format, IFormatProvider formatProvider)
        {
            return string.Format("half2({0}, {1})", x.ToString(format, formatProvider), y.ToString(format, formatProvider));
        }

        internal sealed class DebuggerProxy
        {
            public half x;
            public half y;
            public DebuggerProxy(half2 v)
            {
                x = v.x;
                y = v.y;
            }
        }

    }

    public static partial class math
    {
        /// <summary>Returns a half2 vector constructed from two half values.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static half2 half2(half x, half y) { return new half2(x, y); }

        /// <summary>Returns a half2 vector constructed from a half2 vector.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static half2 half2(half2 xy) { return new half2(xy); }

        /// <summary>Returns a half2 vector constructed from a single half value by assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static half2 half2(half v) { return new half2(v); }

        /// <summary>Returns a half2 vector constructed from a single float value by converting it to half and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static half2 half2(float v) { return new half2(v); }

        /// <summary>Return a half2 vector constructed from a float2 vector by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static half2 half2(float2 v) { return new half2(v); }

        /// <summary>Returns a half2 vector constructed from a single double value by converting it to half and assigning it to every component.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static half2 half2(double v) { return new half2(v); }

        /// <summary>Return a half2 vector constructed from a double2 vector by componentwise conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static half2 half2(double2 v) { return new half2(v); }

        /// <summary>Returns a uint hash code of a half2 vector.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint hash(half2 v)
        {
            return csum(uint2(v.x.value, v.y.value) * uint2(0x6E624EB7u, 0x7383ED49u)) + 0xDD49C23Bu;
        }

        /// <summary>
        /// Returns a uint2 vector hash code of a half2 vector.
        /// When multiple elements are to be hashes together, it can more efficient to calculate and combine wide hash
        /// that are only reduced to a narrow uint hash at the very end instead of at every step.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint2 hashwide(half2 v)
        {
            return (uint2(v.x.value, v.y.value) * uint2(0xEBD0D005u, 0x91475DF7u)) + 0x55E84827u;
        }

    }
}
