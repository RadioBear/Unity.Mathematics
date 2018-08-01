// GENERATED CODE
using NUnit.Framework;
using static Unity.Mathematics.math;
namespace Unity.Mathematics.Tests
{
    [TestFixture]
    public class TestBool2x2
    {
        [Test]
        public void bool2x2_operator_equal_wide_wide()
        {
            bool2x2 a0 = bool2x2(true, false, true, false);
            bool2x2 b0 = bool2x2(true, false, false, false);
            bool2x2 r0 = bool2x2(true, true, false, true);
            TestUtils.AreEqual(a0 == b0, r0);

            bool2x2 a1 = bool2x2(false, true, false, false);
            bool2x2 b1 = bool2x2(true, false, false, true);
            bool2x2 r1 = bool2x2(false, false, true, false);
            TestUtils.AreEqual(a1 == b1, r1);

            bool2x2 a2 = bool2x2(true, false, true, false);
            bool2x2 b2 = bool2x2(false, false, false, false);
            bool2x2 r2 = bool2x2(false, true, false, true);
            TestUtils.AreEqual(a2 == b2, r2);

            bool2x2 a3 = bool2x2(true, true, false, true);
            bool2x2 b3 = bool2x2(true, false, false, false);
            bool2x2 r3 = bool2x2(true, false, true, false);
            TestUtils.AreEqual(a3 == b3, r3);
        }

        [Test]
        public void bool2x2_operator_equal_wide_scalar()
        {
            bool2x2 a0 = bool2x2(false, true, false, false);
            bool b0 = (true);
            bool2x2 r0 = bool2x2(false, true, false, false);
            TestUtils.AreEqual(a0 == b0, r0);

            bool2x2 a1 = bool2x2(false, true, false, false);
            bool b1 = (false);
            bool2x2 r1 = bool2x2(true, false, true, true);
            TestUtils.AreEqual(a1 == b1, r1);

            bool2x2 a2 = bool2x2(false, true, false, true);
            bool b2 = (true);
            bool2x2 r2 = bool2x2(false, true, false, true);
            TestUtils.AreEqual(a2 == b2, r2);

            bool2x2 a3 = bool2x2(false, false, false, false);
            bool b3 = (true);
            bool2x2 r3 = bool2x2(false, false, false, false);
            TestUtils.AreEqual(a3 == b3, r3);
        }

        [Test]
        public void bool2x2_operator_equal_scalar_wide()
        {
            bool a0 = (false);
            bool2x2 b0 = bool2x2(true, false, true, false);
            bool2x2 r0 = bool2x2(false, true, false, true);
            TestUtils.AreEqual(a0 == b0, r0);

            bool a1 = (false);
            bool2x2 b1 = bool2x2(true, false, false, true);
            bool2x2 r1 = bool2x2(false, true, true, false);
            TestUtils.AreEqual(a1 == b1, r1);

            bool a2 = (false);
            bool2x2 b2 = bool2x2(true, true, false, true);
            bool2x2 r2 = bool2x2(false, false, true, false);
            TestUtils.AreEqual(a2 == b2, r2);

            bool a3 = (false);
            bool2x2 b3 = bool2x2(true, true, true, false);
            bool2x2 r3 = bool2x2(false, false, false, true);
            TestUtils.AreEqual(a3 == b3, r3);
        }

        [Test]
        public void bool2x2_operator_not_equal_wide_wide()
        {
            bool2x2 a0 = bool2x2(true, true, true, false);
            bool2x2 b0 = bool2x2(true, false, false, false);
            bool2x2 r0 = bool2x2(false, true, true, false);
            TestUtils.AreEqual(a0 != b0, r0);

            bool2x2 a1 = bool2x2(false, true, false, false);
            bool2x2 b1 = bool2x2(true, false, false, false);
            bool2x2 r1 = bool2x2(true, true, false, false);
            TestUtils.AreEqual(a1 != b1, r1);

            bool2x2 a2 = bool2x2(false, false, true, true);
            bool2x2 b2 = bool2x2(false, true, true, true);
            bool2x2 r2 = bool2x2(false, true, false, false);
            TestUtils.AreEqual(a2 != b2, r2);

            bool2x2 a3 = bool2x2(true, true, true, true);
            bool2x2 b3 = bool2x2(false, false, true, true);
            bool2x2 r3 = bool2x2(true, true, false, false);
            TestUtils.AreEqual(a3 != b3, r3);
        }

        [Test]
        public void bool2x2_operator_not_equal_wide_scalar()
        {
            bool2x2 a0 = bool2x2(false, true, false, true);
            bool b0 = (false);
            bool2x2 r0 = bool2x2(false, true, false, true);
            TestUtils.AreEqual(a0 != b0, r0);

            bool2x2 a1 = bool2x2(true, false, false, false);
            bool b1 = (false);
            bool2x2 r1 = bool2x2(true, false, false, false);
            TestUtils.AreEqual(a1 != b1, r1);

            bool2x2 a2 = bool2x2(false, false, false, true);
            bool b2 = (false);
            bool2x2 r2 = bool2x2(false, false, false, true);
            TestUtils.AreEqual(a2 != b2, r2);

            bool2x2 a3 = bool2x2(false, false, false, true);
            bool b3 = (false);
            bool2x2 r3 = bool2x2(false, false, false, true);
            TestUtils.AreEqual(a3 != b3, r3);
        }

        [Test]
        public void bool2x2_operator_not_equal_scalar_wide()
        {
            bool a0 = (true);
            bool2x2 b0 = bool2x2(false, false, true, false);
            bool2x2 r0 = bool2x2(true, true, false, true);
            TestUtils.AreEqual(a0 != b0, r0);

            bool a1 = (false);
            bool2x2 b1 = bool2x2(false, true, true, true);
            bool2x2 r1 = bool2x2(false, true, true, true);
            TestUtils.AreEqual(a1 != b1, r1);

            bool a2 = (false);
            bool2x2 b2 = bool2x2(false, false, false, false);
            bool2x2 r2 = bool2x2(false, false, false, false);
            TestUtils.AreEqual(a2 != b2, r2);

            bool a3 = (true);
            bool2x2 b3 = bool2x2(false, false, true, false);
            bool2x2 r3 = bool2x2(true, true, false, true);
            TestUtils.AreEqual(a3 != b3, r3);
        }

        [Test]
        public void bool2x2_operator_bitwise_and_wide_wide()
        {
            bool2x2 a0 = bool2x2(false, false, true, true);
            bool2x2 b0 = bool2x2(false, false, true, false);
            bool2x2 r0 = bool2x2(false, false, true, false);
            TestUtils.AreEqual(a0 & b0, r0);

            bool2x2 a1 = bool2x2(false, false, true, true);
            bool2x2 b1 = bool2x2(true, true, false, false);
            bool2x2 r1 = bool2x2(false, false, false, false);
            TestUtils.AreEqual(a1 & b1, r1);

            bool2x2 a2 = bool2x2(true, false, true, true);
            bool2x2 b2 = bool2x2(true, true, false, false);
            bool2x2 r2 = bool2x2(true, false, false, false);
            TestUtils.AreEqual(a2 & b2, r2);

            bool2x2 a3 = bool2x2(true, true, false, false);
            bool2x2 b3 = bool2x2(false, false, true, false);
            bool2x2 r3 = bool2x2(false, false, false, false);
            TestUtils.AreEqual(a3 & b3, r3);
        }

        [Test]
        public void bool2x2_operator_bitwise_and_wide_scalar()
        {
            bool2x2 a0 = bool2x2(true, false, false, true);
            bool b0 = (true);
            bool2x2 r0 = bool2x2(true, false, false, true);
            TestUtils.AreEqual(a0 & b0, r0);

            bool2x2 a1 = bool2x2(true, false, false, false);
            bool b1 = (false);
            bool2x2 r1 = bool2x2(false, false, false, false);
            TestUtils.AreEqual(a1 & b1, r1);

            bool2x2 a2 = bool2x2(false, true, true, false);
            bool b2 = (true);
            bool2x2 r2 = bool2x2(false, true, true, false);
            TestUtils.AreEqual(a2 & b2, r2);

            bool2x2 a3 = bool2x2(false, true, true, false);
            bool b3 = (true);
            bool2x2 r3 = bool2x2(false, true, true, false);
            TestUtils.AreEqual(a3 & b3, r3);
        }

        [Test]
        public void bool2x2_operator_bitwise_and_scalar_wide()
        {
            bool a0 = (false);
            bool2x2 b0 = bool2x2(false, false, true, true);
            bool2x2 r0 = bool2x2(false, false, false, false);
            TestUtils.AreEqual(a0 & b0, r0);

            bool a1 = (true);
            bool2x2 b1 = bool2x2(false, true, false, false);
            bool2x2 r1 = bool2x2(false, true, false, false);
            TestUtils.AreEqual(a1 & b1, r1);

            bool a2 = (false);
            bool2x2 b2 = bool2x2(true, false, false, false);
            bool2x2 r2 = bool2x2(false, false, false, false);
            TestUtils.AreEqual(a2 & b2, r2);

            bool a3 = (true);
            bool2x2 b3 = bool2x2(true, true, true, false);
            bool2x2 r3 = bool2x2(true, true, true, false);
            TestUtils.AreEqual(a3 & b3, r3);
        }

        [Test]
        public void bool2x2_operator_bitwise_or_wide_wide()
        {
            bool2x2 a0 = bool2x2(true, true, true, false);
            bool2x2 b0 = bool2x2(false, false, false, false);
            bool2x2 r0 = bool2x2(true, true, true, false);
            TestUtils.AreEqual(a0 | b0, r0);

            bool2x2 a1 = bool2x2(true, false, true, true);
            bool2x2 b1 = bool2x2(true, true, true, false);
            bool2x2 r1 = bool2x2(true, true, true, true);
            TestUtils.AreEqual(a1 | b1, r1);

            bool2x2 a2 = bool2x2(false, true, true, true);
            bool2x2 b2 = bool2x2(false, true, false, false);
            bool2x2 r2 = bool2x2(false, true, true, true);
            TestUtils.AreEqual(a2 | b2, r2);

            bool2x2 a3 = bool2x2(true, true, true, false);
            bool2x2 b3 = bool2x2(true, true, true, true);
            bool2x2 r3 = bool2x2(true, true, true, true);
            TestUtils.AreEqual(a3 | b3, r3);
        }

        [Test]
        public void bool2x2_operator_bitwise_or_wide_scalar()
        {
            bool2x2 a0 = bool2x2(true, true, false, true);
            bool b0 = (true);
            bool2x2 r0 = bool2x2(true, true, true, true);
            TestUtils.AreEqual(a0 | b0, r0);

            bool2x2 a1 = bool2x2(true, true, false, true);
            bool b1 = (true);
            bool2x2 r1 = bool2x2(true, true, true, true);
            TestUtils.AreEqual(a1 | b1, r1);

            bool2x2 a2 = bool2x2(false, true, false, false);
            bool b2 = (false);
            bool2x2 r2 = bool2x2(false, true, false, false);
            TestUtils.AreEqual(a2 | b2, r2);

            bool2x2 a3 = bool2x2(true, true, true, true);
            bool b3 = (false);
            bool2x2 r3 = bool2x2(true, true, true, true);
            TestUtils.AreEqual(a3 | b3, r3);
        }

        [Test]
        public void bool2x2_operator_bitwise_or_scalar_wide()
        {
            bool a0 = (true);
            bool2x2 b0 = bool2x2(true, true, false, false);
            bool2x2 r0 = bool2x2(true, true, true, true);
            TestUtils.AreEqual(a0 | b0, r0);

            bool a1 = (true);
            bool2x2 b1 = bool2x2(true, true, false, false);
            bool2x2 r1 = bool2x2(true, true, true, true);
            TestUtils.AreEqual(a1 | b1, r1);

            bool a2 = (true);
            bool2x2 b2 = bool2x2(true, false, true, true);
            bool2x2 r2 = bool2x2(true, true, true, true);
            TestUtils.AreEqual(a2 | b2, r2);

            bool a3 = (false);
            bool2x2 b3 = bool2x2(false, false, false, true);
            bool2x2 r3 = bool2x2(false, false, false, true);
            TestUtils.AreEqual(a3 | b3, r3);
        }

        [Test]
        public void bool2x2_operator_bitwise_xor_wide_wide()
        {
            bool2x2 a0 = bool2x2(true, false, false, true);
            bool2x2 b0 = bool2x2(true, true, false, true);
            bool2x2 r0 = bool2x2(false, true, false, false);
            TestUtils.AreEqual(a0 ^ b0, r0);

            bool2x2 a1 = bool2x2(false, false, false, true);
            bool2x2 b1 = bool2x2(false, true, false, true);
            bool2x2 r1 = bool2x2(false, true, false, false);
            TestUtils.AreEqual(a1 ^ b1, r1);

            bool2x2 a2 = bool2x2(false, false, true, true);
            bool2x2 b2 = bool2x2(false, false, false, true);
            bool2x2 r2 = bool2x2(false, false, true, false);
            TestUtils.AreEqual(a2 ^ b2, r2);

            bool2x2 a3 = bool2x2(false, false, true, false);
            bool2x2 b3 = bool2x2(false, false, true, true);
            bool2x2 r3 = bool2x2(false, false, false, true);
            TestUtils.AreEqual(a3 ^ b3, r3);
        }

        [Test]
        public void bool2x2_operator_bitwise_xor_wide_scalar()
        {
            bool2x2 a0 = bool2x2(false, false, true, true);
            bool b0 = (false);
            bool2x2 r0 = bool2x2(false, false, true, true);
            TestUtils.AreEqual(a0 ^ b0, r0);

            bool2x2 a1 = bool2x2(false, false, false, false);
            bool b1 = (false);
            bool2x2 r1 = bool2x2(false, false, false, false);
            TestUtils.AreEqual(a1 ^ b1, r1);

            bool2x2 a2 = bool2x2(false, true, false, true);
            bool b2 = (false);
            bool2x2 r2 = bool2x2(false, true, false, true);
            TestUtils.AreEqual(a2 ^ b2, r2);

            bool2x2 a3 = bool2x2(true, true, false, true);
            bool b3 = (true);
            bool2x2 r3 = bool2x2(false, false, true, false);
            TestUtils.AreEqual(a3 ^ b3, r3);
        }

        [Test]
        public void bool2x2_operator_bitwise_xor_scalar_wide()
        {
            bool a0 = (true);
            bool2x2 b0 = bool2x2(true, false, true, true);
            bool2x2 r0 = bool2x2(false, true, false, false);
            TestUtils.AreEqual(a0 ^ b0, r0);

            bool a1 = (false);
            bool2x2 b1 = bool2x2(true, true, false, false);
            bool2x2 r1 = bool2x2(true, true, false, false);
            TestUtils.AreEqual(a1 ^ b1, r1);

            bool a2 = (true);
            bool2x2 b2 = bool2x2(true, false, false, true);
            bool2x2 r2 = bool2x2(false, true, true, false);
            TestUtils.AreEqual(a2 ^ b2, r2);

            bool a3 = (false);
            bool2x2 b3 = bool2x2(true, true, false, true);
            bool2x2 r3 = bool2x2(true, true, false, true);
            TestUtils.AreEqual(a3 ^ b3, r3);
        }


    }
}