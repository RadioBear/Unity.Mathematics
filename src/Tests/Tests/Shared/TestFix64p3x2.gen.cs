//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------
using NUnit.Framework;
using static Unity.Mathematics.math;
using Burst.Compiler.IL.Tests;

namespace Unity.Mathematics.Tests
{
    [TestFixture]
    public class TestFix64p3x2
    {
        [TestCompiler]
        public static void fix64p3x2_zero()
        {
            TestUtils.AreEqual(fix64p3x2.zero.c0.x, fix64p.zero);
            TestUtils.AreEqual(fix64p3x2.zero.c0.y, fix64p.zero);
            TestUtils.AreEqual(fix64p3x2.zero.c0.z, fix64p.zero);
            TestUtils.AreEqual(fix64p3x2.zero.c1.x, fix64p.zero);
            TestUtils.AreEqual(fix64p3x2.zero.c1.y, fix64p.zero);
            TestUtils.AreEqual(fix64p3x2.zero.c1.z, fix64p.zero);
        }

        [TestCompiler]
        public static void fix64p3x2_operator_equal_wide_wide()
        {
            fix64p3x2 a0 = fix64p3x2(new fix64p(-135.18924f), new fix64p(-49.0941162f), new fix64p(169.129822f), new fix64p(240.8053f), new fix64p(314.7392f), new fix64p(442.393f));
            fix64p3x2 b0 = fix64p3x2(new fix64p(-220.014648f), new fix64p(66.98004f), new fix64p(499.2016f), new fix64p(-371.1131f), new fix64p(208.448669f), new fix64p(390.8037f));
            bool3x2 r0 = bool3x2(false, false, false, false, false, false);
            TestUtils.AreEqual(a0 == b0, r0);

            fix64p3x2 a1 = fix64p3x2(new fix64p(177.924438f), new fix64p(335.5334f), new fix64p(168.15448f), new fix64p(350.729553f), new fix64p(367.178467f), new fix64p(46.9414673f));
            fix64p3x2 b1 = fix64p3x2(new fix64p(-72.44382f), new fix64p(362.97644f), new fix64p(194.678345f), new fix64p(471.644836f), new fix64p(-404.044678f), new fix64p(-144.696747f));
            bool3x2 r1 = bool3x2(false, false, false, false, false, false);
            TestUtils.AreEqual(a1 == b1, r1);

            fix64p3x2 a2 = fix64p3x2(new fix64p(188.76416f), new fix64p(-97.2113953f), new fix64p(-293.320984f), new fix64p(-234.822937f), new fix64p(417.0337f), new fix64p(26.3864136f));
            fix64p3x2 b2 = fix64p3x2(new fix64p(-494.446655f), new fix64p(-252.970367f), new fix64p(234.417114f), new fix64p(398.724f), new fix64p(260.4287f), new fix64p(370.144226f));
            bool3x2 r2 = bool3x2(false, false, false, false, false, false);
            TestUtils.AreEqual(a2 == b2, r2);

            fix64p3x2 a3 = fix64p3x2(new fix64p(269.245728f), new fix64p(29.4741821f), new fix64p(479.485229f), new fix64p(-237.230957f), new fix64p(-221.9837f), new fix64p(-506.672546f));
            fix64p3x2 b3 = fix64p3x2(new fix64p(89.579834f), new fix64p(-434.816833f), new fix64p(-109.845337f), new fix64p(336.973022f), new fix64p(-409.154968f), new fix64p(500.387573f));
            bool3x2 r3 = bool3x2(false, false, false, false, false, false);
            TestUtils.AreEqual(a3 == b3, r3);
        }

        [TestCompiler]
        public static void fix64p3x2_operator_equal_wide_scalar()
        {
            fix64p3x2 a0 = fix64p3x2(new fix64p(65.6712f), new fix64p(404.415527f), new fix64p(-269.730164f), new fix64p(83.6306152f), new fix64p(152.9945f), new fix64p(-155.868286f));
            fix64p b0 = (new fix64p(-155.815765f));
            bool3x2 r0 = bool3x2(false, false, false, false, false, false);
            TestUtils.AreEqual(a0 == b0, r0);

            fix64p3x2 a1 = fix64p3x2(new fix64p(314.671265f), new fix64p(290.04895f), new fix64p(-132.6352f), new fix64p(-65.66748f), new fix64p(-69.68326f), new fix64p(-191.190765f));
            fix64p b1 = (new fix64p(386.365173f));
            bool3x2 r1 = bool3x2(false, false, false, false, false, false);
            TestUtils.AreEqual(a1 == b1, r1);

            fix64p3x2 a2 = fix64p3x2(new fix64p(186.845215f), new fix64p(-319.144043f), new fix64p(-49.70108f), new fix64p(-300.8819f), new fix64p(333.396851f), new fix64p(386.3775f));
            fix64p b2 = (new fix64p(-232.895691f));
            bool3x2 r2 = bool3x2(false, false, false, false, false, false);
            TestUtils.AreEqual(a2 == b2, r2);

            fix64p3x2 a3 = fix64p3x2(new fix64p(-296.7019f), new fix64p(141.542358f), new fix64p(-227.323334f), new fix64p(83.87286f), new fix64p(-410.91687f), new fix64p(110.501282f));
            fix64p b3 = (new fix64p(-309.1172f));
            bool3x2 r3 = bool3x2(false, false, false, false, false, false);
            TestUtils.AreEqual(a3 == b3, r3);
        }

        [TestCompiler]
        public static void fix64p3x2_operator_equal_scalar_wide()
        {
            fix64p a0 = (new fix64p(36.38391f));
            fix64p3x2 b0 = fix64p3x2(new fix64p(-400.4892f), new fix64p(-71.2868347f), new fix64p(156.978088f), new fix64p(-225.238739f), new fix64p(499.141785f), new fix64p(-211.979919f));
            bool3x2 r0 = bool3x2(false, false, false, false, false, false);
            TestUtils.AreEqual(a0 == b0, r0);

            fix64p a1 = (new fix64p(428.311951f));
            fix64p3x2 b1 = fix64p3x2(new fix64p(-489.501343f), new fix64p(-5.691559f), new fix64p(-30.8659363f), new fix64p(-362.9831f), new fix64p(184.503174f), new fix64p(-160.470612f));
            bool3x2 r1 = bool3x2(false, false, false, false, false, false);
            TestUtils.AreEqual(a1 == b1, r1);

            fix64p a2 = (new fix64p(316.668823f));
            fix64p3x2 b2 = fix64p3x2(new fix64p(390.369263f), new fix64p(505.1051f), new fix64p(-294.6487f), new fix64p(443.1991f), new fix64p(96.5592651f), new fix64p(-257.012939f));
            bool3x2 r2 = bool3x2(false, false, false, false, false, false);
            TestUtils.AreEqual(a2 == b2, r2);

            fix64p a3 = (new fix64p(-245.054962f));
            fix64p3x2 b3 = fix64p3x2(new fix64p(326.464844f), new fix64p(-23.9599f), new fix64p(-168.694885f), new fix64p(386.2486f), new fix64p(-227.090637f), new fix64p(-336.612427f));
            bool3x2 r3 = bool3x2(false, false, false, false, false, false);
            TestUtils.AreEqual(a3 == b3, r3);
        }

        [TestCompiler]
        public static void fix64p3x2_operator_not_equal_wide_wide()
        {
            fix64p3x2 a0 = fix64p3x2(new fix64p(279.994141f), new fix64p(-43.34201f), new fix64p(-465.724731f), new fix64p(317.466553f), new fix64p(85.7149658f), new fix64p(360.8905f));
            fix64p3x2 b0 = fix64p3x2(new fix64p(-460.9121f), new fix64p(-476.009033f), new fix64p(468.1364f), new fix64p(-341.012543f), new fix64p(-62.65805f), new fix64p(-458.801666f));
            bool3x2 r0 = bool3x2(true, true, true, true, true, true);
            TestUtils.AreEqual(a0 != b0, r0);

            fix64p3x2 a1 = fix64p3x2(new fix64p(366.081543f), new fix64p(154.542847f), new fix64p(332.4262f), new fix64p(397.11322f), new fix64p(-431.374969f), new fix64p(489.0108f));
            fix64p3x2 b1 = fix64p3x2(new fix64p(-457.730225f), new fix64p(-59.5232544f), new fix64p(3.024231f), new fix64p(155.812744f), new fix64p(-19.8399048f), new fix64p(-6.01693726f));
            bool3x2 r1 = bool3x2(true, true, true, true, true, true);
            TestUtils.AreEqual(a1 != b1, r1);

            fix64p3x2 a2 = fix64p3x2(new fix64p(398.4336f), new fix64p(-489.817932f), new fix64p(171.4049f), new fix64p(-67.82968f), new fix64p(-192.278717f), new fix64p(227.84082f));
            fix64p3x2 b2 = fix64p3x2(new fix64p(-406.207916f), new fix64p(-102.420715f), new fix64p(-40.362915f), new fix64p(452.6754f), new fix64p(93.25757f), new fix64p(-258.378052f));
            bool3x2 r2 = bool3x2(true, true, true, true, true, true);
            TestUtils.AreEqual(a2 != b2, r2);

            fix64p3x2 a3 = fix64p3x2(new fix64p(62.1381836f), new fix64p(262.186462f), new fix64p(-404.0531f), new fix64p(34.449585f), new fix64p(-204.795776f), new fix64p(-285.4118f));
            fix64p3x2 b3 = fix64p3x2(new fix64p(-184.0498f), new fix64p(-379.2353f), new fix64p(-370.687317f), new fix64p(-255.947235f), new fix64p(29.0557861f), new fix64p(322.407654f));
            bool3x2 r3 = bool3x2(true, true, true, true, true, true);
            TestUtils.AreEqual(a3 != b3, r3);
        }

        [TestCompiler]
        public static void fix64p3x2_operator_not_equal_wide_scalar()
        {
            fix64p3x2 a0 = fix64p3x2(new fix64p(-155.4411f), new fix64p(-19.4266052f), new fix64p(174.633057f), new fix64p(507.920715f), new fix64p(59.177063f), new fix64p(171.151489f));
            fix64p b0 = (new fix64p(-393.413544f));
            bool3x2 r0 = bool3x2(true, true, true, true, true, true);
            TestUtils.AreEqual(a0 != b0, r0);

            fix64p3x2 a1 = fix64p3x2(new fix64p(-58.92328f), new fix64p(492.20105f), new fix64p(-165.241516f), new fix64p(270.341f), new fix64p(-380.243256f), new fix64p(501.899048f));
            fix64p b1 = (new fix64p(-398.176849f));
            bool3x2 r1 = bool3x2(true, true, true, true, true, true);
            TestUtils.AreEqual(a1 != b1, r1);

            fix64p3x2 a2 = fix64p3x2(new fix64p(-134.345459f), new fix64p(46.7709961f), new fix64p(161.459961f), new fix64p(261.514221f), new fix64p(-145.6124f), new fix64p(-0.449920654f));
            fix64p b2 = (new fix64p(458.400452f));
            bool3x2 r2 = bool3x2(true, true, true, true, true, true);
            TestUtils.AreEqual(a2 != b2, r2);

            fix64p3x2 a3 = fix64p3x2(new fix64p(350.461426f), new fix64p(242.664f), new fix64p(382.677063f), new fix64p(-468.967957f), new fix64p(-497.459473f), new fix64p(-80.93225f));
            fix64p b3 = (new fix64p(202.221008f));
            bool3x2 r3 = bool3x2(true, true, true, true, true, true);
            TestUtils.AreEqual(a3 != b3, r3);
        }

        [TestCompiler]
        public static void fix64p3x2_operator_not_equal_scalar_wide()
        {
            fix64p a0 = (new fix64p(478.353149f));
            fix64p3x2 b0 = fix64p3x2(new fix64p(459.553223f), new fix64p(436.453247f), new fix64p(-488.714172f), new fix64p(392.767944f), new fix64p(-266.736633f), new fix64p(338.557861f));
            bool3x2 r0 = bool3x2(true, true, true, true, true, true);
            TestUtils.AreEqual(a0 != b0, r0);

            fix64p a1 = (new fix64p(-338.100128f));
            fix64p3x2 b1 = fix64p3x2(new fix64p(-152.314545f), new fix64p(-452.820679f), new fix64p(209.439331f), new fix64p(50.10797f), new fix64p(372.4344f), new fix64p(-488.0213f));
            bool3x2 r1 = bool3x2(true, true, true, true, true, true);
            TestUtils.AreEqual(a1 != b1, r1);

            fix64p a2 = (new fix64p(489.740784f));
            fix64p3x2 b2 = fix64p3x2(new fix64p(270.4001f), new fix64p(-472.846771f), new fix64p(-286.850464f), new fix64p(-384.691864f), new fix64p(443.423523f), new fix64p(358.7472f));
            bool3x2 r2 = bool3x2(true, true, true, true, true, true);
            TestUtils.AreEqual(a2 != b2, r2);

            fix64p a3 = (new fix64p(-15.4140625f));
            fix64p3x2 b3 = fix64p3x2(new fix64p(-342.179169f), new fix64p(468.967529f), new fix64p(-130.568085f), new fix64p(401.785828f), new fix64p(-268.352264f), new fix64p(-239.231018f));
            bool3x2 r3 = bool3x2(true, true, true, true, true, true);
            TestUtils.AreEqual(a3 != b3, r3);
        }


    }
}
