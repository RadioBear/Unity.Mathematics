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
    public class TestFix64p3x4
    {
        [TestCompiler]
        public static void fix64p3x4_zero()
        {
            TestUtils.AreEqual(fix64p3x4.zero.c0.x, fix64p.zero);
            TestUtils.AreEqual(fix64p3x4.zero.c0.y, fix64p.zero);
            TestUtils.AreEqual(fix64p3x4.zero.c0.z, fix64p.zero);
            TestUtils.AreEqual(fix64p3x4.zero.c1.x, fix64p.zero);
            TestUtils.AreEqual(fix64p3x4.zero.c1.y, fix64p.zero);
            TestUtils.AreEqual(fix64p3x4.zero.c1.z, fix64p.zero);
            TestUtils.AreEqual(fix64p3x4.zero.c2.x, fix64p.zero);
            TestUtils.AreEqual(fix64p3x4.zero.c2.y, fix64p.zero);
            TestUtils.AreEqual(fix64p3x4.zero.c2.z, fix64p.zero);
            TestUtils.AreEqual(fix64p3x4.zero.c3.x, fix64p.zero);
            TestUtils.AreEqual(fix64p3x4.zero.c3.y, fix64p.zero);
            TestUtils.AreEqual(fix64p3x4.zero.c3.z, fix64p.zero);
        }

        [TestCompiler]
        public static void fix64p3x4_operator_equal_wide_wide()
        {
            fix64p3x4 a0 = fix64p3x4(new fix64p(-135.18924f), new fix64p(-49.0941162f), new fix64p(169.129822f), new fix64p(240.8053f), new fix64p(314.7392f), new fix64p(442.393f), new fix64p(177.924438f), new fix64p(335.5334f), new fix64p(168.15448f), new fix64p(350.729553f), new fix64p(367.178467f), new fix64p(46.9414673f));
            fix64p3x4 b0 = fix64p3x4(new fix64p(-220.014648f), new fix64p(66.98004f), new fix64p(499.2016f), new fix64p(-371.1131f), new fix64p(208.448669f), new fix64p(390.8037f), new fix64p(-72.44382f), new fix64p(362.97644f), new fix64p(194.678345f), new fix64p(471.644836f), new fix64p(-404.044678f), new fix64p(-144.696747f));
            bool3x4 r0 = bool3x4(false, false, false, false, false, false, false, false, false, false, false, false);
            TestUtils.AreEqual(a0 == b0, r0);

            fix64p3x4 a1 = fix64p3x4(new fix64p(188.76416f), new fix64p(-97.2113953f), new fix64p(-293.320984f), new fix64p(-234.822937f), new fix64p(417.0337f), new fix64p(26.3864136f), new fix64p(269.245728f), new fix64p(29.4741821f), new fix64p(479.485229f), new fix64p(-237.230957f), new fix64p(-221.9837f), new fix64p(-506.672546f));
            fix64p3x4 b1 = fix64p3x4(new fix64p(-494.446655f), new fix64p(-252.970367f), new fix64p(234.417114f), new fix64p(398.724f), new fix64p(260.4287f), new fix64p(370.144226f), new fix64p(89.579834f), new fix64p(-434.816833f), new fix64p(-109.845337f), new fix64p(336.973022f), new fix64p(-409.154968f), new fix64p(500.387573f));
            bool3x4 r1 = bool3x4(false, false, false, false, false, false, false, false, false, false, false, false);
            TestUtils.AreEqual(a1 == b1, r1);

            fix64p3x4 a2 = fix64p3x4(new fix64p(-22.98944f), new fix64p(487.260864f), new fix64p(-419.731964f), new fix64p(337.2033f), new fix64p(245.043884f), new fix64p(390.215881f), new fix64p(84.4129639f), new fix64p(434.2079f), new fix64p(-68.7284241f), new fix64p(485.769958f), new fix64p(413.169739f), new fix64p(-418.2693f));
            fix64p3x4 b2 = fix64p3x4(new fix64p(-174.081818f), new fix64p(395.101135f), new fix64p(350.3393f), new fix64p(-243.144592f), new fix64p(-416.397369f), new fix64p(151.576477f), new fix64p(-18.2243347f), new fix64p(-431.677917f), new fix64p(-468.330963f), new fix64p(429.495728f), new fix64p(477.389221f), new fix64p(-433.4254f));
            bool3x4 r2 = bool3x4(false, false, false, false, false, false, false, false, false, false, false, false);
            TestUtils.AreEqual(a2 == b2, r2);

            fix64p3x4 a3 = fix64p3x4(new fix64p(-346.795868f), new fix64p(504.159668f), new fix64p(345.186279f), new fix64p(-434.713043f), new fix64p(-499.7741f), new fix64p(282.019043f), new fix64p(259.15625f), new fix64p(306.455933f), new fix64p(435.2254f), new fix64p(-386.8997f), new fix64p(211.364014f), new fix64p(-7.229828f));
            fix64p3x4 b3 = fix64p3x4(new fix64p(273.5464f), new fix64p(-34.9762268f), new fix64p(221.968445f), new fix64p(85.91913f), new fix64p(-85.59894f), new fix64p(392.7608f), new fix64p(-117.924072f), new fix64p(-445.3056f), new fix64p(-242.468964f), new fix64p(173.643066f), new fix64p(389.897766f), new fix64p(-14.2904663f));
            bool3x4 r3 = bool3x4(false, false, false, false, false, false, false, false, false, false, false, false);
            TestUtils.AreEqual(a3 == b3, r3);
        }

        [TestCompiler]
        public static void fix64p3x4_operator_equal_wide_scalar()
        {
            fix64p3x4 a0 = fix64p3x4(new fix64p(65.6712f), new fix64p(404.415527f), new fix64p(-269.730164f), new fix64p(83.6306152f), new fix64p(152.9945f), new fix64p(-155.868286f), new fix64p(314.671265f), new fix64p(386.365173f), new fix64p(290.04895f), new fix64p(-132.6352f), new fix64p(-65.66748f), new fix64p(-69.68326f));
            fix64p b0 = (new fix64p(-155.815765f));
            bool3x4 r0 = bool3x4(false, false, false, false, false, false, false, false, false, false, false, false);
            TestUtils.AreEqual(a0 == b0, r0);

            fix64p3x4 a1 = fix64p3x4(new fix64p(-191.190765f), new fix64p(-232.895691f), new fix64p(-319.144043f), new fix64p(-49.70108f), new fix64p(-300.8819f), new fix64p(333.396851f), new fix64p(386.3775f), new fix64p(-296.7019f), new fix64p(-309.1172f), new fix64p(141.542358f), new fix64p(-227.323334f), new fix64p(83.87286f));
            fix64p b1 = (new fix64p(186.845215f));
            bool3x4 r1 = bool3x4(false, false, false, false, false, false, false, false, false, false, false, false);
            TestUtils.AreEqual(a1 == b1, r1);

            fix64p3x4 a2 = fix64p3x4(new fix64p(-410.91687f), new fix64p(-390.103577f), new fix64p(36.57434f), new fix64p(-427.541443f), new fix64p(-268.170837f), new fix64p(175.8117f), new fix64p(-193.47995f), new fix64p(291.051941f), new fix64p(423.97168f), new fix64p(-429.8739f), new fix64p(-275.156952f), new fix64p(-56.3323669f));
            fix64p b2 = (new fix64p(110.501282f));
            bool3x4 r2 = bool3x4(false, false, false, false, false, false, false, false, false, false, false, false);
            TestUtils.AreEqual(a2 == b2, r2);

            fix64p3x4 a3 = fix64p3x4(new fix64p(-95.83597f), new fix64p(253.006165f), new fix64p(-300.509521f), new fix64p(314.866516f), new fix64p(195.616211f), new fix64p(-26.1289063f), new fix64p(-284.7747f), new fix64p(-242.672058f), new fix64p(140.3606f), new fix64p(505.644348f), new fix64p(506.537964f), new fix64p(-502.3698f));
            fix64p b3 = (new fix64p(-124.865326f));
            bool3x4 r3 = bool3x4(false, false, false, false, false, false, false, false, false, false, false, false);
            TestUtils.AreEqual(a3 == b3, r3);
        }

        [TestCompiler]
        public static void fix64p3x4_operator_equal_scalar_wide()
        {
            fix64p a0 = (new fix64p(36.38391f));
            fix64p3x4 b0 = fix64p3x4(new fix64p(-400.4892f), new fix64p(-71.2868347f), new fix64p(156.978088f), new fix64p(-225.238739f), new fix64p(499.141785f), new fix64p(-211.979919f), new fix64p(428.311951f), new fix64p(-489.501343f), new fix64p(-5.691559f), new fix64p(-30.8659363f), new fix64p(-362.9831f), new fix64p(184.503174f));
            bool3x4 r0 = bool3x4(false, false, false, false, false, false, false, false, false, false, false, false);
            TestUtils.AreEqual(a0 == b0, r0);

            fix64p a1 = (new fix64p(-160.470612f));
            fix64p3x4 b1 = fix64p3x4(new fix64p(316.668823f), new fix64p(390.369263f), new fix64p(505.1051f), new fix64p(-294.6487f), new fix64p(443.1991f), new fix64p(96.5592651f), new fix64p(-257.012939f), new fix64p(-245.054962f), new fix64p(326.464844f), new fix64p(-23.9599f), new fix64p(-168.694885f), new fix64p(386.2486f));
            bool3x4 r1 = bool3x4(false, false, false, false, false, false, false, false, false, false, false, false);
            TestUtils.AreEqual(a1 == b1, r1);

            fix64p a2 = (new fix64p(-227.090637f));
            fix64p3x4 b2 = fix64p3x4(new fix64p(-336.612427f), new fix64p(365.108154f), new fix64p(-405.390839f), new fix64p(-473.995483f), new fix64p(298.435364f), new fix64p(-149.86322f), new fix64p(450.0664f), new fix64p(153.47644f), new fix64p(56.28778f), new fix64p(39.3421021f), new fix64p(-350.403717f), new fix64p(-482.717224f));
            bool3x4 r2 = bool3x4(false, false, false, false, false, false, false, false, false, false, false, false);
            TestUtils.AreEqual(a2 == b2, r2);

            fix64p a3 = (new fix64p(239.9654f));
            fix64p3x4 b3 = fix64p3x4(new fix64p(-3.40603638f), new fix64p(-1.49484253f), new fix64p(105.960449f), new fix64p(151.537537f), new fix64p(63.2832031f), new fix64p(-289.693176f), new fix64p(137.553772f), new fix64p(-247.666473f), new fix64p(-339.420563f), new fix64p(23.2382813f), new fix64p(21.1778564f), new fix64p(477.03656f));
            bool3x4 r3 = bool3x4(false, false, false, false, false, false, false, false, false, false, false, false);
            TestUtils.AreEqual(a3 == b3, r3);
        }

        [TestCompiler]
        public static void fix64p3x4_operator_not_equal_wide_wide()
        {
            fix64p3x4 a0 = fix64p3x4(new fix64p(279.994141f), new fix64p(-43.34201f), new fix64p(-465.724731f), new fix64p(317.466553f), new fix64p(85.7149658f), new fix64p(360.8905f), new fix64p(366.081543f), new fix64p(154.542847f), new fix64p(332.4262f), new fix64p(397.11322f), new fix64p(-431.374969f), new fix64p(489.0108f));
            fix64p3x4 b0 = fix64p3x4(new fix64p(-460.9121f), new fix64p(-476.009033f), new fix64p(468.1364f), new fix64p(-341.012543f), new fix64p(-62.65805f), new fix64p(-458.801666f), new fix64p(-457.730225f), new fix64p(-59.5232544f), new fix64p(3.024231f), new fix64p(155.812744f), new fix64p(-19.8399048f), new fix64p(-6.01693726f));
            bool3x4 r0 = bool3x4(true, true, true, true, true, true, true, true, true, true, true, true);
            TestUtils.AreEqual(a0 != b0, r0);

            fix64p3x4 a1 = fix64p3x4(new fix64p(398.4336f), new fix64p(-489.817932f), new fix64p(171.4049f), new fix64p(-67.82968f), new fix64p(-192.278717f), new fix64p(227.84082f), new fix64p(62.1381836f), new fix64p(262.186462f), new fix64p(-404.0531f), new fix64p(34.449585f), new fix64p(-204.795776f), new fix64p(-285.4118f));
            fix64p3x4 b1 = fix64p3x4(new fix64p(-406.207916f), new fix64p(-102.420715f), new fix64p(-40.362915f), new fix64p(452.6754f), new fix64p(93.25757f), new fix64p(-258.378052f), new fix64p(-184.0498f), new fix64p(-379.2353f), new fix64p(-370.687317f), new fix64p(-255.947235f), new fix64p(29.0557861f), new fix64p(322.407654f));
            bool3x4 r1 = bool3x4(true, true, true, true, true, true, true, true, true, true, true, true);
            TestUtils.AreEqual(a1 != b1, r1);

            fix64p3x4 a2 = fix64p3x4(new fix64p(-72.20682f), new fix64p(444.749268f), new fix64p(238.81781f), new fix64p(365.1801f), new fix64p(-437.9229f), new fix64p(-362.442627f), new fix64p(445.954346f), new fix64p(-0.417480469f), new fix64p(-506.828369f), new fix64p(245.477051f), new fix64p(-173.571045f), new fix64p(390.338562f));
            fix64p3x4 b2 = fix64p3x4(new fix64p(415.071716f), new fix64p(-467.726135f), new fix64p(-433.784668f), new fix64p(-212.165924f), new fix64p(474.674927f), new fix64p(452.483215f), new fix64p(-92.11273f), new fix64p(-385.9221f), new fix64p(420.2151f), new fix64p(-239.176056f), new fix64p(-99.0791f), new fix64p(4.476013f));
            bool3x4 r2 = bool3x4(true, true, true, true, true, true, true, true, true, true, true, true);
            TestUtils.AreEqual(a2 != b2, r2);

            fix64p3x4 a3 = fix64p3x4(new fix64p(252.837769f), new fix64p(47.8658447f), new fix64p(457.7105f), new fix64p(-313.22113f), new fix64p(391.203857f), new fix64p(481.786133f), new fix64p(26.8878174f), new fix64p(-298.1424f), new fix64p(240.077454f), new fix64p(-332.455139f), new fix64p(-333.607178f), new fix64p(-313.1897f));
            fix64p3x4 b3 = fix64p3x4(new fix64p(264.348572f), new fix64p(451.312317f), new fix64p(232.958008f), new fix64p(-142.6222f), new fix64p(-300.2256f), new fix64p(268.333252f), new fix64p(-112.103546f), new fix64p(-270.494019f), new fix64p(-71.9932251f), new fix64p(99.46326f), new fix64p(321.7033f), new fix64p(200.059631f));
            bool3x4 r3 = bool3x4(true, true, true, true, true, true, true, true, true, true, true, true);
            TestUtils.AreEqual(a3 != b3, r3);
        }

        [TestCompiler]
        public static void fix64p3x4_operator_not_equal_wide_scalar()
        {
            fix64p3x4 a0 = fix64p3x4(new fix64p(-155.4411f), new fix64p(-19.4266052f), new fix64p(174.633057f), new fix64p(507.920715f), new fix64p(59.177063f), new fix64p(171.151489f), new fix64p(-58.92328f), new fix64p(-398.176849f), new fix64p(492.20105f), new fix64p(-165.241516f), new fix64p(270.341f), new fix64p(-380.243256f));
            fix64p b0 = (new fix64p(-393.413544f));
            bool3x4 r0 = bool3x4(true, true, true, true, true, true, true, true, true, true, true, true);
            TestUtils.AreEqual(a0 != b0, r0);

            fix64p3x4 a1 = fix64p3x4(new fix64p(501.899048f), new fix64p(458.400452f), new fix64p(46.7709961f), new fix64p(161.459961f), new fix64p(261.514221f), new fix64p(-145.6124f), new fix64p(-0.449920654f), new fix64p(350.461426f), new fix64p(202.221008f), new fix64p(242.664f), new fix64p(382.677063f), new fix64p(-468.967957f));
            fix64p b1 = (new fix64p(-134.345459f));
            bool3x4 r1 = bool3x4(true, true, true, true, true, true, true, true, true, true, true, true);
            TestUtils.AreEqual(a1 != b1, r1);

            fix64p3x4 a2 = fix64p3x4(new fix64p(-497.459473f), new fix64p(-328.587769f), new fix64p(-506.490326f), new fix64p(449.348145f), new fix64p(210.771f), new fix64p(249.181824f), new fix64p(-338.468536f), new fix64p(229.670654f), new fix64p(-76.5433044f), new fix64p(317.286072f), new fix64p(401.939575f), new fix64p(210.984863f));
            fix64p b2 = (new fix64p(-80.93225f));
            bool3x4 r2 = bool3x4(true, true, true, true, true, true, true, true, true, true, true, true);
            TestUtils.AreEqual(a2 != b2, r2);

            fix64p3x4 a3 = fix64p3x4(new fix64p(-147.096313f), new fix64p(207.731384f), new fix64p(284.3921f), new fix64p(-509.0853f), new fix64p(414.307617f), new fix64p(-52.2944641f), new fix64p(-140.437927f), new fix64p(-316.787781f), new fix64p(-358.696838f), new fix64p(312.31897f), new fix64p(270.629456f), new fix64p(-140.016724f));
            fix64p b3 = (new fix64p(-193.399048f));
            bool3x4 r3 = bool3x4(true, true, true, true, true, true, true, true, true, true, true, true);
            TestUtils.AreEqual(a3 != b3, r3);
        }

        [TestCompiler]
        public static void fix64p3x4_operator_not_equal_scalar_wide()
        {
            fix64p a0 = (new fix64p(478.353149f));
            fix64p3x4 b0 = fix64p3x4(new fix64p(459.553223f), new fix64p(436.453247f), new fix64p(-488.714172f), new fix64p(392.767944f), new fix64p(-266.736633f), new fix64p(338.557861f), new fix64p(-338.100128f), new fix64p(-152.314545f), new fix64p(-452.820679f), new fix64p(209.439331f), new fix64p(50.10797f), new fix64p(372.4344f));
            bool3x4 r0 = bool3x4(true, true, true, true, true, true, true, true, true, true, true, true);
            TestUtils.AreEqual(a0 != b0, r0);

            fix64p a1 = (new fix64p(-488.0213f));
            fix64p3x4 b1 = fix64p3x4(new fix64p(489.740784f), new fix64p(270.4001f), new fix64p(-472.846771f), new fix64p(-286.850464f), new fix64p(-384.691864f), new fix64p(443.423523f), new fix64p(358.7472f), new fix64p(-15.4140625f), new fix64p(-342.179169f), new fix64p(468.967529f), new fix64p(-130.568085f), new fix64p(401.785828f));
            bool3x4 r1 = bool3x4(true, true, true, true, true, true, true, true, true, true, true, true);
            TestUtils.AreEqual(a1 != b1, r1);

            fix64p a2 = (new fix64p(-268.352264f));
            fix64p3x4 b2 = fix64p3x4(new fix64p(-239.231018f), new fix64p(411.386536f), new fix64p(139.769348f), new fix64p(334.522034f), new fix64p(-223.629242f), new fix64p(-12.4884644f), new fix64p(113.468872f), new fix64p(-189.652252f), new fix64p(-212.846558f), new fix64p(306.1256f), new fix64p(-178.330383f), new fix64p(382.141968f));
            bool3x4 r2 = bool3x4(true, true, true, true, true, true, true, true, true, true, true, true);
            TestUtils.AreEqual(a2 != b2, r2);

            fix64p a3 = (new fix64p(-340.8656f));
            fix64p3x4 b3 = fix64p3x4(new fix64p(-17.58023f), new fix64p(-409.874847f), new fix64p(-349.70166f), new fix64p(275.8543f), new fix64p(-229.371948f), new fix64p(-127.505737f), new fix64p(90.75342f), new fix64p(-422.087128f), new fix64p(-2.44754028f), new fix64p(-280.5517f), new fix64p(-484.374359f), new fix64p(-33.7634277f));
            bool3x4 r3 = bool3x4(true, true, true, true, true, true, true, true, true, true, true, true);
            TestUtils.AreEqual(a3 != b3, r3);
        }


    }
}
