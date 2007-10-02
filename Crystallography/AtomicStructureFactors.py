
# Atomic structure factors in Five-Gaussian approximation.
# The formula for the atomic structure factor is
#   f(s) = N.sum(a*N.exp(-b*s))
# where (a, b) are the two arrays stored in atomic_structure_factors
# and s is the length of the scattering vector (in 1/nm).
#

from Scientific import N

atomic_structure_factors = {

('h', 0): (N.array([  0.49300200,   0.32291200,   0.14019100,   0.04081000,   0.00303800]),
             N.array([  0.02627725,   0.06531425,   0.00785590,   0.14449924,   0.00000000])),
('h', -1): (N.array([  0.89766100,   0.56561600,   0.41581500,   0.11697300,   0.00238900]),
             N.array([  0.13284200,   0.03796750,   0.46643997,   0.00891772,   0.00000000])),
('he', 0): (N.array([  0.87340000,   0.63090000,   0.31120000,   0.17800000,   0.00640000]),
             N.array([  0.02275925,   0.00839200,   0.05731900,   0.00245525,   0.00000000])),
('li', 0): (N.array([  1.12820000,   0.75080000,   0.61750000,   0.46530000,   0.03770000]),
             N.array([  0.00988650,   0.00263100,   0.21347626,   0.42065250,   0.00000000])),
('be', 0): (N.array([  1.59190000,   1.12780000,   0.53910000,   0.70290000,   0.03850000]),
             N.array([  0.10910675,   0.00465575,   0.25870750,   0.00135500,   0.00000000])),
('be', 2): (N.array([  6.26030000,   0.88490000,   0.79930000,   0.16470000,  -6.10920000]),
             N.array([  0.00000675,   0.00207825,   0.00568950,   0.01278650,   0.00000000])),
('b', 0): (N.array([  2.05450000,   1.33260000,   1.09790000,   0.70680000,  -0.19320000]),
             N.array([  0.05804625,   0.00255250,   0.15087450,   0.00035075,   0.00000000])),
('c', 0): (N.array([  2.31000000,   1.02000000,   1.58860000,   0.86500000,   0.21560000]),
             N.array([  0.05210975,   0.02551875,   0.00142175,   0.12912800,   0.00000000])),
('cv', 0): (N.array([  2.26069000,   1.56165000,   1.05075000,   0.83925900,   0.28697700]),
             N.array([  0.05672675,   0.00164166,   0.02439045,   0.13898724,   0.00000000])),
('n', 0): (N.array([ 12.21260000,   3.13220000,   2.01250000,   1.16630000, -11.52899900]),
             N.array([  0.00001425,   0.02473325,   0.07249375,   0.00145650,   0.00000000])),
('o', 0): (N.array([  3.04850000,   2.28680000,   1.54630000,   0.86700000,   0.25080000]),
             N.array([  0.03319275,   0.01425275,   0.00080975,   0.08227224,   0.00000000])),
('o', -1): (N.array([  4.19160000,   1.63969000,   1.52673000, -20.30699900,  21.94120000]),
             N.array([  0.03214325,   0.01043090,   0.11754475,  -0.00003510,   0.00000000])),
('f', 0): (N.array([  3.53920000,   2.64120000,   1.51700000,   1.02430000,   0.27760000]),
             N.array([  0.02570625,   0.01073600,   0.00065375,   0.06536900,   0.00000000])),
('f', -1): (N.array([  3.63220000,   3.51057000,   1.26064000,   0.94070600,   0.65339600]),
             N.array([  0.01319390,   0.03683825,   0.00110564,   0.11835925,   0.00000000])),
('ne', 0): (N.array([  3.95530000,   3.11250000,   1.45460000,   1.12510000,   0.35150000]),
             N.array([  0.02101050,   0.00856550,   0.00057650,   0.05429600,   0.00000000])),
('na', 0): (N.array([  4.76260000,   3.17360000,   1.26740000,   1.11280000,   0.67600000]),
             N.array([  0.00821250,   0.02210550,   0.00078400,   0.32355999,   0.00000000])),
('na', 1): (N.array([  3.25650000,   3.93620000,   1.39980000,   1.00320000,   0.40400000]),
             N.array([  0.00666775,   0.01528825,   0.00050025,   0.03509750,   0.00000000])),
('mg', 0): (N.array([  5.42040000,   2.17350000,   1.22690000,   2.30730000,   0.85840000]),
             N.array([  0.00706875,   0.19815275,   0.00095200,   0.01798425,   0.00000000])),
('mg', 2): (N.array([  3.49880000,   3.83780000,   1.32840000,   0.84970000,   0.48530000]),
             N.array([  0.00541900,   0.01188550,   0.00046250,   0.02535275,   0.00000000])),
('al', 0): (N.array([  6.42020000,   1.90020000,   1.59360000,   1.96460000,   1.11510000]),
             N.array([  0.00759675,   0.00185650,   0.07886800,   0.21272150,   0.00000000])),
('al', 3): (N.array([  4.17448000,   3.38760000,   1.20296000,   0.52813700,   0.70678600]),
             N.array([  0.00484540,   0.01036382,   0.00057188,   0.02071310,   0.00000000])),
('si', 0): (N.array([  6.29150000,   3.03530000,   1.98910000,   1.54100000,   1.14070000]),
             N.array([  0.00609650,   0.08083424,   0.00169625,   0.20423424,   0.00000000])),
('siv', 0): (N.array([  5.66269000,   3.07164000,   2.62446000,   1.39320000,   1.24707000]),
             N.array([  0.00666300,   0.09665850,   0.00229236,   0.23386450,   0.00000000])),
('si', 4): (N.array([  4.43918000,   3.20345000,   1.19453000,   0.41653000,   0.74629700]),
             N.array([  0.00410417,   0.00859392,   0.00053725,   0.01663412,   0.00000000])),
('p', 0): (N.array([  6.43450000,   4.17910000,   1.78000000,   1.49080000,   1.11490000]),
             N.array([  0.00476675,   0.06789250,   0.00131500,   0.17041124,   0.00000000])),
('s', 0): (N.array([  6.90530000,   5.20340000,   1.43790000,   1.58630000,   0.86690000]),
             N.array([  0.00366975,   0.05553775,   0.00063400,   0.14043000,   0.00000000])),
('cl', 0): (N.array([ 11.46040000,   7.19640000,   6.25560000,   1.64550000,  -9.55740000]),
             N.array([  0.00002600,   0.00291550,   0.04629850,   0.11944600,   0.00000000])),
('cl', -1): (N.array([ 18.29150000,   7.20840000,   6.53370000,   2.33860000, -16.37800000]),
             N.array([  0.00001650,   0.00292925,   0.04885600,   0.15112150,   0.00000000])),
('ar', 0): (N.array([  7.48450000,   6.77230000,   0.65390000,   1.64420000,   1.44450000]),
             N.array([  0.00226800,   0.03710175,   0.10974575,   0.08348225,   0.00000000])),
('k', 0): (N.array([  8.21859900,   7.43980000,   1.05190000,   0.86590000,   1.42280000]),
             N.array([  0.03198725,   0.00193700,   0.53296749,   0.10421024,   0.00000000])),
('k', 1): (N.array([  7.95780000,   7.49170000,   6.35900000,   1.19150000,  -4.99780000]),
             N.array([  0.03158275,   0.00191850,  -0.00000500,   0.07978200,   0.00000000])),
('ca', 0): (N.array([  8.62660000,   7.38730000,   1.58990000,   1.02110000,   1.37510000]),
             N.array([  0.02610525,   0.00164975,   0.21437099,   0.44609249,   0.00000000])),
('ca', 2): (N.array([ 15.63480000,   7.95180000,   8.43720000,   0.85370000, -14.87500000]),
             N.array([ -0.00001850,   0.00152225,   0.02577900,   0.06497625,   0.00000000])),
('sc', 0): (N.array([  9.18900000,   7.36790000,   1.64090000,   1.46800000,   1.33290000]),
             N.array([  0.02255325,   0.00143225,   0.34027000,   0.12838275,   0.00000000])),
('sc', 3): (N.array([ 14.40080000,   8.02730000,   1.65943000,   1.57936000,  -6.66670000]),
             N.array([  0.00074635,   0.01990725,  -0.00071510,   0.04016550,   0.00000000])),
('ti', 0): (N.array([  9.75950000,   7.35580000,   1.69910000,   1.90210000,   1.28070000]),
             N.array([  0.01962700,   0.00125000,   0.08908450,   0.29026249,   0.00000000])),
('ti', 2): (N.array([  9.11423000,   7.62174000,   2.27930000,   0.08789900,   0.89715500]),
             N.array([  0.01881075,   0.00114396,   0.04884025,   0.15413950,   0.00000000])),
('ti', 3): (N.array([ 17.73440000,   8.73816000,   5.25691000,   1.92134000, -14.65200000]),
             N.array([  0.00055152,   0.01761790,  -0.00039405,   0.03994200,   0.00000000])),
('ti', 4): (N.array([ 19.51140000,   8.23473000,   2.01341000,   1.52080000, -13.28000000]),
             N.array([  0.00044712,   0.01667545,  -0.00073157,   0.03236600,   0.00000000])),
('v', 0): (N.array([ 10.29710000,   7.35110000,   2.07030000,   2.05710000,   1.21990000]),
             N.array([  0.01716425,   0.00109625,   0.06723450,   0.25619499,   0.00000000])),
('v', 2): (N.array([ 10.10600000,   7.35410000,   2.28840000,   0.02230000,   1.22980000]),
             N.array([  0.01720450,   0.00110225,   0.05075100,   0.28780500,   0.00000000])),
('v', 5): (N.array([ 15.68870000,   8.14208000,   2.03081000,  -9.57600000,   1.71430000]),
             N.array([  0.00169751,   0.01350337,   0.02493195,   0.00235116,   0.00000000])),
('cr', 0): (N.array([ 10.64060000,   7.35370000,   3.32400000,   1.49220000,   1.18320000]),
             N.array([  0.01525950,   0.00098000,   0.05065650,   0.24684975,   0.00000000])),
('cr', 2): (N.array([  9.54034000,   7.75090000,   3.58274000,   0.50910700,   0.61689800]),
             N.array([  0.01415195,   0.00086065,   0.03326875,   0.08105600,   0.00000000])),
('cr', 3): (N.array([  9.68090000,   7.81136000,   2.87603000,   0.11357500,   0.51827500]),
             N.array([  0.01398657,   0.00083598,   0.03207200,   0.08219025,   0.00000000])),
('mn', 0): (N.array([ 11.28190000,   7.35730000,   3.01930000,   2.24410000,   1.08960000]),
             N.array([  0.01335225,   0.00085800,   0.04466850,   0.20938576,   0.00000000])),
('mn', 2): (N.array([ 10.80610000,   7.36200000,   3.52680000,   0.21840000,   1.08740000]),
             N.array([  0.01319900,   0.00085875,   0.03585750,   0.10330875,   0.00000000])),
('mn', 3): (N.array([  9.84521000,   7.87194000,   3.56531000,   0.32361300,   0.39397400]),
             N.array([  0.01229492,   0.00073598,   0.02704275,   0.06032025,   0.00000000])),
('mn', 4): (N.array([  9.96253000,   7.97057000,   2.76067000,   0.05444700,   0.25187700]),
             N.array([  0.01212125,   0.00070826,   0.02621300,   0.06893250,   0.00000000])),
('fe', 0): (N.array([ 11.76950000,   7.35730000,   3.52220000,   2.30450000,   1.03690000]),
             N.array([  0.01190275,   0.00076800,   0.03838375,   0.19220125,   0.00000000])),
('fe', 2): (N.array([ 11.04240000,   7.37400000,   4.13460000,   0.43990000,   1.00970000]),
             N.array([  0.01163450,   0.00076325,   0.03013650,   0.07820225,   0.00000000])),
('fe', 3): (N.array([ 11.17640000,   7.38630000,   3.39480000,   0.07240000,   0.97070000]),
             N.array([  0.01153675,   0.00075125,   0.02918225,   0.09639150,   0.00000000])),
('co', 0): (N.array([ 12.28410000,   7.34090000,   4.00340000,   2.34880000,   1.01180000]),
             N.array([  0.01069775,   0.00069600,   0.03383975,   0.17792299,   0.00000000])),
('co', 2): (N.array([ 11.22960000,   7.38830000,   4.73930000,   0.71080000,   0.93240000]),
             N.array([  0.01030775,   0.00068150,   0.02561075,   0.06411650,   0.00000000])),
('co', 3): (N.array([ 10.33799900,   7.88173000,   4.76795000,   0.72559100,   0.28666700]),
             N.array([  0.00977422,   0.00059667,   0.02088957,   0.04587275,   0.00000000])),
('ni', 0): (N.array([ 12.83760000,   7.29200000,   4.44380000,   2.38000000,   1.03410000]),
             N.array([  0.00969625,   0.00064125,   0.03044075,   0.16585525,   0.00000000])),
('ni', 2): (N.array([ 11.41660000,   7.40050000,   5.34420000,   0.97730000,   0.86140000]),
             N.array([  0.00919150,   0.00061225,   0.02218250,   0.05540650,   0.00000000])),
('ni', 3): (N.array([ 10.78060000,   7.75868000,   5.22746000,   0.84711400,   0.38604400]),
             N.array([  0.00886925,   0.00055785,   0.01911170,   0.04241825,   0.00000000])),
('cu', 0): (N.array([ 13.33799900,   7.16760000,   5.61580000,   1.67350000,   1.19100000]),
             N.array([  0.00895700,   0.00061750,   0.02849150,   0.16203150,   0.00000000])),
('cu', 1): (N.array([ 11.94750000,   7.35730000,   6.24550000,   1.55780000,   0.89000000]),
             N.array([  0.00841725,   0.00056850,   0.02165625,   0.06462175,   0.00000000])),
('cu', 2): (N.array([ 11.81680000,   7.11181000,   5.78135000,   1.14523000,   1.14431000]),
             N.array([  0.00843710,   0.00061019,   0.01996900,   0.04974250,   0.00000000])),
('zn', 0): (N.array([ 14.07430000,   7.03180000,   5.16250000,   2.41000000,   1.30410000]),
             N.array([  0.00816375,   0.00058325,   0.02579075,   0.14677425,   0.00000000])),
('zn', 2): (N.array([ 11.97190000,   7.38620000,   6.46680000,   1.39400000,   0.78070000]),
             N.array([  0.00748650,   0.00050775,   0.01770650,   0.04524875,   0.00000000])),
('ga', 0): (N.array([ 15.23540000,   6.70060000,   4.35910000,   2.96230000,   1.71890000]),
             N.array([  0.00766725,   0.00060300,   0.02695125,   0.15353374,   0.00000000])),
('ga', 3): (N.array([ 12.69199900,   6.69883000,   6.06692000,   1.00660000,   1.53545000]),
             N.array([  0.00703155,   0.00056972,   0.01591102,   0.03603050,   0.00000000])),
('ge', 0): (N.array([ 16.08160000,   6.37470000,   3.70680000,   3.68300000,   2.13130000]),
             N.array([  0.00712725,   0.00062900,   0.02861700,   0.13690625,   0.00000000])),
('ge', 4): (N.array([ 12.91720000,   6.70003000,   6.06791000,   0.85904100,   1.45572000]),
             N.array([  0.00634295,   0.00051464,   0.01369782,   0.02900750,   0.00000000])),
('as', 0): (N.array([ 16.67230000,   6.07010000,   3.43130000,   4.27790000,   2.53100000]),
             N.array([  0.00658625,   0.00066175,   0.03236975,   0.11949300,   0.00000000])),
('se', 0): (N.array([ 17.00059900,   5.81960000,   3.97310000,   4.35430000,   2.84090000]),
             N.array([  0.00602450,   0.00068150,   0.03809300,   0.10954075,   0.00000000])),
('br', 0): (N.array([ 17.17890000,   5.23580000,   5.63770000,   3.98510000,   2.95570000]),
             N.array([  0.00543075,   0.04144900,   0.00065225,   0.10358200,   0.00000000])),
('br', -1): (N.array([ 17.17179900,   6.33380000,   5.57540000,   3.72720000,   3.17760000]),
             N.array([  0.00551475,   0.04833625,   0.00071775,   0.14538375,   0.00000000])),
('kr', 0): (N.array([ 17.35549900,   6.72860000,   5.54930000,   3.53750000,   2.82500000]),
             N.array([  0.00484600,   0.04140575,   0.00056525,   0.09849300,   0.00000000])),
('rb', 0): (N.array([ 17.17840000,   9.64349900,   5.13990000,   1.52920000,   3.48730000]),
             N.array([  0.00447200,   0.04328775,   0.00068700,   0.41233497,   0.00000000])),
('rb', 1): (N.array([ 17.58160000,   7.65980000,   5.89810000,   2.78170000,   2.07820000]),
             N.array([  0.00428475,   0.03698925,   0.00040075,   0.07802175,   0.00000000])),
('sr', 0): (N.array([ 17.56629900,   9.81839900,   5.42200000,   2.66940000,   2.50640000]),
             N.array([  0.00389100,   0.03524700,   0.00041600,   0.33094002,   0.00000000])),
('sr', 2): (N.array([ 18.08740000,   8.13730000,   2.56540000, -34.19300100,  41.40250000]),
             N.array([  0.00372675,   0.03174075,   0.06141275,  -0.00003450,   0.00000000])),
('y', 0): (N.array([ 17.77599900,  10.29460000,   5.72629000,   3.26588000,   1.91213000]),
             N.array([  0.00350725,   0.03200150,   0.00031400,   0.26088499,   0.00000000])),
('y', 3): (N.array([ 17.92680000,   9.15310000,   1.76795000, -33.10800200,  40.26020100]),
             N.array([  0.00338542,   0.02803625,   0.05664975,  -0.00003297,   0.00000000])),
('zr', 0): (N.array([ 17.87649900,  10.94800000,   5.41732000,   3.65721000,   2.06929000]),
             N.array([  0.00319045,   0.02979000,   0.00029405,   0.21915674,   0.00000000])),
('zr', 4): (N.array([ 18.16680000,  10.05620000,   1.01118000,  -2.64790000,   9.41453900]),
             N.array([  0.00303700,   0.02537075,   0.05401350,  -0.00025690,   0.00000000])),
('nb', 0): (N.array([ 17.61420100,  12.01440000,   4.04183000,   3.53346000,   3.75591000]),
             N.array([  0.00297162,   0.02941500,   0.00051196,   0.17448925,   0.00000000])),
('nb', 3): (N.array([ 19.88119900,  18.06530000,  11.01770000,   1.94715000, -12.91200000]),
             N.array([  0.00004794,   0.00283262,   0.02540525,   0.07084725,   0.00000000])),
('nb', 5): (N.array([ 17.91630000,  13.34170000,  10.79900000,   0.33790500,  -6.39340000]),
             N.array([  0.00281115,   0.00007195,   0.02320515,   0.06430700,   0.00000000])),
('mo', 0): (N.array([  3.70250000,  17.23560000,  12.88760000,   3.74290000,   4.38750000]),
             N.array([  0.00069300,   0.00273950,   0.02751000,   0.15414600,   0.00000000])),
('mo', 3): (N.array([ 21.16640100,  18.20170000,  11.74230000,   2.30951000, -14.42100000]),
             N.array([  0.00003683,   0.00257577,   0.02384147,   0.06657675,   0.00000000])),
('mo', 5): (N.array([ 21.01490000,  18.09919900,  11.46320000,   0.74062500, -14.31600000]),
             N.array([  0.00003586,   0.00255595,   0.02197022,   0.05836300,   0.00000000])),
('mo', 6): (N.array([ 17.88710000,  11.17500000,   6.57891000,   0.00000000,   0.34494100]),
             N.array([  0.00259122,   0.02120152,   0.00014720,   0.00000000,   0.00000000])),
('tc', 0): (N.array([ 19.13010000,  11.09480000,   4.64901000,   2.71263000,   5.40428000]),
             N.array([  0.00216033,   0.02036217,   0.05392675,   0.21711799,   0.00000000])),
('ru', 0): (N.array([ 19.26739900,  12.91820000,   4.86337000,   1.56756000,   5.37874000]),
             N.array([  0.00202130,   0.02108667,   0.06199925,   0.23573200,   0.00000000])),
('ru', 3): (N.array([ 18.56380100,  13.28850000,   9.32601900,   3.00964000,  -3.18920000]),
             N.array([  0.00211832,   0.02092910,   0.00004415,   0.05721750,   0.00000000])),
('ru', 4): (N.array([ 18.50029900,  13.17869900,   4.71304000,   2.18535000,   1.42357000]),
             N.array([  0.00211145,   0.02031335,   0.00091237,   0.05212600,   0.00000000])),
('rh', 0): (N.array([ 19.29570000,  14.35010000,   4.73425000,   1.28918000,   5.32800000]),
             N.array([  0.00187884,   0.02054395,   0.06468725,   0.24651550,   0.00000000])),
('rh', 3): (N.array([ 18.87850000,  14.12590000,   3.32515000,  -6.19890000,  11.86780000]),
             N.array([  0.00191063,   0.01961095,   0.05312175,  -0.00002590,   0.00000000])),
('rh', 4): (N.array([ 18.85450000,  13.98060000,   2.53464000,  -5.65260000,  11.28350000]),
             N.array([  0.00190206,   0.01906090,   0.04832925,  -0.00002550,   0.00000000])),
('pd', 0): (N.array([ 19.33190000,  15.50169900,   5.29537000,   0.60584400,   5.26593000]),
             N.array([  0.00174664,   0.01997322,   0.06301300,   0.19224649,   0.00000000])),
('pd', 2): (N.array([ 19.17009900,  15.20960000,   4.32234000,   0.00000000,   5.29160000]),
             N.array([  0.00174055,   0.01888932,   0.05626425,   0.00000000,   0.00000000])),
('pd', 4): (N.array([ 19.24930000,  14.79000000,   2.89289000,  -7.94920000,  13.01740000]),
             N.array([  0.00170960,   0.01787082,   0.04478600,   0.00001282,   0.00000000])),
('ag', 0): (N.array([ 19.28080000,  16.68849900,   4.80450000,   1.04630000,   5.17900000]),
             N.array([  0.00161150,   0.01868150,   0.06165125,   0.24953899,   0.00000000])),
('ag', 1): (N.array([ 19.18120000,  15.97190000,   5.27475000,   0.35753400,   5.21572000]),
             N.array([  0.00161545,   0.01797807,   0.05433150,   0.16528675,   0.00000000])),
('ag', 2): (N.array([ 19.16429900,  16.24560000,   4.37090000,   0.00000000,   5.21404000]),
             N.array([  0.00161411,   0.01796360,   0.05351800,   0.00000000,   0.00000000])),
('cd', 0): (N.array([ 19.22139900,  17.64440000,   4.46100000,   1.60290000,   5.06940000]),
             N.array([  0.00148650,   0.01727225,   0.06175200,   0.21870624,   0.00000000])),
('cd', 2): (N.array([ 19.15140000,  17.25350000,   4.47128000,   0.00000000,   5.11937000]),
             N.array([  0.00149480,   0.01701597,   0.05063025,   0.00000000,   0.00000000])),
('in', 0): (N.array([ 19.16239900,  18.55960100,   4.29480000,   2.03960000,   4.93910000]),
             N.array([  0.00136900,   0.01594400,   0.06462475,   0.23200725,   0.00000000])),
('in', 3): (N.array([ 19.10450000,  18.11080000,   3.78897000,   0.00000000,   4.99635000]),
             N.array([  0.00137880,   0.01581175,   0.04339875,   0.00000000,   0.00000000])),
('sn', 0): (N.array([ 19.18890000,  19.10050000,   4.45850000,   2.46630000,   4.78210000]),
             N.array([  0.01457575,   0.00125775,   0.06722725,   0.20989275,   0.00000000])),
('sn', 2): (N.array([ 19.10940000,  19.05480000,   4.56480000,   0.48700000,   4.78610000]),
             N.array([  0.00125900,   0.01459450,   0.05843800,   0.15551525,   0.00000000])),
('sn', 4): (N.array([ 18.93330000,  19.71310000,   3.41820000,   0.01930000,   3.91820000]),
             N.array([  0.01441000,   0.00116375,   0.03501225,  -0.00189575,   0.00000000])),
('sb', 0): (N.array([ 19.64180000,  19.04550000,   5.03710000,   2.68270000,   4.59090000]),
             N.array([  0.01325850,   0.00115175,   0.06976850,   0.18820625,   0.00000000])),
('sb', 3): (N.array([ 18.97550000,  18.93299900,   5.10789000,   0.28875300,   4.69626000]),
             N.array([  0.00116799,   0.01305315,   0.04897550,   0.13877825,   0.00000000])),
('sb', 5): (N.array([ 19.86850000,  19.03019900,   2.41253000,   0.00000000,   4.69263000]),
             N.array([  0.01362132,   0.00116993,   0.03531475,   0.00000000,   0.00000000])),
('te', 0): (N.array([ 19.96439900,  19.01380000,   6.14487000,   2.52390000,   4.35200000]),
             N.array([  0.01204355,   0.00105221,   0.07132100,   0.17710075,   0.00000000])),
('i', 0): (N.array([ 20.14720000,  18.99490000,   7.51380000,   2.27350000,   4.07120000]),
             N.array([  0.01086750,   0.00095350,   0.06941500,   0.16719400,   0.00000000])),
('i', -1): (N.array([ 20.23320000,  18.99700000,   7.80690000,   2.88680000,   4.07140000]),
             N.array([  0.01089475,   0.00095375,   0.07381475,   0.21232599,   0.00000000])),
('xe', 0): (N.array([ 20.29330100,  19.02980000,   8.97670000,   1.99000000,   3.71180000]),
             N.array([  0.00982050,   0.00086000,   0.06616475,   0.16066450,   0.00000000])),
('cs', 0): (N.array([ 20.38920000,  19.10619900,  10.66200000,   1.49530000,   3.33520000]),
             N.array([  0.00892250,   0.00077675,   0.06096975,   0.53475998,   0.00000000])),
('cs', 1): (N.array([ 20.35240000,  19.12780000,  10.28210000,   0.96150000,   3.27910000]),
             N.array([  0.00888000,   0.00077150,   0.05928200,   0.14864124,   0.00000000])),
('ba', 0): (N.array([ 20.33610000,  19.29700100,  10.88800000,   2.69590000,   2.77310000]),
             N.array([  0.00804000,   0.00068900,   0.05051825,   0.41800499,   0.00000000])),
('ba', 2): (N.array([ 20.18070000,  19.11360000,  10.90539900,   0.77634000,   3.02902000]),
             N.array([  0.00803417,   0.00070827,   0.05013950,   0.12936499,   0.00000000])),
('la', 0): (N.array([ 20.57799900,  19.59899900,  11.37270000,   3.28719000,   2.14678000]),
             N.array([  0.00737042,   0.00061119,   0.04693150,   0.33280998,   0.00000000])),
('la', 3): (N.array([ 20.24889900,  19.37630100,  11.63229900,   0.33604800,   2.40860000]),
             N.array([  0.00730175,   0.00062674,   0.04455275,   0.13736324,   0.00000000])),
('ce', 0): (N.array([ 21.16709900,  19.76950100,  11.85129900,   3.33049000,   1.86264000]),
             N.array([  0.00703047,   0.00056709,   0.04402075,   0.31778250,   0.00000000])),
('ce', 3): (N.array([ 20.80360000,  19.55900000,  11.93690000,   0.61237600,   2.09013000]),
             N.array([  0.00694227,   0.00057885,   0.04135200,   0.10792300,   0.00000000])),
('ce', 4): (N.array([ 20.32350000,  19.81860000,  12.12330000,   0.14458300,   1.59180000]),
             N.array([  0.00664852,   0.00054712,   0.03949800,   0.15558875,   0.00000000])),
('pr', 0): (N.array([ 22.04399900,  19.66970100,  12.38560000,   2.82428000,   2.05830000]),
             N.array([  0.00693482,   0.00055522,   0.04191725,   0.35910999,   0.00000000])),
('pr', 3): (N.array([ 21.37270000,  19.74910000,  12.13290000,   0.97518000,   1.77132000]),
             N.array([  0.00661300,   0.00053575,   0.03830750,   0.09101625,   0.00000000])),
('pr', 4): (N.array([ 20.94129900,  20.05390000,  12.46680000,   0.29668900,   1.24285000]),
             N.array([  0.00636167,   0.00050620,   0.03703425,   0.11366074,   0.00000000])),
('nd', 0): (N.array([ 22.68450000,  19.68470000,  12.77400000,   2.85137000,   1.98486000]),
             N.array([  0.00665620,   0.00052657,   0.03971250,   0.34475750,   0.00000000])),
('nd', 3): (N.array([ 21.96100000,  19.93389900,  12.12000000,   1.51031000,   1.47588000]),
             N.array([  0.00631805,   0.00049809,   0.03544575,   0.07717925,   0.00000000])),
('pm', 0): (N.array([ 23.34050000,  19.60950100,  13.12350000,   2.87516000,   2.02876000]),
             N.array([  0.00640675,   0.00050522,   0.03775225,   0.33180248,   0.00000000])),
('pm', 3): (N.array([ 22.55270000,  20.11080000,  12.06710000,   2.07492000,   1.19499000]),
             N.array([  0.00604350,   0.00046442,   0.03281875,   0.06862275,   0.00000000])),
('sm', 0): (N.array([ 24.00420000,  19.42580000,  13.43960000,   2.89604000,   2.20963000]),
             N.array([  0.00618185,   0.00049113,   0.03599900,   0.32001751,   0.00000000])),
('eu', 0): (N.array([ 24.62739900,  19.08860000,  13.76030000,   2.92270000,   2.57450000]),
             N.array([  0.00596975,   0.00048550,   0.03438650,   0.30793499,   0.00000000])),
('eu', 2): (N.array([ 24.00630000,  19.95039900,  11.80340000,   3.87243000,   1.36389000]),
             N.array([  0.00569457,   0.00043382,   0.02902400,   0.06628900,   0.00000000])),
('eu', 3): (N.array([ 23.74969900,  20.37450000,  11.85090000,   3.26503000,   0.75934400]),
             N.array([  0.00555645,   0.00040985,   0.02827750,   0.05749150,   0.00000000])),
('gd', 0): (N.array([ 25.07090000,  19.07980000,  13.85180000,   3.54545000,   2.41960000]),
             N.array([  0.00563352,   0.00045488,   0.03233275,   0.25349499,   0.00000000])),
('gd', 3): (N.array([ 24.34660000,  20.42079900,  11.87080000,   3.71490000,   0.64508900]),
             N.array([  0.00533882,   0.00038881,   0.02644550,   0.05425725,   0.00000000])),
('tb', 0): (N.array([ 25.89760000,  18.21850000,  14.31670000,   2.95354000,   3.58224000]),
             N.array([  0.00560640,   0.00049036,   0.03166200,   0.28840500,   0.00000000])),
('tb', 3): (N.array([ 24.95590000,  20.32709900,  12.24710000,   3.77300000,   0.69196700]),
             N.array([  0.00514002,   0.00037381,   0.02512475,   0.05319325,   0.00000000])),
('dy', 0): (N.array([ 26.50700000,  17.63830000,  14.55960000,   2.96577000,   4.29728000]),
             N.array([  0.00545050,   0.00050543,   0.03047475,   0.27968500,   0.00000000])),
('dy', 3): (N.array([ 25.53949900,  20.28610000,  11.98120000,   4.50073000,   0.68969000]),
             N.array([  0.00495100,   0.00035846,   0.02337430,   0.04895250,   0.00000000])),
('ho', 0): (N.array([ 26.90490000,  17.29399900,  14.55830000,   3.63837000,   4.56796000]),
             N.array([  0.00517627,   0.00049485,   0.02860175,   0.23164150,   0.00000000])),
('ho', 3): (N.array([ 26.12960100,  20.09940000,  11.97880000,   4.93676000,   0.85279500]),
             N.array([  0.00477680,   0.00034839,   0.02200045,   0.04647700,   0.00000000])),
('er', 0): (N.array([ 27.65630000,  16.42849900,  14.97790000,   2.98233000,   5.92046000]),
             N.array([  0.00518390,   0.00055886,   0.02840100,   0.26425751,   0.00000000])),
('er', 3): (N.array([ 26.72200000,  19.77479900,  12.15060000,   5.17379000,   1.17613000]),
             N.array([  0.00461647,   0.00034322,   0.02090562,   0.04474350,   0.00000000])),
('tm', 0): (N.array([ 28.18190000,  15.88509900,  15.15420000,   2.98706000,   6.75621000]),
             N.array([  0.00507147,   0.00059712,   0.02749375,   0.25740250,   0.00000000])),
('tm', 3): (N.array([ 27.30830000,  19.33200100,  12.33390000,   5.38348000,   1.63929000]),
             N.array([  0.00446777,   0.00034243,   0.01991945,   0.04323050,   0.00000000])),
('yb', 0): (N.array([ 28.66410100,  15.43450000,  15.30870000,   2.98963000,   7.56672000]),
             N.array([  0.00497225,   0.00064280,   0.02666175,   0.25104250,   0.00000000])),
('yb', 2): (N.array([ 28.12089900,  17.68170000,  13.33350000,   5.14657000,   3.70983000]),
             N.array([  0.00446257,   0.00039992,   0.02045760,   0.05097500,   0.00000000])),
('yb', 3): (N.array([ 27.89170000,  18.76140000,  12.60720000,   5.47647000,   2.26001000]),
             N.array([  0.00433180,   0.00034697,   0.01911030,   0.04203825,   0.00000000])),
('lu', 0): (N.array([ 28.94759900,  15.22080000,  15.10000000,   3.71601000,   7.97628000]),
             N.array([  0.00475455,   0.02496297,   0.00065258,   0.21082451,   0.00000000])),
('lu', 3): (N.array([ 28.46279900,  18.12100000,  12.84289900,   5.59415000,   2.97573000]),
             N.array([  0.00420540,   0.00035573,   0.01834317,   0.04088375,   0.00000000])),
('hf', 0): (N.array([ 29.14399900,  15.17260000,  14.75860000,   4.30013000,   8.58154000]),
             N.array([  0.00458155,   0.02399975,   0.00068779,   0.18007250,   0.00000000])),
('hf', 4): (N.array([ 28.81309900,  18.46010000,  12.72850000,   5.59927000,   2.39699000]),
             N.array([  0.00397840,   0.00032226,   0.01690580,   0.03509150,   0.00000000])),
('ta', 0): (N.array([ 29.20240000,  15.22930000,  14.51350000,   4.76492000,   9.24354000]),
             N.array([  0.00443332,   0.02342615,   0.00073994,   0.15841100,   0.00000000])),
('ta', 5): (N.array([ 29.15869900,  18.84070000,  12.82679900,   5.38695000,   1.78555000]),
             N.array([  0.00376777,   0.00029185,   0.01578810,   0.03106100,   0.00000000])),
('w', 0): (N.array([ 29.08180000,  15.43000000,  14.43270000,   5.11982000,   9.88750000]),
             N.array([  0.00430072,   0.02306475,   0.00080426,   0.14264000,   0.00000000])),
('w', 6): (N.array([ 29.49359900,  19.37630100,  13.05439900,   5.06412000,   1.01074000]),
             N.array([  0.00356887,   0.00026155,   0.01484167,   0.02799300,   0.00000000])),
('re', 0): (N.array([ 28.76210000,  15.71890000,  14.55640000,   5.44174000,  10.47200000]),
             N.array([  0.00417977,   0.02273067,   0.00087625,   0.13021524,   0.00000000])),
('os', 0): (N.array([ 28.18940000,  16.15499900,  14.93050000,   5.67589000,  11.00050000]),
             N.array([  0.00407257,   0.02244870,   0.00095665,   0.12041175,   0.00000000])),
('os', 4): (N.array([ 30.41899900,  15.26370000,  14.74580000,   5.06795000,   6.49804000]),
             N.array([  0.00342782,   0.01711765,   0.00041298,   0.04500750,   0.00000000])),
('ir', 0): (N.array([ 27.30489900,  16.72959900,  15.61150000,   5.83377000,  11.47220000]),
             N.array([  0.00398197,   0.02216382,   0.00104479,   0.11250275,   0.00000000])),
('ir', 3): (N.array([ 30.41560000,  15.86200000,  13.61450000,   5.82008000,   8.27903000]),
             N.array([  0.00335807,   0.01777272,   0.00051158,   0.05081350,   0.00000000])),
('ir', 4): (N.array([ 30.70579900,  15.55120000,  14.23260000,   5.53672000,   6.96824000]),
             N.array([  0.00327307,   0.01679957,   0.00041813,   0.04372775,   0.00000000])),
('pt', 0): (N.array([ 27.00589900,  17.76390100,  15.71310000,   5.78370000,  11.68830000]),
             N.array([  0.00378232,   0.02202935,   0.00106148,   0.09652574,   0.00000000])),
('pt', 2): (N.array([ 29.84289900,  16.72240100,  13.21530000,   6.35234000,   9.85329000]),
             N.array([  0.00332317,   0.01847447,   0.00065824,   0.05735650,   0.00000000])),
('pt', 4): (N.array([ 30.96120100,  15.98290000,  13.73480000,   5.92034000,   7.39534000]),
             N.array([  0.00312032,   0.01652085,   0.00042160,   0.04234800,   0.00000000])),
('au', 0): (N.array([ 16.88190100,  18.59129900,  25.55820100,   5.86000000,  12.06580000]),
             N.array([  0.00115275,   0.02155400,   0.00370650,   0.09098900,   0.00000000])),
('au', 1): (N.array([ 28.01089900,  17.82040000,  14.33589900,   6.58077000,  11.22990000]),
             N.array([  0.00338302,   0.01934875,   0.00089188,   0.06601075,   0.00000000])),
('au', 3): (N.array([ 30.68859900,  16.90290100,  12.78010000,   6.52354000,   9.09680000]),
             N.array([  0.00304975,   0.01707180,   0.00053217,   0.04664750,   0.00000000])),
('hg', 0): (N.array([ 20.68090100,  19.04170000,  21.65749900,   5.96760000,  12.60890000]),
             N.array([  0.00136250,   0.02112100,   0.00393225,   0.09581150,   0.00000000])),
('hg', 1): (N.array([ 25.08530000,  18.49729900,  16.88830000,   6.48216000,  12.02050000]),
             N.array([  0.00348767,   0.01912762,   0.00110844,   0.07056550,   0.00000000])),
('hg', 2): (N.array([ 29.56410000,  18.05999900,  12.83740000,   6.89912000,  10.62680000]),
             N.array([  0.00302880,   0.01764097,   0.00071184,   0.05187050,   0.00000000])),
('tl', 0): (N.array([ 27.54460000,  19.15840000,  15.53800000,   5.52593000,  13.17460000]),
             N.array([  0.00163787,   0.02176877,   0.00490867,   0.11453725,   0.00000000])),
('tl', 1): (N.array([ 21.39850000,  20.47230000,  18.74779900,   6.82847000,  12.52580000]),
             N.array([  0.00367775,   0.00129348,   0.01858657,   0.07212050,   0.00000000])),
('tl', 3): (N.array([ 30.86949900,  18.38410000,  11.93280000,   7.00574000,   9.80270000]),
             N.array([  0.00275200,   0.01634630,   0.00054768,   0.04302850,   0.00000000])),
('pb', 0): (N.array([ 31.06169900,  13.06370000,  18.44199900,   5.96960000,  13.41180000]),
             N.array([  0.00172550,   0.00589400,   0.02154500,   0.11814475,   0.00000000])),
('pb', 2): (N.array([ 21.78860100,  19.56819900,  19.14060000,   7.01107000,  12.47340000]),
             N.array([  0.00334150,   0.00122096,   0.01693175,   0.05953300,   0.00000000])),
('pb', 4): (N.array([ 32.12439700,  18.80030100,  12.01750000,   6.96886000,   8.08428000]),
             N.array([  0.00251415,   0.01527315,   0.00036760,   0.03678500,   0.00000000])),
('bi', 0): (N.array([ 33.36890000,  12.95100000,  16.58770000,   6.46920000,  13.57819900]),
             N.array([  0.00176000,   0.00730950,   0.02198425,   0.12002325,   0.00000000])),
('bi', 3): (N.array([ 21.80530000,  19.50260000,  19.10530100,   7.10295000,  12.47110000]),
             N.array([  0.00308900,   0.01560372,   0.00117500,   0.05079625,   0.00000000])),
('bi', 5): (N.array([ 33.53640000,  25.09460100,  19.24969900,   6.91555000,  -6.79940000]),
             N.array([  0.00229135,   0.00097605,   0.01428535,   0.03207125,   0.00000000])),
('po', 0): (N.array([ 34.67260000,  15.47330000,  13.11380000,   7.02588000,  13.67700000]),
             N.array([  0.00175250,   0.00887695,   0.02389105,   0.11751125,   0.00000000])),
('at', 0): (N.array([ 35.31629900,  19.02109900,   9.49887000,   7.42518000,  13.71080000]),
             N.array([  0.00171467,   0.00993645,   0.02845600,   0.11367875,   0.00000000])),
('rn', 0): (N.array([ 35.56309900,  21.28160100,   8.00370000,   7.44330000,  13.69050000]),
             N.array([  0.00165775,   0.01017275,   0.03510550,   0.11061825,   0.00000000])),
('fr', 0): (N.array([ 35.92990100,  23.05469900,  12.14390000,   2.11253000,  13.72470000]),
             N.array([  0.00161613,   0.01044047,   0.05776300,   0.37661247,   0.00000000])),
('ra', 0): (N.array([ 35.76300000,  22.90639900,  12.47390000,   3.21097000,  13.62109900]),
             N.array([  0.00154085,   0.00967837,   0.04997175,   0.35581249,   0.00000000])),
('ra', 2): (N.array([ 35.21500000,  21.67000000,   7.91342000,   7.65078000,  13.54310000]),
             N.array([  0.00151227,   0.00894175,   0.03150250,   0.07460900,   0.00000000])),
('ac', 0): (N.array([ 35.65969800,  23.10319900,  12.59770000,   4.08655000,  13.52660000]),
             N.array([  0.00147273,   0.00912887,   0.04649750,   0.29254999,   0.00000000])),
('ac', 3): (N.array([ 35.17359900,  22.11120000,   8.19216000,   7.05545000,  13.46369900]),
             N.array([  0.00144922,   0.00853592,   0.03229675,   0.06486075,   0.00000000])),
('th', 0): (N.array([ 35.56449900,  23.42190000,  12.74730000,   4.80703000,  13.43140000]),
             N.array([  0.00140840,   0.00865510,   0.04457725,   0.24793049,   0.00000000])),
('th', 4): (N.array([ 35.10070000,  22.44179900,   9.78554000,   5.29444000,  13.37599900]),
             N.array([  0.00138763,   0.00811245,   0.03366525,   0.05988325,   0.00000000])),
('pa', 0): (N.array([ 35.88470100,  23.29480000,  14.18910000,   4.17287000,  13.42869900]),
             N.array([  0.00136938,   0.00853797,   0.04230875,   0.26312750,   0.00000000])),
('u', 0): (N.array([ 36.02280000,  23.41280000,  14.94910000,   4.18800000,  13.39660000]),
             N.array([  0.00132325,   0.00831325,   0.04023175,   0.25153250,   0.00000000])),
('u', 3): (N.array([ 35.57469900,  22.52590000,  12.21649900,   5.37073000,  13.30920000]),
             N.array([  0.00130120,   0.00780732,   0.03178700,   0.06584850,   0.00000000])),
('u', 4): (N.array([ 35.37149800,  22.53260000,  12.02910000,   4.79840000,  13.26709900]),
             N.array([  0.00129149,   0.00762632,   0.03143075,   0.05864550,   0.00000000])),
('u', 6): (N.array([ 34.85089900,  22.75840000,  14.00990000,   1.21457000,  13.16650000]),
             N.array([  0.00126770,   0.00722575,   0.03294175,   0.06300425,   0.00000000])),
('np', 0): (N.array([ 36.18740100,  23.59639900,  15.64020000,   4.18550000,  13.35730000]),
             N.array([  0.00127982,   0.00813490,   0.03840550,   0.24372700,   0.00000000])),
('np', 3): (N.array([ 35.70739700,  22.61299900,  12.98979900,   5.43227000,  13.25440000]),
             N.array([  0.00125581,   0.00759517,   0.03036225,   0.06373200,   0.00000000])),
('np', 4): (N.array([ 35.51030000,  22.57869900,  12.77660000,   4.92159000,  13.21159900]),
             N.array([  0.00124656,   0.00741567,   0.02987100,   0.05687550,   0.00000000])),
('np', 6): (N.array([ 35.01359900,  22.72860000,  14.38840000,   1.75669000,  13.11300000]),
             N.array([  0.00122452,   0.00702747,   0.03082500,   0.05664525,   0.00000000])),
('pu', 0): (N.array([ 36.52539800,  23.80830000,  16.77070000,   3.47947000,  13.38120000]),
             N.array([  0.00124846,   0.00815927,   0.03736375,   0.26494999,   0.00000000])),
('pu', 3): (N.array([ 35.84000000,  22.71690000,  13.58070000,   5.66016000,  13.19910000]),
             N.array([  0.00121234,   0.00740295,   0.02883275,   0.06099800,   0.00000000])),
('pu', 4): (N.array([ 35.64930000,  22.64600000,  13.35950000,   5.18831000,  13.15550000]),
             N.array([  0.00120355,   0.00722550,   0.02829000,   0.05457525,   0.00000000])),
('pu', 6): (N.array([ 35.17359900,  22.71810000,  14.76350000,   2.28678000,  13.05820000]),
             N.array([  0.00118301,   0.00684620,   0.02888250,   0.05232575,   0.00000000])),
('am', 0): (N.array([ 36.67060100,  24.09919900,  17.34149900,   3.49331000,  13.35920000]),
             N.array([  0.00120907,   0.00801617,   0.03578400,   0.25568249,   0.00000000])),
('cm', 0): (N.array([ 36.64880000,  24.40959900,  17.39900000,   4.21665000,  13.28870000]),
             N.array([  0.00116288,   0.00772492,   0.03358650,   0.22120849,   0.00000000])),
('bk', 0): (N.array([ 36.78810100,  24.77360000,  17.89189900,   4.23284000,  13.27540000]),
             N.array([  0.00112754,   0.00761547,   0.03223650,   0.21500749,   0.00000000])),
('cf', 0): (N.array([ 36.91849900,  25.19949900,  18.33169900,   4.24391000,  13.26740000]),
             N.array([  0.00109383,   0.00751937,   0.03101100,   0.20947025,   0.00000000])),
}


for (element, charge), value in atomic_structure_factors.items():
    if charge == 0:
       atomic_structure_factors[element] = value

