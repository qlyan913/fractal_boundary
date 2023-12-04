
SetFactory("OpenCASCADE");

// Define the unit square
Point(1) ={1, 1, 0, 1};
Point(2) = {1, 0, 0, 1};
Point(3) = {0, 0, 0, 1};
Point(4) = {0, 1, 0, 1};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
// Define the Koch snowflake
Point(5)={0.1111111111111111,1.0,0,1};
 Line(4) = {4, 5};
Point(6)={0.16666666666666666,1.0962250448649375,0,1};
 Line(5) = {5, 6};
Point(7)={0.2222222222222222,1.0,0,1};
 Line(6) = {6, 7};
Point(8)={0.3333333333333333,1.0,0,1};
 Line(7) = {7, 8};
Point(9)={0.3888888888888889,1.0962250448649375,0,1};
 Line(8) = {8, 9};
Point(10)={0.33333333333333337,1.192450089729875,0,1};
 Line(9) = {9, 10};
Point(11)={0.4444444444444445,1.192450089729875,0,1};
 Line(10) = {10, 11};
Point(12)={0.5,1.2886751345948129,0,1};
 Line(11) = {11, 12};
Point(13)={0.5555555555555556,1.1924500897298753,0,1};
 Line(12) = {12, 13};
Point(14)={0.6666666666666666,1.1924500897298753,0,1};
 Line(13) = {13, 14};
Point(15)={0.6111111111111112,1.0962250448649378,0,1};
 Line(14) = {14, 15};
Point(16)={0.6666666666666666,1.0,0,1};
 Line(15) = {15, 16};
Point(17)={0.7777777777777778,1.0,0,1};
 Line(16) = {16, 17};
Point(18)={0.8333333333333334,1.0962250448649375,0,1};
 Line(17) = {17, 18};
Point(19)={0.888888888888889,1.0,0,1};
 Line(18) = {18, 19};
Line(19) = {19,1};
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
Plane Surface(1) = {1};

Physical Curve("Right") = {1};
Physical Curve("Bottom") = {2};
Physical Curve("Left") = {3};
Physical Curve("Top")= {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

Physical Surface("sur")={1};