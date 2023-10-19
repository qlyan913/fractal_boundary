
SetFactory("OpenCASCADE");

// Define the unit square
Point(1) ={1, 1, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};
Point(3) = {0, 0, 0, 0.1};
Point(4) = {0, 1, 0, 0.1};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
// Define the Koch snowflake
Point(5)={0.3333333333333333,1.0,0,0.1};
 Line(4) = {4, 5};
Point(6)={0.3333333333333333,1.3333333333333333,0,0.1};
 Line(5) = {5, 6};
Point(7)={0.6666666666666666,1.3333333333333333,0,0.1};
 Line(6) = {6, 7};
Point(8)={0.6666666666666666,1.0,0,0.1};
 Line(7) = {7, 8};
Line(8) = {8,1};
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(1) = {1};

Physical Curve("Right") = {1};
Physical Curve("Bottom") = {2};
Physical Curve("Left") = {3};
Physical Curve("Top")= {4, 5, 6, 7, 8};

Physical Surface("sur")={1};
